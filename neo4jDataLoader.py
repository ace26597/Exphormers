import torch
from torch_geometric.data import HeteroData
from py2neo import Graph

import numpy as np
import openai
import os
import pickle

import sys
import json
import openai
#import aws secrets manager infrastructure
import boto3 
import requests
import pandas as pd
import numpy as np
import neo4j
import pandas as pd
from neo4j import GraphDatabase
import urllib.parse
import os,sys

def get_aws_secret_pws(pw_to_find):

    secret_name = "omealerts_pws/{}".format(pw_to_find)
    region_name = "us-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    # In this sample we only handle the specific exceptions for the 'GetSecretValue' API.
    # See https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
    # We rethrow the exception by default.

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
        return get_secret_value_response
    except ClientError as e:
        print(e)


currentdir = os.path.dirname(os.path.realpath('Jupyterlab/Ankur_Notebooks/Sumi_KG/Neo4j_DS.ipynb'))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(currentdir)

omealerts_kg_access_username = 'omealerts_kg_access'
omealerts_kg_access_pw = json.loads(get_aws_secret_pws('omealerts_kg_access').get('SecretString', False)).get('omealerts_kg_access', None)

host = "bolt://10.115.1.170:7687"
user = omealerts_kg_access_username
password = str(omealerts_kg_access_pw)
database='ome-alerts'
driver = GraphDatabase.driver(host,auth=(user, password))
db =driver.session(database=database)

openai.api_key = json.loads(get_aws_secret_pws('openai_api_key').get('SecretString'))['omealerts_pws/openai_api_key']
MODEL="gpt-4"

def unpack_res(r):
    res =  r['choices'][0]['message']['content']
    tokens = r['usage']['total_tokens']
    response = r
    return res,tokens,response

def run_query_df(query, params={}):
    with driver.session(database=database) as session:
        result = session.run(query, params)
        data = [record.data() for record in result]
        df = pd.json_normalize(data)
        return df


##Read cypher query results into Dataframe
def run_query(query):
        with driver.session(database=database) as session:
            result = session.run(query)
            print(query)
            return pd.DataFrame([r.values() for r in result], columns=result.keys())


# foo = run_query_df("""MATCH (n) RETURN (n) LIMIT 5""")
from neo4j import GraphDatabase
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
from neo4j import GraphDatabase
from sklearn.linear_model import LogisticRegression
import numpy as np

import pandas as pd
from time import sleep

class Neo4jHeteroGraphStore:
    def __init__(self, uri, user, password, database, data_dir='data', use_file_storage=True):
        self.graph = Graph(uri, name=database, auth=(user, password))
        self.data_dir = data_dir
        self.use_file_storage = use_file_storage
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def _save_data(self, data, filename):
        if filename.endswith('.json'):
            with open(os.path.join(self.data_dir, filename), 'w') as f:
                json.dump(data, f)
    
    def _load_data(self, filename):
        if filename.endswith('.json'):
            with open(os.path.join(self.data_dir, filename)) as f:
                return json.load(f)
        return None
    
    def _data_exists(self, filename):
        return os.path.exists(os.path.join(self.data_dir, filename))
    
    def fetch_nodes(self):
        node_data = {}
        node_types = self.graph.run("CALL db.labels()").data()
        for node_type in node_types:
            label = node_type["label"]
            filename = f'nodes_{label}.json'
            if self.use_file_storage and self._data_exists(filename):
                node_data[label] = self._load_data(filename)
            else:
                query = f"""
                    MATCH (n:{label})
                    RETURN id(n) AS id, n.sbert_embedding AS embedding, n.name AS name
                """
                results = self.graph.run(query).data()
                print(label,len(results))
                embeddings = [result['embedding'] for result in results]
                names = [result['name'] for result in results]
                node_data[label] = {'embeddings': embeddings, 'names': names}
                if self.use_file_storage:
                    self._save_data(node_data[label], filename)
        return node_data


    def fetch_relationships(self):
        edge_data = {}
        rel_types = self.graph.run("CALL db.relationshipTypes()").data()
        # This could be a class attribute
        self.rel_type_mapping = {rel_type['relationshipType']: i for i, rel_type in enumerate(self.graph.run("CALL db.relationshipTypes()").data())}


        for rel_type in rel_types:
            type_ = rel_type["relationshipType"]
            query = f"""
                MATCH ()-[r:{type_}]->()
                RETURN id(startNode(r)) AS source, id(endNode(r)) AS target, r.weight AS weight
            """
            results = self.graph.run(query).data()
            print(rel_type,len(results))
            edge_index = torch.tensor([(result['source'], result['target']) for result in results], dtype=torch.long).t().contiguous()
            weights = torch.tensor([result['weight'] for result in results], dtype=torch.float)
            # Assume rel_type_mapping is defined to map relationship types to integers
            rel_type_idx = torch.full((edge_index.size(1),), self.rel_type_mapping[type_], dtype=torch.long)
            edge_data[type_] = {'edge_index': edge_index, 'weights': weights, 'rel_type': rel_type_idx}
            if self.use_file_storage:
                    self._save_data(edge_data[type_], filename)
        return edge_data
        
    def to_pyg_hetero_data(self):
        hetero_data = HeteroData()
        
        node_data = self.fetch_nodes()
        for node_type, data in node_data.items():
            hetero_data[node_type].x = data['embeddings']  # Node features
            hetero_data[node_type].name = data['names']  # Node names as features
            
        edge_data = self.fetch_relationships()
        for rel_type, data in edge_data.items():
            hetero_data[rel_type].edge_index = data['edge_index']
            hetero_data[rel_type].edge_attr = torch.stack([data['weights'], data['rel_type']], dim=1)  # Stack weights and relationship types
        
        return hetero_data

def convert_Neo4j2pyG(database):
    neo4j_store = Neo4jHeteroGraphStore(uri=host, user=user, password=password, database=database)
    pyg_hetero_data = neo4j_store.to_pyg_hetero_data()
    print(pyg_hetero_data.metadata())
    return pyg_hetero_data

    
# # Access node names for a specific node type
# print(pyg_hetero_data['Article'].name)

# # Access relationship weights for a specific relationship type
# print(pyg_hetero_data['SENT_TO'].edge_attr)
