import os
from pymongo import MongoClient


def database_from_env():
    user = os.environ['SACRED_USER']
    password = os.environ['SACRED_PASSWORD']
    database = os.environ['SACRED_DATABASE']
    host = os.environ['SACRED_HOST']
    url = (
        f'mongodb+srv://{user}:{password}@{host}/{database}?retryWrites=true'
        '&w=majority'
    )
    client = MongoClient(url, serverSelectionTimeoutMS=10)
    return client[database]
