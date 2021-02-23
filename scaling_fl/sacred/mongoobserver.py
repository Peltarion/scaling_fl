import os

from sacred.observers import MongoObserver

from .resumable_mongo_observer import ResumableMongoObserver


def observer_from_env(resume_key: str = None):
    user = os.environ['SACRED_USER']
    password = os.environ['SACRED_PASSWORD']
    database = os.environ['SACRED_DATABASE']
    host = os.environ['SACRED_HOST']
    url = (
        f'mongodb+srv://{user}:{password}@{host}/{database}?retryWrites=true&'
        'w=majority'
    )
    if resume_key is None:
        return MongoObserver(url, db_name=database)
    return ResumableMongoObserver(resume_key, url, db_name=database)
