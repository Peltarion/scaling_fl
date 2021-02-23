"""Spooky Author Identification kaggle dataset.

https://www.kaggle.com/c/spooky-author-identification

In order to work, all files relative to spooky author dataset must be present
in subfolder spooky_author inside the main data folder which is mounted.
"""

from typing import Tuple

import pandas as pd

from ..text_classification import TextClassificationDataset


def train_test(train_path: str, test_path: str) -> Tuple[TextClassificationDataset,
                                                         TextClassificationDataset,
                                                         int]:
    # Hard code label map, from  author to float
    LABEL_MAP = {
        'EAP': 0.0,
        'HPL': 1.0,
        'MWS': 2.0,
    }

    df_train = pd.read_csv(train_path, index_col='id')
    df_test = pd.read_csv(test_path, index_col='id')
    trainset = TextClassificationDataset(
        texts=df_train['text'].tolist(),
        labels=df_train['author'].map(LABEL_MAP).tolist()
    )
    testset = TextClassificationDataset(
        texts=df_test['text'].tolist(),
        labels=df_test['author'].map(LABEL_MAP).tolist()
    )
    return trainset, testset, len(LABEL_MAP)
