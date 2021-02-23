"""Yelp-5 dataset."""

from typing import Tuple

import pandas as pd

from ..text_classification import TextClassificationDataset


def train_test(train_path: str, test_path: str) -> Tuple[TextClassificationDataset,
                                                         TextClassificationDataset,
                                                         int]:
    """Return the Yelp-5 train data, test data and number of stars."""
    train_data = pd.read_csv(train_path, header=None, names=['stars', 'text'])
    test_data = pd.read_csv(test_path, header=None, names=['stars', 'text'])
    num_labels = len(
        set(train_data['stars'].unique()).union(test_data['stars'].unique()))

    def dataset(data: pd.DataFrame) -> TextClassificationDataset:
        return TextClassificationDataset(
            texts=data['text'].tolist(), labels=(data['stars'] - 1).tolist())

    return dataset(train_data), dataset(test_data), num_labels
