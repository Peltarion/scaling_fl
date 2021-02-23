"""AG news dataset."""

from typing import Tuple

import pandas as pd

from ..text_classification import TextClassificationDataset


def train_test(train_path: str,
               test_path: str,
               text_column: str = "Description") -> Tuple[TextClassificationDataset,
                                                          TextClassificationDataset,
                                                          int]:
    """Return the AG news train data, test data and number of classes.

    text_column : str
        The string identifying from which column to fetch the actual text to
        use. Either "Description" or "Title"
    """
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    num_labels = len(
        set(df_train['Class Index'].unique()).union(df_test['Class Index'].unique()))

    def dataset(examples: pd.DataFrame) -> TextClassificationDataset:
        return TextClassificationDataset(
            texts=examples[text_column].tolist(),
            labels=(examples["Class Index"].astype(float) - 1.0).tolist()
        )

    return dataset(df_train), dataset(df_test), num_labels
