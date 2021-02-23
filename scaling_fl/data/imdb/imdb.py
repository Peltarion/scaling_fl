"""The old classic IMDB sentiment analysis dataset."""
from typing import Tuple

from torchtext import data, datasets

from ..text_classification import TextClassificationDataset


def train_test() -> Tuple[TextClassificationDataset, TextClassificationDataset, int]:
    """Return the training dataset, test dataset and number of labels."""
    train_examples, test_examples = datasets.IMDB.splits(
        text_field=data.Field(lower=False, sequential=False),
        label_field=data.Field(sequential=False, is_target=True)
    )

    def dataset(examples: data.dataset.Dataset) -> TextClassificationDataset:
        return TextClassificationDataset(
            texts=[example.text for example in examples],
            labels=[float(example.label == 'pos') for example in examples]
        )

    return dataset(train_examples), dataset(test_examples), 2
