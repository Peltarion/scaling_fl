import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from .split import CollateFnInput, CollateFnOutput


class TextClassificationDataset(Dataset):

    def __init__(self, texts, labels):
        assert len(texts) == len(labels)
        self.texts = texts
        self.labels = labels

    def __getitem__(self, idx):
        item = self.texts[idx], self.labels[idx]
        return list(zip(*item)) if isinstance(idx, slice) else item

    def __len__(self):
        return len(self.texts)


class TextClassificationCollator:
    """Collator for text classification which performs tokenization.

    Args:
        tokenizer: Tokenizer
        seq_len: Sequence length to use for each batch
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, seq_len: int):
        self._seq_len = seq_len
        self._tokenizer = tokenizer

    def __call__(self, batch: CollateFnInput) -> CollateFnOutput:
        """Collate batch and tokenize text."""
        text, labels = zip(*batch)
        tokens = self._tokenizer.batch_encode_plus(
            text,
            max_length=self._seq_len,
            pad_to_max_length=True,
            truncation=True,
            return_tensors='pt'
        )['input_ids']
        return tokens, torch.tensor(labels, dtype=torch.long)
