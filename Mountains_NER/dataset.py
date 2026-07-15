import pandas as pd
from typing import List, Dict
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast


class MountainsDataset(Dataset):

    def __init__(self, df:pd.DataFrame, tokenizer:DistilBertTokenizerFast):
        self.len = len(df)
        self.data = df
        self.tokenizer = tokenizer


    def __align_labels(self, offset_mappings: List[tuple], spans: List[tuple]) -> List[int]:
        labels = []
        for start, end in offset_mappings:
            # Special tokens ([CLS], [SEP]) have (0, 0) offsets
            if start == 0 and end == 0:
                labels.append(-100)  # PyTorch ignore index
                continue

            token_label = 0  # Default to 'O' (Outside)

            for span_start, span_end in spans:
                if start >= span_start and end <= span_end:
                    # If this token is the starting token, it's B-MOUNTAIN (1)
                    if start == span_start:
                        token_label = 1
                    # If it's a continuing token or a subword, it's I-MOUNTAIN (2)
                    else:
                        token_label = 2
                    break

            labels.append(token_label)
        return labels


    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        record = self.data.iloc[index]
        text = record.text
        spans = record.marker

        inputs = self.tokenizer(
            text,
            truncation=True,
            return_offsets_mapping=True
        )

        offsets = inputs['offset_mapping']
        labels = self.__align_labels(offsets, spans)

        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': labels
        }


    def __len__(self):
        return self.len