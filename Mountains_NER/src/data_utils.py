import os
import ast
import pandas as pd
from typing import Tuple
from dataset import MountainsDataset
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast, DataCollatorForTokenClassification


def load_dataloaders(tokenizer:DistilBertTokenizerFast,
                     train_parameters:dict,
                     val_parameters:dict,
                     train_data_path:str,
                     val_data_path:str) -> Tuple[DataLoader, DataLoader]:
    """
    Loads data from s3 bucket and creates dataloaders.
    """
    train_file = os.path.join(train_data_path, "train_dataset.csv")
    val_file = os.path.join(val_data_path, "val_dataset.csv")

    train_df = pd.read_csv(train_file, converters={"marker": ast.literal_eval})
    val_df = pd.read_csv(val_file, converters={"marker": ast.literal_eval})

    train_dataset = MountainsDataset(train_df, tokenizer)
    val_dataset = MountainsDataset(val_df, tokenizer)

    # Function that bundles up batches
    collate_fn = DataCollatorForTokenClassification(tokenizer=tokenizer)

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, **train_parameters)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, **val_parameters)

    return train_dataloader, val_dataloader