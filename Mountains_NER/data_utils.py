import os
import ast
import pandas as pd
from dotenv import load_dotenv
from dataset import MountainsDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from config import TRAIN_PARAMETERS, VAL_PARAMETERS
from transformers import DistilBertTokenizerFast, DataCollatorForTokenClassification

load_dotenv()


def load_datasets(tokenizer:DistilBertTokenizerFast):
    df = pd.read_csv(os.environ["TRAIN_DATA_PATH"], converters={"marker": ast.literal_eval})

    ### Sampling train and validation data ###
    train_data, val_data = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)

    train_dataset = MountainsDataset(train_data, tokenizer)
    val_dataset = MountainsDataset(val_data, tokenizer)

    return train_dataset, val_dataset


def load_dataloaders(tokenizer:DistilBertTokenizerFast):
    train_dataset, val_dataset = load_datasets(tokenizer)
    collate_fn = DataCollatorForTokenClassification(tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, **TRAIN_PARAMETERS)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, **VAL_PARAMETERS)

    return train_dataloader, val_dataloader