import os
from torch.utils.data import DataLoader
from dataset import SatelliteMatchingDataset


def load_dataloader(dataset_dir: str, batch_size: int) -> DataLoader:
    csv_path = os.path.join(dataset_dir, "dataset_labels.csv")
    dataset = SatelliteMatchingDataset(csv_file=csv_path, root_dir=dataset_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader