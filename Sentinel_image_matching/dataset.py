import os
import cv2
import pandas as pd
import kornia as K
from torch.utils.data import Dataset
from kornia.image import image_to_tensor


class SatelliteMatchingDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Load Image A
        img_a_path = os.path.join(self.root_dir, self.annotations.iloc[idx]['image_A_path'])
        img_a = cv2.imread(img_a_path)
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)

        # Load Image B
        img_b_path = os.path.join(self.root_dir, self.annotations.iloc[idx]['image_B_path'])
        img_b = cv2.imread(img_b_path)
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

        # Convert to Tensors and Grayscale (LoFTR requirement)
        tensor_a = image_to_tensor(img_a).float() / 255.0
        tensor_b = image_to_tensor(img_b).float() / 255.0

        tensor_a = K.color.rgb_to_grayscale(tensor_a)
        tensor_b = K.color.rgb_to_grayscale(tensor_b)

        return {'image0': tensor_a, 'image1': tensor_b}