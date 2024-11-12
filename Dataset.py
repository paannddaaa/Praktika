import os, sys
import random
from csv1 import df
import seaborn as sns
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image
import pandas as pd
from torchvision.utils import make_grid

class CelebaTripletDataset(Dataset):
    def __init__(self, path2df, path2root, transforms=None):
        self.path2root = path2root
        self.full_df = pd.read_csv(path2df)
        self.unique_ids = [id for id in self.full_df['person_id'] if
                           len(self.full_df[self.full_df['person_id'] == id]) > 1]
        self.transforms = transforms

    def __len__(self):
        return len(self.unique_ids)

    def __getitem__(self, idx):
        person_id = self.unique_ids[idx]
        base_anchor, pos_anchor = df[df['person_id'] == person_id]['filepath'].sample(2)
        neg_anchor = df[df['person_id'] != person_id]['filepath'].sample(1).item()
        base_anchor = read_image(os.path.join(self.path2root, base_anchor))
        pos_anchor = read_image(os.path.join(self.path2root, pos_anchor))
        neg_anchor = read_image(os.path.join(self.path2root, neg_anchor))
        return {"person_id": person_id,
                "base_anchor": base_anchor,
                "pos_anchor": pos_anchor,
                "neg_anchor": neg_anchor}

    def plot_examples(self, idx):
        return make_grid(torch.stack((self.__getitem__(idx)['base_anchor'], self.__getitem__(idx)['pos_anchor'],
                                      self.__getitem__(idx)['neg_anchor']))).permute(1, 2, 0)

dataset = CelebaTripletDataset(r"D:\Celeba\CelebaTriplets.csv",
                               r"D:\Celeba\img_align_celeba\img_align_celeba",
                               None)

import matplotlib.pyplot as plt
plt.imshow(dataset.plot_examples(0))

if __name__ == '__main__':
        dataset()