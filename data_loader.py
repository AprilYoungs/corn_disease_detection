import torch.utils.data as data
import torch
from PIL import Image
import numpy as np
import pandas as pd
import random
import os


def get_loader(transform, class_size=61, mode='train', batch_size=1, data_path='data'):
    """Return a data loader"""
    
    assert mode in ['train', 'test'], "mode must be one of 'train' or 'test'"
    if mode == 'train':
        img_folder = os.path.join(data_path, '2018_trainingset_20180905/AgriculturalDisease_trainingset/images/')
        notation_file = os.path.join(data_path, '2018_trainingset_20180905/AgriculturalDisease_trainingset/AgriculturalDisease_train_annotations.json')
    if mode == 'test':
        img_folder = os.path.join(data_path, '2018_validationset_20180905/AgriculturalDisease_validationset/images/')
        notation_file = os.path.join(data_path, '2018_validationset_20180905/AgriculturalDisease_validationset/AgriculturalDisease_validation_annotations.json')
    
    dataset = ClassifySet(transform, class_size, notation_file, img_folder, batch_size)
    
#     if mode == 'train':
#         indices = dataset.get_train_indices()
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=dataset.batch_size,
                                  shuffle=True)
    
    return data_loader
    
    
    
class ClassifySet(data.Dataset):
    
    def __init__(self, transforms, class_size, json_file, root_dir, batch_size):
        self.refer = pd.read_json(json_file)
        self.root_dir = root_dir
        self.transform = transforms
        self.batch_size = batch_size
        self.class_size = class_size
        
    def __len__(self):
        return len(self.refer)
    
    def __getitem__(self, idx):
        
        image_path = os.path.join(self.root_dir, self.refer.iloc[idx, 1])
        image = Image.open(image_path).convert("RGB")
        
#         y = np.zeros(self.class_size)
#         y[self.refer.iloc[idx, 0]] = 1
        
#         y = torch.from_numpy(y)
        
        return self.transform(image), self.refer.iloc[idx, 0]
    
    def get_train_indices(self):
        indices = list(np.random.choice(np.arange(len(self.refer)), size=self.batch_size))
        return indices
    