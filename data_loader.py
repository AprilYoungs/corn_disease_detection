import torch.utils.data as data
import torch
from PIL import Image
import numpy as np
import pandas as pd
import time
import random
import os
import sys
import glob


def get_loader(transform, mode='train', batch_size=1, data_path='data'):
    """Return a data loader"""
    
    assert mode in ['train', 'valid'], "mode must be one of 'train' or 'valid'"
    if mode == 'train':
        img_folder = os.path.join(data_path, '2018_trainingset_20180905/AgriculturalDisease_trainingset/images/')
        notation_file = os.path.join(data_path, '2018_trainingset_20180905/AgriculturalDisease_trainingset/AgriculturalDisease_train_annotations.json')
    if mode == 'valid':
        img_folder = os.path.join(data_path, '2018_validationset_20180905/AgriculturalDisease_validationset/images/')
        notation_file = os.path.join(data_path, '2018_validationset_20180905/AgriculturalDisease_validationset/AgriculturalDisease_validation_annotations.json')
    
    dataset = ClassifySet(transform, 1000, notation_file, img_folder, batch_size)
    
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=dataset.batch_size,
                                  shuffle=True)
    
    return data_loader


def get_encoder_loader(transform, encoder, device, mode='train', batch_size=1, data_path='data', file=None):
    """Return a data loader that contain only embedding images"""
    
    assert mode in ['train', 'valid'], "mode must be one of 'train' or 'valid'"
    if mode == 'train':
        img_folder = os.path.join(data_path, '2018_trainingset_20180905/AgriculturalDisease_trainingset/images/')
        notation_file = os.path.join(data_path, '2018_trainingset_20180905/AgriculturalDisease_trainingset/AgriculturalDisease_train_annotations.json')
    if mode == 'valid':
        img_folder = os.path.join(data_path, '2018_validationset_20180905/AgriculturalDisease_validationset/images/')
        notation_file = os.path.join(data_path, '2018_validationset_20180905/AgriculturalDisease_validationset/AgriculturalDisease_validation_annotations.json')
    
    dataset = EncoderSet(transform, encoder, device, notation_file, img_folder, batch_size, file)
    

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=dataset.batch_size,
                                  shuffle=True)
    
    return data_loader

def get_encoder_loader_fold(transform, encoder, device, fileFold, load=False, mode='train', batch_size=1, data_path='data'):
    """Return a data loader that contain only embedding images,
       embedding files save in separate files,
       load: should load the images?
    """
    
    assert mode in ['train', 'valid'], "mode must be one of 'train' or 'valid'"
    if mode == 'train':
        img_folder = os.path.join(data_path, '2018_trainingset_20180905/AgriculturalDisease_trainingset/images/')
        notation_file = os.path.join(data_path, '2018_trainingset_20180905/AgriculturalDisease_trainingset/AgriculturalDisease_train_annotations.json')
    if mode == 'valid':
        img_folder = os.path.join(data_path, '2018_validationset_20180905/AgriculturalDisease_validationset/images/')
        notation_file = os.path.join(data_path, '2018_validationset_20180905/AgriculturalDisease_validationset/AgriculturalDisease_validation_annotations.json')
    
    dataset = EncoderSeparateSet(transform, encoder, device, notation_file, img_folder, batch_size, fileFold, load=load)
    
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=dataset.batch_size,
                                  shuffle=True)
    
    return data_loader
    

class EncoderSeparateSet(data.Dataset):
    """save the embedding data to separate files
    """
    def __init__(self, transform, encoder, device, json_file, root_dir, batch_size, fileFold, load=False):
        self.refer = pd.read_json(json_file)
        self.root_dir = root_dir
        self.transform = transform
        self.encoder = encoder
        self.batch_size = batch_size
        self.fileFold = fileFold
  
        
        if load == True:
            self.embeddings = []
            self.encode_data(device)
            self.indexs = self.refer.copy()
            self.indexs['embed'] = self.embeddings
            self.indexs.to_csv(os.path.join(self.fileFold, 'indexs.csv'), index=False)
        else:
            self.indexs = pd.read_csv(os.path.join(self.fileFold, 'indexs.csv'))

        
    def encode_data(self, device):
        total = len(self.refer)
        start_time = time.time()
        for idx in range(len(self.refer)):
            image_path = os.path.join(self.root_dir, self.refer.iloc[idx, 1])
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image).unsqueeze(0).to(device)
            embed = self.encoder(image)
            embed_path = os.path.join(self.fileFold, '{}.pkl'.format(idx))
            torch.save(embed, embed_path)
            self.embeddings.append(embed_path)
            spent_time = time.time() - start_time
            spent_time = "%d:%2.2f" % (spent_time//60, spent_time%60)
            print('\r'+"encoding {}/{}->{:.2f}%, spent_time:{}".format(idx, total, 100*idx/total, spent_time), end='')
            sys.stdout.flush()
            
                               
    def __len__(self):
        return len(self.indexs)
    
    def __getitem__(self, idx):
        embed = self.indexs['embed'][idx]
        embed = torch.load(embed)
        return embed, self.indexs['disease_class'][idx]
    
    def get_train_indices(self):
        indices = list(np.random.choice(np.arange(len(self.refer)), size=self.batch_size))
        return indices
    

class EncoderSet(data.Dataset):
    def __init__(self, transforms, encoder, device, json_file, root_dir, batch_size, file=None):
        self.refer = pd.read_json(json_file)
        self.root_dir = root_dir
        self.transform = transforms
        self.encoder = encoder
        self.batch_size = batch_size
        
        if file == None:
            self.embeddings = []
            self.encode_data(device)
        else:
            self.embeddings = torch.load(file)
        
    def encode_data(self, device):
        total = len(self.refer)
        start_time = time.time()
        for idx in range(len(self.refer)):
            image_path = os.path.join(self.root_dir, self.refer.iloc[idx, 1])
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image).unsqueeze(0).to(device)
            embed = self.encoder(image)
            self.embeddings.append(embed)
            spent_time = time.time() - start_time
            spent_time = "%d:%2.2f" % (spent_time//60, spent_time%60)
            print('\r'+"encoding {}/{}->{:.2f}%, spent_time:{}".format(idx, total, 100*idx/total, spent_time), end='')
            sys.stdout.flush()
            
    def save_to(self, file):
        """Save the encoded tensor to a specific file"""
        torch.save(self.embeddings, file)
        
    def __len__(self):
        return len(self.refer)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.refer.iloc[idx, 0]
    
    def get_train_indices(self):
        indices = list(np.random.choice(np.arange(len(self.refer)), size=self.batch_size))
        return indices
    
    
    
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
        
        
        return self.transform(image), self.refer.iloc[idx, 0]
    
    def get_train_indices(self):
        indices = list(np.random.choice(np.arange(len(self.refer)), size=self.batch_size))
        return indices
    
def get_test_data(transform, encoder, device):
    """ Return the datas to generate test result """
    path = './data/2018_testA_20180905/AgriculturalDisease_testA/images/*'
    images = glob.glob(path)
    start_time = time.time()
    total = len(images)
    dataSet = []
    for idx in range(total):
        image_path = images[idx]
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        embed = encoder(image)
        img_id = image_path.split('/')[-1]
        dataSet.append((embed, img_id))
        spent_time = time.time() - start_time
        spent_time = "%d:%2.2f" % (spent_time//60, spent_time%60)
        print('\r'+"encoding {}/{}->{:.2f}%, spent_time:{}".format(idx, total, 100*idx/total, spent_time), end='')
        sys.stdout.flush()
    return dataSet