import numpy as np
import pandas as pd

from PIL import Image, ImageEnhance
import torchvision.transforms as transforms

import os
from tqdm import tqdm 

data_dir = "../data/"
train_dir = '../data/train/hands'
test_dir = '../data/test/hands'

T = pd.read_csv('../data/train/hands/hands_labels.csv')
V = pd.read_csv('../data/test/hands/hands_labels.csv')

val_transform = transforms.Compose([
    transforms.Scale(299, Image.LANCZOS),
    transforms.CenterCrop(299)
])

val_size = len(V)

for i, row in tqdm(V.iterrows()):
    # get image
    file_path = os.path.join(test_dir,row.filename)
    image = Image.open(file_path)
    
    # transform it
    image = val_transform(image)
    
    # save
    save_path = os.path.join(data_dir, 'test/', row.filename)
    image.save(save_path, 'jpeg')





train_transform = transforms.Compose([
    transforms.Scale(384, Image.LANCZOS),
    transforms.RandomCrop(299),
    transforms.RandomHorizontalFlip(),
])

# resize RGB images
for i, row in tqdm(T.iterrows()):
    # get image
    file_path = os.path.join(train_dir,row.filename)
    image = Image.open(file_path)
    
 
    image = train_transform(image)
    
    # save
    
    save_path = os.path.join(data_dir, 'train/',  row.filename)
    image.save(save_path, 'jpeg')