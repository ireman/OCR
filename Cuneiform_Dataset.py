# Imports
import numpy as np
from PIL import Image
import torch
import pandas as pd
from torch.utils.data import Dataset


class CYRUS(Dataset):
    def __init__(self, data_df, transform=None, sign_to_label={}):
        self.annotations = data_df
        self.transform = transform
        self.sign_to_label = sign_to_label
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        #img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0].split("\\")[1])
        img_path = str(self.annotations.at[index,'image']).replace("\\", "/")
        #img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1] + '/'+ self.annotations.iloc[index, 0] )
        #image = np.asarray(io.imread(fname=img_path)).reshape(64,64,3).astype(np.float64)/255
        #image = io.imread(fname=img_path, as_gray=True)
        image = Image.open(img_path).convert('RGB')
        x = self.annotations.at[index,'x']
        y = self.annotations.at[index,'y']
        w = self.annotations.at[index,'w']
        h = self.annotations.at[index,'h']
        image = image.crop((x,y,w,h))
        image = image.resize((64, 64))
        try:
          image = np.asarray(image).reshape(64,64,3).astype(np.float64)/255
        except:
          #print(self.annotations.iloc[index, 0])
          print(img_path)
        #image = np.asarray(image).reshape(64,64,3).astype(np.float64)/255
        #image = ImageOps.grayscale(image)
        #print(self.annotations.iloc[index, 1])
        #print(self.annotations.iloc[index, :])
        try:
          y_label = torch.tensor(self.sign_to_label.get(self.annotations.at[index,'label']))
        except:
          y_label = torch.tensor(-1000)
          #print(self.annotations.iloc[index, 1])
        #print(y_label)
        if self.transform:
            image = self.transform(image)

        return (image, y_label)