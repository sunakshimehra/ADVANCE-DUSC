from google.colab import drive
drive.mount('/content/drive')

imagefolderpath = 'IMAGE'


import torch
import torchvision.models as models
from tqdm import tqdm
swin = models.swin_t()

import os
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn

swin.head = nn.Linear(768, 10)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

swin.to(device)
swin

allimagenames = os.listdir(imagefolderpath)
allimagetensor = []
transformation1 = transforms.ToTensor()
transformation2 = transforms.Resize((256,256))


for name in allimagenames:
    
    img = Image.open(imagefolderpath + '/' + name)
    img = transformation1(img)
    img = transformation2(img)
    image_without_alpha = img[:3,:,:]
    
    allimagetensor.append(image_without_alpha)
print(len(allimagetensor))

allimagetensor2 = torch.stack(allimagetensor)
outputs = []
for i in tqdm(allimagetensor2):
  i = i[None, :, :, :]
  i = i.to(device)
  output = swin(i)
  outputs.append(output.detach())
  
  
output10 = torch.stack(outputs)
print(output10.size())

targets = torch.Tensor(output10).clone().detach()
print(targets)

import numpy as np 
probability = [];
            
probability = targets

a_file = open("swin_scores.txt", "w")
for row in probability:
    row = row.cpu().numpy()
    np.savetxt(a_file, row)
    

a_file.close()


