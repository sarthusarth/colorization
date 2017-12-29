import os
import cv2
import numpy as np
import torch
def np_images(path,size): # Path to the folder containing images
    x,y=[],[]  # c contains b/w images and y contains colored images
    images=os.listdir(path)[:100]
    for img in images:
        im=cv2.resize(cv2.imread(path+img),(128,128))
        im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        y.append(im)
        im=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
        im=np.reshape(im,(1,128,128))
        x.append(im)
    y=np.array(y).transpose((0,3,1,2)) #Pytorch requires channel first for CNN
    x=np.array(x)
    return x/255.0 , y/255.0

class dataset():
    def __init__(self,path,size=128):
        self.X,self.Y=np_images(path,size)
        self.len=self.X.shape[0]
    def batch(self,batch_size):
        ch=np.random.choice(range(self.len),batch_size,replace=False)
        return torch.FloatTensor(self.X[ch]),torch.FloatTensor(self.Y[ch])
