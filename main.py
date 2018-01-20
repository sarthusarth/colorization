from models import Discriminator,Generator
from inputs import dataset
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

lr=1e-4
betas=(0.9,0.99)
batch_size=10
path= "/input/"
g_loss=[]   # storing Generator loss
d_loss=[]   # storing Discriminator loss
Epochs=2

G=Generator()
D=Discriminator()

g_optimizer=optim.Adam(G.parameters(),lr=lr,betas=betas)
d_optimizer=optim.Adam(D.parameters(),lr=lr,betas=betas)

criteria=nn.BCELoss(size_average=False) #loss function
#print "1"
dt=dataset(path)
#print "2"
# Training Generator and Discriminator

for i in range(Epochs):
    t1=datetime.now()
    # Discriminator train :
    for ix in range(1):
        D.zero_grad()
        bw,clr=dt.batch(batch_size)
        out=D(Variable(clr),Variable(bw))
        loss=criteria(out,Variable(torch.ones(batch_size,1,30,30)))
        loss.backward()
        d_loss.append(loss.data.numpy()[0])
        fake_images=G(Variable(bw)).detach()
        fake_out=D(fake_images,Variable(bw))
        loss=criteria(fake_out,Variable(torch.zeros(batch_size,1,30,30)))
        d_loss[i]+=loss.data.numpy()[0]
        loss.backward()
        d_optimizer.step()

    #Generator Train :
    for ix in range(1):
        G.zero_grad()
        bw,clr=dt.batch(batch_size)
        result=D(G(Variable(bw)),Variable(bw))
        loss=criteria(result,Variable(torch.ones(batch_size,1,30,30)))
        g_loss.append(loss.data.numpy()[0])
        loss.backward()
        g_optimizer.step()
    t2=datetime.now()
    print i,"->",t2-t1
    t1=datetime.now()
plt.plot(g_loss)
plt.plot(d_loss)
plt.legend(["Generator","Discriminator"])
plt.savefig("/output/fig.png")
