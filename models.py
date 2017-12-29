import torch
import torch.nn as nn
def conv_block(in_dim,out_dim):
  return nn.Sequential(nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                       nn.BatchNorm2d(in_dim),
                       nn.LeakyReLU(True),
                       nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.BatchNorm2d(in_dim),
                       nn.Conv2d(in_dim,out_dim,kernel_size=1,stride=1,padding=0),
                       nn.BatchNorm2d(out_dim),
                       nn.ELU(True),
                       nn.AvgPool2d(kernel_size=2,stride=2))
def deconv_block(in_dim,out_dim):
  return nn.Sequential(#nn.ConvTranspose2d(in_dim,out_dim,2,2),
                       nn.Conv2d(in_dim,out_dim,kernel_size=3,stride=1,padding=1),
                       nn.BatchNorm2d(out_dim),
                       nn.LeakyReLU(True),
                       nn.Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding=1),
                       nn.BatchNorm2d(out_dim),
                       nn.ELU(True),
                       nn.Upsample(scale_factor=2)
                       )


class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.layer1=nn.Sequential(nn.Conv2d(1,3,3,1,1),nn.LeakyReLU(True))
        self.layer2=conv_block(3,64)
        self.layer3=conv_block(64,128)
        self.layer4=conv_block(128,256)
        self.layer5=conv_block(256,512)
        self.layer6=deconv_block(512,256)
        self.layer7=deconv_block(256,128)
        self.layer8=deconv_block(128,64)
        self.layer9=deconv_block(64,3)
        self.layer10=nn.Sequential(nn.Conv2d(3,3,3,1,1),nn.Tanh())
    def forward(self,input):
        o1=self.layer1(input)   #input : 1,128,128
        o2=self.layer2(o1)  #output : 64,64,64
        o3=self.layer3(o2)  #output : 32,32,128
        o4=self.layer4(o3)   # output : 16,16,256
        o5=self.layer5(o4)   #output : 8,8,512
        o6= self.layer6(o5) + o4  # output 16,16,256
        o7=self.layer7(o6)  + o3 #output 32,32,128
        o8=self.layer8(o7)  + o2 #output 64,64,64
        o9=self.layer9(o8)  + o1 #output 128,128,3
        o10=self.layer10(o9) #output 128,128,3
        return o10

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.layer1=conv_block(4,64)
        self.layer2=conv_block(64,128)
        self.layer3=nn.Sequential(nn.Conv2d(128,256,1),nn.LeakyReLU(True))
        self.layer4=nn.Sequential(nn.Conv2d(256,1,3),nn.Sigmoid())
    def forward(self,input1,input2):
        combined=torch.cat((input1,input2),1)
        out1=self.layer1(combined) # 64,64,64
        out2=self.layer2(out1)     # 128,32,32
        out3=self.layer3(out2)     # 256,32,32
        out4=self.layer4(out3)     # 1,30,30
        return out4
