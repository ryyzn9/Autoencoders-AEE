import torch
import torch.nn.functional as F
from torch import nn 




class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,kernel_size=(3,3),padding ="same")
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(2,2))
        self.conv2 = nn.Conv2d(16,8,kernel_size=(3,3),padding ='same')
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(2,2))
        self.conv3 = nn.Conv2d(8,8,kernel_size=(2,2),padding="same")
        self.maxpooling3 = nn.MaxPool2d(kernel_size=(2,2),padding=(1,1))
        self.relu = nn.ReLU()
    def forward(self,x):
        #here x is the input 
        hidden1 = self.maxpooling1(self.relu(self.conv1(x)))
        hidden2 = self.maxpooling2(self.relu(self.conv2(hidden1)))
        encoded = self.maxpooling3(self.relu(self.conv3(hidden2)))
        return encoded




class Decoder(nn.Module):#1
    def __init__(self)->None:
        super().__init__()
        self.conv1 = nn.Conv2d(8,8,kernel_size=(3,3),padding ='same')
        self.up1 = nn.Upsample(scale_factor=(2,2))
        self.conv2 = nn.Conv2d(8,8,kernel_size=(3,3),padding='same')
        self.up2 = nn.Upsample(scale_factor=(2,2))
        self.conv3 = nn.Conv2d(8,16,kernel_size=(3,3))
        self.up3 = nn.Upsample(scale_factor=(2,2))
        self.conv4 = nn.Conv2d(16,1,kernel_size=(3,3),padding='same')
        self.relu = nn.ReLU()
        self.Sig = nn.Sigmoid()
    def forward(self,input):
        h1 = self.up1(self.relu(self.conv1(input)))
        h2 = self.up2(self.relu(self.conv2(h1)))
        h3 = self.up3(self.relu(self.conv3(h2)))
        decode = self.Sig(self.conv4(h3))
        return decode



class Autoencoder(nn.Module):
    def __init__(self,super_resolution=False)->None:
        super().__init__()
        if not super_resolution:
            self.encode = Encoder()
        else:
            self.encode = SuperResolutionEncoder()
        self.decode = Decoder()        
    def forward(self,x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded
      
class SuperResolutionEncoder(nn.Module):
    def __init__(self)->None:
        super().__init__()
        return None