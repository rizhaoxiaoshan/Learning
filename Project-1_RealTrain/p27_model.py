import torch.nn as nn
import torch
import torchvision

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),#结成一串
            nn.Linear(64,10) #输入-->输出
        )

    def forward(self,x):
        x=self.model(x)
        return x

if __name__=='__main__':
     tudui=Tudui()
     input=torch.ones((64,3,32,32))
     output=tudui(input)
     print(output.shape)