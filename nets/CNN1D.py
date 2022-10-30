import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

# 搭建神经网络
class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        # 1X800-64X800  通道X信号尺寸
        self.conv1 = nn.Conv1d(in_channels=1,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm1d(num_features=64)
        # 64X800-64X400  通道X信号尺寸
        self.pool1=nn.MaxPool1d(kernel_size=2,stride=2,padding=0)

        # 64X400-256X400  通道X信号尺寸
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=256)
        # 256X400-256X200  通道X信号尺寸
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # 256X200-512X200  通道X信号尺寸
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=512)
        # 512X200-512X100  通道X信号尺寸
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        # 100-----13
        self.fc1 = nn.Linear(in_features=100, out_features=13, bias=True)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.fc1(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Model=CNN1D() #实例化网络模型
    wang_DS_RGB=Model.to(device) #将模型转移到cuda上
    input=torch.ones((64,1,800)) #生成一个batchsize为64的，通道数为1，宽度为2048的信号
    input=input.to(device) #将数据转移到cuda上
    output=Model(input) #将输入喂入网络中进行处理
    print(output.shape)
    summary(Model,input_size=(1,800)) #输入一个通道为1的宽度为2048，并展示出网络模型结构和参数
