import torch
from torch import nn
from torchvision import models


class ResNet(nn.Module):
    def __init__(self, num_classes=6):
        super(ResNet, self).__init__()
        # 定义残差网络中的残差块
        # 3 * 32 * 32 --> 64 *
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第一个残差块
        self.res_block1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.conv2,
            self.bn2,
            self.relu
        )

        # 第二个残差块
        self.res_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )

        # 全连接层
        self.fc = nn.Linear(8192, num_classes)

    def forward(self, x):
        # 通过第一个残差块
        x = self.res_block1(x)
        x = self.pool(x)  # 添加池化层

        # 通过第二个残差块
        x = self.res_block2(x)
        x = self.pool(x)  # 添加池化层
        # 展平操作应该在这里，确保x的尺寸是正确的
        x = x.view(x.shape[0], -1)

        # 通过全连接层
        out = self.fc(x)
        return out


if __name__ == '__main__':
    model = ResNet()
    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    print(out.shape)