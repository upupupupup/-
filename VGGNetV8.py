import torch
import torch.nn as nn

# 定义基本的残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        # 注意确保第二个卷积层的输出通道数等于输入通道数
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入通道数与输出通道数不相等，则需要使用额外的卷积层调整维度
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # 如果输入通道数与输出通道数不相等，则使用downsample调整维度
        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity  # 残差连接
        out = self.relu(out)
        return out

# 定义带有残差块的VGG11网络结构
class VGG11ResNet(nn.Module):
    def __init__(self, num_classes=6):
        super(VGG11ResNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(256, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    # 创建带有残差块的VGG11模型实例
    vgg11_resnet = VGG11ResNet(num_classes=6)

    x = torch.randn(1, 3, 32, 32)
    y = vgg11_resnet(x)
    print(y.size())
