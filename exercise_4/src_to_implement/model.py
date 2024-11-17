import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv = nn.Conv2d(3, 64, 7, 2, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.rel = nn.ReLU()
        self.max_pool = nn.MaxPool2d(3, 2)
        self.res_block1 = ResBlock(64, 64, 1)
        self.res_block2 = ResBlock(64, 128, 2)
        self.res_block3 = ResBlock(128, 256, 2)
        self.res_block4 = ResBlock(256, 512, 2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        output = self.conv(input_tensor)
        output = self.bn(output)
        output = self.rel(output)
        output = self.max_pool(output)
        output = self.res_block1(output)
        output = self.res_block2(output)
        output = self.res_block3(output)
        output = self.res_block4(output)
        output = self.global_avg_pool(output)
        output = self.flatten(output)
        output = self.fc(output)
        output = self.sigmoid(output)
        return output


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.rel1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.rel2 = nn.ReLU()

        self.conv1_1 = nn.Conv2d(in_channel, out_channel, 1, stride)

    def forward(self, input_tensor):
        in_tensor = self.conv1_1(input_tensor)
        # in_tensor = self.bn2(in_tensor)
        output = self.conv1(input_tensor)
        output = self.bn1(output)
        output = self.rel1(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output += in_tensor
        output = self.rel2(output)
        return output
