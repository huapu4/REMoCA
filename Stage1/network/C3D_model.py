# -*- coding: utf-8 -*-
"""
@Time    : 2021/10/28
@Author  : Lin Zhenzhe, Zhang Shuyi
"""
import torch
import torch.nn as nn


class C3D(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, num_classes):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm3a = nn.BatchNorm3d(128)
        self.conv3b = nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm3b = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm4a = nn.BatchNorm3d(256)
        self.conv4b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm4b = nn.BatchNorm3d(256)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm5a = nn.BatchNorm3d(512)
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm5b = nn.BatchNorm3d(512)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(32768, 4096)
        self.fc7 = nn.Linear(4096, 256)
        self.fc8 = nn.Linear(256, 16)
        self.fc9 = nn.Linear(16, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.__init_weight()

    def forward(self, x):
        # print ('1:',x.size())
        x = self.relu(self.norm1(self.conv1(x)))
        # print ('2:',x.size())
        x = self.pool1(x)
        # print ('3:',x.size())

        x = self.relu(self.norm2(self.conv2(x)))
        # print ('4:',x.size())
        x = self.pool2(x)
        # print ('5:',x.size())

        x = self.relu(self.norm3a(self.conv3a(x)))
        # print ('6:',x.size())
        x = self.relu(self.norm3b(self.conv3b(x)))
        # print ('7:',x.size())
        x = self.pool3(x)
        # print ('8:',x.size())

        x = self.relu(self.norm4a(self.conv4a(x)))
        # print ('9:',x.size())
        x = self.relu(self.norm4b(self.conv4b(x)))
        # print ('10:',x.size())
        x = self.pool4(x)
        # print ('11:',x.size())

        x = self.relu(self.norm5a(self.conv5a(x)))
        # print ('12:',x.size())
        x = self.relu(self.norm5b(self.conv5b(x)))
        # print ('13:',x.size())
        x = self.pool5(x)
        # print ('14:',x.size())
        # x = x.view(-1, 8192)

        x = x.view(-1, 32768)
        # print ('15:',x.size())
        x = self.relu(self.fc6(x))
        # print ('16:',x.size())
        # x = self.dropout(x)
        x = self.relu(self.fc7(x))
        # x = self.dropout(x)

        x = self.fc8(x)
        logits = self.fc9(x)
        # print ('17:',logits.size())
        return logits

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [model.conv1, model.conv2, model.conv3a, model.conv3b, model.conv4a, model.conv4b,
         model.conv5a, model.conv5b, model.fc6, model.fc7]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc8]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


if __name__ == "__main__":
    inputs = torch.rand(1, 3, 16, 112, 112)
    net = C3D(num_classes=2, pretrained=True)

    outputs = net.forward(inputs)
    print(outputs.size())
