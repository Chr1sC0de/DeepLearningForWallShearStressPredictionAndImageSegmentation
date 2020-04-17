import torch.nn as _nn
from myTorch.Models import Utils as _Utils
import torch
"""
https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/linknet.py
https://github.com/e-lab/pytorch-linknet
https://arxiv.org/pdf/1707.03718.pdf

"""


class LinkNet(_nn.Module):
    
    def __init__(
        self, n_classes=1, in_channels=4, filters=[64,128,256,512],
        output_activation=_nn.LeakyReLU(inplace=True)
    ):
        super(LinkNet, self).__init__()
        self.conv1 = _nn.Conv2d(in_channels, filters[0], kernel_size=7,stride=1,padding=3, bias=False)
        self.bn1 = _nn.BatchNorm2d(filters[0])
        self.relu = _nn.ReLU(inplace=True)
        self.maxpool = _nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.encoder1 = _Utils.Encoder(filters[0], filters[0], kernel_size=3,stride=1,padding=1)
        self.encoder2 = _Utils.Encoder(filters[0], filters[1], kernel_size=3,stride=2,padding=1)
        self.encoder3 = _Utils.Encoder(filters[1], filters[2], kernel_size=3, stride=2, padding=1)
        self.encoder4 = _Utils.Encoder(filters[2], filters[3], kernel_size=3, stride=2, padding=1)

        self.decoder1 = _Utils.Decoder(filters[0], filters[0], kernel_size=3, stride=1, padding=1, output_padding=0)
        self.decoder2 = _Utils.Decoder(filters[1], filters[0], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder3 = _Utils.Decoder(filters[2], filters[1], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder4 = _Utils.Decoder(filters[3], filters[2], kernel_size=3, stride=2, padding=1, output_padding=1)

        # Classifier
        self.tp_conv1 = _nn.Sequential(_nn.ConvTranspose2d(filters[0], filters[0]//2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                      _nn.BatchNorm2d(filters[0]//2),
                                      _nn.ReLU(inplace=True),)
        self.conv2 = _nn.Sequential(_nn.Conv2d(filters[0]//2, filters[0]//2, kernel_size=3, stride=2, padding=1),
                                _nn.BatchNorm2d(filters[0]//2),
                                _nn.ReLU(inplace=True),)
        self.tp_conv2 = _nn.ConvTranspose2d(filters[0]//2, n_classes, kernel_size=2, stride=2, padding=0)
        self.output_activation = output_activation

    def forward(self, x):
        # Initial block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder blocks
        d4 = e3 + self.decoder4(e4)
        d3 = e2 + self.decoder3(d4)
        d2 = e1 + self.decoder2(d3)
        d1 = x + self.decoder1(d2)

        # Classifier
        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)

        if self.output_activation is not None:
            y = self.output_activation(y)

        return y

if __name__ == "__main__":
    import torch as _torch
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()

    input_example = torch.randn(1,3,512,512)
    input_example = torch.tensor(input_example,dtype = torch.float32)

    model = LinkNet()
    output = model(input_example)

    writer.add_graph(output, input_example)
    writer.close()

    print('done')