import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import functools
import argparse


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

class Estimator(nn.Module):
    def __init__(self, args):
        super(Estimator, self).__init__()

        in_nc=3
        nf=args.n_feats//2
        num_blocks=3
        self.ksize = args.kernel_size

        basic_block = functools.partial(ResidualBlock_noBN, nf=nf)

        self.head = nn.Sequential(
            nn.Conv2d(in_nc, nf, 5, 1, 2),
            make_layer(basic_block, num_blocks)
        )
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 0),
            nn.ReLU(inplace=True),
        )

        self.tail = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 0),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(nf, self.ksize ** 2, 1, 1, 0),
            nn.Softmax(1),
        )

    def forward(self, LR):
        f = self.head(LR)
        f = self.body(f)
        f = self.tail(f)

        return f.view(f.shape[0], 1, self.ksize, self.ksize)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

	## Mode
    parser.add_argument("--n_feats", type=int, default=64, help="conv channel")
    parser.add_argument("--kernel_size", type=int, default=19, help="predict kernel size")

    args = parser.parse_args()

    model = Estimator(args)
    
    input = torch.randn((4,3,128, 128))
    output = model(input)
    print(output.shape)