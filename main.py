from time import perf_counter
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchgpipe import GPipe


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class FakeData(Dataset):
    def __init__(self, size, width):
        self.size = size
        self.data = torch.rand(size, width)
        self.target = torch.rand(size, 128)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


class FakeImageData(Dataset):
    def __init__(self, size, width):
        self.size = size
        self.data = torch.rand(size, 3, width, width)
        self.target = torch.rand(size, 1)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


def large_cnn(width, depth):
    tmp = nn.ModuleList()
    for _ in range(depth):
        tmp.append(nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1))
        tmp.append(nn.ReLU())
    return nn.Sequential(*tmp)


def large_fc(width, depth):
    tmp = nn.ModuleList()
    for _ in range(depth):
        tmp.append(nn.Linear(width, width))
        tmp.append(nn.ReLU())
    return nn.Sequential(*tmp)


def get_model(args):
    WIDTH, DEPTH = args.w, args.l
    if args.m == "fc":
        return nn.Sequential(
            large_fc(WIDTH, DEPTH),
            large_fc(WIDTH, DEPTH),
            large_fc(WIDTH, DEPTH),
            nn.Sequential(nn.Linear(WIDTH, 128))
        )
    elif args.m == "cnn":
        return nn.Sequential(
            nn.Sequential(nn.Conv2d(3, WIDTH, 3, 1), large_cnn(WIDTH, DEPTH)),
            large_cnn(WIDTH, DEPTH),
            nn.Sequential(large_cnn(WIDTH, DEPTH), nn.Conv2d(WIDTH, 8, 3, 1)),
            nn.Sequential(nn.AdaptiveMaxPool2d(64, 64), Flatten(),
                          nn.Linear(64*64*8, 1))
            )
    else:
        raise NotImplementedError


def get_data(args):
    DSZ, WIDTH = args.d, args.w
    if args.m == "fc":
        return FakeData(DSZ, WIDTH)
    elif args.m == "cnn":
        return FakeImageData(DSZ, 256)
    else:
        raise NotImplementedError


parser = argparse.ArgumentParser()
parser.add_argument("-m", choices=["fc", "cnn"], help="choose arch")
parser.add_argument("-b", default=1 << 5, type=int, help="batch size")
parser.add_argument("-c", default=1 << 4, type=int, help="chunk size")
parser.add_argument("-d", default=1 << 12, type=int, help="data size")
parser.add_argument("-w", default=1 << 12, type=int, help="fc layer width")
parser.add_argument("-l", default=1 << 3, type=int, help="fc layer depth")
parser.add_argument("-e", default=1, type=int, help="epochs")
args = parser.parse_args()


BSZ, CSZ, DSZ = args.b, args.c, args.d
WIDTH, DEPTH, EPOCH = args.w, args.l, args.e
ARCH = args.m


data_set = get_data(args)
data_loader = DataLoader(data_set, batch_size=BSZ)
model = get_model(args)
model = GPipe(model, balance=[1, 1, 2], chunks=CSZ)

crit = nn.MSELoss()
optm = Adam(model.parameters(), lr=0.001)
t1 = perf_counter()
for _ in range(EPOCH):
    for x, y in data_loader:
        x = x.to(model.devices[0])
        y = y.to(model.devices[-1])
        optm.zero_grad()
        output = model(x)
        loss = crit(output, y)
        loss.backward()
t2 = perf_counter()
print(f"total time cost: {t2-t1} sec")
