from time import perf_counter
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchgpipe import GPipe
import torch.cuda.profiler as profiler


class FakeData(Dataset):
    def __init__(self, size, width):
        self.size = size
        self.data = torch.rand(size, width)
        self.target  = torch.rand(size, 128)
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


def large_fc(width, depth):
    tmp = nn.ModuleList()
    for _ in range(depth):
        tmp.append(nn.Linear(width, width))
        tmp.append(nn.ReLU())
    return nn.Sequential(*tmp)

parser = argparse.ArgumentParser()
parser.add_argument("-b", default=1<<5, type=int, help="batch size")
parser.add_argument("-c", default=1<<4, type=int, help="chunk size")
parser.add_argument("-d", default=1<<12, type=int, help="data size")
parser.add_argument("-w", default=1<<12, type=int, help="fc layer width")
parser.add_argument("-l", default=1<<3, type=int, help="fc layer depth")
parser.add_argument("-e", default=1, type=int, help="epochs")
args = parser.parse_args()


BSZ, CSZ, DSZ = args.b, args.c, args.d
WIDTH, DEPTH, EPOCH = args.w, args.l, args.e
model = nn.Sequential(
    nn.Sequential(large_fc(WIDTH, DEPTH)),
    nn.Sequential(large_fc(WIDTH, DEPTH)),
    nn.Sequential(large_fc(WIDTH, DEPTH)),
    nn.Linear(WIDTH, 128)
)
print("num parameters =", sum( p.numel() for p in model.parameters()))
print(f"BSZ = {BSZ}, CSZ = {CSZ}")
model = GPipe(model, balance=[1, 1, 2], chunks=CSZ)

data_set = FakeData(DSZ, WIDTH)
data_loader = DataLoader(data_set, batch_size=BSZ)
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
