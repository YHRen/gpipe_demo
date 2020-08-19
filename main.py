from time import perf_counter
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

try:
    from torchgpipe import GPipe
except:
    raise UserWarning("GPipe not installed")

# for distributed training
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


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


class CNNPipeline(nn.Module):
    def __init__(self, devices, WIDTH, DEPTH):
        super(CNNPipeline, self).__init__()
        self.devices = devices
        self.width, self.depth = WIDTH, DEPTH
        self.m1 = nn.Sequential(nn.Conv2d(3, WIDTH, 3, 1),
                                large_cnn(WIDTH, DEPTH))
        self.m2 = large_cnn(WIDTH, DEPTH)
        self.m3 = nn.Sequential(large_cnn(WIDTH, DEPTH),
                                nn.Conv2d(WIDTH, 8, 3, 1)),
        self.m4 = nn.Sequential(nn.AdaptiveMaxPool2d(64), Flatten(),
                                nn.Linear(64*64*8, 1))
        self.m1 = self.m1.to(devices[0])
        self.m2 = self.m1.to(devices[1])
        self.m3 = self.m1.to(devices[2])
        self.m4 = self.m1.to(devices[2])

    def forward(self, x):
        x = self.m1(x)
        x = self.m2(x)
        x = self.m3(x)
        x = self.m4(x)
        return x


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
            nn.Sequential(nn.AdaptiveMaxPool2d(64), Flatten(),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", choices=["fc", "cnn"], help="choose arch")
    parser.add_argument("-b", default=1 << 5, type=int, help="batch size")
    parser.add_argument("-c", default=1 << 4, type=int, help="chunk size")
    parser.add_argument("-d", default=1 << 12, type=int, help="data size")
    parser.add_argument("-w", default=1 << 12, type=int, help="fc layer width")
    parser.add_argument("-l", default=1 << 3, type=int, help="fc layer depth")
    parser.add_argument("-e", default=1, type=int, help="epochs")
    # turn off gpipe
    parser.add_argument("--nogpipe", action="store_true",
                        help="turn off gpipe")
    # for distributed training
    parser.add_argument("--dist", action='store_true',
                        help="use distributed training")
    parser.add_argument('--gpus_per_group', default=1, type=int,
                        help="num. of GPUs per resource group")
    parser.add_argument('--group_per_node', default=4, type=int,
                        help="num. of model replicas a node can accomondate")
    parser.add_argument('--local_rank', default=0, type=int)

    args = parser.parse_args()

    start_dev_id = None
    if args.dist:
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        gs, gpn, rk = args.gpus_per_group, args.group_per_node, dist.get_rank()
        start_dev_id = gs*(rk % gpn)
        torch.cuda.set_device(start_dev_id)
        print(f"start dev id {start_dev_id}")
        print(f"total device count: {torch.cuda.device_count()}")

    BSZ, CSZ, DSZ = args.b, args.c, args.d
    WIDTH, DEPTH, EPOCH = args.w, args.l, args.e
    ARCH = args.m

    dataset = get_data(args)
    start_dev, end_dev = 0, 0

    if args.dist:
        devices = [start_dev_id+i for i in range(args.gpus_per_group)]
        sampler = DistributedSampler(dataset)
        data_loader = DataLoader(
            dataset, batch_size=BSZ, shuffle=False, sampler=sampler)
        if args.nogpipe:
            model = CNNPipeline(devices, args.w, args.l)
        else:
            model = get_model(args)
            model = GPipe(model, balance=[1, 1, 2], devices=devices, chunks=CSZ)
            model = DDP(model)
        start_dev, end_dev = devices[0], devices[-1]
    else:
        devices = list(range(3))
        dataloader = DataLoader(dataset, batch_size=BSZ)
        if args.nogpipe:
            model = CNNPipeline(devices, args.w, args.l)
        else:
            model = get_model(args)
            model = GPipe(model, balance=[1, 1, 2], devices=devices, chunks=CSZ)
        start_dev, end_dev = devices[0], devices[-1]

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

    if args.dist:
        dist.destroy_process_group()
