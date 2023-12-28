import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR, StepLR

import timm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Config:

    batch_size = 64
    epochs = 300
    lr = 1e-3

    channel = 3

    height = 32
    width = 32

    data_root = '../dataset/cifar10'

    dropout_rate = 0.1
    attn_dropout = 0

    patch_size = 4
    num_patches = int((height * width) / (patch_size ** 2))

    layers = 12
    embedding_d = 768
    mlp_size = 1024
    heads = 8

    num_classes = 10

    log_f = 100


transform = transforms.Compose([
  transforms.Resize(224),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])


trainset = torchvision.datasets.CIFAR10(root=Config.data_root, train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=Config.batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=Config.data_root, train=False,
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=Config.batch_size,
                                         shuffle=False, num_workers=2)


model = timm.create_model("vit_base_patch32_224-in21k", pretrained=True)
model.head = nn.Linear(model.head.in_features, 10)

optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
criterion = nn.CrossEntropyLoss()

scheduler = StepLR(optimizer, 5)
writer = SummaryWriter()

def test(epoch):

    model.eval()

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):

            if Config.batch_size != len(inputs):
                inputs = inputs.repeat(Config.batch_size // inputs.size(0) + 1, 1, 1, 1)[:Config.batch_size]
                targets = targets.repeat(Config.batch_size // targets.size(0) + 1)[:Config.batch_size]

            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total

    print(f'Epoch {epoch} val loss: {test_loss:.5f}, test acc: {(acc):.5f}')

    return test_loss, acc


def test(epoch):

    model.eval()

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):

            if Config.batch_size != len(inputs):
                inputs = inputs.repeat(Config.batch_size // inputs.size(0) + 1, 1, 1, 1)[:Config.batch_size]
                targets = targets.repeat(Config.batch_size // targets.size(0) + 1)[:Config.batch_size]

            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total

    print(f'Epoch {epoch} test loss: {test_loss:.5f}, test acc: {(acc):.5f}')

    return test_loss, acc


def train(epoch):

    model.train()

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):

        if Config.batch_size != len(inputs):
            inputs = inputs.repeat(Config.batch_size // inputs.size(0) + 1, 1, 1, 1)[:Config.batch_size]
            targets = targets.repeat(Config.batch_size // targets.size(0) + 1)[:Config.batch_size]

        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        acc = 100.*correct/total

        if batch_idx % Config.log_f == 0:
            print(f'Epoch {epoch}, batch index {batch_idx} || train loss: {train_loss/(batch_idx+1)}, train acc: {acc}')

    return train_loss/(batch_idx+1), acc


model = model.to(device)


for epoch in range(Config.epochs):

    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)

    scheduler.step()

    writer.add_scalar('train/loss', train_loss, epoch)
    writer.add_scalar('train/acc', train_acc, epoch)
    writer.add_scalar('test/loss', test_loss, epoch)
    writer.add_scalar('test/acc', test_acc, epoch)


