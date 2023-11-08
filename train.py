from config import Config
from dataset.load_dataset import *
from model import *

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader


writer = SummaryWriter()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data, test_data = load_data()
trainloader = DataLoader(CICIDSDataset(train_data), batch_size=Config.batch_size, shuffle=True)
testloader = DataLoader(CICIDSDataset(test_data), batch_size=Config.batch_size, shuffle=True)

model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
criterion = nn.CrossEntropyLoss()


def get_mask(batch_size, heads, seq_size):
    mask_prob = 0.2
    mask = torch.rand((batch_size, heads, seq_size, seq_size)) > mask_prob
    return mask.cuda()


def test(e):
    model.eval()

    with torch.no_grad():

        for idx, (x_test, y_test) in enumerate(testloader):
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            mask_test = get_mask(Config.batch_size, Config.heads, Config.seq_size)

            test_output = model(x_test, mask_test)
            test_loss = criterion(test_output, y_test)

            writer.add_scalar('loss/test', test_loss.item(), e * len(testloader) + idx)


for epoch in range(Config.epochs):

    model.train()

    for i, (x, y) in enumerate(trainloader):

        x = x.to(device)
        y = y.to(device)

        mask = get_mask(Config.batch_size, Config.heads, Config.seq_size)

        output = model(x, mask)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('loss/train', loss.item(), epoch * len(trainloader) + i)

        if i % Config.log_f == 0:
            print('Epoch: {}, Iter: {}, Loss: {}'.format(epoch, i, loss.item()))

    test(epoch)
