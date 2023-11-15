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

step = 0


def get_mask(batch_size, heads, seq_size):
    mask_prob = 0.2
    mask = torch.rand((batch_size, heads, seq_size, seq_size)) > mask_prob
    return mask.cuda()


def test(e):
    model.eval()

    losses = []
    _correct = 0

    with torch.no_grad():
        for idx, b in enumerate(testloader):
            data, target, _ = b
            data, target = data.cuda(), target.cuda()
            test_output = model(data)

            test_loss = criterion(test_output, target)

            losses.append(test_loss.item())
            _correct += torch.eq(torch.argmax(test_output, dim=1), torch.argmax(target, dim=1)).cpu().sum().item()

        eval_acc = 100. * _correct / len(testloader.dataset)
        eval_loss = float(np.mean(losses))
        writer.add_scalar('test/acc', eval_acc, e)
        writer.add_scalar('test/loss', eval_loss.item(), e)


for epoch in range(Config.epochs):

    model.train()

    correct = 0

    for i, batch in enumerate(trainloader):

        src, trg, _ = batch
        src, trg = src.cuda(), trg.cuda()

        if isinstance(model, Model):
            trg_mask = get_mask(128, 8, 78)
        else:
            trg_mask = None

        output = model(src, trg_mask)
        loss = criterion(output, trg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('train/loss', loss.item(), step)

        step += 1

        if i % Config.log_f == 0:
            print('Epoch: {}, Iter: {}, Loss: {}'.format(epoch, i, loss.item()))

        pred = torch.argmax(output, dim=1)
        correct += (pred == trg).sum().item()
    train_acc = 100. * correct / len(trainloader.dataset)
    writer.add_scalar('train/acc', train_acc, epoch)

    test(epoch)