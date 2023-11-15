from config import Config
from dataset.load_images import *
from model_images import *

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


writer = SummaryWriter()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = CIDDataset()
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
trainloader = DataLoader(train_data, batch_size=Config.batch_size, shuffle=True)
testloader = DataLoader(test_data, batch_size=Config.batch_size, shuffle=True)


print('Data loaded successfully!')

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

        for batch_idx, test_sample in enumerate(testloader):

            x_test = test_sample['input'].to(device)
            y_test = test_sample['label'].to(device)

            test_output = model(x_test)

            test_loss = criterion(test_output, y_test)

            losses.append(test_loss.item())

            test_pred = torch.argmax(test_output, dim=1)
            _correct += (test_pred == y_test).sum().item()

        eval_acc = 100. * _correct / len(testloader.dataset)
        eval_loss = float(np.mean(losses))
        writer.add_scalar('test/acc', eval_acc, e)
        writer.add_scalar('test/loss', eval_loss.item(), e)


for epoch in range(Config.epochs):
    model.train()

    correct = 0

    for i, inputs in enumerate(trainloader):

        x_train = inputs['input'].to(device)
        y_train = inputs['label'].to(device)

        optimizer.zero_grad()

        output = model(x_train)

        loss = criterion(output, y_train)

        loss.backward()
        optimizer.step()

        step += 1

        writer.add_scalar('train/loss', loss.item(), step)

        if i % Config.log_f == 0:
            print(f'Epoch {epoch}, step {step}, loss {loss.item()}')

        pred = torch.argmax(output, dim=1)
        correct += (pred == y_train).sum().item()
    train_acc = 100. * correct / len(trainloader.dataset)
    writer.add_scalar('train/acc', train_acc, epoch)

    test(epoch)