from dataset.load_chd import *
from model.model_chd import *

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


data = CHDDataset()
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)
trainloader = DataLoader(train_data, batch_size=Config.batch_size, shuffle=True)
valloader = DataLoader(val_data, batch_size=Config.batch_size, shuffle=True)
testloader = DataLoader(test_data, batch_size=Config.batch_size, shuffle=True)

print('Data loaded successfully!',
      'Train data length:', len(train_data),
      'Test data length:', len(test_data))

model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
criterion = nn.CrossEntropyLoss()

scheduler = StepLR(optimizer, step_size=Config.step_size, gamma=Config.gamma)

step = 0

writer = SummaryWriter()


def validation(e):
    model.eval()

    losses = []
    _correct = 0

    with torch.no_grad():

        for b_idx, val_sample in enumerate(tqdm(valloader, desc='Test')):

            x_val = val_sample['input'].to(device)
            y_val = val_sample['label'].to(device)

            if len(y_val) != Config.batch_size:
                continue

            val_output = model(x_val)

            val_loss = criterion(val_output, y_val)

            losses.append(val_loss.item())

            val_pred = torch.argmax(val_output, dim=1)
            _correct += (val_pred == y_val).sum().item()

        val_acc = 100. * _correct / len(valloader.dataset)
        val_loss = float(np.mean(losses))
        writer.add_scalar('val/acc', val_acc, e)
        writer.add_scalar('val/loss', val_loss, e)


for epoch in range(Config.epochs):
    model.train()

    correct = 0

    for i, inputs in enumerate(tqdm(trainloader, desc='Train')):

        x_train = inputs['input'].to(device)
        y_train = inputs['label'].to(device)

        if len(y_train) != Config.batch_size:
            continue

        optimizer.zero_grad()

        output = model(x_train)

        loss = criterion(output, y_train)

        loss.backward()
        optimizer.step()

        writer.add_scalar('train/loss', loss.item(), step)
        step += 1

        if i % Config.log_f == 0:
            print(f'Epoch {epoch}, step {step}, loss {loss.item()}')

        pred = torch.argmax(output, dim=1)
        correct += (pred == y_train).sum().item()
    train_acc = 100. * correct / len(trainloader.dataset)
    writer.add_scalar('train/acc', train_acc, epoch)

    validation(epoch)

    scheduler.step()


# test
model.eval()

testlosses = []
test_correct = 0

with torch.no_grad():

    for batch_idx, test_sample in enumerate(tqdm(testloader, desc='Test')):

        x_test = test_sample['input'].to(device)
        y_test = test_sample['label'].to(device)

        if len(y_test) != Config.batch_size:
            continue

        test_output = model(x_test)

        test_loss = criterion(test_output, y_test)

        testlosses.append(test_loss.item())

        test_pred = torch.argmax(test_output, dim=1)
        test_correct += (test_pred == y_test).sum().item()

    test_acc = 100. * test_correct / len(testloader.dataset)
    test_loss = float(np.mean(testlosses))
    writer.add_scalar('test/acc', test_acc)
    writer.add_scalar('test/loss', test_loss)
    