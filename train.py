from config import Config
from dataset.load_dataset import *
from model import *

<<<<<<< HEAD
from model import *

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tensorboardX import SummaryWriter

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


step = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = Encoder().to(device)
decoder = Decoder().to(device)

encoder_optimizer = Adam(encoder.encoder.parameters(), lr=Config.lr)
decoder_optimizer = Adam(decoder.decoder.parameters(), lr=Config.lr)

dataset = Dataset()

smote = SMOTE(sampling_strategy={
    label: 7 * 100000
}, k_neighbors=5, random_state=42)
X, y = smote.fit_resample(dataset.data, dataset.label)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

writer = SummaryWriter()


def evaluate():
    accuracy = accuracy_score()
    f1 = f1_score()
    precision = precision_score()
    recall = recall_score()

    return accuracy, f1, precision, recall


def train():
    encoder.train()
    decoder.train()

    for batch_idx, x in enumerate(zip(x_train, y_train)):
        x, y = x

        x = torch.tensor(x).to(device)
        y = torch.tensor(y).to(device)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_output = encoder(x)
        decoder_output = decoder(encoder_output)

        loss = F.cross_entropy(decoder_output, y)

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        if batch_idx % 1000 == 0:
            accuracy, f1, precision, recall = evaluate(something)
            
            writer.add_scalar('test/Accuracy', accuracy, step)
            writer.add_scalar('test/F1', f1, step)
            writer.add_scalar('test/Precision', precision, step)
            writer.add_scalar('test/Recall', recall, step)


def test():
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for batch_idx, x in enumerate(zip(x_test, y_test)):

            if batch_idx % 1000 == 0:
                accuracy, f1, precision, recall = evaluate(something)

                writer.add_scalar('test/Accuracy', accuracy, step)
                writer.add_scalar('test/F1', f1, step)
                writer.add_scalar('test/Precision', precision, step)
                writer.add_scalar('test/Recall', recall, step)


for epoch in range(Config.epcohs):
    for batch_idx, x in enumerate(zip(x_train, y_train)):

        train()

    test()

    step += 1
=======
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
    correct = 0

    with torch.no_grad():
        for idx, b in enumerate(testloader):
            data, target, _ = b
            data, target = data.cuda(), target.cuda()
            test_output = model(data)

            test_loss = criterion(test_output, target)

            losses.append(test_loss.item())
            correct += torch.eq(torch.argmax(test_output, dim=1), torch.argmax(target, dim=1)).cpu().sum().item()

        eval_acc = 100. * correct / len(testloader.dataset)
        eval_loss = float(np.mean(losses))
        writer.add_scalar('test/acc', eval_acc, e)
        writer.add_scalar('test/loss', test_loss.item(), e)


for epoch in range(Config.epochs):

    model.train()

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

        if i % Config.log_f == 0:
            print('Epoch: {}, Iter: {}, Loss: {}'.format(epoch, i, loss.item()))

        step += 1

    test(epoch)
>>>>>>> f35e0785efa6a80da62d42b0569b5d1e9f294745
