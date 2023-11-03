from config import Config
from dataset.load_dataset import Dataset

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
