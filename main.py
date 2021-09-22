import time
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from model import CharRNN, CharLSTM
from dataset import Shakespeare
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
    """

    model.train()
    model.to(device)

    trn_loss = 0

    for i, (input, target) in enumerate(trn_loader):
        batch_size = input.shape[0]

        input, target = input.to(device), target.to(device)
        target = target.contiguous().view(-1, 1).squeeze(-1)
        h = model.init_hidden(batch_size)

        optimizer.zero_grad()

        output, h = model(input, h)

        loss_ = criterion(output, target)

        loss_.backward()
        optimizer.step()

        trn_loss += loss_

    trn_loss = trn_loss / len(trn_loader)

    print(f"train loss : {trn_loss}")

    return trn_loss

def validate(model, val_loader, device, criterion):
    """ Validate function

    Args:
        model: network
        val_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        val_loss: average loss value
    """

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            batch_size = input.shape[0]

            input, target = input.to(device), target.to(device)
            target = target.contiguous().view(-1, 1).squeeze(-1)
            h = model.init_hidden(batch_size)
            prediction, h = model(input, h)

            v_loss = criterion(prediction, target).item()
            val_loss += v_loss

    val_loss = val_loss / len(val_loader)

    print(f"validation loss = {val_loss}")

    return val_loss


def main():
    """ Main function

        Here, you should instantiate
        1) DataLoaders for training and validation. 
           Try SubsetRandomSampler to create these DataLoaders.
        3) model
        4) optimizer
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='char_rnn', help='determining the kind of model')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device} is available")

    with open( "./shakespeare_train.txt", 'r') as f:
        text = f.read()
    input_file = text
    seq_length = 30

    data_set = Shakespeare(input_file, seq_length)

    dataset_size = len(data_set)
    dataset_indices = list(range(dataset_size))
    val_split_index = int(np.floor(0.2 * dataset_size))
    train_idx, val_idx = dataset_indices[val_split_index:], dataset_indices[:val_split_index]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_iter = DataLoader(data_set, batch_size=256, shuffle=False, num_workers=0, sampler=train_sampler)
    val_iter = DataLoader(data_set, batch_size=256, shuffle=False, num_workers=0, sampler=val_sampler)

    inputs = data_set.chars
    hidden_size = 256
    epochs = 10
    stand_val_loss = np.inf

    if args.model == 'char_rnn':
        model = CharRNN(inputs, hidden_size, n_layers=4, drop_prob=0.1).to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        trn_loss_total, val_loss_total = [], []
        start_time = time.time()
        for e in range(epochs):
            epoch_start_time = time.time()
            print(f"------{e + 1} epoch in progress------")
            rnn_trn_loss = train(model, train_iter, device, criterion, optimizer)
            rnn_val_loss = validate(model, val_iter, device, criterion)
            trn_loss_total.append(rnn_trn_loss)
            val_loss_total.append(rnn_val_loss)

            print(f'RNN {e + 1} epoch time : {time.time() - epoch_start_time:.4f}')

            if rnn_val_loss < stand_val_loss:
                stand_val_loss = rnn_val_loss
                if not os.path.isdir('./models/'):
                    os.mkdir('models/')
                torch.save(model.state_dict(), f'models/rnn.pt')

        loss, idx = np.array(trn_loss_total).min(), np.array(val_loss_total).argmin()
        print(f"min epochs: {idx}\n"
              f"min valid loss: {loss}")
        plt.figure(figsize=(8, 6))
        plt.title('Char RNN train losses & validation losses')
        plt.plot(np.arange(1, epochs+1), trn_loss_total, 'b', label='train loss')
        plt.plot(np.arange(1, epochs+1), val_loss_total, 'r', label='validation loss')
        plt.grid(True)
        plt.legend(loc='upper right')
        if not os.path.isdir('results/'):
            os.mkdir('results/')

        plt.savefig('results/rnn.png', dpi=300)
        print(f'RNN total learning time : {time.time() - start_time:.4f}')

    elif args.model == 'char_lstm':
        model = CharLSTM(inputs, hidden_size, n_layers=4, drop_prob=0.1).to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        trn_loss_total, val_loss_total = [], []

        start_time = time.time()

        for e in range(epochs):
            epoch_start_time = time.time()
            print(f"------{e + 1} epoch in progress------")
            rnn_trn_loss = train(model, train_iter, device, criterion, optimizer)
            rnn_val_loss = validate(model, val_iter, device, criterion)
            trn_loss_total.append(rnn_trn_loss)
            val_loss_total.append(rnn_val_loss)

            print(f'LSTM {e + 1} epoch time : {time.time() - epoch_start_time:.4f}')

            if rnn_val_loss < stand_val_loss:
                stand_val_loss = rnn_val_loss
                if not os.path.isdir('./models/'):
                    os.mkdir('models/')
                torch.save(model.state_dict(), f'models/lstm.pt')

        loss, idx = np.array(trn_loss_total).min(), np.array(val_loss_total).argmin()
        print(f"min epochs: {idx}\n"
              f"min valid loss: {loss}")
        plt.figure(figsize=(8, 6))
        plt.title('Char LSTM train losses & validation losses')
        plt.plot(np.arange(1, epochs+1), trn_loss_total, 'b', label='train loss')
        plt.plot(np.arange(1, epochs+1), val_loss_total, 'r', label='validation loss')
        plt.grid(True)
        plt.legend(loc='upper right')

        if not os.path.isdir('results/'):
            os.mkdir('results/')

        plt.savefig('results/lstm.png', dpi=300)

        print(f'LSTM total learning time : {time.time() - start_time:.4f}')

    else :
        print("try again")

if __name__ == '__main__':
    main()