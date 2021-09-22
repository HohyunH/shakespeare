import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import argparse

from dataset import Shakespeare
from model import CharRNN, CharLSTM

def generate(model, seed_characters, temperature, length=200):

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device} is available")
    with open("./shakespeare_train.txt", 'r') as f:
        text = f.read()
    input_file = text
    seq_length = length

    data_set = Shakespeare(input_file, seq_length)
    test_ex = seed_characters
    h = model.init_hidden(1)
    input = torch.Tensor(np.eye(len(data_set.char2int))[data_set.char2int[test_ex]])

    plot_idx, plot_score = [], []
    while len(test_ex) < length:
        input, h = model(input.view(1, 1, -1).to(device), h)
        pred = F.softmax(input / temperature, dim=1)
        plot_score.append(torch.max(pred))
        pred = pred.squeeze().to('cpu').detach().numpy()

        next_char = data_set.int2char[np.random.choice(np.arange(len(data_set.char2int)), p=pred/pred.sum())]
        input = torch.Tensor(np.eye(len(data_set.char2int))[data_set.char2int[next_char]])

        plot_idx.append(next_char)


        test_ex += next_char

    plt.ylim(0,1)
    plt.bar(plot_idx, plot_score)
    plt.savefig('results/plot.png', dpi=300)

    return test_ex

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=str, default="T", help='Input the test characters')
    parser.add_argument('--model', type=str, default='char_rnn', help='determining the kind of model')
    args = parser.parse_args()

    with open( "./shakespeare_train.txt", 'r') as f:
        text = f.read()
    input_file = text
    seq_length = 30

    data_set = Shakespeare(input_file, seq_length)
    seed_characters = args.seed

    temperature = 0.1

    inputs = data_set.chars
    hidden_size = 256

    if args.model == 'char_rnn':
        model = CharRNN(inputs, hidden_size, n_layers=4, drop_prob=0.1).to(device)
        model.load_state_dict(torch.load('models/rnn.pt'), strict=False)
        model = model.to(device)
        print(generate(model, seed_characters, temperature, length=30))

    elif args.model == 'char_lstm':
        model = CharLSTM(inputs, hidden_size, n_layers=4, drop_prob=0.1).to(device)
        model.load_state_dict(torch.load('models/lstm.pt'), strict=False)
        model = model.to(device)
        print(generate(model, seed_characters, temperature, length=30))
    else :
        print("try again")


if __name__=='__main__':
    main()