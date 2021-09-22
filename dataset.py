import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

class Shakespeare(Dataset):
    """ Shakespeare dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
             You need this dictionary to generate characters.
        2) Make list of character indices using the dictionary
        3) Split the data into chunks of sequence length 30.
   You should create targets appropriately.
    """

    def __init__(self, input_file, seq_length=30):

        self.seq_length = seq_length
        self.data = input_file
        self.chars = tuple(sorted(set(input_file)))
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        self.char_list = [self.char2int[c] for c in input_file]

    def __len__(self):
        length = len(self.char_list)-self.seq_length-1
        return length

    def __getitem__(self, idx):

        x_sample = self.char_list[idx: idx + self.seq_length]
        input = [np.eye(len(self.char2int))[x] for x in x_sample]
        input = torch.tensor(input, dtype=torch.float)

        y_sample = self.char_list[idx+1: idx+ self.seq_length+1]
        target = torch.LongTensor(y_sample)

        return input, target

if __name__ == '__main__':

    with open( "./shakespeare_train.txt", 'r') as f:
        text = f.read()
    input_file = text
    seq_length = 30

    train_set = Shakespeare(input_file, seq_length)

    dataset_size = len(train_set)
    dataset_indices = list(range(dataset_size))
    val_split_index = int(np.floor(0.2 * dataset_size))
    train_idx, val_idx = dataset_indices[val_split_index:], dataset_indices[:val_split_index]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_iter = DataLoader(train_set, batch_size=4, shuffle=False, num_workers=0, sampler=train_sampler)
    val_iter = DataLoader(train_set, batch_size=4, shuffle=False, num_workers=0, sampler=val_sampler)

    for text, target in train_iter:
        print(text)
        print(target)
        print(text.size())

        import sys;
        sys.exit(0)
