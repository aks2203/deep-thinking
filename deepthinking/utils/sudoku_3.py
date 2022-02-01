""" mazes_data.py
    Maze related dataloaders

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

import torch
from torch.utils import data
import random
import urllib.request
from torch.utils.data import Dataset, DataLoader
import csv
import numpy as np

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115),
#     Unused import (W0611).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, W0611
def permute(sample):
    q, a = sample
    keys = list(map(str, range(1, 10)))
    values = random.sample(keys, len(keys))
    mapping = dict(zip(keys, values))
    mapping['0'] = '0'

    def perm(digits):
        return ''.join([mapping[x] for x in digits])

    return perm(q), perm(a)


def add(sample, n):
    q, a = sample
    for _ in range(n):
        empty_places = [k for k, v in enumerate(q) if v == '0']
        idx = random.choice(empty_places)
        q = q[:idx] + a[idx] + q[(idx + 1):]
    return q, a

def sub(sample, n):
    q, a = sample
    for _ in range(n):
        empty_places = [k for k, v in enumerate(q) if v != '0']
        idx = random.choice(empty_places)
        q = q[:idx] + '0' + q[(idx + 1):]
    return q, a

def generate(pool, givens_max, n_per_givens):
    generated = []
    for i in range(n_per_givens):
        sample = random.choice(pool)
        if len(sample[0]) == 81:
            sample = permute(sample)
            missing_zeros = sample[0].count('0')
            present = 81 - missing_zeros

            if present < givens_max:
                sample = add(sample, givens_max - present)
            else:
                sample = sub(sample, present - givens_max)
            generated.append(sample)

    random.shuffle(generated)
    return generated

def dump(fname, samples):
    with open(fname, 'w') as f:
        for q, a in samples:
            f.write(q + "," + a + "\n")

def create_data():
    fname = './sudoku.csv'
    with open(fname) as f:
        lines = f.readlines()

    hard = [line.strip().split(',') for line in lines]
    hard = random.sample(hard, len(hard))  # shuffled copy

    n_test = 10000
    n_valid = 20000

    test_pool = hard[:n_test]
    valid_pool = hard[n_test:n_test + n_valid]
    train_pool = hard[n_test + n_valid:]

    givens_max = 17
    train = generate(train_pool, givens_max, 900000)
    valid = generate(valid_pool, givens_max, 20000)
    givens_max = 17
    test = generate(test_pool, givens_max, 10000)

    dump('train_3.csv', train)
    dump('valid_3.csv', valid)
    dump('test_3.csv', test)

def read_csv(fname):
    print("Reading %s..." % fname)
    with open(fname) as f:
        reader = csv.reader(f, delimiter=',')
        return [(q, a) for q, a in reader]

class dataset(Dataset):
    def __init__(self, root_dir, dataset_type, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_names = self.root_dir + dataset_type + ".csv"
        self.csv_data = read_csv(self.file_names)
        self.data = []
        for d in self.csv_data:
            input = np.asarray(list(map(int, list(d[0]))))
            input = torch.from_numpy(input)
            input = input.reshape(9, 9)
            input = torch.unsqueeze(input, 0).float()
            # input = torch.nn.functional.one_hot(input,num_classes=10)
            # input = input.permute(2, 1, 0).float()

            target = np.asarray(list(map(int, list(d[1]))))
            target = torch.from_numpy(target)
            target = target.reshape(9, 9)
            #target = torch.nn.functional.one_hot(target,num_classes=10)
            #target = target.permute(2, 1, 0)

            self.data.append((input, target))
        #self.data = torch.FloatTensor(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input, target = self.data[idx]
        return input, target

def prepare_sudoku_3_loader(train_batch_size, test_batch_size, train_data, test_data, shuffle=True):

    #create_data()

    trainset = dataset(root_dir = './', dataset_type='train_3')
    valset = dataset(root_dir='./', dataset_type='valid_3')
    testset = dataset(root_dir='./', dataset_type='test_3')

    trainloader = data.DataLoader(trainset,
                                  num_workers=0,
                                  batch_size=train_batch_size,
                                  shuffle=shuffle,
                                  drop_last=False)
    valloader = data.DataLoader(valset,
                                num_workers=0,
                                batch_size=test_batch_size,
                                shuffle=False,
                                drop_last=False)
    testloader = data.DataLoader(testset,
                                 num_workers=0,
                                 batch_size=test_batch_size,
                                 shuffle=False,
                                 drop_last=False)

    loaders = {"train": trainloader, "test": testloader, "val": valloader}

    return loaders
