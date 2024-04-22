#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import os
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid


def create_user_groups(X_train, num_users):
    user_groups = []
    num_samples_per_user = len(X_train) // num_users
    
    for i in range(num_users):
        start_idx = i * num_samples_per_user
        end_idx = (i + 1) * num_samples_per_user
        
        user_data = X_train[start_idx:end_idx]
        user_groups.append(user_data)
    
    return user_groups


def ts_array_create(dirname, time_seq):
    features = ['LTE_HO', 'MN_HO',  'SCG_RLF',
                'num_of_neis', 'RSRP', 'RSRQ', 'RSRP1', 'RSRQ1', 'nr-RSRP', 'nr-RSRQ', 'nr-RSRP1', 'nr-RSRQ1']
    target = ['LTE_HO', 'MN_HO']
    split_time = []
    for file_name in os.listdir(dirname):
        file_path = os.path.join(dirname, file_name)
        df = pd.read_csv(file_path)
        
        if not set(features).issubset(df.columns):
            print(f"Skipping {file_name} because it doesn't contain all the required features.")
            continue
      
        if 'Timestamp' in df.columns:
            del df['Timestamp']

        X = df[features]
        Y = df[target]

        Xt_list = []

        for j in range(time_seq):
            X_t = X.shift(periods=-j)
            Xt_list.append(X_t)

        X_ts = np.array(Xt_list)
        X_ts = np.transpose(X_ts, (1, 0, 2))
        X_ts = X_ts[:-(time_seq), :, :]
        X_ts = X_ts.reshape(-1, 40)

        Y = Y.to_numpy()
        Y = [1 if sum(y) > 0 else 0 for y in Y]

        YY = []

        for j in range(time_seq, len(Y)):
            count = 0
            for k in range(j, len(Y)):
                count += 1
                if Y[k] != 0:
                    break
            YY.append(count)

        YY = np.array(YY)

        split_time.append(len(X_ts))
        i=0
        if i == 0:
            X_final = X_ts
            Y_final = YY
            i=1
        else:
            X_final = np.concatenate((X_final, X_ts), axis=0)
            Y_final = np.concatenate((Y_final, YY), axis=0)
    split_time = [(sum(split_time[:i]), sum(split_time[:i])+x)
                  for i, x in enumerate(split_time)]

    return X_final, Y_final
    

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'v22':
        dirname = '../data/v22'
        dirlist = os.listdir(dirname)
       
        X, y = ts_array_create(dirname,time_seq=20)

        # Split data into train and test sets
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

        # Select corresponding targets (y) for train and test sets
        y_train = y[:len(X_train)//6]  # Assuming y is a list or numpy array
        y_test = y[len(X_train)//6:]

        # Convert data to PyTorch Dataset objects
        # train_dataset = CustomDataset(X_train, y_train)
        # test_dataset = CustomDataset(X_test, y_test)
        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # Assuming y_train contains labels
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)    # Assuming y_test contains labels

        # Create PyTorch Dataset objects using TensorDataset
        train_dataset = TensorDataset(*X_train_tensor)
        test_dataset = TensorDataset(*X_test_tensor)

        user_groups = create_user_groups(X_train, args.num_users)


        # return train_dataset, test_dataset, user_groups

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
