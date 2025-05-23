import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score

import argparse

from tqdm import tqdm


""" Define Command Line Parser """
def parse_cmd_line_params():
    parser = argparse.ArgumentParser(description="batch_size")
    parser.add_argument(
        "--ssl_model_name",
        help="SSL model name",
        default="facebook/hubert-base-ls960",
        type=str,
        required=False)
    parser.add_argument(
        "--learning_rate",
        help="learning rate for training",
        default=1e-5,
        type=float,
        required=False)
    parser.add_argument(
        "--batch_size",
        help="batch size for training",
        default=32,
        type=int,
        required=False)
    parser.add_argument(
        "--hidden_size",
        help="hidden size for the model",
        default=128,
        type=int,
        required=False)
    parser.add_argument(
        "--num_epochs",
        help="number of epochs for training",
        default=10,
        type=int,
        required=False)
    parser.add_argument(
        "--early_stopping",
        help="early stopping patience",
        default=5,
        type=int,
        required=False)
    parser.add_argument(
        "--max_audio_duration",
        help="maximum duration for audio files",
        default=8.0,
        type=float,
        required=False)
    parser.add_argument(
        "--feature_dim",
        help="feature dimensionality",
        default=768,
        type=int,
        required=False)
    parser.add_argument(
        "--num_samples_per_class",
        help="number of samples per class for training",
        default=1000,
        type=int,
        required=False) 
    parser.add_argument(
        "--seed",
        help="seed for reproducibility",
        default=0,
        type=int,
        required=False)
    args = parser.parse_args()
    return args


## Set the random seeds for reproducibility pytorch
def set_seed(seed=0):
    import random
    random.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


## Count the number of parameters in a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


## Training Function
def train_model(
    model, 
    data_loader,    
    criterion,
    optimizer,
    device='cuda'
    ):

    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
        
    for inputs, labels in tqdm(data_loader):

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return total_loss/len(data_loader), acc, f1


## Evaluation Function
def evaluate_model(
    model, 
    data_loader, 
    criterion, 
    device='cuda'
    ):
    
    model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():

        for inputs, labels in data_loader:

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return total_loss / len(data_loader), acc, f1