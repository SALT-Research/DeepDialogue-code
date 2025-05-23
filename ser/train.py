import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import AutoFeatureExtractor, AutoModel

import numpy as np
import pandas as pd 

from dataset import EmotionDataset
from utils import set_seed, count_parameters, parse_cmd_line_params 
from utils import train_model, evaluate_model

import warnings
warnings.filterwarnings("ignore")

class SSLMLPModel(nn.Module):
    def __init__(
        self, 
        hidden_size, 
        dropout_ratio, 
        num_classes, 
        feature_dim,
        ssl_model=None, 
        device='cuda'
        ):
        super(SSLMLPModel, self).__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.dropout_ratio = dropout_ratio
        self.num_classes = num_classes
        self.ssl_model = ssl_model
        self.feature_dim = feature_dim

        ## Define MLP
        self.dense = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_size, num_classes),
            ).to(device)

    def forward(self, x):
        ## Forward pass through SSL
        x = self.ssl_model(x).last_hidden_state
        x = x.mean(dim=1)
        ## Forward pass through dense layers
        x = self.dense(x)
        return x


##################################################################
##########                Main Function                 ##########
##################################################################

if __name__ == "__main__":

    torch.multiprocessing.set_start_method('spawn')

    ##################################################################
    ##########               Set the device                 ##########
    ##################################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Is CUDA available? ", torch.cuda.is_available())
    print("Current device: ", torch.cuda.current_device())
    print("Device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))
    ##################################################################


    ##################################################################
    ##########               Parse Command Line             ##########
    ##################################################################
    args = parse_cmd_line_params()
    ##################################################################


    ##################################################################
    ##########         Set seed for reproducibility         ##########
    ##################################################################
    print('\n---------------------')
    print("Setting random seed...")
    SEED = args.seed
    set_seed(SEED)
    print(f"Random seed set to {SEED}")
    print('---------------------\n')
    ##################################################################


    ##################################################################
    #########      Load SSL Model and Feature Extractor      #########
    ##################################################################
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.ssl_model_name)
    ssl_model = AutoModel.from_pretrained(args.ssl_model_name)
    ##################################################################


    ##################################################################
    ##########               Load datasets                  ##########
    ##################################################################
    print('---------------------')
    print("Loading dataset...")
    df_train = pd.read_csv('data/train.csv', index_col=None)
    df_val = pd.read_csv('data/val.csv', index_col=None)
    df_test = pd.read_csv('data/test.csv', index_col=None)
    ## Print the number of samples in each set
    print(f"Number of samples in train set: {len(df_train)}")
    print(f"Number of samples in val set: {len(df_val)}")
    print(f"Number of samples in test set: {len(df_test)}")
    ## Print the number of samples per class
    print("Number of samples per class in train set:")
    print(df_train['emotion'].value_counts())
    print("Number of samples per class in val set:")
    print(df_val['emotion'].value_counts())
    print("Number of samples per class in test set:")
    print(df_test['emotion'].value_counts())
    ## Print the number of classes
    print(f"Number of classes: {len(df_train['emotion'].unique())}")
    print("---------------------\n")

    emotion_col = 'emotion'
    labels = df_train[emotion_col].unique()
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
    num_labels = len(id2label)

    ## Train
    for index in range(0,len(df_train)):
        df_train.loc[index,'label'] = label2id[df_train.loc[index, emotion_col]]
    df_train['label'] = df_train['label'].astype(int)
    train_dataset = EmotionDataset(
        data=df_train,
        feature_extractor=feature_extractor,
        max_audio_duration=args.max_audio_duration
        )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        )

    ## Validation
    for index in range(0,len(df_val)):
        df_val.loc[index,'label'] = label2id[df_val.loc[index, emotion_col]]
    df_val['label'] = df_val['label'].astype(int)
    val_dataset = EmotionDataset(
        data=df_val,
        feature_extractor=feature_extractor,
        max_audio_duration=args.max_audio_duration
        )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        )

    ## Test
    for index in range(0,len(df_test)):
        df_test.loc[index,'label'] = label2id[df_test.loc[index, emotion_col]]
    df_test['label'] = df_test['label'].astype(int)
    test_dataset = EmotionDataset(
        data=df_test,
        feature_extractor=feature_extractor,
        max_audio_duration=args.max_audio_duration
        )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        )

    print("Datasets loaded.")
    print('---------------------\n')
    ##################################################################

    ##################################################################
    ##########        Instantiate and Train Models          ##########
    ##################################################################      
    print('-----------------------')
    print("Instantiating model...")
    model = SSLMLPModel(
        hidden_size=args.hidden_size, 
        dropout_ratio=0.1,
        feature_dim=args.feature_dim,
        num_classes=num_labels,
        ssl_model=ssl_model,
        device=device
        ).to(device)
    print(model)
    print(f"Number of model parameters: {count_parameters(model)}")

    ## Define criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)

    ## Train the model
    print("Training the model...")
    best_acc = 0
    for epoch in range(args.num_epochs):

        train_loss, train_acc, train_f1 = train_model(
            model=model, 
            data_loader=train_loader, 
            criterion=criterion,
            optimizer=optimizer,
            device=device
            )

        val_loss, val_acc, val_f1 = evaluate_model(
            model=model, 
            data_loader=val_loader, 
            criterion=criterion,
            device=device
            )
        scheduler.step(val_acc)

        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model
            best_epoch = epoch + 1
            no_improvement_count = 0  ## Reset counter when improvement is found
        else:
            no_improvement_count += 1

        ## Check for early stopping
        if no_improvement_count >= args.early_stopping:
            print(f"Early stopping triggered. No improvement for {args.early_stopping} epochs.")
            break

    print(f"Best validation accuracy achieved: {best_acc:.4f} at epoch {best_epoch}")
    print("---------------------")


    ## Test the model
    print("Testing the model...")
    _, test_acc, test_f1 = evaluate_model(
        model=best_model, 
        data_loader=test_loader, 
        criterion=criterion, 
        device=device
        )

    print(f"Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")
    print('---------------------\n')

    with open(f'results.txt', 'a+') as f:
        f.write(f"Model: {args.ssl_model_name.split('/')[-1]}\n")
        f.write(f"Number of model parameters: {count_parameters(model)}\n")
        f.write(f"Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}\n\n")

    ## Save model
    torch.save(best_model.state_dict(), f'{args.ssl_model_name.split("/")[-1]}_deepdialogue{args.num_samples_per_class}.pth')