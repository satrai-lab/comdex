import argparse
import json
import os
import http.server
import socketserver
import threading
import time
import subprocess
import sys
import paho.mqtt.client as mqtt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import SubsetRandomSampler
import flwr as fl
from comdex_action import post_entity
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
import requests

import csv
from datetime import datetime


def weighted_average(metrics):
    """Compute weighted average for federated metrics aggregation."""
    num_examples_total = sum(num_examples for num_examples, _ in metrics)
    avg_metrics = {
        key: sum(num_examples * m[key] for num_examples, m in metrics) / num_examples_total
        for key in metrics[0][1]
    }
    return avg_metrics

def weighted_average_evaluation(metrics):
    """Compute weighted average for federated evaluation."""
    num_examples_total = sum(num_examples for num_examples, _ in metrics)
    avg_metrics = {
        key: sum(num_examples * m[key] for num_examples, m in metrics) / num_examples_total
        for key in metrics[0][1]
    }
    return avg_metrics


###############################################################################
#                 REGISTRIES FOR MODELS AND DATASETS                         #
###############################################################################

class MNISTNet(nn.Module):
    """A simple CNN for MNIST (1×28×28)."""
    def __init__(self, hidden_size=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(hidden_size * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, x.shape[1] * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class HAR_CNN(nn.Module):
    def __init__(self, input_dim=561, num_classes=6):
        super(HAR_CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * (input_dim // 2 // 2 // 2), 128)  # Adjust based on sequence length
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
class TinyCNN(nn.Module):
    """Another demonstration model (for CIFAR-10)."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 6, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        # After conv: (batch, 6, 30, 30) then pool gives (batch, 6, 15, 15) → 6*15*15 = 1350.
        self.fc = nn.Linear(6 * 15 * 15, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(-1, 6 * 15 * 15)
        x = self.fc(x)
        return x


class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.dropout(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        out += self.shortcut(x)
        return out

class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(WideResNet, self).__init__()
        self.in_planes = 16
        assert ((depth-4) % 6 == 0), "Depth should be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor
        print(f"WideResNet: depth {depth}, widen_factor {widen_factor}, dropout {dropout_rate}")
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.layer1 = self._wide_layer(WideBasic, 16*k, n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasic, 32*k, n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasic, 64*k, n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(64*k)
        self.fc = nn.Linear(64*k, num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# Synthetic Port Model (for our synthetic port experiments)
class SyntheticPortModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class ResNet18(nn.Module):
    """ResNet-18 adapted for CIFAR-10."""
    def __init__(self, num_classes=10):
        super().__init__()
        # Load a vanilla ResNet-18 without pretrained weights.
        self.model = models.resnet18(pretrained=False)
        # Change the first conv layer: original (7x7, stride=2, padding=3) -> (3x3, stride=1, padding=1)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Remove the maxpool layer (or replace it with identity) to keep the small input resolution.
        self.model.maxpool = nn.Identity()
        # Adjust the final fully connected layer to output the desired number of classes.
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)    
    


# Occupancy Model: a simple MLP for binary occupancy detection.
#class OccupancyModel(nn.Module):
#    def __init__(self, input_dim=5, hidden_dim=32, num_classes=2):
#        super().__init__()
#        self.fc1 = nn.Linear(input_dim, hidden_dim)
#        self.relu = nn.ReLU()
#        self.fc2 = nn.Linear(hidden_dim, num_classes)
#    def forward(self, x):
#        x = self.fc1(x)
#        x = self.relu(x)
#        x = self.fc2(x)
#        return x
class OccupancyMLP(nn.Module):
    def __init__(self, input_dim=5, hidden_dims=[64, 32], num_classes=2, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dims[1], num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x




MODEL_REGISTRY = {
    "mnistnet": MNISTNet,
    "tinycnn": TinyCNN,
    "synthetic_port": SyntheticPortModel,
    "occupancy_model": OccupancyMLP,
    "resnet18": ResNet18,
    "har_cnn": HAR_CNN,
    "wideresnet28_10": lambda params=None: WideResNet(depth=28, widen_factor=10, dropout_rate=0.3, 
                                                       num_classes=params.get("num_classes", 100) if params else 100),
}


def create_model(model_type: str, model_params: dict = None) -> nn.Module:
    model_type = model_type.lower()
    if model_type in MODEL_REGISTRY:
        constructor = MODEL_REGISTRY[model_type]
        return constructor(**(model_params or {}))
    else:
        raise ValueError(f"Unknown model type '{model_type}'. Available: {list(MODEL_REGISTRY.keys())}")

###############################################################################
#               DATA LOADING FUNCTIONS (MNIST, CIFAR10, PORT, OCCUPANCY)       #
###############################################################################

#############################################
# Helper: Split indices for client-level IID split
#############################################
def split_indices_iid(total_len, total_clients, client_index):
    all_indices = list(range(total_len))
    np.random.shuffle(all_indices)
    per_client = total_len // total_clients
    start = client_index * per_client
    end = start + per_client
    return all_indices[start:end]

#CIFAR 100 dataset
def load_cifar100(data_path, train_batch_size=32, test_batch_size=32,
                  client_id=None, total_clients=None,
                  federation_id=None, total_federations=None,
                  data_distribution="iid", experiment_mode=True):
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform)
    
    # Federation-level split for 100 classes.
    if experiment_mode and (federation_id is not None and total_federations is not None and total_federations != 0):
        n_classes = 100
        classes_per_fed = n_classes // total_federations
        start_label = int(federation_id * classes_per_fed)
        # Ensure the last federation gets all remaining classes
        end_label = start_label + classes_per_fed if federation_id < total_federations - 1 else n_classes
        fed_allowed = list(range(start_label, end_label))
        print(f"[load_cifar100] Federation {federation_id}/{total_federations} allowed classes: {fed_allowed}")
        train_indices = [i for i, label in enumerate(train_set.targets) if label in fed_allowed]
        test_indices = [i for i, label in enumerate(test_set.targets) if label in fed_allowed]
        train_set = data.Subset(train_set, train_indices)
        test_set = data.Subset(test_set, test_indices)
    else:
        fed_allowed = list(range(100))
    
    # Client-level split.
    if experiment_mode and total_clients is not None and client_id is not None:
        if data_distribution.lower() == "non-iid":
            allowed_labels = fed_allowed
            L = len(allowed_labels)
            per_client = L // total_clients
            remainder = L % total_clients
            try:
                client_index = int(''.join(filter(str.isdigit, client_id)))
            except:
                client_index = 0
            start = client_index * per_client + min(client_index, remainder)
            end = start + per_client + (1 if client_index < remainder else 0)
            client_allowed = allowed_labels[start:end]
            print(f"[load_cifar100] Client {client_id} non-iid allowed classes: {client_allowed}")
            full_train = train_set.dataset if hasattr(train_set, "dataset") else train_set
            full_test = test_set.dataset if hasattr(test_set, "dataset") else test_set
            train_indices = [i for i, label in enumerate(full_train.targets) if label in client_allowed]
            test_indices = [i for i, label in enumerate(full_test.targets) if label in client_allowed]
            train_set = data.Subset(full_train, train_indices)
            test_set = data.Subset(full_test, test_indices)
        elif data_distribution.lower() == "iid":
            total_train = len(train_set)
            total_test = len(test_set)
            try:
                client_index = int(''.join(filter(str.isdigit, client_id)))
            except:
                client_index = 0
            train_indices = split_indices_iid(total_train, total_clients, client_index)
            test_indices = split_indices_iid(total_test, total_clients, client_index)
            train_set = data.Subset(train_set, train_indices)
            test_set = data.Subset(test_set, test_indices)
    
    trainloader = data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
    testloader = data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False)
    metadata = {"datasetName": "CIFAR-100", "inputShape": [3, 32, 32], "numClasses": 100}
    return trainloader, testloader, metadata



#############################################
# MNIST
#############################################
def load_mnist(data_path, train_batch_size=32, test_batch_size=32,
               client_id=None, total_clients=None,
               federation_id=None, total_federations=None,
               data_distribution="iid", experiment_mode=True):
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
    
    # Federation-level split: determine which classes belong to this federation.
    if experiment_mode and (federation_id is not None and total_federations is not None and total_federations != 0):
        n_classes = 10
        classes_per_fed = n_classes // total_federations
        start_label = int(federation_id * classes_per_fed)
        end_label = start_label + classes_per_fed if federation_id < total_federations - 1 else n_classes
        fed_allowed = list(range(start_label, end_label))
        print(f"[load_mnist] Federation {federation_id}/{total_federations} allowed classes: {fed_allowed}")
        train_indices = [i for i, label in enumerate(train_set.targets.tolist()) if label in fed_allowed]
        test_indices  = [i for i, label in enumerate(test_set.targets.tolist()) if label in fed_allowed]
        train_set = data.Subset(train_set, train_indices)
        test_set = data.Subset(test_set, test_indices)
    else:
        fed_allowed = list(range(10))
    
    # Client-level split: assign each client a random subset of the already filtered data.
    if experiment_mode and total_clients is not None and client_id is not None:
        if data_distribution.lower() == "non-iid":
            # Instead of further splitting the allowed labels, we split the data indices.
            total_train = len(train_set)
            total_test = len(test_set)
            try:
                # <-- FIX: extract client index from client_id by splitting on '_' 
                client_index = int(client_id.split('_')[-1])
            except:
                client_index = 0
            train_indices = split_indices_iid(total_train, total_clients, client_index)
            test_indices = split_indices_iid(total_test, total_clients, client_index)
            train_set = data.Subset(train_set, train_indices)
            test_set = data.Subset(test_set, test_indices)
        elif data_distribution.lower() == "iid":
            total_train = len(train_set)
            total_test = len(test_set)
            try:
                client_index = int(client_id.split('_')[-1])
            except:
                client_index = 0
            train_indices = split_indices_iid(total_train, total_clients, client_index)
            test_indices = split_indices_iid(total_test, total_clients, client_index)
            train_set = data.Subset(train_set, train_indices)
            test_set = data.Subset(test_set, test_indices)
    
    trainloader = data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
    testloader  = data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False)
    metadata = {"datasetName": "MNIST", "inputShape": [1, 28, 28], "numClasses": 10}
    return trainloader, testloader, metadata

#############################################
# CIFAR-10
#############################################
def load_cifar10(data_path, train_batch_size=32, test_batch_size=32,
                 client_id=None, total_clients=None,
                 federation_id=None, total_federations=None,
                 data_distribution="iid", experiment_mode=True):
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
    
    if experiment_mode and (federation_id is not None and total_federations is not None and total_federations != 0):
        n_classes = 10
        classes_per_fed = n_classes // total_federations
        start_label = int(federation_id * classes_per_fed)
        end_label = start_label + classes_per_fed if federation_id < total_federations - 1 else n_classes
        fed_allowed = list(range(start_label, end_label))
        print(f"[load_cifar10] Federation {federation_id}/{total_federations} allowed classes: {fed_allowed}")
        train_indices = [i for i, label in enumerate(train_set.targets) if label in fed_allowed]
        test_indices  = [i for i, label in enumerate(test_set.targets) if label in fed_allowed]
        train_set = data.Subset(train_set, train_indices)
        test_set = data.Subset(test_set, test_indices)
    else:
        fed_allowed = list(range(10))
    
    if experiment_mode and total_clients is not None and client_id is not None:
        total_train = len(train_set)
        total_test = len(test_set)
        try:
            client_index = int(client_id.split('_')[-1])
        except:
            client_index = 0
        if data_distribution.lower() == "iid":
            train_indices = split_indices_iid(total_train, total_clients, client_index)
            test_indices = split_indices_iid(total_test, total_clients, client_index)
        elif data_distribution.lower() == "non-iid":
            full_train = train_set.dataset if hasattr(train_set, "dataset") else train_set
            full_test = test_set.dataset if hasattr(test_set, "dataset") else test_set
            sorted_train = sorted(range(len(full_train)), key=lambda i: full_train.targets[i])
            sorted_test = sorted(range(len(full_test)), key=lambda i: full_test.targets[i])
            per_client_train = len(sorted_train) // total_clients
            per_client_test = len(sorted_test) // total_clients
            start_train = client_index * per_client_train
            end_train = start_train + per_client_train
            start_test = client_index * per_client_test
            end_test = start_test + per_client_test
            train_indices = sorted_train[start_train:end_train]
            test_indices = sorted_test[start_test:end_test]
        train_set = data.Subset(train_set, train_indices)
        test_set = data.Subset(test_set, test_indices)
    
    trainloader = data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
    testloader = data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False)
    metadata = {"datasetName": "CIFAR-10", "inputShape": [3, 32, 32], "numClasses": 10}
    return trainloader, testloader, metadata

#############################################
# Occupancy Dataset
#############################################
def generate_occupancy_data(num_samples, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    # Generate a time-of-day feature.
    time_of_day = np.random.uniform(0, 24, (num_samples, 1)).astype(np.float32)
    # Simulate occupancy with a sinusoidal daily pattern plus noise.
    occupancy = (100 + 50 * np.sin((time_of_day - 12) / 24 * 2 * np.pi) +
                 np.random.normal(0, 10, (num_samples, 1))).astype(np.float32)
    # Four additional random features.
    extra = np.random.rand(num_samples, 4).astype(np.float32)
    X = np.concatenate([time_of_day, extra], axis=1)
    threshold = np.median(occupancy)
    y = (occupancy > threshold).astype(np.int64).squeeze()
    return X, y




def load_har_data(data_path, train_batch_size=32, test_batch_size=32,
                  client_id=None, total_clients=None,
                  federation_id=None, total_federations=None,
                  data_distribution="iid", experiment_mode=True):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    print("Current Working Directory:", os.getcwd())
    print("Expected File Path:", os.path.abspath("./data/train/X_train.txt"))
    # Paths for the dataset files
    train_features_file = os.path.join(data_path, "train", "X_train.txt")
    train_labels_file = os.path.join(data_path, "train", "y_train.txt")
    test_features_file = os.path.join(data_path, "test", "X_test.txt")
    test_labels_file = os.path.join(data_path, "test", "y_test.txt")
    
    # Load the dataset
    X_train = np.loadtxt(train_features_file)
    y_train = np.loadtxt(train_labels_file).astype(int) - 1  # Convert to zero-based indexing
    X_test = np.loadtxt(test_features_file)
    y_test = np.loadtxt(test_labels_file).astype(int) - 1  # Convert to zero-based indexing

    # Convert to DataFrame
    df_train = pd.DataFrame(X_train)
    df_train["Activity"] = y_train
    df_test = pd.DataFrame(X_test)
    df_test["Activity"] = y_test
    df = pd.concat([df_train, df_test])

    # Federation-level class partitioning
    if experiment_mode and (federation_id is not None and total_federations is not None and total_federations != 0):
        num_classes = len(df["Activity"].unique())
        classes_per_fed = num_classes // total_federations
        start_class = federation_id * classes_per_fed
        end_class = start_class + classes_per_fed if federation_id < total_federations - 1 else num_classes
        fed_classes = list(range(start_class, end_class))
        
        print(f"[load_har] Federation {federation_id}/{total_federations} assigned classes: {fed_classes}")

        # Keep only data corresponding to the assigned classes
        df = df[df["Activity"].isin(fed_classes)]

    # Split into train and test sets
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    X_train = df_train.drop(columns=["Activity"]).values.astype(np.float32)
    y_train = df_train["Activity"].values.astype(np.int64)
    X_test = df_test.drop(columns=["Activity"]).values.astype(np.float32)
    y_test = df_test["Activity"].values.astype(np.int64)

    # Client-level class partitioning (Non-IID or IID)
    if experiment_mode and total_clients is not None and client_id is not None:
        if data_distribution.lower() == "non-iid":
            allowed_labels = df["Activity"].unique()
            L = len(allowed_labels)
            per_client = L // total_clients
            remainder = L % total_clients
            try:
                client_index = int(''.join(filter(str.isdigit, client_id)))
            except:
                client_index = 0
            start = client_index * per_client + min(client_index, remainder)
            end = start + per_client + (1 if client_index < remainder else 0)
            client_labels = allowed_labels[start:end]

            print(f"[load_har] Client {client_id} assigned labels: {client_labels}")

            df_train = df_train[df_train["Activity"].isin(client_labels)]
            df_test = df_test[df_test["Activity"].isin(client_labels)]
        elif data_distribution.lower() == "iid":
            total_train = len(X_train)
            total_test = len(X_test)
            try:
                client_index = int(''.join(filter(str.isdigit, client_id)))
            except:
                client_index = 0
            train_indices = split_indices_iid(total_train, total_clients, client_index)
            test_indices = split_indices_iid(total_test, total_clients, client_index)
            X_train = np.array(X_train)[train_indices]
            y_train = np.array(y_train)[train_indices]
            X_test = np.array(X_test)[test_indices]
            y_test = np.array(y_test)[test_indices]

    # Create PyTorch datasets
    train_dataset = data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                       torch.tensor(y_train, dtype=torch.long))
    test_dataset = data.TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                      torch.tensor(y_test, dtype=torch.long))

    trainloader = data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    testloader = data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    metadata = {"datasetName": "HAR", "inputShape": [X_train.shape[1]], "numClasses": 6}
    return trainloader, testloader, metadata




def load_occupancy_data(data_path, train_batch_size=32, test_batch_size=32,
                        client_id=None, total_clients=None,
                        federation_id=None, total_federations=None,
                        data_distribution="iid", experiment_mode=True):

    csv_file = os.path.join(data_path, "occupancy", "occupancy.csv")
    if not os.path.exists(csv_file):
        print("Warning: Occupancy CSV not found. Falling back to synthetic occupancy data.")
        X, y = generate_occupancy_data(7000, random_seed=42)
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        # Create a DataFrame with default sensor names
        df_train = pd.DataFrame(X_train, columns=["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])
        df_train["Occupancy"] = y_train
        df_test = pd.DataFrame(X_test, columns=["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])
        df_test["Occupancy"] = y_test
        df = pd.concat([df_train, df_test])
    else:
        df = pd.read_csv(csv_file)
        if "date" in df.columns:
            df = df.drop(columns=["date"])
        df["Occupancy"] = df["Occupancy"].astype(int)

    # --- NEW: Federation-level feature (sensor) split ---
    # If federation-level information is provided, split the sensor columns.
    # Assume that all columns except "Occupancy" are sensor features.
    if experiment_mode and (federation_id is not None and total_federations is not None and total_federations!=0):
        all_sensors = [col for col in df.columns if col != "Occupancy"]
        num_sensors = len(all_sensors)
        # Compute how many sensors per federation (use ceiling so that every federation gets at least one)
        import math
        sensors_per_fed = math.ceil(num_sensors / total_federations)
        start_idx = federation_id * sensors_per_fed
        end_idx = min(start_idx + sensors_per_fed, num_sensors)
        fed_sensors = all_sensors[start_idx:end_idx]
        print(f"[load_occupancy] Federation {federation_id}/{total_federations} assigned sensors: {fed_sensors}")
        # Ensure all federations return a full feature vector
        # Ensure all required features exist before subsetting
        full_feature_set = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]

        # Add missing columns and fill them with zeros
        for sensor in full_feature_set:
            if sensor not in df.columns:
                df[sensor] = 0.0  

        # Maintain correct feature order
        df = df[full_feature_set + ["Occupancy"]]
    # --- End of federation-level feature split ---

    # Now split into train/test (if not already split)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    X_train = df_train.drop(columns=["Occupancy"]).values.astype(np.float32)
    y_train = df_train["Occupancy"].values.astype(np.int64)
    X_test = df_test.drop(columns=["Occupancy"]).values.astype(np.float32)
    y_test = df_test["Occupancy"].values.astype(np.int64)

    # Client-level split (either IID or non‑IID) remains as before.
    if experiment_mode and total_clients is not None and client_id is not None:
        if data_distribution.lower() == "non-iid":
            # For non‑IID, we sort by one feature (say, the first sensor) and partition.
            X_train, y_train = zip(*sorted(zip(X_train, y_train), key=lambda x: x[0][0]))
            X_test, y_test = zip(*sorted(zip(X_test, y_test), key=lambda x: x[0][0]))
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            X_test = np.array(X_test)
            y_test = np.array(y_test)
            total_train = len(X_train)
            total_test = len(X_test)
            try:
                client_index = int(''.join(filter(str.isdigit, client_id)))
            except:
                client_index = 0
            start_train = client_index * (total_train // total_clients)
            end_train = start_train + (total_train // total_clients)
            start_test = client_index * (total_test // total_clients)
            end_test = start_test + (total_test // total_clients)
            X_train = X_train[start_train:end_train]
            y_train = y_train[start_train:end_train]
            X_test = X_test[start_test:end_test]
            y_test = y_test[start_test:end_test]
        elif data_distribution.lower() == "iid":
            total_train = len(X_train)
            total_test = len(X_test)
            try:
                client_index = int(''.join(filter(str.isdigit, client_id)))
            except:
                client_index = 0
            train_indices = split_indices_iid(total_train, total_clients, client_index)
            test_indices = split_indices_iid(total_test, total_clients, client_index)
            X_train = np.array(X_train)[train_indices]
            y_train = np.array(y_train)[train_indices]
            X_test = np.array(X_test)[test_indices]
            y_test = np.array(y_test)[test_indices]

    train_dataset = data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                         torch.tensor(y_train, dtype=torch.long))
    test_dataset = data.TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                        torch.tensor(y_test, dtype=torch.long))
    trainloader = data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    testloader = data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    metadata = {"datasetName": "Occupancy", "inputShape": [X_train.shape[1]], "numClasses": 2}
    return trainloader, testloader, metadata


#############################################
# Port Data (Synthetic Port)
#############################################
def load_port_data(data_path, train_batch_size=32, test_batch_size=32,
                   client_id=None, total_clients=None,
                   federation_id=None, total_federations=None,
                   data_distribution="iid", experiment_mode=True, client_type="occupancy"):
    num_train = 6000
    num_test = 1000
    seed = hash(client_id) % 10000 if client_id is not None else None
    if client_type == "occupancy":
        X_train, y_train = generate_occupancy_data(num_train, random_seed=seed)
        X_test, y_test = generate_occupancy_data(num_test, random_seed=seed)
    elif client_type in ["bus", "sensor"]:
        X_train, y_train = generate_occupancy_data(num_train, random_seed=seed)
        X_test, y_test = generate_occupancy_data(num_test, random_seed=seed)
    else:
        raise ValueError("Unknown client_type for port data")
    
    # Federation-level: assume full label space [0,1] for port.
    fed_allowed = [0, 1]
    
    if experiment_mode and total_clients is not None and client_id is not None:
        total_train = len(X_train)
        total_test = len(X_test)
        try:
            client_index = int(client_id.split('_')[-1])
        except:
            client_index = 0
        if data_distribution.lower() == "iid":
            indices_train = split_indices_iid(total_train, total_clients, client_index)
            indices_test = split_indices_iid(total_test, total_clients, client_index)
            X_train = np.array(X_train)[indices_train]
            y_train = np.array(y_train)[indices_train]
            X_test = np.array(X_test)[indices_test]
            y_test = np.array(y_test)[indices_test]
        elif data_distribution.lower() == "non-iid":
            sorted_train = sorted(zip(X_train, y_train), key=lambda x: x[0][0])
            sorted_test = sorted(zip(X_test, y_test), key=lambda x: x[0][0])
            X_train, y_train = zip(*sorted_train)
            X_test, y_test = zip(*sorted_test)
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            X_test = np.array(X_test)
            y_test = np.array(y_test)
            per_client_train = len(X_train) // total_clients
            per_client_test = len(X_test) // total_clients
            start_train = client_index * per_client_train
            end_train = start_train + per_client_train
            start_test = client_index * per_client_test
            end_test = start_test + per_client_test
            X_train = X_train[start_train:end_train]
            y_train = y_train[start_train:end_train]
            X_test = X_test[start_test:end_test]
            y_test = y_test[start_test:end_test]
    train_dataset = data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                         torch.tensor(y_train, dtype=torch.long))
    test_dataset = data.TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                        torch.tensor(y_test, dtype=torch.long))
    trainloader = data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    testloader = data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    metadata = {"datasetName": "SyntheticPort", "inputShape": [X_train.shape[1]], "numClasses": 2}
    return trainloader, testloader, metadata

#############################################
# FashionMNIST
#############################################
def load_fashion_mnist(data_path, train_batch_size=32, test_batch_size=32,
                       client_id=None, total_clients=None,
                       federation_id=None, total_federations=None,
                       data_distribution="iid", experiment_mode=True):
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.FashionMNIST(root=data_path, train=False, download=True, transform=transform)
    
    if experiment_mode and (federation_id is not None and total_federations is not None and total_federations != 0):
        n_classes = 10
        classes_per_fed = n_classes // total_federations
        start_label = int(federation_id * classes_per_fed)
        end_label = start_label + classes_per_fed if federation_id < total_federations - 1 else n_classes
        fed_allowed = list(range(start_label, end_label))
        print(f"[load_fashion_mnist] Federation {federation_id}/{total_federations} allowed classes: {fed_allowed}")
        train_indices = [i for i, label in enumerate(train_set.targets.tolist()) if label in fed_allowed]
        test_indices = [i for i, label in enumerate(test_set.targets.tolist()) if label in fed_allowed]
        train_set = data.Subset(train_set, train_indices)
        test_set = data.Subset(test_set, test_indices)
    else:
        fed_allowed = list(range(10))
    
    if experiment_mode and total_clients is not None and client_id is not None:
        total_train = len(train_set)
        total_test = len(test_set)
        try:
            client_index = int(client_id.split('_')[-1])
        except:
            client_index = 0
        if data_distribution.lower() == "iid":
            train_indices = split_indices_iid(total_train, total_clients, client_index)
            test_indices = split_indices_iid(total_test, total_clients, client_index)
        elif data_distribution.lower() == "non-iid":
            full_train = train_set.dataset if hasattr(train_set, "dataset") else train_set
            full_test = test_set.dataset if hasattr(test_set, "dataset") else test_set
            sorted_train = sorted(range(len(full_train)), key=lambda i: full_train.targets[i])
            sorted_test = sorted(range(len(full_test)), key=lambda i: full_test.targets[i])
            per_client_train = len(sorted_train) // total_clients
            per_client_test = len(sorted_test) // total_clients
            start_train = client_index * per_client_train
            end_train = start_train + per_client_train
            start_test = client_index * per_client_test
            end_test = start_test + per_client_test
            train_indices = sorted_train[start_train:end_train]
            test_indices = sorted_test[start_test:end_test]
        train_set = data.Subset(train_set, train_indices)
        test_set = data.Subset(test_set, test_indices)
    
    trainloader = data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
    testloader = data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False)
    metadata = {"datasetName": "FashionMNIST", "inputShape": [1, 28, 28], "numClasses": 10}
    return trainloader, testloader, metadata

#############################################
# Unified loader: choose dataset based on a key.
#############################################
def load_dataset(dataset_type: str, data_path: str, train_batch_size=32, test_batch_size=32,
                 client_id=None, total_clients=None,
                 federation_id=None, total_federations=None,
                 data_distribution="iid", experiment_mode=True):
    dataset_type = dataset_type.lower()
    if dataset_type == "mnist":
        return load_mnist(data_path, train_batch_size, test_batch_size,
                          client_id, total_clients, federation_id, total_federations,
                          data_distribution, experiment_mode)
    elif dataset_type == "cifar10":
        return load_cifar10(data_path, train_batch_size, test_batch_size,
                            client_id, total_clients, federation_id, total_federations,
                            data_distribution, experiment_mode)
    elif dataset_type == "cifar100":
        return load_cifar100(data_path, train_batch_size, test_batch_size,
                             client_id, total_clients, federation_id, total_federations,
                             data_distribution, experiment_mode)
    #elif dataset_type == "port":
    #    return load_port_data(data_path, train_batch_size, test_batch_size,
    #                          client_id, total_clients, federation_id, total_federations,
    #                          data_distribution, experiment_mode, client_type="occupancy")
    # elif dataset_type == "occupancy":
    #     return load_occupancy_data(data_path, train_batch_size, test_batch_size,
    #                                client_id, total_clients, federation_id, total_federations,
    #                                data_distribution, experiment_mode)
    elif dataset_type == "fashionmnist":
        return load_fashion_mnist(data_path, train_batch_size, test_batch_size,
                                  client_id, total_clients, federation_id, total_federations,
                                  data_distribution, experiment_mode)
    elif dataset_type == "har":
        return load_har_data(data_path, train_batch_size, test_batch_size,
                                  client_id, total_clients, federation_id, total_federations,
                                  data_distribution, experiment_mode) 
    else:
        raise ValueError(f"Unknown dataset '{dataset_type}'.")




###############################################################################
#                           BROKER / CONTEXT LOGIC                            #
###############################################################################

def parse_broker_url(broker_url: str):
    if broker_url.startswith("mqtt://"):
        broker_url = broker_url.replace("mqtt://", "")
    parts = broker_url.split(":")
    if len(parts) == 2:
        return parts[0], int(parts[1])
    return broker_url, 1026

def send_fl_job_advertisement_to_broker(broker_url: str, fl_job_info: dict):
    broker_host, broker_port = parse_broker_url(broker_url)
    job_entity = {
        "id": "urn:ngsi-ld:FLJob:" + fl_job_info.get("job_id", "unknown_job"),
        "type": "FederatedLearningJob",
        "@context": ["https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld"],
        "description": {"value": fl_job_info.get("description", "No description")},
        "serverAddress": {"value": fl_job_info.get("server_address", "")},
        "numRounds": {"value": fl_job_info.get("num_rounds", 1)},
        "strategy": {"value": fl_job_info.get("strategy", "fedavg")},
        "status": {"value": "advertised"},
        "modelSpec": {
            "value": {
                "modelType": fl_job_info.get("model_type", "unknown_model"),
                "modelParams": fl_job_info.get("model_params", {}),
            }
        },
        "dataSpec": {"value": fl_job_info.get("data_spec", {})},
        "minNumSamples": {"value": fl_job_info.get("min_num_samples", 1)},
        "targetAccuracy": {"value": fl_job_info.get("target_accuracy", 0.0)},
        "requiredParticipants": {"value": fl_job_info.get("required_participants", 2)},
        "dataDistribution": {"value": fl_job_info.get("data_distribution", "IID")},
        "localEpochs": {"value": fl_job_info.get("local_epochs", 1)},
        "localBatchSize": {"value": fl_job_info.get("local_batch_size", 32)},
    }
    my_area = "federation_area"
    my_loc = "production_location"
    qos = 1
    client = mqtt.Client(clean_session=True)
    client.connect(broker_host, broker_port)
    post_entity(
        data=job_entity,
        my_area=my_area,
        broker=broker_host,
        port=broker_port,
        qos=qos,
        my_loc=my_loc,
        bypass_existence_check=0,
        client=client
    )
    print("[send_fl_job_advertisement_to_broker] Posted FL job advertisement to broker.")

def update_global_model_context(broker_url, round_number, global_model_uri, done=False, job_id="unknown_job"):
    broker_host, broker_port = parse_broker_url(broker_url)
    client = mqtt.Client(clean_session=True)
    client.connect(broker_host, broker_port)
    entity_id = f"urn:ngsi-ld:GlobalModel:{job_id}:round{round_number}"
    data = {
        "id": entity_id,
        "type": "GlobalModel",
        "@context": ["https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld"],
        "roundNumber": {"value": round_number},
        "modelURI": {"value": global_model_uri},
        "status": {"value": "completed" if done else "ongoing"},
        "belongsToJob": {"value": job_id},
    }
    my_area = "federation_area"
    my_loc  = "production_location"
    qos     = 1
    post_entity(
        data=data,
        my_area=my_area,
        broker=broker_host,
        port=broker_port,
        qos=qos,
        my_loc=my_loc,
        bypass_existence_check=1,
        client=client
    )
    print(f"[update_global_model_context] Round {round_number}: done={done}, URI={global_model_uri}")

def send_client_capability_advertisement(broker_url: str, client_info: dict):
    broker_host, broker_port = parse_broker_url(broker_url)
    client_entity = {
        "id": "urn:ngsi-ld:FLClient:" + client_info.get("client_id", "unknown_client"),
        "type": "FederatedLearningClient",
        "@context": ["https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld"],
        "clientID": {"value": client_info.get("client_id", "unknown_client")},
        "availableData": {"value": client_info.get("available_data", "unknown")},
        "computeSpec": {"value": client_info.get("compute_spec", "default")},
        "clientMode": {"value": client_info.get("client_mode", "active")},
        "status": {"value": "advertised"},
        "description": {"value": client_info.get("description", "FL client advertising its capabilities.")}
    }
    if "notificationEndpoint" in client_info:
        client_entity["notificationEndpoint"] = {"value": client_info["notificationEndpoint"]}
    client = mqtt.Client(clean_session=True)
    client.connect(broker_host, broker_port)
    post_entity(
        data=client_entity,
        my_area="federation_area",
        broker=broker_host,
        port=broker_port,
        qos=1,
        my_loc="production_location",
        bypass_existence_check=0,
        client=client
    )
    print("[send_client_capability_advertisement] Advertised client capabilities to broker.")

###############################################################################
#                      ADDED: SERVER INVITATION FUNCTION                      #
###############################################################################

def send_server_invitations(client_endpoints, server_address, job_id):
    for endpoint in client_endpoints:
        try:
            data = {"job_id": job_id, "server_address": server_address}
            response = requests.post(endpoint, json=data)
            print(f"[send_server_invitations] Invitation sent to {endpoint}, response code {response.status_code}")
        except Exception as e:
            print(f"[send_server_invitations] Error sending invitation to {endpoint}: {e}")

###############################################################################
#                      SIMPLE HTTP SERVER FOR MODELS                          #
###############################################################################

def serve_local_directory(directory="models", port=8000):
    handler = http.server.SimpleHTTPRequestHandler
    def serve_forever():
        os.chdir(directory)
        httpd = socketserver.TCPServer(("", port), handler)
        print(f"[serve_local_directory] Serving '{directory}' on port {port}")
        httpd.serve_forever()
    os.makedirs(directory, exist_ok=True)
    thread = threading.Thread(target=serve_forever, daemon=True)
    thread.start()
    return thread

def save_global_model(model: nn.Module, round_number: int, save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"global_round_{round_number}.pt")
    torch.save(model.state_dict(), filename)
    print(f"[save_global_model] Saved model checkpoint: {filename}")
    return filename

###############################################################################
#                UPDATED: SUPPORT FOR MULTIPLE SERVER STRATEGIES              #
###############################################################################

def _get_base_strategy(strategy_name="fedavg", **kwargs):
    strategy = strategy_name.lower()
    if strategy == "fedavg":
        return fl.server.strategy.FedAvg(
            fraction_fit=kwargs.get("fraction_fit", 1.0),
            fraction_evaluate=kwargs.get("fraction_eval", 1.0),
            min_fit_clients=kwargs.get("min_fit_clients", 2),
            min_evaluate_clients=kwargs.get("min_eval_clients", 2),
            min_available_clients=kwargs.get("min_available_clients", 2),
            on_fit_config_fn=kwargs.get("on_fit_config_fn"),
        )
    elif strategy == "fedprox":
        return fl.server.strategy.FedProx(
            fraction_fit=kwargs.get("fraction_fit", 1.0),
            fraction_evaluate=kwargs.get("fraction_eval", 1.0),
            min_fit_clients=kwargs.get("min_fit_clients", 2),
            min_evaluate_clients=kwargs.get("min_eval_clients", 2),
            min_available_clients=kwargs.get("min_available_clients", 2),
            on_fit_config_fn=kwargs.get("on_fit_config_fn"),
            proximal_mu=0.1,
        )
    elif strategy == "fedopt":
        lr = kwargs.get("lr", 0.001)
        return fl.server.strategy.FedOpt(
            server_optimizer=kwargs.get("server_optimizer", optim.Adam),
            server_optimizer_args={"lr": lr},
            fraction_fit=kwargs.get("fraction_fit", 1.0),
            fraction_evaluate=kwargs.get("fraction_eval", 1.0),
            min_fit_clients=kwargs.get("min_fit_clients", 2),
            min_evaluate_clients=kwargs.get("min_eval_clients", 2),
            min_available_clients=kwargs.get("min_available_clients", 2),
            on_fit_config_fn=kwargs.get("on_fit_config_fn"),
        )
    elif strategy == "fedasync":
        return fl.server.strategy.FedAsync(
            fraction_fit=kwargs.get("fraction_fit", 1.0),
            min_fit_clients=kwargs.get("min_fit_clients", 2),
            min_available_clients=kwargs.get("min_available_clients", 2),
        )
    else:
        print(f"Unknown strategy '{strategy_name}', defaulting to FedAvg")
        return fl.server.strategy.FedAvg(
            fraction_fit=kwargs.get("fraction_fit", 1.0),
            fraction_evaluate=kwargs.get("fraction_eval", 1.0),
            min_fit_clients=kwargs.get("min_fit_clients", 2),
            min_evaluate_clients=kwargs.get("min_eval_clients", 2),
            min_available_clients=kwargs.get("min_available_clients", 2),
            on_fit_config_fn=kwargs.get("on_fit_config_fn"),
        )

def evaluate_model(model, testloader, device):
    # For classification tasks (like occupancy detection), use CrossEntropyLoss.
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

def create_custom_strategy(
    global_model,
    broker_url,
    job_id,
    num_rounds,
    base_strategy,
    host_ip="127.0.0.1",
    http_port=8000,
    evaluation_data=None  # Tuple: (testloader, metadata)
):
    # Define a filename for per-round logging based on the job_id.
    log_filename = f"{job_id}_round_metrics.csv"
    # Create the CSV file and write the header if it doesn't exist.
    if not os.path.exists(log_filename):
        with open(log_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "timestamp", "loss", "accuracy", "round_duration", "num_clients", "num_failures"])
    
    class CustomStrategy(base_strategy.__class__):
        def __init__(
            self,
            fraction_fit,
            fraction_evaluate,
            min_fit_clients,
            min_evaluate_clients,
            min_available_clients,
            initial_parameters,
            global_model,
            broker_url,
            host_ip,
            http_port,
            job_id,
            num_rounds,
            evaluation_data=None,
            **kwargs
        ):
            super().__init__(
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_available_clients,
                initial_parameters=initial_parameters,
                **kwargs
            )
            self.global_model = global_model
            self.broker_url = broker_url
            self.host_ip = host_ip
            self.http_port = http_port
            self.job_id = job_id
            self.num_rounds = num_rounds
            self.evaluation_data = evaluation_data

        def aggregate_fit(self, rnd, results, failures):
            # Record the starting time of the round.
            round_start = datetime.now()
            
            # Perform the default aggregation.
            agg_result = super().aggregate_fit(rnd, results, failures)
            if agg_result is None:
                return agg_result

            aggregated_params_proto, _ = agg_result
            aggregated_params = parameters_to_ndarrays(aggregated_params_proto)
            params_dict = zip(self.global_model.state_dict().keys(), aggregated_params)
            with torch.no_grad():
                if aggregated_params is not None:  # Ensure parameters exist
                    params_dict = zip(self.global_model.state_dict().keys(), aggregated_params)
                    state_dict = {k: torch.tensor(v) for k, v in params_dict}
                    self.global_model.load_state_dict(state_dict, strict=True)
                else:
                    print("[Error] aggregated_params is None, skipping state_dict assignment!")

            # Save the updated global model and update the broker.
            filename = save_global_model(self.global_model, rnd)
            model_uri = f"http://{self.host_ip}:{self.http_port}/{os.path.basename(filename)}"
            update_global_model_context(
                broker_url=self.broker_url,
                round_number=rnd,
                global_model_uri=model_uri,
                done=(rnd >= self.num_rounds),
                job_id=self.job_id
            )
            # Evaluate the model if evaluation data is provided.
            if self.evaluation_data is not None:
                testloader, _ = self.evaluation_data  # Get the test loader
                dataset_size = len(testloader.dataset)  # Total test samples

                reduced_size = min(1000, dataset_size)  # Use only 1000 test samples (or less if dataset is smaller)
                indices = np.random.choice(dataset_size, reduced_size, replace=False)  # Randomly pick indices

                testloader = torch.utils.data.DataLoader(testloader.dataset, batch_size=32, sampler=SubsetRandomSampler(indices))
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.global_model.to(device)
                global_loss, global_accuracy = evaluate_model(self.global_model, testloader, device)
                # Print the evaluation in the expected format.
                print(f"[CustomStrategy] Global model evaluation - Loss: {global_loss:.4f}, Accuracy: {global_accuracy:.2f}%")
            else:
                global_loss, global_accuracy = None, None

            # Record round end time and calculate duration.
            round_end = datetime.now()
            round_duration = (round_end - round_start).total_seconds()

            # Log the number of clients that participated and failures.
            num_clients = len(results)
            num_failures = len(failures)

            # Append the per-round metrics to the CSV file.
            with open(log_filename, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    rnd,
                    datetime.now().isoformat(),
                    global_loss if global_loss is not None else "COULDNTRECORD",
                    global_accuracy if global_accuracy is not None else "COULDNTRECORD",
                    round_duration,
                    num_clients,
                    num_failures
                ])
            print(f"[CustomStrategy] Round {rnd} | Duration: {round_duration:.2f}s | Clients: {num_clients} | Failures: {num_failures}")
            return agg_result
    proximal_mu_value = 0.1 
    if isinstance(base_strategy, fl.server.strategy.FedProx):
        return CustomStrategy(
            fraction_fit=base_strategy.fraction_fit,
            fraction_evaluate=base_strategy.fraction_evaluate,
            min_fit_clients=base_strategy.min_fit_clients,
            min_evaluate_clients=base_strategy.min_evaluate_clients,
            min_available_clients=base_strategy.min_available_clients,
            initial_parameters=base_strategy.initial_parameters,
            global_model=global_model,
            broker_url=broker_url,
            host_ip=host_ip,
            http_port=http_port,
            job_id=job_id,
            num_rounds=num_rounds,
            evaluation_data=evaluation_data,
            proximal_mu=proximal_mu_value,
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average_evaluation,
        )
    else:
        return CustomStrategy(
            fraction_fit=base_strategy.fraction_fit,
            fraction_evaluate=base_strategy.fraction_evaluate,
            min_fit_clients=base_strategy.min_fit_clients,
            min_evaluate_clients=base_strategy.min_evaluate_clients,
            min_available_clients=base_strategy.min_available_clients,
            initial_parameters=base_strategy.initial_parameters,
            global_model=global_model,
            broker_url=broker_url,
            host_ip=host_ip,
            http_port=http_port,
            job_id=job_id,
            num_rounds=num_rounds,
            evaluation_data=evaluation_data,
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average_evaluation,)
###############################################################################
#                 ADDED: CLIENT NOTIFICATION SERVER LOGIC                     #
###############################################################################

from http.server import HTTPServer, BaseHTTPRequestHandler

class NotificationHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        print(f"[NotificationHandler] Received invitation: {post_data.decode('utf-8')}")
        self.send_response(200)
        self.end_headers()
        try:
            data = json.loads(post_data)
            new_server_address = data.get("server_address")
            if new_server_address:
                self.server.client_manager.server_address = new_server_address
                print(f"[NotificationHandler] Updated server address to {new_server_address}")
            self.server.client_manager.start_client_event.set()
        except Exception as e:
            print(f"[NotificationHandler] Error processing invitation: {e}")
    def log_message(self, format, *args):
        return

def start_notification_server(client_manager, port):
    server_address = ('', port)
    httpd = HTTPServer(server_address, NotificationHandler)
    httpd.client_manager = client_manager
    print(f"[start_notification_server] Listening for notifications on port {port}")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd

###############################################################################
#                           SERVER AND CLIENT LOGIC                           #
###############################################################################

class FLServerManager:
    def __init__(
        self,
        broker_url,
        server_address,
        strategy_name,
        num_rounds,
        model_type,
        model_params,
        job_id,
        host_ip,
        http_port,
        local_epochs=1,
        local_batch_size=32,
        experiment_mode=True,
        dataset="mnist",
        data_path="./data",
        required_participants=2
    ):
        self.broker_url = broker_url
        self.server_address = server_address
        self.strategy_name = strategy_name
        self.num_rounds = num_rounds
        self.model_type = model_type
        self.model_params = model_params or {}
        self.job_id = job_id
        self.host_ip = host_ip
        self.http_port = http_port
        self.local_epochs = local_epochs
        self.local_batch_size = local_batch_size
        self.experiment_mode = experiment_mode
        self.dataset = dataset
        self.data_path = data_path
        self.required_participants = required_participants

    def on_fit_config_fn(self, rnd: int) -> dict:
        return {"local_epochs": self.local_epochs, "batch_size": self.local_batch_size}

    def start_server(self):
        serve_local_directory("models", self.http_port)
        global_model = create_model(self.model_type, self.model_params)
        base_strategy = _get_base_strategy(
            self.strategy_name,
            on_fit_config_fn=self.on_fit_config_fn,
            min_fit_clients=self.required_participants,
            min_evaluate_clients=self.required_participants,
            min_available_clients=self.required_participants,
        )
        evaluation_data = None
        if self.experiment_mode:
            #  Force Full Test Data Loading (Ignore Federation or Client Splits)
            _, testloader, metadata = load_dataset(
                self.dataset, self.data_path,
                train_batch_size=32, test_batch_size=32,
                client_id=None,               #  Ensure it's `None` so it doesn't filter clients
                total_clients=None,           #  Ensure full dataset is loaded
                federation_id=None,           #  Avoid federation-based filtering
                total_federations=None,       #  Ensure all federations are included
                data_distribution="iid",      #  Avoid potential non-IID partitioning
                experiment_mode=False         #  Disable experiment mode to load everything
            )

            evaluation_data = (testloader, None)
        custom_strategy = create_custom_strategy(
            global_model=global_model,
            broker_url=self.broker_url,
            job_id=self.job_id,
            num_rounds=self.num_rounds,
            base_strategy=base_strategy,
            host_ip=self.host_ip,
            http_port=self.http_port,
            evaluation_data=evaluation_data
        )
        print(f"[FLServerManager] Starting server at {self.server_address} with strategy {self.strategy_name}, rounds {self.num_rounds}, model {self.model_type} {self.model_params}, local_epochs {self.local_epochs}, local_batch_size {self.local_batch_size}")
        fl.server.start_server(
            server_address=self.server_address,
            strategy=custom_strategy,
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
        )

class FLClientManager:
    def __init__(
        self,
        broker_url,
        server_address,
        client_id,
        dataset,
        data_path,
        model_type,
        model_params,
        client_host_ip="127.0.0.1",
        notification_port=None,
        client_mode="active",
        data_distribution="iid",
        experiment_mode=True,
        total_clients = 5,
        federation_id = 0,
        total_federations=1

    ):
        self.broker_url = broker_url
        self.server_address = server_address
        self.client_id = client_id
        self.dataset = dataset
        self.data_path = data_path
        self.model_type = model_type
        self.model_params = model_params or {}
        self.client_host_ip = client_host_ip
        self.notification_port = notification_port
        self.client_mode = client_mode
        self.data_distribution = data_distribution
        self.experiment_mode = experiment_mode
        self.total_clients = total_clients
        self.federation_id = federation_id
        self.total_federations = total_federations


        if self.notification_port:
            self.notification_endpoint = f"http://{self.client_host_ip}:{self.notification_port}/{self.client_id}"
        else:
            self.notification_endpoint = None

        self.trainloader, self.testloader, self.dataset_meta = load_dataset(
        self.dataset, self.data_path,
        train_batch_size=32, test_batch_size=32,
        client_id=self.client_id,
        total_clients=self.total_clients,
        federation_id=self.federation_id,
        total_federations=self.total_federations,
        data_distribution=self.data_distribution,
        experiment_mode=self.experiment_mode
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = create_model(self.model_type, self.model_params)
        self.model.to(self.device)
        self.start_client_event = threading.Event()

        client_info = {
            "client_id": self.client_id,
            "available_data": self.dataset,
            "compute_spec": "default",
            "client_mode": self.client_mode,
            "description": f"FL client {self.client_id} running dataset {self.dataset} with model {self.model_type}",
        }
        if self.notification_endpoint:
            client_info["notificationEndpoint"] = self.notification_endpoint
        send_client_capability_advertisement(self.broker_url, client_info)

    def rebuild_dataloader(self, train_batch_size: int, test_batch_size: int):
        self.trainloader, self.testloader, _ = load_dataset(
            self.dataset, self.data_path, train_batch_size, test_batch_size,
            client_id=self.client_id,total_clients=self.total_clients,
        federation_id=self.federation_id,
        total_federations=self.total_federations, data_distribution=self.data_distribution, experiment_mode=self.experiment_mode
        )

    def train_one_epoch(self):
        criterion = nn.CrossEntropyLoss()
        # Increase the learning rate from 0.01 to 0.05
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.model.train()
        running_loss = 0.0
        for images, labels in self.trainloader:
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(self.trainloader)
        print(f"[FLClientManager] Client {self.client_id} training loss: {avg_loss:.4f}")



    def evaluate_local(self):
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100.0 * correct / total
        print(f"[FLClientManager] Client {self.client_id} test accuracy: {accuracy:.2f}%")
        return accuracy

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, params):
        params_dict = zip(self.model.state_dict().keys(), params)
        state_dict = {k: torch.tensor(v).to(self.device) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def start_client(self):
        print(f"[FLClientManager] Client {self.client_id} connecting to {self.server_address}")
        class RealClient(fl.client.NumPyClient):
            def get_parameters(iclient, config):
                return self_server.get_parameters()
            def fit(iclient, parameters, config):
                self_server.set_parameters(parameters)
                local_epochs = config.get("local_epochs", 1)
                batch_size   = config.get("batch_size", 32)
                #self_server.rebuild_dataloader(batch_size, batch_size)
                if not hasattr(self_server, "dataloader_initialized") or not self_server.dataloader_initialized:
                    self_server.rebuild_dataloader(batch_size, batch_size)
                    self_server.dataloader_initialized = True
                for _ in range(local_epochs):
                    self_server.train_one_epoch()
                new_params  = self_server.get_parameters()
                num_samples = len(self_server.trainloader.dataset)
                return new_params, num_samples, {}
            def evaluate(iclient, parameters, config):
                self_server.set_parameters(parameters)
                accuracy = self_server.evaluate_local()
                return 0.0, len(self_server.testloader.dataset), {"accuracy": accuracy}
        self_server = self
        fl.client.start_client(
            server_address=self.server_address,
            client=RealClient().to_client(),
        )

    def start_notification_server(self, port):
        start_notification_server(self, port)

###############################################################################
#                     ORCHESTRATOR & MAIN (CLI)                               #
###############################################################################

def select_collaborative_ml_paradigm(paradigm, **kwargs):
    if paradigm == "federated_learning":
        role = kwargs.get("role", "server")
        if role == "server":
            srv = FLServerManager(
                broker_url=kwargs["broker_url"],
                server_address=kwargs["server_address"],
                strategy_name=kwargs.get("strategy_name", "fedavg"),
                num_rounds=kwargs.get("num_rounds", 5),
                model_type=kwargs.get("model_type", "mnistnet"),
                model_params=kwargs.get("model_params", {}),
                job_id=kwargs.get("job_id", "experiment_job"),
                host_ip=kwargs.get("host_ip", "127.0.0.1"),
                http_port=kwargs.get("http_port", 8000),
                local_epochs=kwargs.get("local_epochs", 10),
                local_batch_size=kwargs.get("local_batch_size", 32),
                experiment_mode=kwargs.get("experiment_mode", False),
                dataset=kwargs.get("dataset", "mnist"),
                data_path=kwargs.get("data_path", "./data"),
                required_participants=kwargs.get("required_participants",2)
            )
            srv.start_server()
        elif role == "client":
            cl = FLClientManager(
                broker_url=kwargs["broker_url"],
                server_address=kwargs["server_address"],
                client_id=kwargs.get("client_id", "client-1"),
                dataset=kwargs.get("dataset", "mnist"),
                data_path=kwargs.get("data_path", "./data"),
                model_type=kwargs.get("model_type", "mnistnet"),
                model_params=kwargs.get("model_params", {}),
                client_host_ip=kwargs.get("client_host_ip", "127.0.0.1"),
                notification_port=kwargs.get("notification_port", None),
                client_mode=kwargs.get("client_mode", "active"),
                data_distribution=kwargs.get("data_distribution", "iid"),
                experiment_mode=kwargs.get("experiment_mode", False),
                total_clients=kwargs.get("total_clients", 5),         
                federation_id=kwargs.get("federation_id", 0),          
                total_federations=kwargs.get("total_federations", None)
            )
            if kwargs.get("membership_approach") in ["server_driven"] or (kwargs.get("membership_approach") == "hybrid" and kwargs.get("client_mode") == "passive"):
                cl.start_notification_server(kwargs.get("notification_port"))
                print("[Facilitator] Running in passive mode. Waiting for server invitation...")
                while not cl.start_client_event.is_set():
                    time.sleep(1)
                cl.start_client()
            else:
                cl.start_client()
        else:
            print(f"[Facilitator] Unknown role '{role}' for paradigm '{paradigm}'.")
    else:
        print(f"[Facilitator] Paradigm '{paradigm}' not supported.")

def parse_data_spec(data_spec_str=None, data_spec_file=None):
    if data_spec_str:
        try:
            return json.loads(data_spec_str)
        except:
            raise ValueError(f"Invalid JSON in --data-spec='{data_spec_str}'")
    elif data_spec_file:
        with open(data_spec_file, "r") as f:
            return json.load(f)
    else:
        return {}

def parse_model_params(model_params_str=None, model_params_file=None):
    if model_params_str:
        try:
            return json.loads(model_params_str)
        except:
            raise ValueError(f"Invalid JSON in --model-params='{model_params_str}'")
    elif model_params_file:
        with open(model_params_file, "r") as f:
            return json.load(f)
    else:
        return {}

def main():
    parser = argparse.ArgumentParser("Selective Federated Learning CLI")
    parser.add_argument("--paradigm", default="federated_learning")
    parser.add_argument("--role", choices=["server", "client"], default="client")
    parser.add_argument("--broker-url", default="mqtt://localhost:1026")
    parser.add_argument("--server-address", default="[::]:8080")
    parser.add_argument("--num-rounds", type=int, default=5)
    parser.add_argument("--strategy-name", default="fedavg")
    parser.add_argument("--job-id", default="experiment_clientDriven")
    parser.add_argument("--description", default="Experiment FL job with 5 clients in client-driven mode.")
    parser.add_argument("--host-ip", default="127.0.0.1", help="Used in model download URI")
    parser.add_argument("--http-port", type=int, default=8000)
    parser.add_argument("--local-epochs", type=int, default=10, help="Number of epochs each client trains locally per round.")
    parser.add_argument("--local-batch-size", type=int, default=32, help="Batch size used in local client training.")
    parser.add_argument("--data-spec", type=str, default=None, help="JSON string specifying data requirements.")
    parser.add_argument("--data-spec-file", type=str, default=None, help="File containing JSON data spec.")
    parser.add_argument("--dataset", default="mnist", help="Which dataset to load (mnist, cifar10,har).")
    parser.add_argument("--data-path", default="./data", help="Path for local dataset storage.")
    parser.add_argument("--min-num-samples", type=int, default=1)
    parser.add_argument("--required-participants", type=int, default=5)
    parser.add_argument("--data-distribution", default="iid", help="Data distribution: 'iid' or 'non-iid'.")
    parser.add_argument("--experiment-mode", action="store_true", help="Enable experiment mode with data partitioning.")
    parser.add_argument("--model-type", default="mnistnet", help="Which model to create (e.g., 'mnistnet', 'synthetic_port', 'occupancy_model').")
    parser.add_argument("--model-params", type=str, default=None, help="JSON string for additional model hyperparams.")
    parser.add_argument("--model-params-file", type=str, default=None, help="File containing JSON for model hyperparams.")
    parser.add_argument("--client-id", default="client-1")
    parser.add_argument("--client-mode", choices=["active", "passive"], default="active", help="For hybrid mode: active clients connect immediately; passive clients wait for invitation.")
    parser.add_argument("--membership-approach", choices=["client_driven", "server_driven", "hybrid"], default="client_driven", help="Membership approach for FL participants.")
    parser.add_argument("--notification-port", type=int, default=9000, help="Port on which the client listens for notifications.")
    parser.add_argument("--client-host-ip", default="127.0.0.1", help="Client host IP for the notification endpoint.")
    parser.add_argument("--client-endpoints", type=str, default=None, help="JSON list of client endpoints for server-driven approach.")
    parser.add_argument("--num-clients", type=int, default=1, help="Number of client processes to spawn.")
    parser.add_argument("--client-type", choices=["occupancy", "bus", "sensor"], default="occupancy", help="For port dataset: type of client data to simulate.")
    parser.add_argument("--total-clients", type=int, default=5, help="Total number of clients in the federation.")
    parser.add_argument("--federation-id", type=int, default=0, help="Federation ID (0-indexed).")
    parser.add_argument("--total-federations", type=int, default=3, help="Total number of federations.")

    
    args = parser.parse_args()
    data_spec = parse_data_spec(args.data_spec, args.data_spec_file)
    model_params = parse_model_params(args.model_params, args.model_params_file)
    
    if args.role == "server":
        if args.membership_approach != "server_driven":
            fl_job_info = {
                "job_id": args.job_id,
                "description": args.description,
                "strategy": args.strategy_name,
                "num_rounds": args.num_rounds,
                "server_address": args.server_address,
                "model_type": args.model_type,
                "model_params": model_params,
                "data_spec": data_spec,
                "min_num_samples": args.min_num_samples,
                "required_participants": args.required_participants,
                "data_distribution": args.data_distribution,
                "target_accuracy": 0.85,
                "local_epochs": args.local_epochs,
                "local_batch_size": args.local_batch_size,
            }
            send_fl_job_advertisement_to_broker(args.broker_url, fl_job_info)
        select_collaborative_ml_paradigm(
            paradigm=args.paradigm,
            role=args.role,
            broker_url=args.broker_url,
            server_address=args.server_address,
            strategy_name=args.strategy_name,
            num_rounds=args.num_rounds,
            model_type=args.model_type,
            model_params=model_params,
            job_id=args.job_id,
            host_ip=args.host_ip,
            http_port=args.http_port,
            local_epochs=args.local_epochs,
            local_batch_size=args.local_batch_size,
            experiment_mode=args.experiment_mode,
            dataset=args.dataset,
            data_path=args.data_path,
            required_participants=args.required_participants
        )
    elif args.role == "client":
        if args.num_clients > 1:
            processes = []
            for i in range(args.num_clients):
                unique_client_id = f"{args.client_id}_{i}"
                cmd = [
                    sys.executable, __file__,
                    "--role", "client",
                    "--membership-approach", args.membership_approach,
                    "--server-address", args.server_address,
                    "--broker-url", args.broker_url,
                    "--dataset", args.dataset,
                    "--data-path", args.data_path,
                    "--model-type", args.model_type,
                    "--client-id", unique_client_id,
                    "--client-host-ip", args.client_host_ip,
                    "--notification-port", str(args.notification_port),
                    "--data-distribution", args.data_distribution,
                    "--client-type", args.client_type,
                    "--total-clients", str(args.total_clients),    
                    "--federation-id", str(args.federation_id),       
                    "--total-federations", str(args.total_federations)  
                ]
                if args.experiment_mode:
                    cmd.append("--experiment-mode")
                cmd.extend(["--client-mode", args.client_mode])
                print(f"[Launcher] Starting client process with ID: {unique_client_id}")
                p = subprocess.Popen(cmd)
                processes.append(p)
            for p in processes:
                p.wait()
        else:
            select_collaborative_ml_paradigm(
                paradigm=args.paradigm,
                role=args.role,
                broker_url=args.broker_url,
                server_address=args.server_address,
                dataset=args.dataset,
                data_path=args.data_path,
                client_id=args.client_id,
                model_type=args.model_type,
                model_params=model_params,
                client_host_ip=args.client_host_ip,
                notification_port=args.notification_port,
                membership_approach=args.membership_approach,
                client_mode=args.client_mode,
                data_distribution=args.data_distribution,
                experiment_mode=args.experiment_mode,
                **({"client_type": args.client_type} if args.dataset.lower() == "port" else {}),
                total_clients=args.total_clients,         
                federation_id=args.federation_id,         
                total_federations=args.total_federations    
            )

if __name__ == "__main__":
    main()
