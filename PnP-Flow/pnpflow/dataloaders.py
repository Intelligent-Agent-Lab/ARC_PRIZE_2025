import torch
import torchvision
import torchvision.transforms as v2
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision import datasets
from PIL import Image
import pandas as pd
import os
import warnings
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pickle
import logging
import json

class DataLoaders:
    def __init__(self, dataset_name, batch_size_train, batch_size_test):
        self.dataset_name = dataset_name
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test

    def load_data(self):

        if self.dataset_name == 'celeba':
            transform = v2.Compose([
                v2.CenterCrop(178),
                v2.Resize((128, 128)),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            # Paths
            img_dir = './data/celeba/img_align_celeba/'
            partition_csv = './data/celeba/list_eval_partition.csv'

            # Datasets
            train_dataset = CelebADataset(
                img_dir, partition_csv, partition=0, transform=transform)
            val_dataset = CelebADataset(
                img_dir, partition_csv, partition=1, transform=transform)
            test_dataset = CelebADataset(
                img_dir, partition_csv, partition=2, transform=transform)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size_train,
                shuffle=True,
                collate_fn=custom_collate)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)

        elif self.dataset_name == 'celebahq':

            transform = v2.Compose([
                v2.Resize(256),
                v2.ToTensor(),         # Convert images to PyTorch tensor
            ])

            test_dir = './data/celebahq/test/'
            test_dataset = CelebAHQDataset(
                test_dir, batchsize=self.batch_size_test, transform=transform)
            train_loader = None
            val_loader = None
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)

        elif self.dataset_name == 'afhq_cat':
            # transform should include a linear transform 2x - 1
            transform = v2.Compose([
                v2.Resize((256, 256)),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

            # transform = False
            img_dir_test = './data/afhq_cat/val/cat/'
            img_dir_val = './data/afhq_cat/val/cat/'
            img_dir_train = './data/afhq_cat/train/cat/'
            test_dataset = AFHQDataset(
                img_dir_test, batchsize=self.batch_size_test, transform=transform)
            val_dataset = AFHQDataset(
                img_dir_val, batchsize=self.batch_size_test, transform=transform)
            train_dataset = AFHQDataset(
                img_dir_train, batchsize=self.batch_size_test, transform=transform)
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size_train,
                shuffle=True,
                collate_fn=custom_collate, drop_last=True)
        
        elif self.dataset_name == 'arc_agi':
            # transform should include a linear transform 2x - 1
            # transform = False
            json_dir_test = './data/arc_agi/arc-agi_test_challenges.json'
            json_dir_val = './data/arc_agi/arc-agi_evaluation_challenges.json'
            json_dir_train = './data/arc_agi/arc-agi_training_challenges.json'
            test_dataset = ARCAGIDataset(json_dir_test)
            val_dataset = ARCAGIDataset(json_dir_test)
            train_dataset = ARCAGIDataset(json_dir_train)
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size_train,
                shuffle=True,
                collate_fn=custom_collate, drop_last=True)
            
        elif self.dataset_name == 'mnist':
            test_dataset = datasets.MNIST(
                root="./mnist",
                train=False, 
                transform=ToTensor(),
                download=True,
                )
            val_dataset = datasets.MNIST(
                root="./mnist",
                train=False, 
                transform=ToTensor(),
                download=False,
                )
            train_dataset = datasets.MNIST(
                root="./mnist",
                train=True, 
                transform=ToTensor(),
                download=False,
                )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size_train,
                shuffle=True,
                collate_fn=custom_collate, drop_last=True)
        else:
            raise ValueError("The dataset your entered does not exist")

        data_loaders = {'train': train_loader,
                        'test': test_loader, 'val': val_loader}

        return data_loaders


class CelebADataset(Dataset):
    def __init__(self, img_dir, partition_csv, partition, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.partition = partition

        # Load the partition file correctly
        partition_df = pd.read_csv(
            partition_csv, header=0, names=[
                'image', 'partition'], skiprows=1)
        self.img_names = partition_df[partition_df['partition']
                                      == partition]['image'].values

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            warnings.warn(f"File not found: {img_path}. Skipping.")
            return None, None

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, 0


class CelebAHQDataset(Dataset):
    """CelebA HQ dataset."""

    def __init__(self, data_dir, batchsize, transform=None):
        self.files = os.listdir(data_dir)
        self.root_dir = data_dir
        self.num_imgs = len(os.listdir(self.root_dir))
        self.transform = transform
        self.batchsize = batchsize

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.root_dir, img_name)

        if not os.path.exists(img_path):
            warnings.warn(f"File not found: {img_path}. Skipping.")
            return None, None

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
            image = 2 * image - 1
        image = image.float()

        return image, 0


class AFHQDataset(Dataset):
    """AFHQ Cat dataset."""

    def __init__(self, img_dir, batchsize, category='cat', transform=None):
        self.files = os.listdir(img_dir)
        self.num_imgs = len(self.files)
        self.batchsize = batchsize
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            warnings.warn(f"File not found: {img_path}. Skipping.")
            return None, None

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, 0

def preprocess_data(challenges):
    inputs, outputs = [], []
    
    # Iterate over each task (ID) in the challenge dataset
    for task_id, task_data in challenges.items():
        # Extract training pairs
        for pair in task_data.get('train', []):
            input_grid = np.array(pair['input'])
            output_grid = np.array(pair['output'])
            
            # Append input and output grids to the respective lists
            inputs.append(input_grid)
            outputs.append(output_grid)
    
    # Optionally, pad arrays to make them of uniform shape
    max_input_shape = max([input.shape for input in inputs], key=lambda x: np.prod(x))  # Get max shape
    max_output_shape = max([output.shape for output in outputs], key=lambda x: np.prod(x))  # Get max shape
    
    # Pad inputs to max_input_shape
    padded_inputs = [np.pad(input, [(0, max_input_shape[0] - input.shape[0]), (0, max_input_shape[1] - input.shape[1])], mode='constant', constant_values=0) for input in inputs]
    padded_outputs = [np.pad(output, [(0, max_output_shape[0] - output.shape[0]), (0, max_output_shape[1] - output.shape[1])], mode='constant', constant_values=0) for output in outputs]
    padded_inputs = (np.array(padded_inputs).astype(np.float32) / 255.0)
    padded_outputs = (np.array(padded_outputs).astype(np.float32) / 255.0)
    return padded_inputs, padded_outputs


class ARCAGIDataset:
    def __init__(self, 
                json_path: str,):
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        self.X, self.Y = preprocess_data(data)
        self.X = torch.tensor(self.X)
        self.Y = torch.tensor(self.Y)
        self.X = self.Y.unsqueeze(1)
        self.Y = self.Y.unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # print(f'self.X[idx].shape: {self.X[idx].shape}')
        XY = torch.cat([self.X[idx], self.Y[idx]], dim=-1)
        return XY, self.Y[idx]



def custom_collate(batch):
    # Filter out None values

    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data._utils.collate.default_collate(batch)


logging.basicConfig(level=logging.INFO)
