import torch
import torchvision
from torchsummary import summary
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
from PIL import Image

import glob2
import pandas as pd

import torch.optim as optim

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import json
import os

import wandb


train_losses = []
test_losses = []
train_acc = []
test_acc = []


class CatsandDogs(Dataset):
  def __init__(self, file_list, transform=None):
    self.file_list = file_list
    self.transform = transform


  def __len__(self):
    self.file_length = len(self.file_list)
    return(self.file_length)

  def __getitem__(self, idx):
    self.img_path = self.file_list[idx]
    self.image = np.array(Image.open(self.img_path))
    label = self.img_path.split("/")[-1].split(".")[0]
    label = 1 if label == "dog" else 0

    if self.transform is not None:
      self.image = self.transform(image=self.image)

    return(self.image, label)



class Params:
    """Load hyperparameters from a json file
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def __str__(self) -> str:
        return str(self.__dict__)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def get_summary(model, device):
  """
  Args:
      model (torch.nn Model): 
      device (str): cuda/CPU
  """
  print(summary(model.to(device), input_size=(3, 224, 224)))


def get_stats(images_array):
  """
  Args:
      images_array (numpy array): Image array
  Returns:
      mean: per channel mean
      std: per channel std
  """

  print('[Train]')
  print(' - Numpy Shape:', images_array.shape)
  #print(' - Tensor Shape:', images_array.shape)
  print(' - min:', np.min(images_array))
  print(' - max:', np.max(images_array))

  print('Divide by 255')
  images_array = images_array / 255.0

  mean = np.mean(images_array, axis=tuple(range(images_array.ndim-1)))
  std = np.std(images_array, axis=tuple(range(images_array.ndim-1)))

  print(f'\nDataset Mean - {mean}')
  print(f'Dataset Std - {std} ')

  return([mean, std])

def get_stats(train_loader):
  """
  Args:
      images_array (numpy array): Image array
  Returns:
      mean: per channel mean
      std: per channel std
  """

  print('Computing mean & std')

  mean = 0.
  std = 0.

  for images, _ in train_loader:

      images = images['image'] / 255.0
      batch_samples = images.size(0)
      images = images.view(batch_samples, images.size(1), -1)
      mean += images.mean(2).sum(0)
      std += images.std(2).sum(0)

  mean /= len(train_loader.dataset)
  std /= len(train_loader.dataset)

  print(f'\nDataset Mean - {mean}')
  print(f'Dataset Std - {std} ')

  return(mean, std)


def get_train_transforms(h, w, mu, std):
    """
    Args:
        h (int): image height
        w (int):image width
        mu (array): image mean
        std (array): standard deviation
    Returns:
        train_transforms (Albumentation): Transform Object
    """

    train_transform = A.Compose([
                            A.Resize(h, w, cv2.INTER_NEAREST),
                            A.Normalize(mean=(mu), 
                                        std=std),
                            ToTensorV2()
    ])

    return(train_transform)

def get_val_transforms(h, w, mu, std):
    """
    Args:
        h (int): image height
        w (int):image width
        mu (array): image mean
        std (array): standard deviation
    Returns:
        val_transforms (Albumentation): Transform Object
    """
    val_transforms = A.Compose([
                            A.Resize(h, w, cv2.INTER_NEAREST),
                            A.Normalize(mean=(mu), 
                                        std=std),
                            ToTensorV2()
    ])

    return(val_transforms)

def get_device():
  """
  Returns:
      device (str): device type
  """
  SEED = 1

  # CUDA?
  cuda = torch.cuda.is_available()
  print("CUDA Available?", cuda)

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  # For reproducibility
  if cuda:
      torch.cuda.manual_seed(SEED)
  else:
    torch.manual_seed(SEED)

  return(device)

def train_model(model, criterion, device, train_loader, val_loader, optimizer, scheduler, EPOCHS, use_wandb):
  """
  Args:
      model (torch.nn Model): 
      criterion (criterion) - Loss Function
      device (str): cuda/CPU
      train_loader (DataLoader) - DataLoader Object
      optimizer (optimizer) - Optimizer Object
      scheduler (scheduler) - scheduler object
      EPOCHS (int) - Number of epochs
      use_wandb (bool) - use weights & biases for logging
  Returns:
      results (list): Train/test - Accuracy/Loss 
  """

  for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    epoch_loss = 0
    epoch_accuracy = 0
    for data, label in tqdm(train_loader):
        data = data['image'].to(device)
        label = label.to(device)
        output = model(data)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)
        
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in val_loader:
            data = data['image'].to(device)
            label = label.to(device)
            val_output = model(data)
            val_loss = criterion(val_output, label)
            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(val_loader)
            epoch_val_loss += val_loss / len(val_loader)

    train_losses.append(epoch_loss)
    train_acc.append(epoch_accuracy)
    test_losses.append(epoch_val_loss)
    test_acc.append(epoch_val_accuracy)
    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )

    if use_wandb:
        wandb.log({
            "epoch": epoch,
            "Train Loss": train_losses[-1],
            "Train Acc": train_acc[-1],
            "Valid Loss": test_losses[-1], 
            "Valid Acc": test_acc[-1]
        })

  results = [train_losses, test_losses, train_acc, test_acc]

  return(results)


  for epoch in range(EPOCHS):
      print("EPOCH:", epoch)
      train(model, device, criterion, train_loader, optimizer, epoch)
      scheduler.step()
      test(model, device, criterion, test_loader)


  return(results)

 
def make_plot(results):
    """
    Args:
        images (list of list): Loss & Accuracy List
    """
    tr_losses = results[0]
    te_losses = results[1]
    tr_acc = results[2]
    te_acc = results[3]


    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(tr_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(tr_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(te_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(te_acc)
    axs[1, 1].set_title("Test Accuracy")
    plt.show()

