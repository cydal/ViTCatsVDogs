from models.resnet import *
from utils import *

from sklearn.model_selection import train_test_split


from torch.optim.lr_scheduler import StepLR, OneCycleLR
import torch.nn as nn
import torch.optim as optim

import argparse

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

import wandb


import torch   
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms   
from torch.optim.lr_scheduler import StepLR   
from torch.utils.data import DataLoader, Dataset
import opendatasets as od


import glob2
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from PIL import Image
import cv2



import albumentations as A
from albumentations.pytorch import ToTensorV2


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",
                    default=".",
                    help="Directory containing params file")

wandb.init(project="ViTCatsVDogs")


dataset_url = 'https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition'

if __name__ == "__main__":

    od.download(dataset_url)

    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, "PARAMS.JSON")
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    if params.USE_WANDB:
        with open(json_path, "r") as f:
            data = json.load(f)
        wandb.config.update(data)

    file_list = glob2.glob(os.path.join(params.train_path, "*.jpg"))
    labels = [l.split("/")[-1].split(".")[0] for l in file_list]

    data = pd.DataFrame({
        "file_names": file_list, 
        "labels": labels
    })

    print(data.head())

    # split data
    train_set, valid_set = train_test_split(data, 
                                            test_size=params.SPLIT_RATIO,
                                            stratify=data["labels"],
                                            random_state=params.RANDOM_SEED)

    print(
        "[INFO] Training shape:",
        train_set.shape,
        np.unique(train_set.labels, return_counts=True)
    )

    print(
        "[INFO] Validation shape:",
        valid_set.shape,
        np.unique(valid_set.labels, return_counts=True)
    )

    print("[INFO] labels length:", len(labels))
    print("[INFO] Label Encoding:", labelEncoder.classes_)

    print(f"----------Load and Transform Images----------")
    train_tranform = get_train_transforms(params.HEIGHT,
                                      params.WIDTH,
                                      params.MEAN,
                                      params.STD)

    val_tranform = get_val_transforms(params.HEIGHT,
                                      params.WIDTH,
                                      params.MEAN,
                                      params.STD)

    train_data = CatsandDogs(train_set["file_names"].to_list(), transform=train_tranform)
    val_data = CatsandDogs(valid_set["file_names"].to_list(), transform=val_tranform)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=params.BATCH_SIZE,
                                               pin_memory=True,
                                               shuffle=True,
                                               num_workers=params.NUM_WORKERS)

    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=params.BATCH_SIZE,
                                             pin_memory=True,
                                             shuffle=False,
                                             num_workers=params.NUM_WORKERS)

    print(next(iter(train_loader)))
    im, lbl = next(iter(train_loader))
    print(im['image'].shape, lbl.shape)

    print("[INFO] Training length:", len(train_loader.dataset))
    print("[INFO] Validation length:", len(val_loader.dataset))


    device = get_device()
    print(f"----------Device type - {device}----------")
    
    # Set optimzer & lr_scheduler
    if params.NUM_CLASSES < 2:
        criterion = nn.BCEWithLogitsLoss(weight=torch.from_numpy(cws)).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)


    efficient_transformer = Linformer(
        dim=128,
        seq_len=49+1,
        depth=12,
        heads=8,
        k=64
    )

    model = ViT(
    dim=128,
    image_size=params.WIDTH,
    patch_size=32,
    num_classes=params.NUM_CLASSES,
    transformer=efficient_transformer,
    channels=params.INPUT_CHANNELS
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.8)

    print(f"----------Model Summary----------")
    get_summary(model, device)

    print(f"----------Training Model----------")
    results = train_model(model, criterion, device, train_loader, val_loader, optimizer, scheduler, params.EPOCHS, params.USE_WANDB)

    torch.save(model, params.SAVE_MODEL_PATH)

    print(f"----------Loss & Accuracy Plots----------")
    make_plot(results)