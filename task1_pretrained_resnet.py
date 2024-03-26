import os
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models import resnet18

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import lightning as pl
import random
import numpy as np
from sklearn.model_selection import train_test_split

import optuna
from lightning.pytorch.loggers import MLFlowLogger

from collections import Counter
import matplotlib.pyplot as plt

from data_augmentation2 import ImageDataset_2

import torchmetrics

from PIL import Image
from torchvision.transforms import Compose, ConvertImageDtype, Normalize, PILToTensor, Resize
from torchvision.transforms.functional import InterpolationMode
import holocron
#from holocron.models import model_from_hf_hub
from transformers import AutoModel
from torch.utils.data import Subset

batch_size = 512  # Define your batch size

######################### 1. 
class ResNetLightning(pl.LightningModule):
    def __init__(self, num_classes):
        super(ResNetLightning, self).__init__()

        self.epochh = 0
        # Initialize ResNet18

        self.relu = nn.ReLU()

        model_id = "frgfm/resnet18"
        self.model = AutoModel.from_pretrained(model_id)
        self.conv5 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=(3, 4), padding=1, stride=2)
        self.norm5 = nn.BatchNorm2d(1024)
        self.final_mlp = nn.Sequential(
          nn.Linear(1024, 1536),
          nn.BatchNorm1d(1536),
          nn.ReLU(),
          nn.Linear(1536, n_classes)
        )

        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.4)
        #self.final_mlp = nn.Linear(2048, n_classes)
 
        print(self.model.state_dict().keys())
        #self.model = resnet18()

        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_classes)

    def forward(self, x, cam=False):
        # Forward pass
            
            with torch.no_grad():
                x = self.model(x).last_hidden_state

            #print(x.shape)
            x = self.dropout1(x)
            x = self.relu(self.norm5(self.conv5(x)))
            #print(x.shape)
            x = x.view(x.shape[0], -1)
            #print(x.shape)
            x = self.dropout2(x)
            x = self.final_mlp(x)

            return x

    def training_step(self, batch, batch_idx):
        # Implement the training logic here
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.train_acc(self(batch[0]),batch[-1] )
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Implement the validation logic here
        x, y = batch
        logits = self.forward(x)
        val_loss = F.cross_entropy(logits, y)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.val_acc(self(batch[0]), batch[-1] )
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        # Define and return optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    


######################### 2.
def get_files_from_directory(directory):
    """Recursively gets all image files from a directory and its subdirectories."""
    image_files = []
    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(subdir, file))
    return image_files

def split_files_by_class(root_dir):

    total = 0
    class_directories = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    train_files = []
    test_files = []
    
    for class_dir in class_directories:
        class_path = os.path.join(root_dir, class_dir)
        files = get_files_from_directory(class_path)

        if len(files)>60:

            total+=1

            # Perform the split
            class_train, class_test = train_test_split(files, test_size=0.3)
            
            train_files.extend(class_train)
            test_files.extend(class_test)

    print("total", total)
    
    return train_files, test_files, total

dir = '/Users/mathieugierski/Nextcloud/Macbook M3/vision/data_treated'
train_files, test_files, n_classes = split_files_by_class(dir)


# Assuming you modify ImageDataset to accept a list of files:
train_dataset = ImageDataset_2(train_files)
test_dataset = ImageDataset_2(test_files)


"""
# Create a counter for class frequencies
print(train_dataset.counting_classes)
print("new:")
print(train_dataset.new_counting_classes)
class_counts = Counter(train_dataset.new_counting_classes)

# Plotting
classes = list(class_counts.keys())
counts = list(class_counts.values())
"""

classes = list(train_dataset.summary.keys())
counts = list(train_dataset.summary.values())


plt.figure(figsize=(10, 6))
plt.bar(classes, counts, color='skyblue')
plt.xlabel('Class')
plt.ylabel('Number of Elements')
plt.title('Number of Elements in Each Class')
plt.xticks(rotation=45)
#plt.show()


#print("Training set size:", len(train_dataset))
#print("Test set size:", len(test_dataset))

#indices = range(0, 600)
#Subset(train_dataset, indices)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader= DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

#Devise:
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    mps_device = torch.device("mps")

epochs=40

model = ResNetLightning(num_classes=n_classes)
#print("features", model.model.fc)
#num_ftrs = model.model.fc.in_features
#model.fc = nn.Linear(num_ftrs, n_classes)  # Assuming 10 classes in your dataset
model.to(mps_device)

mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./ml-runs")

trainer = pl.Trainer(max_epochs=epochs, accelerator="mps", logger=mlf_logger, log_every_n_steps=1)
trainer.fit(model, train_loader , test_loader)

val_loss = trainer.callback_metrics["val_loss"].item()