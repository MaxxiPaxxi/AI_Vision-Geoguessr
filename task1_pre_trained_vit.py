import os
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models import resnet50

import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L
import random
from sklearn.model_selection import train_test_split

import optuna
from lightning.pytorch.loggers import MLFlowLogger

from collections import Counter
import matplotlib.pyplot as plt

from data_augmentation2 import ImageDataset_2

import torchmetrics

import torch.optim as optim

from transformers import ViTImageProcessor, ViTForImageClassification
import requests




######################### 1. 
class PreTrainedViT(L.LightningModule):
    def __init__(self, num_classes):
        super(PreTrainedViT, self).__init__()

        #Resize the images
        self.resizing = transforms.Resize((224, 224))

        #Pre trained ViT
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.model.save_pretrained('/Users/mathieugierski/Nextcloud/Macbook M3/vision/AI_Vision-Geoguessr')
        print(self.model)
        self.final_mlp = nn.Linear(1000, num_classes)

        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_classes)

    def forward(self, x):
        # Forward pass
        x = self.model(x).logits
        
        return self.final_mlp(x)

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
train_dataset = ImageDataset_2(train_files, resizing=(224, 224))
test_dataset = ImageDataset_2(test_files, resizing=(224, 224))


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



batch_size = 16
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

epochs=15

model = PreTrainedViT(num_classes=n_classes)
model.to(mps_device)

mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./ml-runs")

trainer = L.Trainer(max_epochs=epochs, accelerator="mps", logger=mlf_logger, log_every_n_steps=1)
trainer.fit(model, train_loader , test_loader)

val_loss = trainer.callback_metrics["val_loss"].item()