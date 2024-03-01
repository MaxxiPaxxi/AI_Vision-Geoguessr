import os
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, random_split

import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as pl
import random
from sklearn.model_selection import train_test_split

import optuna
from lightning.pytorch.loggers import MLFlowLogger

from collections import Counter
import matplotlib.pyplot as plt

from data_augmentation import ImageDataset

batch_size = 32  # Define your batch size

############################################# 1.
class Classifier(pl.LightningModule):
    def __init__(self, a, b, c, d, e):
        super().__init__()

        self.epochh=0

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=a, kernel_size=4, padding=1, stride=2) #80, 40
        self.norm1 = nn.BatchNorm2d(a)

        self.conv2 = nn.Conv2d(in_channels=a, out_channels=b, kernel_size=4, padding=1, stride=2) #40, 20
        self.norm2 = nn.BatchNorm2d(b)

        self.conv3 = nn.Conv2d(in_channels=b, out_channels=c, kernel_size=4, padding=1, stride=2) #20, 10
        self.norm3 = nn.BatchNorm2d(c)

        self.conv4 = nn.Conv2d(in_channels=c, out_channels=d, kernel_size=4, padding=1, stride=2) #10, 5
        self.norm4 = nn.BatchNorm2d(d)

        self.conv5 = nn.Conv2d(in_channels=d, out_channels=e, kernel_size=(2, 5))
        self.norm5 = nn.BatchNorm2d(e)

        self.linear = nn.Linear(e, 48)

        self.L = nn.CrossEntropyLoss()


    def forward(self, x):
        
        #print("0", x.shape)
        x = self.relu(self.norm1(self.conv1(x)))
        #print("1", x.shape)
        x = self.relu(self.norm2(self.conv2(x)))
        #print("2", x.shape) 
        x = self.relu(self.norm3(self.conv3(x)))
        #print("3", x.shape)
        x = self.relu(self.norm4(self.conv4(x)))
        #print("4", x.shape)
        x = self.relu(self.norm5(self.conv5(x)))
        #print("5", x.shape)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        #print("6", x.shape)
        return x
    
    def _step(self, batch, batch_idx):

        out = self(batch[0])

        return self.L(out, batch[-1])
    

    def training_step(self,batch, batch_idx):

        # Called to compute and log the training loss
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):

        # Called to compute and log the validation loss
        val_loss = self._step(batch, batch_idx)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def configure_optimizers(self):

        # Optimizer and LR scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
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

        if len(files)>90 and len(files)<1000:

            total+=1

            # Perform the split
            class_train, class_test = train_test_split(files, test_size=0.3)
            
            train_files.extend(class_train)
            test_files.extend(class_test)

    print("total", total)
    
    return train_files, test_files

dir = '/Users/mathieugierski/Nextcloud/Macbook M3/vision/data_treated'
train_files, test_files = split_files_by_class(dir)


# Assuming you modify ImageDataset to accept a list of files:
train_dataset = ImageDataset(train_files)
test_dataset = ImageDataset(test_files)

# Create a counter for class frequencies
print(train_dataset.counting_classes)
class_counts = Counter(train_dataset.counting_classes)

# Plotting
classes = list(class_counts.keys())
counts = list(class_counts.values())

plt.figure(figsize=(10, 6))
plt.bar(classes, counts, color='skyblue')
plt.xlabel('Class')
plt.ylabel('Number of Elements')
plt.title('Number of Elements in Each Class')
plt.xticks(rotation=45)
plt.show()


#print("Training set size:", len(train_dataset))
#print("Test set size:", len(test_dataset))

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

epochs=20

def objective(trial: optuna.trial.Trial) -> float:

    a = trial.suggest_int("a", 80, 100, 5)
    b = trial.suggest_int("b", 120, 160, 10)
    c = trial.suggest_int("c", 200, 220, 10)
    d = trial.suggest_int("d", 240, 300, 20)
    e = trial.suggest_int("e", 200, 500, 100)

    model = Classifier(a, b, c, d, e)
    model.to(mps_device)

    mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./ml-runs")

    trainer = pl.Trainer(max_epochs=epochs, accelerator="mps", logger=mlf_logger, log_every_n_steps=1)
    trainer.fit(model, train_loader , test_loader)

    val_loss = trainer.callback_metrics["val_loss"].item()

    return val_loss

pruner = optuna.pruners.MedianPruner()

study = optuna.create_study(direction="minimize", pruner=pruner)
study.optimize(objective, n_trials=10)

print(f"Number of finished trials: {len(study.trials)}")

print("Best trial: ")
trial = study.best_trial

print(f" Value: {trial.value}")

print(" Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")