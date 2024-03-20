import os
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, random_split

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

import copy

batch_size = 512  # Define your batch size

############################################# 1.
class Classifier(pl.LightningModule):
    def __init__(self, a, b, c, d, n_classes, cam):
        super().__init__()

        self.epochh=0
        self.cam = cam

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.4)
        self.dropout3 = nn.Dropout(0.4)

        #Starting with:
        #40, 80
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=a, kernel_size=4, padding=1, stride=2)#20, 40
        self.norm1 = nn.BatchNorm2d(a)
        self.pool1 = nn.MaxPool2d(2, 2) #10, 20

        self.conv2 = nn.Conv2d(in_channels=a, out_channels=b, kernel_size=4, padding=1, stride=2) #5, 10
        self.norm2 = nn.BatchNorm2d(b)
        self.pool2 = nn.MaxPool2d(2, 2) #2, 5

        ###
        #Without cam
        self.conv5 = nn.Conv2d(in_channels=b, out_channels=c, kernel_size=(2, 5))
        self.norm5 = nn.BatchNorm2d(c)

        self.linear = nn.Linear(c, n_classes)

        ###

        self.L = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_classes)


    def forward(self, x):

        x = x.to(mps_device)

        if x.shape[0]>1:
        
            x = self.relu(self.norm1(self.conv1(x)))
            x = self.pool1(x)
            x = self.dropout1(x)

            x = self.relu(self.norm2(self.conv2(x)))
            x = self.pool2(x)
            x = self.dropout2(x)

            x = self.relu(self.norm5(self.conv5(x)))
            x = self.dropout3(x)

            x = x.view(x.shape[0], -1)
            x = self.linear(x)

        else:

            x = self.relu(self.conv1(x))
            x = self.pool1(x)
            x = self.dropout1(x)

            x = self.relu(self.conv2(x))
            x = self.pool2(x)
            x = self.dropout2(x)

            x = self.relu(self.conv5(x))
            x = self.dropout3(x)

            x = x.view(x.shape[0], -1)
            x = self.linear(x)

        return x
    
    def _step(self, batch, batch_idx):

        out = self(batch[0])

        return self.L(out, batch[-1])
    

    def training_step(self,batch, batch_idx):

        # Called to compute and log the training loss
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)


        self.train_acc(self(batch[0]),batch[-1] )
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):

        # Called to compute and log the validation loss
        val_loss = self._step(batch, batch_idx)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.val_acc(self(batch[0]), batch[-1] )
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True)
        return val_loss

    def configure_optimizers(self):

        # Optimizer and LR scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)
        return optimizer
    
    def on_train_epoch_start(self):


        if self.epochh==0:
            self.eval()  # Make sure the model is in evaluation mode
            y_true = []
            y_pred = []

            with torch.no_grad():
                for i in range(len(test_dataset)):

                    print(i/len(test_dataset))
                    x, y = test_dataset[i]
                    x = x.unsqueeze(0)  # Assuming x needs to be batched
                    output = self(x)  # Get model output
                    _, predicted = torch.max(output, 1)  # Convert output to predicted class index
                    y_true.append(y)
                    y_pred.append(predicted.item())

            plot_mispredictions(np.array(y_true), np.array(y_pred))
            self.epochh+=1

        else:
            self.epochh+=1
    

def plot_mispredictions(y_true, y_pred):
    # Invert the class dictionary to map indices to class names
    index_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}
    
    num_classes = len(test_dataset.class_to_idx)
    # Initialize confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes))

    # Populate confusion matrix
    for true, pred in zip(y_true, y_pred):
        confusion_matrix[true, pred] += 1

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    class_names = [index_to_class[i] for i in range(num_classes)]  # Sorted class names by index

    for i in range(num_classes):
        # Mispredicted as other classes
        mispred_as_others = np.sum(confusion_matrix[i, :]) - confusion_matrix[i, i]
        # Mispredicted from other classes
        mispred_from_others = np.sum(confusion_matrix[:, i]) - confusion_matrix[i, i]

        axs[0].bar(class_names[i], mispred_as_others, color='skyblue')
        axs[1].bar(class_names[i], mispred_from_others, color='lightgreen')

    axs[0].set_title('Number of times each class was misclassified as another class')
    axs[1].set_title('Number of times other classes were misclassified as this class')
    for ax in axs:
        ax.set_ylabel('Number of Mispredictions')
        ax.set_xlabel('Class')
        ax.set_xticklabels(class_names, rotation=45)

    plt.tight_layout()
    plt.show()

# After collecting y_true and y_pred
# plot_mispredictions(y_true, y_pred, class_names)



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

epochs=100

def objective(trial: optuna.trial.Trial) -> float:

    a = trial.suggest_int("a", 22, 23, 1)
    b = trial.suggest_int("b", 40, 41, 1)
    c = trial.suggest_int("c", 52, 53, 1)
    #c = 0
    #d = trial.suggest_int("d", 10, 120, 10)
    d=0
    #e = trial.suggest_int("e", 10, 120, 10)

    model = Classifier(a, b, c, d, n_classes, cam = True)
    model.to(mps_device)

    mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./ml-runs")

    trainer = pl.Trainer(max_epochs=epochs, accelerator="mps", logger=mlf_logger, log_every_n_steps=1)
    trainer.fit(model, train_loader , test_loader)

    val_acc = trainer.callback_metrics["val_acc"].item()

    return val_acc

pruner = optuna.pruners.MedianPruner()

study = optuna.create_study(direction="maximize", pruner=pruner)
study.optimize(objective, n_trials=30)

print(f"Number of finished trials: {len(study.trials)}")

print("Best trial: ")
trial = study.best_trial

print(f" Value: {trial.value}")

print(" Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")