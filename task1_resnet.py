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

batch_size = 512  # Define your batch size

######################### 1. 
class ResNetLightning(pl.LightningModule):
    def __init__(self, num_classes):
        super(ResNetLightning, self).__init__()

        self.epochh = 0
        # Initialize ResNet18
        self.model = resnet18()
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        #print("state_dict", self.model.state_dict())
        dropout_p=0.4
        self.model.layer1 = nn.Sequential(self.model.layer1, nn.Dropout(dropout_p))
        self.model.layer2 = nn.Sequential(self.model.layer2, nn.Dropout(dropout_p))
        self.model.layer3 = nn.Sequential(self.model.layer3, nn.Dropout(dropout_p))
        self.model.layer4 = nn.Sequential(self.model.layer4, nn.Dropout(dropout_p))

        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_classes)

        self.gap_mlp = nn.Linear(256, n_classes)

    def forward(self, x, cam=False):
        # Forward pass

        if cam:

            with torch.no_grad():
                x = x.to(mps_device)
                x = self.model.conv1(x)
                x = self.model.bn1(x)
                x = self.model.relu(x)
                x = self.model.maxpool(x)
                x = self.model.layer1(x)
                x = self.model.layer2(x)
                x = self.model.layer3(x)
                x_cam = torch.mean(x , dim = (2,3))
                x_cam = self.gap_mlp(x_cam)
            return x,x_cam

        else:
            return self.model(x)

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
    
    def on_train_epoch_start(self):

        #img, classe = test_dataset[0]
        img= Image.open("/Users/mathieugierski/Nextcloud/Macbook M3/vision/AI_Vision-Geoguessr/cam_outputs/original_image_['Poland'].png").convert('RGB')
        transform = transforms.ToTensor()
        img = transform(img)
        dico = test_dataset.class_to_idx
        country = "['Poland']"

        pil_image = TF.to_pil_image(img)
        

        with torch.no_grad():
            # Get predictions and CAMs for these images
            conved_img, pred = self(img.unsqueeze(0), cam=True)
            cam = self.generate_cam(conved_img, pred.argmax(dim=1))  # Implement generate_cam based on the steps provided
            # Plotting code here: Display original image with CAM overlay

            pred_country = [key for key, value in dico.items() if value == pred.argmax(dim=1)]

        
        np_img = img.cpu().detach().numpy()
        np_img = np.moveaxis(np_img, [0, 1, 2], [2, 0, 1])
        np_img = (np_img *255).astype(np.uint8)

        cam_resized = cam.cpu().detach().numpy()
        cam_resized = (cam_resized *255).astype(np.uint8)
        cam_resized = np.array(Image.fromarray(cam_resized).resize((img.shape[2], img.shape[1]), Image.BILINEAR)) 

        colored_overlay = np.zeros((40, 80, 3)).astype(np.uint8) 
        colored_overlay[:, :, 0] = cam_resized.astype(np.uint8)  
        colored_overlay[:, :, 1] = cam_resized.astype(np.uint8)  
        colored_overlay[:, :, 2] = cam_resized.astype(np.uint8)  
        
        overlay_img = ((colored_overlay.astype(np.float32) * 0.5) + (np_img.astype(np.float32) * 0.5)).astype(np.uint8)
        #overlay_img = cam_resized.astype(np.uint8) 

        # Ensure the output directory exists
        output_dir = 'cam_outputs_resnet'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.epochh>=0:
            output_path = os.path.join(output_dir, f'original_image_{country}.png')
            pil_image.save(output_path, inplace=True)

        # Save the image
        output_path = os.path.join(output_dir, f'CAM_{self.epochh}{pred_country}.png')

        #colored_overlay = np.zeros((40, 80, 3)).astype(np.uint8) 
        #colored_overlay[:, :, 0] = overlay_img.astype(np.uint8) 
        Image.fromarray(overlay_img).save(output_path)

        self.epochh+=1

    
    def generate_cam(self, feature_maps, pred_class):
        # Get the weight of the `linear` layer for the predicted class
        W = self.gap_mlp.weight[pred_class,:].view(self.gap_mlp.weight.shape[1], 1, 1)
        feature_maps = torch.squeeze(feature_maps, 0)
        cam = feature_maps*W
        cam = cam.sum(dim=0)
        return cam


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

epochs=40

model = ResNetLightning(num_classes=n_classes)
print("features", model.model.fc)
num_ftrs = model.model.fc.in_features
model.fc = nn.Linear(num_ftrs, n_classes)  # Assuming 10 classes in your dataset
model.to(mps_device)

mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./ml-runs")

trainer = pl.Trainer(max_epochs=epochs, accelerator="mps", logger=mlf_logger, log_every_n_steps=1)
trainer.fit(model, train_loader , test_loader)

val_loss = trainer.callback_metrics["val_loss"].item()