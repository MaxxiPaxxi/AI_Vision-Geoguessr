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

import copy

batch_size = 512  # Define your batch size

############################################# 1.
class Classifier(pl.LightningModule):
    def __init__(self, a, b, c, d, n_classes):
        super().__init__()

        self.epochh=0

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        self.epochh = 0
        # Initialize ResNet18
        self.model = resnet18()
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, n_classes)

        #print("state_dict", self.model.state_dict())
        dropout_p=0.4
        self.model.layer1 = nn.Sequential(self.model.layer1, nn.Dropout(dropout_p))
        self.model.layer2 = nn.Sequential(self.model.layer2, nn.Dropout(dropout_p))
        self.model.layer3 = nn.Sequential(self.model.layer3, nn.Dropout(dropout_p))
        self.model.layer4 = nn.Sequential(self.model.layer4, nn.Dropout(dropout_p))

        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_classes)

        self.gap_mlp = nn.Linear(256, n_classes)

        ###

        self.L = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_classes)


    def forward(self, x):

        x = x.to(mps_device)

        if x.shape[0]>1:

            x = x.to(mps_device)
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            print("XX", x.shape)
            x_cam = torch.mean(x , dim = (2,3))
            x_cam = self.gap_mlp(x_cam)

        else:

            x = x.to(mps_device)
            x = self.model.conv1(x)
            #x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)

            print("XX", x.shape)
            x_cam = torch.mean(x , dim = (2,3))
            x_cam = self.gap_mlp(x_cam)

        return x, x_cam
    
    
    def _step(self, batch, batch_idx):

        _, out = self(batch[0])

        return self.L(out, batch[-1])
    

    def training_step(self,batch, batch_idx):

        # Called to compute and log the training loss
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.train_acc(self(batch[0])[1],batch[-1] )
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):

        # Called to compute and log the validation loss
        val_loss = self._step(batch, batch_idx)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.val_acc(self(batch[0])[1], batch[-1] )
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True)
        return val_loss

    def configure_optimizers(self):

        # Optimizer and LR scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)
        return optimizer
    
    def on_train_epoch_start(self):

        """
        img, classe = test_dataset[0]
        dico = test_dataset.class_to_idx
        country = [key for key, value in dico.items() if value == classe]
        """

        img= Image.open("/Users/mathieugierski/Nextcloud/Macbook M3/vision/AI_Vision-Geoguessr/original_image_Poland.png").convert('RGB')
        #img, _ = test_dataset[0]
        transform = transforms.ToTensor()
        img = transform(img)
        dico = test_dataset.class_to_idx
        country = "Poland"
        classe = test_dataset.class_to_idx[country]

        pil_image = TF.to_pil_image(img)
        

        with torch.no_grad():
            # Get predictions and CAMs for these images
            conved_img, pred = self(img.unsqueeze(0))
            pred = self.softmax(pred)

            cam = self.generate_cam(conved_img, pred.argmax(dim=1))

            pred_country = [key for key, value in dico.items() if value == pred.argmax(dim=1)]
            cam_true_country = self.generate_cam(conved_img, classe)

        
        np_img = img.cpu().detach().numpy()
        np_img = np.moveaxis(np_img, [0, 1, 2], [2, 0, 1])
        np_img = (np_img *255).astype(np.uint8)

        ###
        cam_resized = cam.cpu().detach().numpy()
        cam_resized = (cam_resized *255).astype(np.uint8)
        cam_resized = np.array(Image.fromarray(cam_resized).resize((img.shape[2], img.shape[1]), Image.BILINEAR)) 

        overlay_img = overlay_attention_mask(np_img, cam_resized)
        #plt.imshow(overlay_img)
        #plt.axis('off')  # Hide axes ticks
        #plt.show()

        overlay_img = np.array(overlay_img)

        #overlay_img = ((colored_overlay.astype(np.float32) * 0.5) + (np_img.astype(np.float32) * 0.5)).astype(np.uint8)

        ###
        cam_resized2 = cam_true_country.cpu().detach().numpy()
        cam_resized2 = (cam_resized2 *255).astype(np.uint8)
        cam_resized2 = np.array(Image.fromarray(cam_resized2).resize((img.shape[2], img.shape[1]), Image.BILINEAR)) 

        overlay_img2 = overlay_attention_mask(np_img, cam_resized2)
        overlay_img2 = np.array(overlay_img2)

        #print("SHAPES", type(colored_overlay), type(np_img), np.max(np_img))
        
        #overlay_img = cam_resized.astype(np.uint8) 

        # Ensure the output directory exists
        output_dir = 'cam_outputs_resnet'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.epochh>=0:
            output_path = os.path.join(output_dir, f'original_image_{country}.png')
            pil_image.save(output_path, inplace=True)

        # Save the image
        output_path = os.path.join(output_dir, f'CAM_{self.epochh}{pred_country}{torch.max(pred)}.png')
        Image.fromarray(overlay_img).save(output_path)

        output_path = os.path.join(output_dir, f'CAM_{self.epochh}{country}{pred[0, classe]}.png')
        Image.fromarray(overlay_img2).save(output_path)

        self.epochh+=1

    
    def generate_cam(self, feature_maps, pred_class):
        # Get the weight of the `linear` layer for the predicted class
        #print("WW", self.gap_mlp.weight.shape, self.gap_mlp.weight[pred_class,:].shape)
        W = self.gap_mlp.weight[pred_class,:].view(self.gap_mlp.weight.shape[1], 1, 1)
        feature_maps = torch.squeeze(feature_maps, 0)

        #print(W.shape, feature_maps.shape)
        cam = feature_maps*W
        #print(cam.shape)
        #adding relu
        cam = self.relu(cam)
        cam = cam.sum(dim=0)

        #print("MAX", torch.max(cam))
        return cam
    
def overlay_attention_mask(image, attention_mask):

    # Ensure the attention mask is in the range [0, 1]
    attention_mask_normalized = attention_mask / 255.0

    # Create a colormap (from blue to red)
    colormap = plt.get_cmap('bwr')
    
    # Apply the colormap to the attention mask
    attention_colored = colormap(attention_mask_normalized)[:, :, :3]  # Drop alpha channel
    
    # Blend the attention overlay with the original image
    overlayed_image = (0.5 * image + 0.5 * attention_colored * 255).astype(np.uint8)
    
    return Image.fromarray(overlayed_image)

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
            class_train, class_test = train_test_split(files, test_size=0.3, random_state=2)
            
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

print(train_dataset.summary)



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

    a = trial.suggest_int("a", 25, 26, 1)
    b = trial.suggest_int("b", 100, 101, 1)
    c = trial.suggest_int("c", 62, 63, 1)
    #c = 0
    #d = trial.suggest_int("d", 10, 120, 10)
    d=0
    #e = trial.suggest_int("e", 10, 120, 10)

    model = Classifier(a, b, c, d, n_classes)
    model.to(mps_device)

    mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./ml-runs")

    trainer = pl.Trainer(max_epochs=epochs, accelerator="mps", logger=mlf_logger, log_every_n_steps=1)
    trainer.fit(model, train_loader , test_loader)

    val_acc = trainer.callback_metrics["val_acc"].item()

    return val_acc

pruner = optuna.pruners.MedianPruner()

study = optuna.create_study(direction="maximize", pruner=pruner)
study.optimize(objective, n_trials=1)

print(f"Number of finished trials: {len(study.trials)}")

print("Best trial: ")
trial = study.best_trial

print(f" Value: {trial.value}")

print(" Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")