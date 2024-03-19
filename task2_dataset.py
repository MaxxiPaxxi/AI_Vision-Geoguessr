from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader, Dataset, random_split
import random
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import numpy as np
import copy
import os
import pickle 
from sklearn.model_selection import train_test_split
import torch

#1. Dataloader
class ImageDataset_task2(Dataset):
    def __init__(self, root_dir, resizing = (80, 160)):

        # Define your transformations here, if any

        if resizing is None:
            self.transform = transforms.Compose([transforms.ToTensor()], )
        else:
            self.transform = transforms.Compose([
                transforms.Resize(resizing),
                transforms.ToTensor()
            ])

        corresp = np.load('/Users/mathieugierski/Nextcloud/Macbook M3/vision/task_2/ImageID_to_GPS.npy')
        corresp, self.means, self.stds = normalize_array(corresp)

        #print(corresp)

        with open('saved_dictionary.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)

        loaded_dict = {int(k): v for k, v in sorted(loaded_dict.items(), key=lambda item: int(item[0]))}

        corresp_dict = {}
        corresp_dict_rev = {}
        iterator = 0

        for k in loaded_dict:
            for i in range(loaded_dict[k]+1):
                corresp_dict[(int(k), int(i))]=iterator
                corresp_dict_rev[iterator]=(int(k), int(i))
                iterator+=1

        self.corresp_dict_rev = corresp_dict_rev
       
        self.root_dir = root_dir
        self.images = []
        self.labels = []
        self.class_to_idx = {}

        self.counting_classes = {}
        self.new_counting_classes = {}

        self.planning = {}
        self.instructions = {} #What to do for get item
        self.summary = {}
        self.summary_before = {}

        # Use a set for faster lookup
        self.existing_classes = set()

        self.maxi = 10000

        # Assume root_dir is a directory. Adjust if root_dir is actually a list of files.

        idx = 0
        for file in root_dir:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):

                #self.images.append(file)
                class_name = file.split("/")[-1]
                print("class name", class_name)
                class_name = class_name.split('.')[0]
                classe = (int(class_name.split('_')[-2]), int(class_name.split('_')[-1]))
                print(classe)

                self.instructions[idx] = (file, corresp[corresp_dict[classe]][0], corresp[corresp_dict[classe]][1])
                idx+=1
  
    def __len__(self):
        return len(list(self.instructions.keys()))
        #return len(self.images)

    def __getitem__(self, idx):

        instruction = self.instructions[idx]
        file = instruction[0]
        lon = instruction[1]
        lat = instruction[2]

        image = Image.open(file).convert('RGB')
            
        return self.transform(image), torch.Tensor([lon, lat])


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
        #print(class_path)
        files = get_files_from_directory(class_path)

        if len(files)>1:

            total+=1

            # Perform the split
            class_train, class_test = train_test_split(files, test_size=0.3)
            
            train_files.extend(class_train)
            test_files.extend(class_test)

    print("total", total)
    
    return train_files, test_files, total

#dir = '/Users/mathieugierski/Nextcloud/Macbook M3/vision/data_treated_task2'
#train_files, test_files, n_classes = split_files_by_class(dir)

# Assuming you modify ImageDataset to accept a list of files:
#train_dataset = ImageDataset_task2(train_files)


def normalize_array(arr):

    # Compute the mean and standard deviation for each column (feature)
    means = arr.mean(axis=0)
    stds = arr.std(axis=0)
    
    # Normalize the array
    normalized_arr = (arr - means) / stds
    
    return normalized_arr, means, stds