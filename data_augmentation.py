from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
import random
from torchvision import transforms
import matplotlib.pyplot as plt
import time

#1. Dataloader
class ImageDataset(Dataset):
    def __init__(self, root_dir):

        # Define your transformations here, if any
        self.transform = transforms.Compose([transforms.ToTensor()], )
       
        self.root_dir = root_dir
        self.images = []
        self.labels = []
        self.class_to_idx = {}

        self.counting_classes = {}

        self.patch_size_classes = {}

        # Use a set for faster lookup
        self.existing_classes = set()

        # Assume root_dir is a directory. Adjust if root_dir is actually a list of files.
        for file in root_dir:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):

                #self.images.append(file)
                class_name = file.split("/")[-2]

                if class_name not in self.existing_classes:
                    self.existing_classes.add(class_name)
                    self.counting_classes[class_name]=1
                    # Directly assign class index
                    self.class_to_idx[class_name] = len(self.existing_classes) - 1
                else:
                    self.counting_classes[class_name]+=1

        #Get the classes that need oversampling:
        for key in self.counting_classes:
            if self.counting_classes[key]>8000:
                self.patch_size_classes[key] = 1
            elif self.counting_classes[key]>4000:
                self.patch_size_classes[key] = 2
            elif self.counting_classes[key]>2600:
                self.patch_size_classes[key] = 3
            elif self.counting_classes[key]>2000:
                self.patch_size_classes[key] = 4
            elif self.counting_classes[key]>1000:
                self.patch_size_classes[key] = 5
            elif self.counting_classes[key]>80:
                self.patch_size_classes[key] = 6
            else:
                self.patch_size_classes[key] = None


        #Now build the image resizing with quantity required by counting classes:
        for file in root_dir:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):

                class_name = file.split("/")[-2]
                image = Image.open(file).convert('RGB')

                if self.counting_classes[class_name] is not None:
                    self.images+=extract_random_patches(image, num_patches=self.counting_classes[class_name])

                    for i in range(self.counting_classes[class_name]):
                        self.labels.append(self.class_to_idx[class_name])
       
            
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        #img_path = self.images[idx]
        #image = Image.open(img_path).convert('RGB')
        #image = extract_random_patches(image, num_patches=1)[0]
        #class_name = img_path.split("/")[-2]
        #label = self.class_to_idx[class_name]

        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)


        #print("LABEL", label, image.shape)
        print('SHAPE', image.shape)
        return image, label
    

def extract_random_patches(image, patch_size=(80, 40), num_patches=5):
    width, height = image.size
    
    # Calculate the valid range for the top-left corner of a patch

    min_x_start = 0
    max_x_start = width - patch_size[0]

    min_y_noise = -10
    max_y_noise = 10

    
    if num_patches==1:
        patch = image.crop((width/2-patch_size[0]/2, height/2-patch_size[1]/2, width/2+patch_size[0]/2, height/2+patch_size[1]/2))
        #patch = image.crop((0, 0, max_x, max_y/2))
        #print(type(patch), type(image), image.size, patch.size)
        #patch.show()
        #image.show()
        #time.sleep(10)
        return [patch]

    else:
        patches = []
        for _ in range(num_patches):

            # Random top-left corner
            x = random.randint(min_x_start, max_x_start)
            y = random.randint(min_y_noise, max_y_noise)

            # Extract the patch
            patch = image.crop((x, y-patch_size[1]/2, x+width, y+patch_size[1]/2))
            patches.append(patch)
        
        return patches
