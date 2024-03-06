from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, random_split
import random
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import numpy as np
import copy

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
        self.new_counting_classes = {}

        self.patch_size_classes = {}
        self.noised_classes = {}

        # Use a set for faster lookup
        self.existing_classes = set()

        self.maxi = 10000

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

        #Get the classes that need oversampling by resizing:
        for key in self.counting_classes:

            if self.counting_classes[key]>1500:#4000
                a, b = max_allowed(8000, self.maxi)

                """
                elif self.counting_classes[key]>3000:
                    a, b = max_allowed(4000, self.maxi)

                elif self.counting_classes[key]>2000:
                    a, b = max_allowed(3000, self.maxi)

                elif self.counting_classes[key]>1200:
                    a, b = max_allowed(2000, self.maxi)

                elif self.counting_classes[key]>500:
                    a, b = max_allowed(1200, self.maxi)

                elif self.counting_classes[key]>200:
                    a, b = max_allowed(500, self.maxi)

                elif self.counting_classes[key]>60:
                    a, b = max_allowed(200, self.maxi)
                """

            else:
                a, b = None, None


            self.patch_size_classes[key] = b
            self.noised_classes[key] = a


        print("RESIZED")

        #Now build the image resizing with quantity required by counting classes:
        for file in root_dir:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):

                class_name = file.split("/")[-2]
                image = Image.open(file).convert('RGB')

                #print("hello")

                if self.patch_size_classes[class_name] is not None:

                    img = extract_random_patches(image, num_patches=self.patch_size_classes[class_name])
                    if img is not None:
                        self.images+=img

                        #print(len(self.images), class_name, self.patch_size_classes[class_name])

                        for _ in range(len(img)):
                            self.labels.append(self.class_to_idx[class_name])

                        if class_name not in self.new_counting_classes:
                            self.new_counting_classes[class_name]=len(img)
                        else:
                            self.new_counting_classes[class_name]+=len(img)

                else:
                    img = None

                #Add Gaussian noise:
                if img is not None:

                    for i in img:
                        width, height = image.size
                        #print(width, height)
                        noised = add_gaussian_noise(i, n=self.noised_classes[class_name])
                        blacked = add_black_hole(i, n = self.noised_classes[class_name])
                        self.images+=noised
                        self.images+=blacked
                        self.new_counting_classes[class_name]+=len(noised)+len(blacked)

                        for _ in range(len(noised)+len(blacked)):
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
        #print('SHAPE', image.shape)
        return image, label

#################### get how many samples we want
def max_allowed(n, maxi):

    b = 1
    a = 1

    if n*(1+2*a)*b>2*maxi:
        b = 0.8
        return a, b
        

    while n*(1+2*a)*b<=maxi:

        a+=1
        if a>3:
            b+=1
            a=1

    return a, b

    
################## 2. Make sub-images
def extract_random_patches(image, patch_size=(80, 40), num_patches=5):
    width, height = image.size
    
    # Calculate the valid range for the top-left corner of a patch

    min_x_start = 0
    max_x_start = width - patch_size[0]

    min_y_noise = -12
    max_y_noise = 12

    if num_patches<1:

        #In this case num-patches allows to reduce the number of images in the class
        if np.random.random_sample()<=num_patches:
            patch = image.crop((width/2-patch_size[0]/2, height/2-patch_size[1]/2, width/2+patch_size[0]/2, height/2+patch_size[1]/2))
            #print(patch.size)
            return [patch]
        
        else:
            return None
        
    
    else:

        if num_patches==1:
            patch = image.crop((width/2-patch_size[0]/2, height/2-patch_size[1]/2, width/2+patch_size[0]/2, height/2+patch_size[1]/2))
            return [patch]

        else:
            patches = []
            for _ in range(num_patches):

                x = random.randint(min_x_start, max_x_start)
                y = random.randint(min_y_noise, max_y_noise)

                patch = image.crop((x, height/2+y-patch_size[1]/2, x+patch_size[0], height/2+y+patch_size[1]/2))
                patches.append(patch)
            
            return patches

      
################### 3. Add gaussian noise
def add_gaussian_noise(image, mean=0, std=0.05, n=1):

    result = []

    for _ in range(n):

        np_image = np.array(image)
        noise = 255 * np.random.normal(mean, std, np_image.shape)
        noisy_image = np_image + noise
        noisy_image = np.clip(noisy_image, 0, 255)
        noisy_image = Image.fromarray(noisy_image.astype(np.uint8))

        #noisy_image.show()
        #time.sleep(5)
        result.append(noisy_image)
    
    return result

################# 4. Add black hole
def add_black_hole(image, n=1):

    width, height = image.size
    result = []

    for _ in range(n):

        center = (np.random.randint(0, width-1), np.random.randint(0, height-1))
        radius = np.random.randint(8, 25)

        new = copy.deepcopy(image)
        draw = ImageDraw.Draw(new)
        left_up_point = (center[0] - radius, center[1] - radius)
        right_down_point = (center[0] + radius, center[1] + radius)
        
        draw.ellipse([left_up_point, right_down_point], fill="black")

        result.append(new)
        #new.show()
        #print(center, radius)
        #time.sleep(5)
    
    return result
