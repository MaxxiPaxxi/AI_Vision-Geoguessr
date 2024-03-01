from PIL import Image
import os

def resize_images(folder, output_folder, size=(240, 120)):
    
    id = 0

    for filename in os.listdir(folder):
        print("0", filename)

        if not filename.startswith('.'): #For the DS_Store

            country_path = os.path.join(folder, filename)
            saving_path = os.path.join(output_folder, filename)

            if not os.path.exists(saving_path):
                os.makedirs(saving_path)

            print("1", folder, country_path)

            for photo in os.listdir(country_path):
                #print("2", photo)
                img = Image.open(os.path.join(country_path, photo))
                # Resize the image and use ANTIALIAS filter to maintain quality
                img = img.resize(size)

                # Save the resized image to the output folder
                img.save(os.path.join(saving_path, "image_"+str(id)+".jpg"))
                id+=1

# Define the path to the folder containing the images
source_folder = '/Users/mathieugierski/Nextcloud/Macbook M3/vision/data'
# Define the path to the folder where resized images will be saved
destination_folder = '/Users/mathieugierski/Nextcloud/Macbook M3/vision/data_treated'

# Call the function with the updated folder paths
resize_images(source_folder, destination_folder)


