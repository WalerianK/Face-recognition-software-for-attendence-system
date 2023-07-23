'''
Note : 

Your script must have all the images pooled 
in one single folder and be named like this : 

Assuming users are A,B,C,D...
data/ 
- A_1.jpg
- A_2.jpg 
- A_3.jpg
....
- B_1.jpg
- B_2.jpg
- B_3.jpg
.....

It will then be sorted into USERs 
and dedicated train/ and test/ folders 
and split the data into each folders based on 
train & test split percentage.

'''

import argparse
import copy 
import os 
import shutil
import sys
import random
from random import shuffle
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="Data Preperation Options")
    parser.add_argument('--parent_folder', type=str, help="Input Data Folder")
    parser.add_argument('--target_folder', type=str, help="Output Data Folder")
    args = parser.parse_args()
    return args

# Checks if a folder exists anf has the elements
def checkFolder(folder_path):
    '''
    Checks 
    '''
    if os.path.exists(folder_path):  # Check if folder exists
        if os.listdir(folder_path):  # Check if folder is not empty
            file_count = sum(1 for _ in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, _)))
            print(f"The folder '{folder_path}' exists and contains {file_count} files.")
            return True
        else:
            print(f"The folder '{folder_path}' exists but is empty.")
            return False
    else:
        print(f"The folder '{folder_path}' does not exist.")
        return False

# Picks up random N elements from the input list
def pick_random_elements(parent_list, N):
    random_elements = random.sample(parent_list, N)
    for element in random_elements:
        parent_list.remove(element)
    return random_elements, parent_list

# Copies the images from image_list from source_folder to destination_folder
def copy_images(image_list, source_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)  # Create the destination folder if it doesn't exist
    for image_name in image_list:
        source_path = os.path.join(source_folder, image_name)
        destination_path = os.path.join(destination_folder, image_name)

        if os.path.isfile(source_path) and image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            shutil.copy2(source_path, destination_path)


# Perform the sorting operation from parent to creat each user's target folder based on test_percent 
def user_folder_creation(args,test_percent=0.15):
    parent_folder=args.parent_folder
    target_folder = args.target_folder
    train_percent = 1-test_percent
    # Check if the parent folder exists and is non empty
    try:
        assert checkFolder(folder_path=parent_folder),'Improper Parent Folder!'
    except Exception:
        sys.exit(0)
    # Make the target folder
    os.makedirs(target_folder,exist_ok=True)
    # Get all the files in the parent folder to get their names 
    classes_=[ f.split('_')[0]  for f in os.listdir(parent_folder)]
    classes=list(set(classes_))
    print('Detected {} user data !'.format(len(classes)))
    for i,imgRoot in enumerate(classes):
        print('Processing USER : {}'.format(imgRoot))
        # Collect all the files with this as imageRoot 
        user_files = [ f for f in os.listdir(parent_folder) if imgRoot in str(f)]
        # Test files 
        test_files,user_files =  pick_random_elements(user_files,N=int(len(user_files)*test_percent))
        # Train files 
        train_files = copy.deepcopy(user_files)

        # Copy and paste them in the target folder seperately 
        user_folder = os.path.join(target_folder,imgRoot)
        test_user_folder = os.path.join(user_folder,'test')
        train_user_folder = os.path.join(user_folder,'train')
        # Create the user folder 
        os.makedirs(user_folder,exist_ok=True)
        os.makedirs(test_user_folder,exist_ok=True)
        os.makedirs(train_user_folder,exist_ok=True)

        # Copy to respective folders 
        copy_images(test_files,parent_folder,test_user_folder)
        copy_images(train_files,parent_folder,train_user_folder)

    print('User folder and sub-folders are created successfully!')


if __name__ == "__main__":
    parsed_args = parse_args()
    user_folder_creation(parsed_args,test_percent=0.15)
    print('Stage0 Completed Successfully!')

