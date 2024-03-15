import os
import shutil
import numpy as np
from pathlib import Path
import sys

def rename_files(folder_path):
    """
    Rename files in the given folder according to the specified format.
    """
    file_names = os.listdir(folder_path)
    name_parts = os.path.basename(folder_path).split('_')  # Extract name parts from folder name
    name = '_'.join(name_parts[1:]).strip().title().replace(' ','_') # Extract name from parts and join with underscore

    # Iterate over files and rename
    for i, file_name in enumerate(file_names, start=1):
        file_ext = os.path.splitext(file_name)[1]  # Extract file extension
        new_name = f"{name}_{i:03}{file_ext}"  # Format new name with leading zeros
        old_path = os.path.join(folder_path, file_name)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)

def move_files(src_dir, dest_dir, folder_name, files):
    """
    Move files from source directory to destination directory.
    """
    for file_name in files:
        src_path = os.path.join(src_dir, file_name)
        dest_path = os.path.join(dest_dir, folder_name, file_name)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.move(src_path, dest_path)

def split_data(folder_path, number_of_validation_images):
    """
    Split data into train, validation and test sets.
    """
    print('folder_path : ',folder_path)
    if os.path.isdir(folder_path):
        image_files = [file for file in os.listdir(folder_path) if file.endswith('.jpg')]
        np.random.shuffle(image_files)
        
        if len(image_files) > 30:
            
            return image_files[:number_of_validation_images], image_files[number_of_validation_images:-2], image_files[-2:]

def main():
    """
    Main function to split data and move files.
    """
    data_dir = os.path.join(Path(sys.path[0]).resolve(), 'data')
    new_data_dir = os.path.join(data_dir, 'new_data')
    train_dir = os.path.join(data_dir, 'train')
    validation_dir = os.path.join(data_dir, 'validation')
    test_dir = os.path.join(data_dir, 'test')
    number_of_validation_images = 15

    

    for new_celebrity in os.listdir(new_data_dir):
        
        if new_celebrity.startswith('pins'):
            
            celebrity_dir = os.path.join(new_data_dir, new_celebrity)

            rename_files(folder_path=celebrity_dir)

            print('rename done for : ',new_celebrity.split('_')[1])
            
            validation_images, train_images, test_images  = split_data(folder_path = celebrity_dir, number_of_validation_images = number_of_validation_images)
            
            folder_name = new_celebrity.split('_')[1].strip().title().replace(' ','_')
            if train_images:
                move_files(src_dir = celebrity_dir, dest_dir = train_dir,folder_name = folder_name, files = train_images)

            if validation_images:
                move_files(src_dir = celebrity_dir, dest_dir = validation_dir,folder_name = folder_name, files = validation_images)

            if test_images:
                move_files(src_dir = celebrity_dir, dest_dir = test_dir,folder_name = folder_name, files = test_images)

if __name__ == "__main__":
    main()