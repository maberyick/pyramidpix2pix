import os
import cv2
import numpy as np
import random
import shutil

numSamplesPerCase = 50
patchSize = 512

# Prompt the user for input folder paths
inputImgFolder1 = input("Enter the input image folder path 1: ")
inputImgFolder2 = input("Enter the input image folder path 2: ")

# Prompt the user for the output folder path
outputFolder = input("Enter the output folder path: ")

# Create train, test, and validation subfolders for full images in the "main" folder
outputTrainFolder = os.path.join(outputFolder, "main", "train")
outputTestFolder = os.path.join(outputFolder, "main", "test")
outputValidationFolder = os.path.join(outputFolder, "main", "validation")

# Create train, test, and validation subfolders for patches in the "patch" folder
outputPatchTrainFolder = os.path.join(outputFolder, "patch", "train")
outputPatchTestFolder = os.path.join(outputFolder, "patch", "test")
outputPatchValidationFolder = os.path.join(outputFolder, "patch", "validation")

folders = [
    outputTrainFolder, outputTestFolder, outputValidationFolder,
    outputPatchTrainFolder, outputPatchTestFolder, outputPatchValidationFolder
]

os.makedirs(outputTrainFolder, exist_ok=True)
os.makedirs(outputTestFolder, exist_ok=True)
os.makedirs(outputValidationFolder, exist_ok=True)
os.makedirs(outputPatchTrainFolder, exist_ok=True)
os.makedirs(outputPatchTestFolder, exist_ok=True)
os.makedirs(outputPatchValidationFolder, exist_ok=True)

# Prompt the user for the percentage splits
train_percent = 0.7
test_percent = 0.15
validation_percent = 0.15

# Iterate over the folders and organize the images
for folder in folders:
    files = [file for file in os.listdir(inputImgFolder1) if file.endswith('_snapshot.png')]
    num_files = len(files)
    num_train = int(train_percent * num_files)
    num_test = int(test_percent * num_files)
    num_validation = num_files - num_train - num_test
    
    # Shuffle the files randomly
    random.shuffle(files)
    
    # Determine the appropriate destination folder
    if folder == outputTrainFolder:
        selected_files = files[:num_train]
    elif folder == outputTestFolder:
        selected_files = files[num_train:num_train + num_test]
    elif folder == outputValidationFolder:
        selected_files = files[num_train + num_test:]
    elif folder == outputPatchTrainFolder:
        selected_files = files[:num_train]
    elif folder == outputPatchTestFolder:
        selected_files = files[num_train:num_train + num_test]
    elif folder == outputPatchValidationFolder:
        selected_files = files[num_train + num_test:]
    
    # Copy the selected files to the destination folder
    for sampleName in selected_files:
        src_path_img = os.path.join(inputImgFolder1, sampleName)
        src_path_label = os.path.join(inputImgFolder2, sampleName.replace('_snapshot.png', '_labels.png'))
        dst_path_img = os.path.join(folder, 'im', sampleName.replace(' ', '_'))
        dst_path_label = os.path.join(folder, 'mask', sampleName.replace(' ', '_').replace('_snapshot.png', '_labels.png'))
        
        os.makedirs(os.path.dirname(dst_path_img), exist_ok=True)
        os.makedirs(os.path.dirname(dst_path_label), exist_ok=True)
        
        shutil.copy(src_path_img, dst_path_img)
        shutil.copy(src_path_label, dst_path_label)
