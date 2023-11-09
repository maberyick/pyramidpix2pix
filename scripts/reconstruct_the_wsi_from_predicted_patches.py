import cv2
import os
import pandas as pd
import numpy as np

# Load the CSV file containing patch information
csv_filename = "/home/maberyick/pCloudDrive/CCIPD_echo/Projects/Immune_SCLC/dl_training/test/smaller_version/patch_tracking.csv"  # Update with your CSV file path
patch_info_df = pd.read_csv(csv_filename)

# Determine the original image filename from the first row
original_image_filename = patch_info_df['image_file_name'].iloc[0]

# Construct the full path to the original image
original_image_path = os.path.join("/home/maberyick/pCloudDrive/CCIPD_echo/Projects/Immune_SCLC/dl_training/test/im/", original_image_filename)

# Load the original image to get its dimensions
original_image = cv2.imread(original_image_path)

# Directory containing the saved patches to reconstruct the image
patch_directory = "/home/maberyick/pCloudDrive/CCIPD_echo/Projects/Immune_SCLC/dl_training/test/smaller_version/results/pyramidpix2pix/test_latest/images/"  # Update with your saved patches directory

# Initialize the canvas with the original image dimensions
canvas = original_image.copy()

# Iterate through patch information and reconstruct the image
for index, row in patch_info_df.iterrows():
    patch_name = row['patch_name']
    patch_name = os.path.splitext(patch_name)[0] + '_fake_B.png'  # Add _fake_B.png to the patch_name
    patch_width = row['width']
    patch_height = row['height']
    status = row['status']

    # Check if the patch is marked as "saved"
    if status == "saved":
        # Load the saved patch image
        patch_image = cv2.imread(os.path.join(patch_directory, patch_name))

        # Check if the patch image was successfully loaded
        if patch_image is not None:
            # Resize the patch to 1024x1024
            patch_image = cv2.resize(patch_image, (1024, 1024))

            # Extract the position information from the patch_name
            y = int(int(patch_name.split('_')[2]))
            x = int(int(patch_name.split('_')[3].split('.')[0]))

            # Paste the resized patch onto the canvas
            canvas[y:y+1024, x:x+1024] = patch_image
        else:
            print(f"Warning: Failed to load patch image '{patch_name}'")

# Save or display the reconstructed image
reconstructed_image_path = "/home/maberyick/pCloudDrive/CCIPD_echo/Projects/Immune_SCLC/dl_training/test/smaller_version/reconstruct/reconstructed_image.png"  # Update with your desired output path
cv2.imwrite(reconstructed_image_path, canvas)

print("Reconstruction complete. Reconstructed image saved.")
