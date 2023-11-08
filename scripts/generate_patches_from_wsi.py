import cv2
import os
import numpy as np
import pandas as pd

# Define the paths to the input image and mask folders
image_folder = "/home/cbarr23/Documents/sclc_annotation/tumor_segmentation_images/pix2pix_split/train/he/"
mask_folder = "/home/cbarr23/pCloudDrive/CCIPD_echo/Projects/Immune_SCLC/dl_training/mask/"

# Define the output folders for patches
output_image_folder = "/home/cbarr23/Documents/sclc_annotation/tumor_segmentation_images/pix2pix_split/patch/train/im/"
output_mask_folder = "/home/cbarr23/Documents/sclc_annotation/tumor_segmentation_images/pix2pix_split/patch/train/mask/"

# Ensure the output folders exist
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_mask_folder, exist_ok=True)

# Define the patch size
patch_size = (256, 256)

# Define the color thresholds (adjust as needed)
blue_threshold = 100  # Adjust as needed
red_threshold = 100   # Adjust as needed

# Define the white pixel threshold (percentage)
white_threshold = 35 # Adjust as needed

# Create a CSV file to track patches
csv_filename = "/home/cbarr23/Documents/sclc_annotation/tumor_segmentation_images/pix2pix_split/patch/patch_tracking.csv"
csv_columns = ["patch_name", "image_file_name", "status", "reason", "width", "height"]

# Initialize a list to store patch information
patch_info = []

# Iterate through all image files in the input folder
image_files = [f for f in os.listdir(image_folder) if f.endswith("snapshot.png")]

total_images = len(image_files)

for i, image_file in enumerate(image_files, start=1):
    print(f"Processing image {i}/{total_images}: {image_file}")

    # Load the image and corresponding mask
    image_path = os.path.join(image_folder, image_file)
    mask_file = image_file.replace("snapshot.png", "labels.png")
    mask_path = os.path.join(mask_folder, mask_file)

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)

    # Split the image and mask into patches
    for x in range(0, image.shape[0], patch_size[0]):
        for y in range(0, image.shape[1], patch_size[1]):
            patch_image = image[x:x + patch_size[0], y:y + patch_size[1]]
            patch_mask = mask[x:x + patch_size[0], y:y + patch_size[1]]

            # Calculate the percentage of white pixels in the patch
            # Define the white pixel threshold (percentage)
            white_threshold = 85  # Adjust as needed
            # Define the range for white or light-colored pixels
            white_range = ([200, 200, 200], [255, 255, 255])  # Adjust as needed
            # Calculate the percentage of white pixels in the patch
            white_percentage = ((patch_image >= white_range[0]) & (patch_image <= white_range[1])).all(axis=2).sum() / patch_image.size * 100
            #white_percentage = (np.sum(patch_image == [255, 255, 255]) / patch_image.size) * 100.0

            # Calculate the percentage of pixels below the color thresholds
            blue_below_threshold = np.sum(patch_image[:, :, 0] < blue_threshold)
            red_below_threshold = np.sum(patch_image[:, :, 2] < red_threshold)
            color_percentage = ((blue_below_threshold + red_below_threshold) / patch_image.size) * 100.0
            print("image type")
            print(np.sum(patch_image == [255,255,255]))
            print("white space")
            print(white_percentage)
            print("color issue")
            print(color_percentage)
            # Check if the patch contains mostly white pixels or light-colored speckles
            if white_percentage >= white_threshold or color_percentage >= white_threshold:
                print("here is white")
                reason = "Too white or light-colored"
                status = "ignored"
                # Create a black patch for the ignored region
                patch_image = np.zeros_like(patch_image)
                patch_mask = np.zeros_like(patch_mask)
            elif patch_image.shape[0] != patch_image.shape[1]:
                reason = "Not square"
                status = "ignored"
            else:
                reason = ""
                status = "saved"

            # Define patch filenames based on image filename
            patch_image_filename = f"{os.path.splitext(image_file)[0]}_{x}_{y}.png"
            patch_mask_filename = f"{os.path.splitext(image_file)[0]}_{x}_{y}.png"

            # Save the patches to the output folders
            if status == "saved":
                cv2.imwrite(os.path.join(output_image_folder, patch_image_filename), patch_image)
                cv2.imwrite(os.path.join(output_mask_folder, patch_mask_filename), patch_mask)

            # Record patch information
            patch_info.append([patch_image_filename, image_file, status, reason, patch_image.shape[1], patch_image.shape[0]])

# Create a DataFrame from the patch information
patch_df = pd.DataFrame(patch_info, columns=csv_columns)

# Save the DataFrame to a CSV file
patch_df.to_csv(csv_filename, index=False)

print("Patching complete.")
