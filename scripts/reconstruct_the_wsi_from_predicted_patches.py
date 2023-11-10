import cv2
import os
import pandas as pd
import numpy as np

# Load the CSV file containing patch information
csv_filename = "/home/maberyick/pCloudDrive/CCIPD_echo/Projects/Immune_SCLC/dl_training/test/smaller_version_2/patch_tracking_test.csv"  # Update with your CSV file path
patch_info_df = pd.read_csv(csv_filename)

# Determine the original image filename from the first row
original_image_filename = patch_info_df['image_file_name'].iloc[0]

# Construct the full path to the original image
original_image_path = os.path.join("/home/maberyick/pCloudDrive/CCIPD_echo/Projects/Immune_SCLC/dl_training/test/im/", original_image_filename)

# Load the original image to get its dimensions
original_image = cv2.imread(original_image_path)

# Directory containing the saved patches to reconstruct the image
patch_directory = "/home/maberyick/pCloudDrive/CCIPD_echo/Projects/Immune_SCLC/dl_training/test/smaller_version_2/result/pyramidpix2pix_1024_1024/test_latest/images/"  # Update with your saved patches directory

# Create a binary mask for the tissue region
# Define color thresholds for tissue detection (adjust as needed)
lower_color_threshold = np.array([50, 50, 50])  # Adjust as needed
upper_color_threshold = np.array([255, 255, 255])  # Adjust as needed
real_tissue_mask = cv2.inRange(original_image, lower_color_threshold, upper_color_threshold)

# Invert the colors of the tissue_mask
tissue_mask = cv2.bitwise_not(real_tissue_mask)

# Initialize the canvas with the original image dimensions
canvas = np.zeros_like(original_image)

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

            # Apply blending to combine overlapping patches
            # Define the overlap region (adjust as needed)
            overlap_width = 0  # Example overlap width, adjust as needed

            if x > 0:
                left_patch = canvas[y:y+1024, x:x+overlap_width]
                right_patch = patch_image[:, :overlap_width]

                alpha_mask = np.linspace(1, 0, overlap_width).reshape(1, -1, 1)
                canvas[y:y+1024, x:x+overlap_width] = (left_patch * alpha_mask + right_patch * (1 - alpha_mask)).astype(np.uint8)

            # Paste the remaining part of the patch onto the canvas
            canvas[y:y+1024, x+overlap_width:x+1024] = patch_image[:, overlap_width:]

        else:
            print(f"Warning: Failed to load patch image '{patch_name}'")

# Post-process the final image
# Convert to binary (if not already)
canvas_binary = cv2.threshold(canvas, 128, 255, cv2.THRESH_BINARY)[1]

canvas_binary = canvas_binary[:,:,1]

# Ensure that the tissue mask has the same dimensions as canvas_binary
if tissue_mask.shape[:2] != canvas_binary.shape[:2]:
    tissue_mask = cv2.resize(tissue_mask, (canvas_binary.shape[1], canvas_binary.shape[0]))

# Fill holes in the binary image
kernel_tissue = np.ones((64, 64), np.uint8)
tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel_tissue)

# Remove small objects (blobs) based on area
min_blob_area = 1024  # Minimum area of a blob to keep (adjust as needed)
contours, _ = cv2.findContours(tissue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    if cv2.contourArea(contour) < min_blob_area:
        cv2.drawContours(tissue_mask, [contour], -1, 0, thickness=cv2.FILLED)

# Save or display the reconstructed and post-processed binary image
reconstructed_image_path = "/home/maberyick/pCloudDrive/CCIPD_echo/Projects/Immune_SCLC/dl_training/test/smaller_version_2/build/tissue.png"  # Update with your desired output path
cv2.imwrite(reconstructed_image_path, tissue_mask)

# Multiply the tissue mask with the binary canvas
canvas_binary_glass = cv2.bitwise_and(canvas_binary, tissue_mask)

# Find contours in the binary image
contours, _ = cv2.findContours(canvas_binary_glass, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Initialize a mask to keep the contours you want
mask = np.zeros_like(canvas_binary_glass)
# Specify the minimum area to keep (e.g., 200 pixels)
min_contour_area = 256
# Iterate through the contours
for contour in contours:
    # Calculate the area of the contour
    area = cv2.contourArea(contour)
    # If the area is greater than or equal to the threshold, keep the contour
    if area >= min_contour_area:
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
# Apply the mask to the binary image to remove small blobs
canvas_binary_glass = cv2.bitwise_and(canvas_binary_glass, mask)

# Apply smoothing (dilation and erosion)
smooth_kernel = np.ones((16, 16), np.uint8)
canvas_binary_glass = cv2.morphologyEx(canvas_binary_glass, cv2.MORPH_CLOSE, smooth_kernel)

# Connect nearby binary blobs by dilation
connect_kernel = np.ones((16, 16), np.uint8)
canvas_binary_glass = cv2.dilate(canvas_binary_glass, connect_kernel, iterations=1)

# Make patches more circular (morphological operation)
circular_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 16))
canvas_binary_glass = cv2.morphologyEx(canvas_binary_glass, cv2.MORPH_CLOSE, circular_kernel)

# Find contours in the binary image
contours, _ = cv2.findContours(canvas_binary_glass, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Initialize a mask to keep the contours you want
mask = np.zeros_like(canvas_binary_glass)
# Specify the minimum area to keep (e.g., 200 pixels)
min_contour_area = 1024
# Iterate through the contours
for contour in contours:
    # Calculate the area of the contour
    area = cv2.contourArea(contour)
    # If the area is greater than or equal to the threshold, keep the contour
    if area >= min_contour_area:
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
# Apply the mask to the binary image to remove small blobs
canvas_binary_glass = cv2.bitwise_and(canvas_binary_glass, mask)

# Fill holes using a larger kernel
kernel = np.ones((16, 16), np.uint8)
canvas_binary_glass = cv2.morphologyEx(canvas_binary_glass, cv2.MORPH_CLOSE, kernel)

# Save or display the reconstructed and post-processed binary image
reconstructed_image_path = "/home/maberyick/pCloudDrive/CCIPD_echo/Projects/Immune_SCLC/dl_training/test/smaller_version_2/build/reconstructed_image.png"  # Update with your desired output path
cv2.imwrite(reconstructed_image_path, canvas_binary)

# Save the canvas binary glass mask
canvas_binary_glass_path = "/home/maberyick/pCloudDrive/CCIPD_echo/Projects/Immune_SCLC/dl_training/test/smaller_version_2/build/reconstructed_image_glass.png"  # Update with your desired output path
cv2.imwrite(canvas_binary_glass_path, canvas_binary_glass)

print("Reconstruction complete. Reconstructed binary image saved.")
print("Tissue mask saved as canvas_binary_glass.")
