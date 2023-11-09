import cv2

# Load the original image
image_path = "/home/maberyick/pCloudDrive/CCIPD_echo/Projects/Immune_SCLC/dl_training/test/pix2pix/mix/test/PID 106.png"
original_image = cv2.imread(image_path)

# Check if the image was loaded successfully
if original_image is not None:
    # Get the original image dimensions
    original_height, original_width = original_image.shape[:2]

    # Resize the image by a factor of 2
    resized_image = cv2.resize(original_image, (original_width // 4, original_height // 4))

    # Save the resized image
    output_path = "/home/maberyick/pCloudDrive/CCIPD_echo/Projects/Immune_SCLC/dl_training/test/pix2pix/mix/test/resized_image.png"  # Replace with your desired output path
    cv2.imwrite(output_path, resized_image)

    print("Image resized and saved.")
else:
    print("Failed to load the original image.")
