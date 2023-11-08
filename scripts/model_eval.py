import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity
from sklearn.metrics import mean_squared_error, roc_auc_score, f1_score, accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Input folder containing pairs of real and fake binary images
inputFolder = "/home/maberyick/Documents/GitHub/pyramidpix2pix/test/train_epoch/"

# Output CSV file to store comparison results
outputCSV = "/home/maberyick/Documents/GitHub/pyramidpix2pix/test/comparison_results.csv"

# Output PNG file to save the plot
outputPNG = "/home/maberyick/Documents/GitHub/pyramidpix2pix/test/comparison_plot.png"

# Function to calculate PSNR
def calculate_psnr(real_img, fake_img):
    mse = np.mean((real_img - fake_img) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Function to calculate image metrics
def calculate_metrics(real_img, fake_img, ground_truth):
    # Convert images to grayscale if not already
    if len(real_img.shape) == 3:
        real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)
    if len(fake_img.shape) == 3:
        fake_img = cv2.cvtColor(fake_img, cv2.COLOR_BGR2GRAY)

    # Normalize images to [0, 1] range
    real_img = real_img / 255.0
    fake_img = fake_img / 255.0

    # Calculate SSIM with data_range parameter
    ssim_value = structural_similarity(real_img, fake_img, data_range=1.0)

    # Calculate MSE
    mse_value = mean_squared_error(real_img, fake_img)

    # Calculate PSNR
    psnr_value = calculate_psnr(real_img, fake_img)

    # Calculate IoU (Intersection over Union)
    intersection = np.logical_and(real_img, fake_img)
    union = np.logical_or(real_img, fake_img)
    iou_value = np.sum(intersection) / np.sum(union)

    # Calculate AUC, F1 score, and accuracy
    fpr, tpr, _ = roc_curve(ground_truth.ravel(), fake_img.ravel())
    auc_value = auc(fpr, tpr)

    fake_bin = (fake_img > 0.5).astype(int)
    f1 = f1_score(ground_truth.ravel(), fake_bin.ravel())
    accuracy = accuracy_score(ground_truth.ravel(), fake_bin.ravel())

# Function to calculate image metrics
def calculate_metrics(real_img, fake_img, ground_truth):
    # Convert images to grayscale if not already
    if len(real_img.shape) == 3:
        real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)
    if len(fake_img.shape) == 3:
        fake_img = cv2.cvtColor(fake_img, cv2.COLOR_BGR2GRAY)

    # Normalize images to [0, 1] range
    real_img = real_img / 255.0
    fake_img = fake_img / 255.0

    # Calculate SSIM with data_range parameter
    ssim_value = structural_similarity(real_img, fake_img, data_range=1.0)

    # Calculate MSE
    mse_value = mean_squared_error(real_img, fake_img)

    # Calculate PSNR
    psnr_value = calculate_psnr(real_img, fake_img)

    # Threshold ground truth to be binary (0 or 1)
    ground_truth_binary = (ground_truth > 0).astype(int)

    # Calculate IoU (Intersection over Union)
    intersection = np.logical_and(real_img > 0.5, fake_img > 0.5)
    union = np.logical_or(real_img > 0.5, fake_img > 0.5)
    iou_value = np.sum(intersection) / np.sum(union)

    # Calculate AUC, F1 score, and accuracy
    fpr, tpr, _ = roc_curve(ground_truth_binary.ravel(), fake_img.ravel())
    auc_value = auc(fpr, tpr)

    fake_bin = (fake_img > 0.5).astype(int)
    f1 = f1_score(ground_truth_binary.ravel(), fake_bin.ravel())
    accuracy = accuracy_score(ground_truth_binary.ravel(), fake_bin.ravel())

    return ssim_value, mse_value, psnr_value, iou_value, auc_value, f1, accuracy

# Open the CSV file for writing
with open(outputCSV, 'w') as csv_file:
    csv_file.write("Real Image, Fake Image, SSIM, MSE, PSNR, IoU, AUC, F1 Score, Accuracy\n")

    # Loop through the images in the input folder
    image_files = os.listdir(inputFolder)
    for real_image_file in image_files:
        if real_image_file.endswith("_real_B.png"):
            fake_image_file = real_image_file.replace("_real_B.png", "_fake_B.png")
            ground_truth_file = real_image_file.replace("_real_B.png", "_real_B.png")

            real_image_path = os.path.join(inputFolder, real_image_file)
            fake_image_path = os.path.join(inputFolder, fake_image_file)
            ground_truth_path = os.path.join(inputFolder, ground_truth_file)

            real_img = cv2.imread(real_image_path)
            fake_img = cv2.imread(fake_image_path)
            ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

            if real_img is not None and fake_img is not None and ground_truth is not None:
                ssim_value, mse_value, psnr_value, iou_value, auc_value, f1, accuracy = calculate_metrics(real_img, fake_img, ground_truth)
                csv_file.write(f"{real_image_file}, {fake_image_file}, {ssim_value:.4f}, {mse_value:.4f}, {psnr_value:.4f}, {iou_value:.4f}, {auc_value:.4f}, {f1:.4f}, {accuracy:.4f}\n")

print(f"Comparison results saved to {outputCSV}")

# Function to plot metrics
def plot_metrics(epoch_labels, metrics_dict, output_filename):
    plt.figure(figsize=(12, 6))
    
    # Create a color map for plotting
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_dict)))

    for metric_name, metric_values in metrics_dict.items():
        plt.plot(epoch_labels, metric_values, label=metric_name, marker='o', color=colors[len(plt.gca().lines) - 1])

    plt.xlabel('Epochs')
    plt.ylabel('Metric Values')
    plt.title('Comparison of Metrics')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Comparison plot saved to {output_filename}")

# Lists to store metric values for plotting
epoch_labels = []
metrics_dict = {
    'SSIM': [],
    'MSE': [],
    'IoU': [],
    'AUC': [],
    'F1 Score': [],
    'Accuracy': []
}

# Open the CSV file for reading
with open(outputCSV, 'r') as csv_file:
    # Skip the header line
    next(csv_file)

    # Loop through the lines in the CSV file
    for line in csv_file:
        parts = line.strip().split(',')
        epoch_labels.append(parts[0])
        metrics_dict['SSIM'].append(float(parts[2]))
        metrics_dict['MSE'].append(float(parts[3]))
        metrics_dict['IoU'].append(float(parts[5]))
        metrics_dict['AUC'].append(float(parts[6]))
        metrics_dict['F1 Score'].append(float(parts[7]))
        metrics_dict['Accuracy'].append(float(parts[8]))

# Create and save the plot using the plot_metrics function
plot_metrics(epoch_labels, metrics_dict, outputPNG)

print(f"Comparison results saved to {outputPNG}")
