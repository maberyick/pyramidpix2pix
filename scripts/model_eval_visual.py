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
