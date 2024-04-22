import os
import pandas as pd
import matplotlib.pyplot as plt

# Function to calculate statistics from CSV files
def calculate_statistics(csv_folder):
    # Initialize dictionaries to store metric values
    metrics_dict = {}
    
    # Loop through CSV files in the folder
    for csv_file in os.listdir(csv_folder):
        if csv_file.endswith(".csv"):
            # Read the CSV file into a DataFrame
            df = pd.read_csv(os.path.join(csv_folder, csv_file))
            
            # Get the last row (assuming it contains the metrics)
            metrics_row = df.iloc[-1]
            
            # Extract metric values
            epoch = metrics_row['Epoch']
            ssim = metrics_row['SSIM']
            mse = metrics_row['MSE']
            psnr = metrics_row['PSNR']
            iou = metrics_row['IoU']
            auc = metrics_row['AUC']
            f1 = metrics_row['F1 Score']
            accuracy = metrics_row['Accuracy']
            
            # Store the metric values in the dictionary
            metrics_dict[epoch] = {
                'SSIM': ssim,
                'MSE': mse,
                'PSNR': psnr,
                'IoU': iou,
                'AUC': auc,
                'F1 Score': f1,
                'Accuracy': accuracy
            }
    
    # Create a DataFrame from the metrics dictionary
    metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index')
    
    # Calculate statistics
    metrics_mean = metrics_df.mean()
    metrics_median = metrics_df.median()
    metrics_std = metrics_df.std()
    
    return metrics_mean, metrics_median, metrics_std

# Function to plot metrics
def plot_metrics(metrics_dict, output_filename):
    metrics_mean = metrics_dict['Mean']
    metrics_median = metrics_dict['Median']
    metrics_std = metrics_dict['Std']
    
    metrics_mean.plot(kind='bar', yerr=metrics_std, alpha=0.7, capsize=5, label='Mean ± Std')
    metrics_median.plot(kind='bar', alpha=0.7, color='orange', label='Median')
    
    plt.xlabel('Metrics')
    plt.ylabel('Metric Values')
    plt.title('Comparison of Metrics')
    plt.legend()
    plt.grid(axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Comparison plot saved to {output_filename}")

# Folder containing CSV files with metrics data
csv_folder = "/home/maberyick/Documents/GitHub/pyramidpix2pix/test/comparison_results.csv"

# Output PNG file to save the plot
outputPNG = "/home/maberyick/Documents/GitHub/pyramidpix2pix/test/metrics_patch_comparison_plot.png"

# Calculate statistics from CSV files
metrics_mean, metrics_median, metrics_std = calculate_statistics(csv_folder)

# Create a dictionary to store metrics statistics
metrics_statistics = {
    'Mean': metrics_mean,
    'Median': metrics_median,
    'Std': metrics_std
}

# Plot and save metrics comparison
plot_metrics(metrics_statistics, outputPNG)
