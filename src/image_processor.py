import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from .custom_kmeans import KMeansClustering

def rgb_to_lab(rgb):
    rgb = rgb / 255.0
    
    # Convert to XYZ
    mask = rgb > 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[~mask] = rgb[~mask] / 12.92
    
    # Apply transformation matrix
    xyz = np.zeros_like(rgb)
    xyz[:, :, 0] = rgb[:, :, 0] * 0.4124 + rgb[:, :, 1] * 0.3576 + rgb[:, :, 2] * 0.1805
    xyz[:, :, 1] = rgb[:, :, 0] * 0.2126 + rgb[:, :, 1] * 0.7152 + rgb[:, :, 2] * 0.0722
    xyz[:, :, 2] = rgb[:, :, 0] * 0.0193 + rgb[:, :, 1] * 0.1192 + rgb[:, :, 2] * 0.9505
    
    # Normalize XYZ
    xyz[:, :, 0] /= 0.95047
    xyz[:, :, 1] /= 1.0
    xyz[:, :, 2] /= 1.08883
    
    # Apply nonlinear transformation
    mask = xyz > 0.008856
    xyz[mask] = xyz[mask] ** (1/3)
    xyz[~mask] = 7.787 * xyz[~mask] + 16 / 116
    
    # Calculate LAB
    lab = np.zeros_like(xyz)
    lab[:, :, 0] = 116 * xyz[:, :, 1] - 16  # L
    lab[:, :, 1] = 500 * (xyz[:, :, 0] - xyz[:, :, 1])  # a
    lab[:, :, 2] = 200 * (xyz[:, :, 1] - xyz[:, :, 2])  # b
    
    return lab

def process_image(image_path, save_path=None):
    image = np.array(Image.open(image_path))
    
    # Convert to LAB color space (simplified version)
    lab = rgb_to_lab(image)
    
    # Extract and normalize the 'a' channel
    a_channel = lab[:, :, 1].astype(np.float64)
    a_channel = (a_channel - a_channel.mean()) / a_channel.std()
    
    # Reshape for clustering
    X = a_channel.reshape(-1, 1)
    
    kmeans = KMeansClustering(n_clusters=2, init='k-means++', n_init=10, random_state=42)

    kmeans.fit(X)
    labels = kmeans.predict(X).reshape(lab.shape[:2])
    
    # Create result image with black background
    result = np.zeros_like(image)
    
    # Find green cluster (lowest 'a' value in LAB space)
    green_cluster = np.argmax(kmeans.cluster_centers_)
    
    # Set detected numbers to bright green
    result[labels == green_cluster] = [0, 255, 0]  # RGB format
    
    # Display results
    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(image), plt.title('Original Image')
    plt.subplot(122), plt.imshow(result), plt.title('Numbers in Green')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    return result, kmeans 