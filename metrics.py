import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import seaborn as sns
from PIL import Image
from scipy import ndimage
from collections import defaultdict
from scipy.stats import gaussian_kde
import random

def read_slices(slice_dir):
    slices = []
    for filename in sorted(os.listdir(slice_dir)):
        if filename.lower().endswith(('.tif', '.tiff')):
            slice_path = os.path.join(slice_dir, filename)
            slice_img = tifffile.imread(slice_path)
            slices.append(slice_img)
        elif filename.lower().endswith(('.png', '.jpeg', '.jpg')):
            slice_path = os.path.join(slice_dir, filename)
            slice_img = np.array(Image.open(slice_path))
            slices.append(slice_img)
    return np.array(slices)

def calculate_grain_sizes(slices):
    grain_sizes = []
    for slice_img in slices:
        # Utiliser toutes les valeurs de pixels pour calculer les tailles des grains
        grain_sizes.extend(slice_img.flatten())
    return np.array(grain_sizes)

def hash_color(pixel):
    return hash(tuple(pixel))

def count_neighbors(image):
    h, w, _ = image.shape
    grains = np.zeros((h, w), dtype=np.int64)
    for i in range(h):
        for j in range(w):
            grains[i, j] = hash_color(image[i, j])
    
    unique_grains = np.unique(grains)
    masks = {grain: (grains == grain) for grain in unique_grains}
    
    neighbors = defaultdict(set)
    for grain in unique_grains:
        dilated = ndimage.binary_dilation(masks[grain])
        for other_grain in unique_grains:
            if grain != other_grain and np.any(dilated & masks[other_grain]):
                neighbors[grain].add(other_grain)
    
    return neighbors

def calculate_neighbor_distribution(slices):
    neighbor_counts = []
    for slice_img in slices:
        neighbors = count_neighbors(slice_img)
        neighbor_counts.extend([len(n) for n in neighbors.values()])
    return neighbor_counts

def plot_distribution(data, color, label):
    sns.kdeplot(data, color=color, label=label, fill=True)

def main(slice_dir1, slice_dir2, output_dir):
    # Créer le dossier de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Lire les slices
    slices1 = read_slices(slice_dir1)
    slices2 = read_slices(slice_dir2)

    # Calculer les tailles des grains
    size_values1 = calculate_grain_sizes(slices1)
    size_values2 = calculate_grain_sizes(slices2)

    # Calculer les moyennes des tailles des grains
    mean_size1 = np.mean(size_values1)
    mean_size2 = np.mean(size_values2)

    # Afficher les distributions des tailles des grains
    plt.figure(figsize=(15, 10))

    # Tracer les distributions
    plot_distribution(size_values1, 'blue', 'Volume 1')
    plot_distribution(size_values2, 'red', 'Volume 2')

    # Tracer les moyennes des tailles des grains
    plt.axvline(mean_size1, color='blue', linestyle='--', label=f'Mean Volume 1: {mean_size1:.2f}')
    plt.axvline(mean_size2, color='red', linestyle='--', label=f'Mean Volume 2: {mean_size2:.2f}')

    plt.title("Comparison of Grain Size Distributions")
    plt.xlabel("Size (in pixels)")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Ajuster la marge supérieure pour faire de la place aux étiquettes de moyenne
    plt.subplots_adjust(top=0.9)

    # Extraire les noms des dossiers des chemins fournis
    volume1_name = os.path.basename(slice_dir1.rstrip('/'))
    volume2_name = os.path.basename(slice_dir2.rstrip('/'))

    # Enregistrer la figure avec les noms des volumes
    output_filename = f"{volume1_name}_&&_{volume2_name}_grain_size_Distribution.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"The comparison of grain size distributions has been successfully saved to '{output_path}'.")

    # Calculer les distributions des voisins
    neighbor_counts1 = calculate_neighbor_distribution(slices1)
    neighbor_counts2 = calculate_neighbor_distribution(slices2)

    # Calculer les moyennes des voisins
    mean_neighbors1 = np.mean(neighbor_counts1)
    mean_neighbors2 = np.mean(neighbor_counts2)

    # Afficher les distributions des voisins
    plt.figure(figsize=(15, 10))

    # Tracer les distributions
    density1 = gaussian_kde(neighbor_counts1)
    x1 = np.linspace(min(neighbor_counts1), max(neighbor_counts1), 1000)
    plt.plot(x1, density1(x1), label='Volume 1')

    density2 = gaussian_kde(neighbor_counts2)
    x2 = np.linspace(min(neighbor_counts2), max(neighbor_counts2), 1000)
    plt.plot(x2, density2(x2), label='Volume 2')

    plt.title("Comparison of Neighbor Distributions")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Tracer les moyennes des voisins
    plt.axvline(mean_neighbors1, color='blue', linestyle='--', label=f'Mean Volume 1: {mean_neighbors1:.2f}')
    plt.axvline(mean_neighbors2, color='red', linestyle='--', label=f'Mean Volume 2: {mean_neighbors2:.2f}')

    # Ajuster la marge supérieure pour faire de la place aux étiquettes de moyenne
    plt.subplots_adjust(top=0.9)

    # Enregistrer la figure avec les noms des volumes
    output_filename = f"{volume1_name}_&&_{volume2_name}_neighbor_distribution.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"The comparison of neighbor distributions has been successfully saved to '{output_path}'.")

# Exemple d'appel de la fonction principale
if __name__ == "__main__":
    slice_dir1 = "data/Inconel_718/slices"  # Remplacez par le chemin de votre premier dossier de slices
    slice_dir2 = "data/VER_Nemrique/x"  # Remplacez par le chemin de votre deuxième dossier de slices
    output_dir = "output_metrics"  # Remplacez par le chemin de votre répertoire de sortie
    main(slice_dir1, slice_dir2, output_dir)