import vtk
import numpy as np
from vtk.util import numpy_support
import os
from PIL import Image
import tifffile

def read_volume(file_name):
    if file_name.endswith('.vti'):
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(file_name)
        reader.Update()
        image_data = reader.GetOutput()
        dims = image_data.GetDimensions()
        numpy_data = numpy_support.vtk_to_numpy(image_data.GetPointData().GetScalars())
        return numpy_data.reshape(dims[2], dims[1], dims[0], 3)
    elif file_name.endswith('.tif') or file_name.endswith('.tiff'):
        return tifffile.imread(file_name)
    else:
        raise ValueError("Unsupported file format. Please provide a .vti or .tif file.")

def save_slices(volume, output_dir, num_slices=100, is_isotropic=True):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    if is_isotropic:
        # Create subdirectory for isotropic slices
        isotropic_dir = os.path.join(output_dir, 'origin_isotropic_slices')
        os.makedirs(isotropic_dir, exist_ok=True)
        
        # Calculate slice indices
        slice_indices = np.linspace(0, volume.shape[0] - 1, num_slices, dtype=int)
        
        # Save slices
        for i, idx in enumerate(slice_indices):
            slice_img = Image.fromarray(volume[idx])
            slice_img.save(os.path.join(isotropic_dir, f'slice_{i:03d}.png'))
    else:
        # Create subdirectories for anisotropic slices
        anisotropic_dirs = {
            'x': os.path.join(output_dir, 'origin_anisotropic_slices_x'),
            'y': os.path.join(output_dir, 'origin_anisotropic_slices_y'),
            'z': os.path.join(output_dir, 'origin_anisotropic_slices_z')
        }
        for direction in anisotropic_dirs.values():
            os.makedirs(direction, exist_ok=True)
        
        # Save slices along x-axis
        slice_indices = np.linspace(0, volume.shape[0] - 1, num_slices, dtype=int)
        for i, idx in enumerate(slice_indices):
            slice_img = Image.fromarray(volume[idx])
            slice_img.save(os.path.join(anisotropic_dirs['x'], f'slice_{i:03d}.png'))
        
        # Save slices along y-axis
        slice_indices = np.linspace(0, volume.shape[1] - 1, num_slices, dtype=int)
        for i, idx in enumerate(slice_indices):
            slice_img = Image.fromarray(volume[:, idx, :])
            slice_img.save(os.path.join(anisotropic_dirs['y'], f'slice_{i:03d}.png'))
        
        # Save slices along z-axis
        slice_indices = np.linspace(0, volume.shape[2] - 1, num_slices, dtype=int)
        for i, idx in enumerate(slice_indices):
            slice_img = Image.fromarray(volume[:, :, idx])
            slice_img.save(os.path.join(anisotropic_dirs['z'], f'slice_{i:03d}.png'))

# Exemple d'appel de la fonction principale
if __name__ == "__main__":
    volume_path = "data/Inconel_718/origin_volume/RGB.tif"  # Remplacez par le chemin de votre fichier VTI ou TIF
    num_slices = 100  # Nombre de slices à générer
    is_isotropic = True  # Indique si le volume est isotropique ou anisotropique
    
    # Déterminer le répertoire de sortie basé sur le chemin du volume
    output_dir = os.path.join(os.path.dirname(volume_path), "../")
    
    # Lire le volume
    volume_numpy = read_volume(volume_path)
    
    # Enregistrer les slices
    save_slices(volume_numpy, output_dir, num_slices, is_isotropic)
