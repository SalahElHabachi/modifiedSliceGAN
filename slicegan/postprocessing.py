import numpy as np
from scipy.ndimage import gaussian_filter
import vtk
from vtk.util import numpy_support

def sharpen_volume(volume, alpha=1.5, sigma=1):
    """
    Applique un filtre de sharpening à un volume 3D.
    
    Parameters:
    - volume: numpy array, le volume 3D à traiter.
    - alpha: float, facteur de sharpening.
    - sigma: float, écart-type pour le filtre gaussien.
    
    Returns:
    - volume_sharpened: numpy array, le volume 3D après sharpening.
    """
    blurred = gaussian_filter(volume, sigma=sigma)
    volume_sharpened = volume + alpha * (volume - blurred)
    return volume_sharpened

def save_volume_as_vti(volume, output_path):
    """
    Enregistre un volume 3D en tant que fichier .vti.
    
    Parameters:
    - volume: numpy array, le volume 3D à enregistrer.
    - output_path: str, chemin du fichier de sortie .vti.
    """
    # Vérifier si le volume a 4 dimensions et convertir en niveaux de gris si nécessaire
    if volume.ndim == 4 and volume.shape[-1] == 3:
        # Convertir le volume RGB en niveaux de gris
        volume = np.dot(volume[..., :3], [0.2989, 0.5870, 0.1140])
    
    # Convertir le volume numpy en vtkImageData
    dims = volume.shape
    vtk_data = numpy_support.numpy_to_vtk(volume.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(dims)
    image_data.GetPointData().SetScalars(vtk_data)
    
    # Enregistrer le volume en tant que fichier .vti
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(image_data)
    writer.Write()

def read_vti(file_name):
    """
    Lit un fichier .vti et le convertit en numpy array.
    
    Parameters:
    - file_name: str, chemin du fichier .vti.
    
    Returns:
    - numpy array, le volume 3D.
    """
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(file_name)
    reader.Update()
    image_data = reader.GetOutput()
    dims = image_data.GetDimensions()
    numpy_data = numpy_support.vtk_to_numpy(image_data.GetPointData().GetScalars())
    return numpy_data.reshape(dims[2], dims[1], dims[0])