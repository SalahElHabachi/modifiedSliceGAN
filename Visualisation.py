from mayavi import mlab
import tifffile


def read_tiff_volume(file_path):
    """
    Lit un volume 3D à partir d'un fichier TIFF.
    :param file_path: Chemin vers le fichier TIFF.
    :return: Le volume 3D sous forme de tableau numpy.
    """
    # Lecture du volume 3D
    volume = tifffile.imread(file_path)
    
    # Affichage de la forme du volume
    print(f"Forme du volume : {volume.shape}")
    
    return volume

def visualize_volume(volume):
    """
    Visualise un volume 3D en utilisant Mayavi.
    :param volume: Le volume 3D sous forme de tableau numpy.
    """
    # Affichage du volume
    mlab.figure(bgcolor=(1, 1, 1))  # Couleur de fond noire
    mlab.pipeline.volume(mlab.pipeline.scalar_field(volume))
    mlab.show()

# Utilisation de l'exemple
file_path = "Trained_Generators/RGB/RGB.tif"  # Remplacez par le chemin de votre fichier TIFF

volume = read_tiff_volume(file_path)

# Visualisation du volume
visualize_volume(volume)
