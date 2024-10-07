### Welcome to SliceGAN ###

'''
Use this file to define your settings for a training run, or
to generate a synthetic image using a trained generator.
'''

import os
from slicegan import model, networks, util,postprocessing
import argparse
# Define project name
Project_name = 'dcdc'

# Specify project folder.
Project_dir = 'Trained_Generators'

# Run with False to show an image during or after training
parser = argparse.ArgumentParser()
parser.add_argument('training', type=int)
args = parser.parse_args()
Training = args.training
# Training = 0

Project_path = util.mkdr(Project_name, Project_dir, Training)

## Data Processing
# Define image  type (colour, grayscale, three-phase or two-phase.
# n-phase materials must be segmented)
image_type = 'colour'

# img_channels should be number of phases for nphase, 3 for colour, or 1 for grayscale
img_channels = 3
# define data type (for colour/grayscale images, must be 'colour' / '
# greyscale. nphase can be, 'tif2D', 'png', 'jpg', tif3D, 'array')
data_type = 'colour'
# Path to your data. One string for isotrpic, 3 for anisotropic
data_path = ['data/Inconel_718/isotropic_slices/slice_001.png']


## Network Architectures
# Training image size, no. channels and scale factor vs raw data
img_size, scale_factor = 128,  1
# z vector depth
z_channels = 64
# Layers in G and D
lays = 7
laysd = 7
dk, gk = [4]*laysd, [4]*lays                                    # kernal sizes
# gk[0]=8
ds, gs = [2]*laysd, [2]*lays                                    # strides
# gs[0] = 4
df, gf = [img_channels, 32, 64, 128, 256, 512, 1], [z_channels, 512, 256, 128, 64, 32, img_channels]  # filter sizes for hidden layers

dp, gp = [1, 1, 1, 1, 1, 0], [2, 2, 2, 2, 2, 3]

## Create Networks
netD, netG = networks.slicegan_nets(Project_path, Training, image_type, dk, ds, df, dp, gk, gs, gf, gp)

# Train
if Training:
    model.train(Project_path, image_type, data_type, data_path, netD, netG, img_channels, img_size, z_channels, scale_factor)
else:
    img, raw, netG = util.test_img(Project_path, image_type, netG(), z_channels, lf=10, periodic=[0, 0, 0])
    
    # # Appliquer le sharpening au volume généré
    # volume_path = os.path.join(Project_path, "generated_volume.vti")  # Chemin du volume généré
    # volume = read_vti(volume_path)
    # volume_sharpened = sharpen_volume(volume)
    
    # # Enregistrer le volume traité en tant que fichier .vti
    # output_path = os.path.join(Project_path, "sharpened_volume.vti")
    # save_volume_as_vti(volume_sharpened, output_path)
    
    # print(f"Le volume traité a été enregistré avec succès dans '{output_path}'.")
