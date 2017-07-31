# qtim_ROP
Code base for preprocessing, segmentation and classification retinal images, and the diagnosis of "plus disease" in retinopathy of prematurity (ROP).
Created by the Quantitative Tumor Imaging Lab at Martinos.

# Installation

The software has been tested on Windows, MacOS and Linux.
We recommend using the Anaconda distribution of Python 2.7: https://www.continuum.io/downloads. 
Once installed, the steps for installing qtim_ROP are as follows:
```bash
git clone https://github.com/QTIM-Lab/qtim_ROP.git
cd qtim_ROP
git submodule update --init --recursive
pip install .
```

If you wish to use a GPU, the process for configuring Theano can
be quite involved depending on the OS. The software will use the 
CPU if no GPU is available.

# Usage
The command line utility `deeprop` can be used to perform various tasks on
retinal images, including vessel segmentation and classification of plus disease.

## Configuration

To set which model(s) to use for segmentation and/or classification:

```bash
deeprop configure -s <path-to-unet> -c <path-to-classifier>
```
This will create and update *config.yaml* in the user's home directory:

```yaml
classifier_directory: <path-to-unet>
unet_directory: <path-to-classifier>
```

## Classification

To classify a retinal image for plus disease:

```bash
deeprop classify_plus -i <image-or-folder> -o <output-folder>
```

If the output folder does not exist it will be created automatically. Subfolders
will be created for the segmented and preprocessed image data. The classification
results will be printed to the terminal and output to a timestamped CSV file.

## Segmentation

To segment the vessels in a set of retinal images:

```bash
deeprop segment -i <directory-of-images> -o <output-directory> -u <path-to-unet>
```

# Acknowledgements
orobix: https://github.com/orobix/retina-unet

# Authors
[@jmbrown89](https://github.com/jmbrown89)