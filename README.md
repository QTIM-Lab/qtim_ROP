# qtim_ROP
Code base for preprocessing, segmentation and classification retinal images, and the diagnosis of "plus disease" in retinopathy of prematurity (ROP).
Created by the Quantitative Tumor Imaging Lab at Martinos.

# Installation

The software has been tested on Windows, MacOS and Linux.
We recommend using the Anaconda distribution of Python 2.7: https://www.continuum.io/downloads. 
The steps for installing are as follows:
```bash
git clone https://github.com/QTIM-Lab/qtim_ROP.git
cd qtim_ROP
git checkout packaging
git submodule update --init --recursive
pip install .
```

If you wish to use a GPU, the process for configuring Theano can
be quite involved depending on the OS. The software will use the 
CPU if no GPU is available.

# Usage

The command line utility `deeprop` can be used to classify an
image as follows:

```bash
deeprop classify_plus -i <path-to-image> -o <output-directory>
```

On the first run, the program will create a `config.yaml` file in the
user's home directory (e.g. /home/<user>/.config/DeepROP/0.2/config.yaml).
The config will need to be updated with the full 
path to (a) a trained U-Net for segmentation and (b) a trained
classifier for plus diagnosis:

```yaml
classifier_directory: path/to/classifier
unet_directory: path/to/unet
```
For vessel segmentation, use the following command:

```bash
deeprop segment -i <directory-of-images> -o <output-directory> -u <path-to-unet>
```

# Future additions
* Batch classification, with CSV output of CNN probabilities
* Have vessel segmentation use U-Net in config, if not supplied
* Allow user to update config from command line, rather than edit manually