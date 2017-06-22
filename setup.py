from setuptools import setup, find_packages

setup(
    name='qtim_ROP',
    packages=find_packages(),
    version='0.2',
    description = 'A package for segmentation and classification of retinal images',
    entry_points = {
        "console_scripts": ['deeprop = qtim_ROP.__main__:main']
    },
    author='James Brown',
    author_email='jbrown97@mgh.harvard.edu',
    url='https://github.com/QTIM-Lab/qtim_ROP', # use the URL to the github repo
    download_url = 'https://github.com/QTIM-Lab/qtim_ROP/tarball/0.2',
    keywords=['retina', 'retinopathy of prematurity', 'plus disease', 'machine learning', 'deep learning', 'CNN'],
    install_requires=['numpy', 'scipy', 'appdirs', 'seaborn', 'pandas', 'matplotlib', 'keras<=1.2.1', 'h5py', 'theano'],
    package_data={'qtim_ROP': ['config/preprocessing.yaml']},
    classifiers=[],
)