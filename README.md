[![DOI](https://zenodo.org/badge/454179890.svg)](https://doi.org/10.5281/zenodo.15547824)

# ml-workflow

## About
This repository contains code and Jupyter notebooks used to train CNN models using labeled IFCB images. To date, all work for this repository has been completed on a Microsoft Azure virtual machine (cloud computing resources). IFCB images accessed by the notebooks for model training are also uploaded to the virtual machine to increase the speed of model training. 

The latest model checkpoint files are located in the model_ckpt folder. Currently in use by the UW team is: ```model-cnn-v1-b3.*```. The model-summary file contains information on the data used to train the model. The model predicts the probability scores of each image belonging to one of ten classes:

0 - Chlorophytes

1 - Ciliates

2 - Cryptophytes

3 - Diatoms

4 - Silicoflagellates

5 - Dinoflagellates

6 - Euglenoids

7 - Unidentified_living

8 - Prymnesiophytes

9 - Inoperable

The Unidentified_living category represents images that are cells but cannot be identified taxonomically due to morphological ambiguity. The Inoperable category is a catch-all category for anthing non-living (e.g., detritus, microplastic particles) and also anything deemed a non-usable image (e.g., due to bad focus).

Deployment of the CNN model can be done using the marimo notebook ```create_dataset_csv.py``` found in the [data-pipeline](https://github.com/ifcb-utopia/data-pipeline) repository of ifcbUTOPIA.

Please cite the use of this model as:

Ali Chase, & Valentina Staneva. (2025). ifcb-utopia/ml-workflow: ifcbUTOPIA CNN for Classification (v0.1-alpha). Zenodo. https://doi.org/10.5281/zenodo.15547825
