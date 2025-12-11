# CourseWork 2: COMP0173: Artificial Intelligence for Sustainable Development - Daniel Huencho
This repository contains a replication of the methodology presented in [Article Title/Citation](https://www.sciencedirect.com/science/article/pii/S0303243422000113), developed as part of a UCL coursework project. The original codebase was cloned and adapted to evaluate the model's performance when applied to Chilean forestry data.

## Main files: 
- 0000_Replication_Experimentation_pytorch.ipynb PyTorch Implementation and Validation: A port of the original TensorFlow/Keras architecture to PyTorch. This module verifies the reproducibility of the study by replicating the baseline methodology and achieving comparable performance metrics.

- 0001_loading_clp_data.ipynb Exploratory Data Analysis (EDA) – Chile Region: Procedures for loading, inspecting, and characterizing the dataset specific to the Chilean study area, establishing the baseline distribution for subsequent modeling.

- 0002_gee_sentinel.ipynb Sentinel-2 and Google Earth Engine Integration: Scripts for interfacing with the Google Earth Engine (GEE) API to acquire, preprocess, and visualize Sentinel-2 satellite imagery for specific regions of interest.

- 0003_creating_mask.ipynb Temporal Mask Generation: Algorithms for synthesizing ground-truth masks representing forest loss across specified temporal intervals and spatial zones.

- 0004_get_data.ipynb Automated Label Acquisition Pipeline: An autonomous workflow designed to detect regions of interest and generate corresponding labeled datasets for training and validation.

- 0005_Experimentation_pytorch_testing_application_in_Chile.ipynb Attention U-Net Evaluation and Optimization: Application of the Attention U-Net architecture to the Chilean dataset. This module utilizes the Optuna framework for hyperparameter optimization to assess model generalization and performance. 



# MSc Data Science Dissertation/Project

**An Attention-Based U-Net for Detecting Deforestation Within Satellite Sensor Imagery.** 
https://www.sciencedirect.com/science/article/pii/S0303243422000113

## Datasets
### Amazon 1 (Regular 3-dim Dataset) -- https://zenodo.org/record/3233081
### Amazon 2 (Larger 4-band Amazon and Atlantic Datasets) -- https://zenodo.org/record/4498086#.YMh3GfKSmCU

## Files
+ **dataset** -- Folder of original dataset from Regular Dataset.
+ **figures** -- Figures for report (amazon-atlantic-forest-mapjpg.jpg from https://pubmed.ncbi.nlm.nih.gov/20433744/).
  + **shapefiles** -- Shapefiles for map. Amazon Shapefile from: (http://worldmap.harvard.edu/data/geonode:amapoly_ivb), rest from: (http://terrabrasilis.dpi.inpe.br/en/download-2/).
+ **models** -- Folder of each of the three types of Attention U-Net model; load into Keras using 'load_model([modelfilename])'.
+ **metrics** -- Folder of metrics (accuracy, precision, recall, F1-score) for each result.
+ Experimentation.ipynb -- Jupyter notebook of data processing, augmentation, model training and testing.
+ Figures.ipynb -- Jupyter notebook of figures found in **figures**.
+ predictor.py -- Takes any input RGB or 4-band image and outputs Attention U-Net-predicted deforestation mask to file.
+ preprocess-4band-amazon-data.py -- Python script to preprocess GeoTIFFs from 4-band Amazon Dataset and export as numpy pickles.
+ preprocess-4band-atlantic-forest-data.py -- Python script to preprocess GeoTIFFs from 4-band Atlantic Forest Dataset and export as numpy pickles.
+ preprocess-rgb-data.py -- Python script to preprocess data in RGB Dataset and export as numpy pickles.
+ requirements.txt -- Required Python libraries.

## How to use
### Obtaining Attention U-Net Deforestation Masks
+ Run pip -r requirements.txt to install libraries.
+ Download 'unet-attention-3d.hdf5', 'unet-attention-4d.hdf5' and 'unet-attention-4d-atlantic.hdf5' models, and place in same directory as script.
+ Run 'python predictor.py [MODEL IDENTIFIER] [INPUT IMAGE PATH]' or 'python3 predictor.py [MODEL IDENTIFIER] [INPUT IMAGE PATH]'.
  + Model identifier for RGB is 1, 4-band Amazon-trained is 2, 4-band Atlantic Forest-trained is 3.
  + e.g. Get mask prediction of image named 'test.tif' from 4-band Amazon model: 'python predictor.py 2 test.tif'.

### Obtaining Pre-Processed Data
+ Run pip -r requirements.txt to install libraries.
+ Run 'preprocess-4band-amazon-data.py' to pre-process 4-band Amazon data.
+ Run 'preprocess-4band-atlantic-forest-data.py' to pre-process 4-band Atlantic Forest data.
+ Run 'preprocess-rgb-data.py' to pre-process RGB Amazon data.
