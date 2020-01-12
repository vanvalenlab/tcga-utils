# tcga-utils
Utilities for retrieving data from The Cancer Genome Atlas (TCGA) and performing image analysis on whole slide images. This repo is a work in progress and draws heavily from the amazing [python-wsi-preprocessing repo](https://github.com/deroneriksson/python-wsi-preprocessing). This repo contains a Dockerfile to facilitate the creation of a Docker image that has all of the required dependencies installed. Tasks currently supported by this library includes:
* Downloading files from tcga by uuid.
* Opening whole slide images using openslide.
* Converting whole slide images into tiles of a given size and resolution.
* Identify which tiles have images using color filters.

In the near future we will be creating a dataset object to link TCGA images with annotations (gene expression, mutation signature etc) and ease the construction of [data generator objects](https://keras.io/preprocessing/image/) that can be used to train deep learning models.

