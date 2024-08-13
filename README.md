# subimg_augmentation

This repository contains code for extracting polygonal segmentation data from ALTO XML files to use in subimage augmentation, as presented in "Evaluating Augmented Training Data for Complex Document Layouts: the Case of Arabic Scientific Manuscripts" (DH2024). The code is available both as a Python script (`extract-regions.py`) and a Jupyter notebook.

The method for creating artificial images using these extracted regions is the choice of the user. A sample workflow that combines together select regions using a [SegmOnto ontology](https://segmonto.github.io/) will soon be uploaded to this repository.
