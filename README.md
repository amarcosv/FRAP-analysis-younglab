# FRAP-analysis-younglab
Pipeline for FRAP analysis for experiments from the Young Lab

## Description

This repository provides an automated workflow for Fluorescence Recovery After Photobleaching (FRAP) analysis. The pipeline processes raw data, calculates key metrics, performs statistical analysis, and generates plots.

It is designed to work with two types of input:
1.  `.czi` files containing labeled bleach and control ROIs.
2.  Zeiss Zen-generated `.csv` files with the mean intensity of bleach and control regions over time.

The workflow can analyze multiple groups at once by processing all subfolders within a parent directory, treating each subfolder as a separate experimental group.

The entire pipeline is run from a Google Colab notebook, making it accessible and easy to use without local setup.

## Features

* **Automated Batch Processing:** Analyzes multiple experimental groups from subfolders containing `.czi` or `.csv` files.
* **Quantitative Analysis:** Automatically calculates and plots recovery curves, half-max recovery time, and mobile fraction.
* **Statistical Comparison:** Runs statistical analysis to compare results between different groups.
* **Google Colab Integration:** The entire workflow is contained in a ready-to-use Google Colab notebook.

## How to Use

This pipeline is run from a Google Colab notebook that can be found at the link below. The Colab comes pre-configured to run a test dataset contained in this repository.

[**FRAP Analysis Colab Notebook**](https://colab.research.google.com/drive/1Sy99HdNc4dcauxnLTJVIe3hPr080o7O-?usp=sharing)

To run the analysis on your own data, follow these steps:
1.  Open the Google Colab notebook using the link above.
2.  Save a copy of the notebook to your own Google Drive by clicking `File` -> `Save a copy in Drive`.
3.  Organize your data on your Google Drive. Create a parent folder, and inside it, create a separate subfolder for each experimental group you want to compare. Place your `.czi` or `.csv` files inside the corresponding group subfolder.
4.  In your copied Colab notebook, find the cell that specifies the input folder path and point it to the parent folder containing your data on Google Drive.
5.  Run the notebook cells in order to perform the analysis.


## Authors

Asier Marcos-Vidal
W.M. Keck Microscopy Facility, Whitehead Institute

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
