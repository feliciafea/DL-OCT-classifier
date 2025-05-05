# Automated Detection of Retinal Diseases in OCT Scans: Improving Diagnosis of AMD, DME, and CNV

This project implements and evaluates different architectures for OCT scan image classification. The three main `.ipynb` files contain the three models that we tested: Pretrained ResNet-50, U-Net Classifier, and Attention-Based CNN.

## Dataset

Our primary dataset is a repository of OCT images published by Mendeley Data. This dataset has many retinal scans, labeled for CNV, DME, Drusen, and normal.
Download the dataset using the link below and use the OCT folder containing the OCT scans split into test and training sets. 

https://data.mendeley.com/datasets/rscbjbr9sj/3

## Setup Instructions

The following steps will show you how to setup the three main notebooks in Google Colab.

1. Download the full dataset using the link above and extract the OCT folder.
2. Zip the OCT folder into a file called OCT.zip and upload that file to Google Drive.
3. Open the notebook in Google Colab and update the second cell to contain the path of your OCT.zip within Google Drive.
4. From here you should be able to successfully run the notebook. We recommend using a GPU for reasonable training times.

## Report 

The CVPR-formatted paper can be found in this repository as a pdf.

