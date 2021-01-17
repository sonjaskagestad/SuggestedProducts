# SuggestedProducts

SuggestedProducts is a program that searches for similar products in a image database given a new query image

## Description

The purpose of this application is to mirror the suggested products feature of an online shop. A CNN classifier
trained on the data in "dataset" is used to classify a test suite of images found in "test_images". Once the
test images are labeled, the application suggests similar products based on that label.

The build_model.pynb contains the code for building and training the CNN and does not need to be run.

## Usage

Please download the Repository and unzip the zip files. Run the program using this command
 ```bash
python SuggestedProducts.py
```
When the images are displayed, there should be two windows: "query" and "suggested products"
press any key to move to the next set of images (there are 17 total)

## Recourses

https://www.kaggle.com/trolukovich/apparel-images-dataset

https://www.pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/

https://towardsdatascience.com/all-the-steps-to-build-your-first-image-classifier-with-code-cf244b015799
