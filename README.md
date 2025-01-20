# UNetSegm: A segmentation tool 

UNetSegm is a pytorch-based implementation of the famous UNet architecture for the purpose of performing segmentation of data.
At its core, this library improves an already existing initial segmentation obtained via Otsu's Method for thresholding, where a binary image is created from a grayscale image, i.e., each pixel is assigned either to class 0 or 1.
It does so by stacking together the original image with its binary classification encoded as an additional channel, this tensor is then used as input for the UNet forward pass.

## Traning Dataset(s)

The training dataset consists of two separate datasets, namely:
- [Blood Cell Segmentation Dataset](https://www.kaggle.com/datasets/jeetblahiri/bccd-dataset-with-mask), a dataset containing blood cells manually annotated for segmentation.  