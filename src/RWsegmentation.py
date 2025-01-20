import argparse
import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
from skimage.segmentation import watershed, random_walker
from skimage.filters import sobel 
import tifffile
import tqdm

def load_img(img_path: str, normalize: bool) -> np.ndarray:
    """
    Load a microscopy image, optionally normalize its intensity.

    Parameters:
    - img_path: str : Path to the image file.
    - normalize: bool : Whether to normalize pixel intensity to [0, 1].

    Returns:
    - np.ndarray : The loaded and processed image.
    """
    img: np.ndarray = ski.io.imread(img_path)

    if img.ndim == 5:
        channel_idx = 3
    elif img.ndim == 4:
        channel_idx = 2
    else:
        raise ValueError("Unsupported image dimensions: Expected 4D or 5D image.")

    img = img[channel_idx, :, :, :]

    if normalize:
        img = (img - np.min(img)) / (np.max(img) - np.min(img))

    return img

def segment_random_walk(image: np.ndarray, output_filename:str):
    labels = np.zeros_like(image)
    for i in tqdm.tqdm(range(len(image))):
        img = image[i, :, :]
        threshold = ski.filters.threshold_otsu(img)

        markers = np.zeros_like(img)
        markers[img < threshold] = 1 # Background
        markers[img > threshold] = 2 # Collagen

        labels[i, :, :] = random_walker(img, markers)
    
    tifffile.imwrite(output_filename, labels)

def main():
    parser = argparse.ArgumentParser(prog="Tool for automated segmentation using the Random Walker Algorithm.")

    parser.add_argument("--file_path", type=str, required=True, 
                        help="Path to the image to load.")
    parser.add_argument("--normalize", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="Flag value to choose if the image should be normalized or not.")
    parser.add_argument("--output_filename", type=str, default="output_label.tiff",
                        help="String value to set the output file name, by default the value is set to output_label.tiff")
    args = parser.parse_args()
    
    img = load_img(args.file_path, args.normalize)
    segment_random_walk(img, args.output_filename)

if __name__ == '__main__':
    main()