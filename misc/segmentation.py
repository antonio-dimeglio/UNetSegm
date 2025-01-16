from argparse import ArgumentParser, BooleanOptionalAction
import matplotlib.pyplot as plt
import napari
import skimage as ski
import numpy as np


def load_image(filepath:str, slices_skipped: int, show_img: bool):
    img:np.ndarray = ski.io.imread(filepath)
    channel_idx = 3 if img.shape[0] == 5 else 2
    img = img[channel_idx, slices_skipped:, :, :]

    # image normalization
    img = (img - np.min(img))/(np.max(img) - np.min(img))
    
    if show_img:
        viewer = napari.view_image(img)
        napari.run()

    return img

def segment_image(img:np.ndarray):
    pass 


def main():
    ap = ArgumentParser(prog="Collagen Segmentation Script",
                        description="Python script for the creation of segmented .tif collagen images.")
    
    ap.add_argument('--input', type=str, help="Input file path", required=True)
    ap.add_argument('--skip', type=int, help="Number of slices to skip", default=0)
    ap.add_argument('--show_img', action=BooleanOptionalAction, default=False)

    args = ap.parse_args()

    img = load_image(args.input, args.skip, args.show_img)


if __name__ == '__main__':
    main()