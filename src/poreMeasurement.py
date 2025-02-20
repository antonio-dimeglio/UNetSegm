import argparse as ap 
import numpy as np 
from skimage.measure import label, regionprops
from skimage.exposure import rescale_intensity
from skimage.filters import gaussian
from skimage.transform import rescale
from skimage.segmentation import chan_vese
from skimage.morphology import binary_closing, disk
import tifffile
from tqdm.contrib.concurrent import process_map
from itertools import repeat
import csv
from typing import Union
from vedo import Volume, show

def process_slice(
        img: np.ndarray,
        crop_value: int,
        normalize: bool,
        rescale_intensities: bool,
        sigma: float,
        down_factor: float,
        mu: float,
        lambda1: float,
        lambda2: float,
        tol: float,
        iterations: int,
        disk_size: int
        ) -> np.ndarray:
    """
    Processes a single 2D slice of the z-stack image using preprocessing and Chan-Vese segmentation.
    
    Parameters:
        img (np.ndarray): The input image slice as a NumPy array.
        crop_value (int): Number of pixels to crop from each side.
        normalize (bool): Whether to normalize image values (assumed 8-bit scale: 0-255).
        rescale_intensities (bool): Whether to rescale image intensity.
        sigma (float): Standard deviation for Gaussian smoothing.
        down_factor (float): Factor by which the image is downscaled for efficiency.
        mu (float): Regularization parameter for Chan-Vese segmentation.
        lambda1 (float): Weight for the foreground region in Chan-Vese.
        lambda2 (float): Weight for the background region in Chan-Vese.
        tol (float): Tolerance threshold for segmentation convergence.
        iterations (int): Maximum iterations before stopping if convergence is not reached.
    
    Returns:
        np.ndarray: The segmented image.
    """
    if crop_value != 0:
        img = img[crop_value:-crop_value, crop_value:-crop_value]

    # Normalize and enhance contrast
    if normalize:
        img = img / 255.0
    if rescale_intensities:
        img = rescale_intensity(img)
    

    img = gaussian(img, sigma=sigma, preserve_range=True)

    # Apply Chan-Vese segmentation on a downsampled image for speed
    small_img = rescale(img, down_factor, anti_aliasing=True)
    cv = chan_vese(small_img, mu=mu, lambda1=lambda1, lambda2=lambda2, tol=tol, 
                max_num_iter=iterations, dt=0.5, init_level_set="checkerboard", extended_output=True)

    segmented_image = np.invert(cv[0]) + 0  # Convert to binary
    # Apply binary closing to smooth segmentation
    segmented_image = binary_closing(segmented_image, disk(disk_size))
    
    return segmented_image.astype(np.uint8)  # Ensure binary format

def process_stack(
        stack_path: str,
        num_workers: int,
        crop_value: int,
        normalize: bool,
        rescale_intensities: bool,
        sigma: float,
        down_factor: float,
        mu: float,
        lambda1: float,
        lambda2: float,
        tol: float,
        iterations: int,
        disk_size: int,
        generate_3d: bool
    ) -> list[dict] | Union[list[dict], np.ndarray]: 
    """
    Processes a z-stack, reconstructs 3D segmentation, and measures pore volumes.

    Parameters:
        stack_path (str): Path to the z-stack image file.
        num_workers (int): Number of workers for parallel processing.
        crop_value (int): Number of pixels to crop from each side.
        normalize (bool): Whether to normalize image values.
        rescale_intensities (bool): Whether to rescale image intensity.
        sigma (float): Standard deviation for Gaussian smoothing.
        down_factor (float): Downscaling factor for efficiency.
        mu (float): Regularization parameter for Chan-Vese segmentation.
        lambda1 (float): Weight for the foreground region in Chan-Vese.
        lambda2 (float): Weight for the background region in Chan-Vese.
        tol (float): Tolerance threshold for segmentation convergence.
        iterations (int): Maximum iterations before stopping if convergence is not reached.
        disk_size (int): Size of the morphological disk used for binary closing.
        generate_3d (bool): If true, function will return 3d array of segmentation.

    Returns:
        list[dict]: List of dictionaries containing pore measurements.
        or 
        list[dict]: List of dictionaries containing pore measurements.
        np.ndarray: 3D segmentation result.
    """
    results: list[dict] = []
    print("Reading stack...")
    z_stack = tifffile.imread(stack_path) 
    print("Segmenting stack...")
    segmented_slices = list(process_map(
            process_slice,
            z_stack,
            repeat(crop_value),
            repeat(normalize),
            repeat(rescale_intensities),
            repeat(sigma),
            repeat(down_factor),
            repeat(mu),
            repeat(lambda1),
            repeat(lambda2),
            repeat(tol),
            repeat(iterations),
            repeat(disk_size), max_workers=num_workers))
    
    print("Measuring pores...")
    segmented_volume = np.stack(segmented_slices, axis=0)
    labeled_volume = label(segmented_volume)
    regions = regionprops(labeled_volume)

    for reg in regions:
        results.append({
            "Volume": reg.area,
            "Centroid_x": reg.centroid[0],
            "Centroid_y": reg.centroid[1],
            "Centroid_z": reg.centroid[2],
            "Euler Number": reg.euler_number,
        })

    if generate_3d:
        return results, segmented_volume * 255
    else:
        return results
    
def export_segmentation(segmented_volume: np.ndarray, output_path: str) -> None:
    """
    Save a 3D numpy array (binary segmentation) as a TIFF stack for ImageJ.

    Parameters:
        segmented_volume (np.ndarray): 3D NumPy array with shape (Z, Y, X) containing values 0 or 255.
        output_path (str): Path to save the TIFF file.
    """
    # Ensure data is uint8 (ImageJ expects 8-bit images)
    segmented_volume = segmented_volume.astype(np.uint8)

    with tifffile.TiffWriter(output_path, ome=True) as tif:
        tif.write(segmented_volume)

    print(f"Segmentation saved to {output_path}")

def export_csv(results: list[dict], filepath:str) -> None:
    with open(filepath, 'w', encoding='utf8', newline='') as output_file:
        fc = csv.DictWriter(output_file, 
                            fieldnames=results[0].keys(),

                        )
        fc.writeheader()
        fc.writerows(results)

def visualize_3d(volume: np.ndarray) -> None:
    volume[volume > 0] = 1

    vol = Volume(volume)
    show(vol, axes=1)


def main():
    parser = ap.ArgumentParser(
        prog="poreMeasurement.py",
        description="Measures pores in a z-stack .tif/.ome.tif image using Chan-Vese segmentation."
    )

    # Input and output files
    parser.add_argument('--input', type=str, required=True, 
                        help="Path to the input .tif or .ome.tif image file (required).")
    parser.add_argument('--output_stats', type=str, default="stats.csv", 
                        help="Path to the output CSV file containing pore measurements (default: %(default)s).")
    parser.add_argument('--generate_3d', action=ap.BooleanOptionalAction, default=False,
                        help="If set, outputs a segmented z-stack image (default: %(default)s).")
    parser.add_argument('--output_3d', type=str, default="pores.tif", 
                        help="Path to save the segmented 3D image if --generate_3d is set (default: %(default)s).")
    parser.add_argument('--visualize_3d', default=False, action=ap.BooleanOptionalAction,
                        help="If set, a window showing the 3D reconstruction of the segmentation is shown.")

    # Image preprocessing
    parser.add_argument('--down_factor', type=float, default=1, 
                        help="Factor by which the image is downscaled for efficiency (default: %(default)s).")
    parser.add_argument('--normalize', action=ap.BooleanOptionalAction, default=True,
                        help="If set, the image is normalized, assuming values between 0-255 (8-bit scale) (default: %(default)s).")
    parser.add_argument('--crop_value', type=int, default=0, 
                        help="Number of pixels cropped from each side of the image (default: %(default)s).")
    parser.add_argument('--rescale_int', action=ap.BooleanOptionalAction, default=True,
                        help="If set, the image intensity is rescaled (default: %(default)s).")
    parser.add_argument('--sigma', type=float, default=3,
                        help="Standard deviation used for Gaussian smoothing (default: %(default)s).")

    # Parallel processing
    parser.add_argument('--num_workers', type=int, default=4, 
                        help="Number of workers used for segmenting the z-stack (default: %(default)s).")

    # Chan-Vese algorithm parameters
    parser.add_argument('--mu', type=float, default=0.1, 
                        help="Regularization parameter controlling segmentation smoothness (default: %(default)s).")
    parser.add_argument('--l1', type=float, default=0.6, 
                        help="Weight for the foreground region in Chan-Vese segmentation (default: %(default)s).")
    parser.add_argument('--l2', type=float, default=1, 
                        help="Weight for the background region in Chan-Vese segmentation (default: %(default)s).")
    parser.add_argument('--tol', type=float, default=1e-3, 
                        help="Tolerance threshold for segmentation convergence (default: %(default)s).")
    parser.add_argument('--iterations', type=int, default=70, 
                        help="Max iterations before stopping if convergence is not reached (default: %(default)s).")
    parser.add_argument('--disk_size', type=int, default=2, 
                        help="Size of the morphological disk used for erosion and dilation (default: %(default)s).")

    args = parser.parse_args()

    if args.generate_3d or args.visualize_3d:
        results, segm = process_stack(
            args.input,
            args.num_workers,
            args.crop_value,
            args.normalize,
            args.rescale_int, 
            args.sigma,
            args.down_factor,
            args.mu,
            args.l1,
            args.l2,
            args.tol,
            args.iterations,
            args.disk_size,
            args.generate_3d or args.visualize_3d)
        export_segmentation(segm, args.output_3d)
    else:
        results = process_stack(
            args.input,
            args.num_workers,
            args.crop_value,
            args.normalize,
            args.rescale_int, 
            args.sigma,
            args.down_factor,
            args.mu,
            args.l1,
            args.l2,
            args.tol,
            args.iterations,
            args.disk_size,
            args.generate_3d)
        
    export_csv(results, filepath=args.output_stats)

    if args.visualize_3d:
        visualize_3d(segm)
if __name__ == '__main__':
    main()
