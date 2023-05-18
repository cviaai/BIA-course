import cv2

import numpy as np

from scipy.signal import fftconvolve


def normxcorr2(template: np.ndarray, image: np.ndarray, mode="full") -> np.ndarray:
    """
    Input arrays should be floating point numbers.
    
    Args:
        template: N-D array, of template or filter you are using for cross-correlation.
            Must be less or equal dimensions to image.
            Length of each dimension must be less than length of image.
        image: N-D array
        mode: Options, "full", "valid", "same"
            full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs. 
                Output size will be image size + 1/2 template size in each dimension.
            valid: The output consists only of those elements that do not rely on the zero-padding.
            same: The output is the same size as image, centered with respect to the ‘full’ output.
    Returns:
        N-D array of same dimensions as image. Size depends on mode parameter.
    """
    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)

    image = fftconvolve(np.square(image), a1, mode=mode) - \
            np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0

    return out


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result