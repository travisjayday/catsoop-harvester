#!/usr/bin/env python3

import sys
import math
import base64
import tkinter

from io import BytesIO
from PIL import Image as Image

# NO ADDITIONAL IMPORTS ALLOWED!


def flat_index(image, x, y):
    """
    Helper function.  Given an image and x and y coordinates, return the
    integer index into image['pixels'] associated with that (x,y) location in
    the image.
    """
    return y * image['width'] + x


def empty_like(image, val=0):
    """
    Helper function.  Given an image, create a new image of the same shape, but
    filled with all val.
    """
    return {
        'width': image['width'],
        'height': image['height'],
        'pixels': [val for _ in image['pixels']],
    }


def get_pixel(image, x, y):
    # FIXED issue with incorrect indexing into the list (need an integer, not
    # an (x, y) tuple)
    return image['pixels'][flat_index(image, x, y)]


def set_pixel(image, x, y, c):
    # FIXED same indexing issue from above
    image['pixels'][flat_index(image, x, y)] = c


def get_coords(image):
    """
    Helper function. Yields all the (x, y) tuple coordinates on the image.
    """
    for y in range(image['height']):
        for x in range(image['width']):
            yield x, y


def apply_per_pixel(image, func):
    result = empty_like(image) # FIXED issue with initializing empty array
    for x, y in get_coords(image): # FIXED issue with confusing width and height
        color = get_pixel(image, x, y)
        newcolor = func(color)
        # FIXED issue with needing the following statement in the loop
        set_pixel(result, x, y, newcolor) # FIXED issue with swapping x and y
    return result


def inverted(image):
    return apply_per_pixel(image, lambda c: 255-c) # FIXED: 255 vs 256


def clip(x, lo, hi):
    """
    Helper function. Returns `x` if it's within the range from
    `lo` to `hi` (inclusive); otherwise returns the cloeser of 
    `lo` and `hi`.
    """
    return max(min(x, hi), lo)


def get_pixel_extended(image, x, y):
    """
    Helper function.  Returns the pixel value at the given coordinate or,
    if the coordinate is out of bounds, the closest coordinate in bounds.
    """
    return get_pixel(image,
                     clip(x, 0, image['width']-1),
                     clip(y, 0, image['height']-1))


def correlate(image, kernel):
    """
    Compute the result of correlating the given image with the given kernel.

    The output of this function has the same form as a 6.009 image (a dictionary 
    with 'height', 'width', and 'pixels' keys), but its pixel values are not 
    necessarily in the range [0,255], and they may not be integers (they are 
    not clipped or rounded at all).

    Does not mutate the input image; rather, creates a separate structure to 
    represent the output.

        kernel: 2-d array (list of lists).
    """
    out = empty_like(image)
    n = len(kernel)
    kern_size = n//2  # the 'half-width' of the kernel
    for x, y in get_coords(image):
        # for each pixel, loop over the surrounding pixels, multiplying
        # each by the associated value in the kernel and accumulating
        # the results.
        v = 0
        for kx in range(n):
            for ky in range(n):
                px = x - kern_size + kx
                py = y - kern_size + ky
                v += get_pixel_extended(image, px, py) * kernel[ky][kx]
        # finally, set the pixel to the accumulated value.
        set_pixel(out, x, y, v)
    return out


def round_and_clip_image(image):
    """
    Given a dictionary, ensures that the values in the 'pixels' list are all
    integers in the range [0, 255].

    All values are converted to integers using Python's `round` function.

    Any locations with values higher than 255 in the input will have value
    255 in the output; and any locations with values lower than 0 in the input
    will have value 0 in the output.

    This version works by mutating the given image, but it also returns the
    image back (to make calling it within our filters slightly more pleasant).
    """
    image['pixels'] = [clip(round(p), 0, 255) for p in image['pixels']]
    return image


def box_blur_kernel(n, scale=1):
    """
    Helper function.  Create a kernel (list-of-lists) for a box blur with the
    given size n, scaled by an additional factor.
    """
    val = 1/n/n * scale
    return [[val]*n for _ in range(n)]


def blurred(image, n):
    """
    Return a new image representing the result of applying a box blur (with
    kernel size n) to the given input image.

    This process does not mutate the input image; rather, it creates a separate
    structure to represent the output.
    """
    return round_and_clip_image(correlate(image, box_blur_kernel(n)))


def sharpened(image, n):
    """
    Return a new image representing the result of applying an unsharp mask
    to the given input image.
    """
    kern = box_blur_kernel(n, -1)
    kern[n//2][n//2] += 2
    return round_and_clip_image(correlate(image, kern))


def edges(image):
    """
    Return a new image representing the result of applying the Sobel operator
    to the given input image.
    """
    sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    # Correlating with the two kernels above results in two images: one
    # that has the vertical edges only, and one that has the horizontal
    # edges only.
    rx = correlate(image, sobel_x)
    ry = correlate(image, sobel_y)
    # Now loop over the pixels and combine
    out = empty_like(image)
    for x, y in get_coords(image):
        val = (get_pixel(rx, x, y)**2 + get_pixel(ry, x, y)**2)**0.5
        set_pixel(out, x, y, clip(round(val), 0, 255))
    return out


# HELPER FUNCTIONS FOR LOADING AND SAVING GREYSCALE IMAGES

def load_image(filename):
    """
    Loads an image from the given file and returns an instance of this class
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_image('test_images/cat.png')
    """
    with open(filename, 'rb') as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith('RGB'):
            pixels = [round(.299 * p[0] + .587 * p[1] + .114 * p[2])
                      for p in img_data]
        elif img.mode == 'LA':
            pixels = [p[0] for p in img_data]
        elif img.mode == 'L':
            pixels = list(img_data)
        else:
            raise ValueError('Unsupported image mode: %r' % img.mode)
        w, h = img.size
        return {'height': h, 'width': w, 'pixels': pixels}


def save_image(image, filename, mode='PNG'):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the 'mode' parameter.
    """
    out = Image.new(mode='L', size=(image['width'], image['height']))
    out.putdata(image['pixels'])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()

if __name__ == '__main__':
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place for
    # generating images, etc.
    pass
