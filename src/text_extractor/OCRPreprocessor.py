import cv2
import pytesseract as tess
import numpy as np


def rescale_for_ocr(image, tesseract_path, font_size=30):
    """
    Performs automatic rescaling for OCR - performs an intiial OCR pass, determines the current letter height,
    and rescales the image to match desired letter height. :param image: image array to rescale. :param
    tesseract_path: the path to the Tesseract OCR. :param font_size: the desired letter height in the output image.
    :return: The rescaled image.
    """
    tess.pytesseract.tesseract_cmd = tesseract_path
    orig_image = image.copy()
    image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    heights = []
    custom_config = r"-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?!-"
    boxes = tess.image_to_boxes(image, config=custom_config)
    h, w, _ = image.shape
    for b in boxes.splitlines():
        b = b.split()
        if len(b) == 6:
            y1, y2 = int(b[2]), int(b[4])
            heights.append(y2 - y1)

    if len(heights) < 5:
        return image

    scale = min(8, 3 * font_size / np.median(heights))
    rescaled_image = cv2.resize(orig_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    return rescaled_image


def binarize_for_ocr(image):
    """
    Binarizes the image using adaptive thresholding.
    :param image: The image to process.
    :return: The binarized image.
    """
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binarized = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
    binarized = cv2.fastNlMeansDenoising(binarized, None, 10, 10, 10)
    return binarized
