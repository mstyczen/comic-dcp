import cv2
import pytesseract as tess
from tqdm import tqdm


class TesseractOCR:
    def __init__(self, tesseract_path):
        """

        :param tesseract_path:
        """
        tess.pytesseract.tesseract_cmd = tesseract_path

    def extract_text(self, image):
        """
        Performs OCR text extraction using Tesseract OCR.
        :param image: image to detect text in.
        :return: tuple of lists, one containing the detected strings, the other containing the b-boxes.
        """
        custom_config = r'-c tessedit_char_whitelist' \
                        r"='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,?!- '"
        boxes = tess.image_to_data(image, config=custom_config)
        words = []
        bboxes = []

        for x, b in enumerate(boxes.splitlines()):
            if x != 0:
                b = b.split()
                if len(b) == 12:
                    x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
                    bboxes.append((x, y, x + w, y + h))
                    words.append(b[11])
        return words, bboxes