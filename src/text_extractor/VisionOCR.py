import io
import os

import cv2
from google.cloud import vision
import numpy as np


class VisionOCR:
    def __init__(self, credentials_path):
        """
        :param credentials_path: a path to the Google Cloud credentials .json file.
        """
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

    def extract_text(self, image):
        """
        Performs OCR text extraction using Google Cloud Vision API OCR.
        :param image: image to detect text in.
        :return: tuple of lists, one containing the detected strings, the other containing the b-boxes.
        """
        client = vision.ImageAnnotatorClient()
        cv2.imwrite("temp.png", image)
        with io.open("temp.png", 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        response = client.text_detection(image=image)
        texts = response.text_annotations

        words = []
        bboxes = []
        for text in texts[1:]:
            words.append(text.description)
            pts = np.array([[vertex.x, vertex.y] for vertex in text.bounding_poly.vertices], np.int32)
            xmin = np.min([x[0] for x in pts])
            ymin = np.min([x[1] for x in pts])
            xmax = np.max([x[0] for x in pts])
            ymax = np.max([x[1] for x in pts])
            bboxes.append((xmin, ymin, xmax, ymax))

        return words, bboxes