import cv2
from tqdm import tqdm

from text_extractor.OCRPreprocessor import rescale_for_ocr, binarize_for_ocr
from text_extractor.OCRPostprocessor import cluster, clear_text, autocorrect_text
from text_extractor.TesseractOCR import TesseractOCR
from text_extractor.VisionOCR import VisionOCR

tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
cloud_credentials = '../../credentials/credentials.json'


def extract_text(img_paths, engine, rescale=False, binarize=False, clustering=False, autocorrect=False):
    """
    Extracts text from images.
    :param img_paths: A list of paths to images for processing.
    :param engine: OCR engine used for extraction (tesseract/vision-api).
    :param rescale: Bool determining if re-scaling pre-processing step should be applied.
    :param binarize: Bool determining if binarization pre-processing step should be applied.
    :param clustering: Bool determining if clustering-based output ordering post-processing step should be applied.
    :param autocorrect: Bool determining if autocorrect post-processing step should be applied.
    :return: A dictionary mapping image paths to the extracted names.
    """

    result = {}
    for path in tqdm(img_paths):
        image = cv2.imread(path)

        if rescale:
            image = rescale_for_ocr(image, tesseract_path)

        if binarize:
            image = binarize_for_ocr(image)

        if engine == "tesseract":
            extractor = TesseractOCR(tesseract_path)
        elif engine == "vision-api":
            extractor = VisionOCR(cloud_credentials)
        else:
            raise Exception("Invalid OCR engine: supported engines are ")

        bboxes, words = extractor.extract_text(image)

        if clustering:
            text = cluster(words, bboxes, image)
        else:
            text = ' '.join(words)

        text = clear_text(text)

        if autocorrect:
            text = autocorrect_text(text)

        result[path] = text

    return result
