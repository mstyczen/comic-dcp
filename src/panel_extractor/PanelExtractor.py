import cv2
import numpy as np
from tqdm import tqdm


class PanelExtractor:
    def extract_and_save_panels(self, img_paths, out_directory, iterations=15, thresh_type='adaptive'):
        """
        Extracts panels and saves them to disk.
        :param img_paths: paths to images.
        :param out_directory: Output directory.
        :param iterations: Number of iterations for morph opening.
        :param thresh_type: Type of thresholding (adaptive/constant).
        :return: A list of paths to the saved images.
        """

        out_paths = []
        for path in tqdm(img_paths):
            img = cv2.imread(path)
            id = path[path.rfind('/') + 1: path.rfind('.')]
            bboxes = self.extract_panels(path, iterations, thresh_type)
            for i, bbox in enumerate(bboxes):
                (x1, y1), (x2, y2) = bbox
                panel = img[y1:y2, x1:x2]
                out_path = f"{out_directory}/{id}_{i}.png"
                cv2.imwrite(out_path, panel)
                out_paths.append(out_path)
        return out_paths

    def extract_panels(self, image_path, iterations, thresh_type):
        """
        Extracts panels from an image.
        :param image_path: Path to the image.
        :param iterations: Number of iterations for morph opening.
        :param thresh_type: Type of thresholding (adaptive/constant).
        :return: A list of bounding boxes for the panels.
        """
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if thresh_type == 'adaptive':
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 9)
        elif thresh_type == 'constant':
            ret, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        else:
            raise Exception(
                f"Invalid threshold type: {thresh_type}. Supported threshold types are \"adaptive\" and \"constant\".")

        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_img = thresh.copy()
        contours = contours[0] if len(contours) == 2 else contours[1]

        for c in contours:
            cv2.drawContours(contours_img, [c], -1, (255, 0, 0), -1)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opening = contours_img.copy()
        opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel, iterations=iterations)

        contours = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        # get the bounding boxes, and sort by y and x coordinates
        bboxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda bbox: 5 * bbox[1] + bbox[0])

        bboxes = self.correct_bboxes(bboxes)
        bboxes = [((x, y), (x + w, y + h)) for (x, y, w, h) in bboxes]

        return bboxes

    def correct_bboxes(self, bboxes):
        for b1 in bboxes:
            for b2 in bboxes:
                if b1 != b2 and self.bbox_contains_other(b1, b2):
                    bboxes.remove(b2)
        if bboxes:
            max_area = max([h * w for (x, y, w, h) in bboxes])
            bboxes = list(filter(lambda box: box[2] * box[3] > 0.15 * max_area, bboxes))

        return bboxes

    @staticmethod
    def merge_two_bboxes(b1, b2):
        (x1, y1, w1, h1) = b1
        (x2, y2, w2, h2) = b2
        x = min(x1, x2)
        w = max(x1 + w1 - x, x2 + w2 - x)
        y = min(y1, y2)
        h = max(y1 + h1 - y, y2 + h2 - y)
        return x, y, w, h

    @staticmethod
    def bboxes_overlap(b1, b2):
        (x1, y1, w1, h1) = b1
        (x2, y2, w2, h2) = b2

        return not (x1 + w1 < x2 or x1 > x2 + w2 or y1 + h1 < y2 or y1 > y2 + w2)

    @staticmethod
    def bbox_contains_other(this, other):
        (x1, y1, w1, h1) = this
        (x2, y2, w2, h2) = other

        return x1 <= x2 and x1 + h1 >= x2 + h2 and y1 <= y2 and y1 + h2 >= y2 + h2

    @staticmethod
    def visualize(image, gray, thresh, contours_img, opening, bboxes):
        # convert to rgb so all images have the same dimensions
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        opening_rgb = cv2.cvtColor(opening, cv2.COLOR_GRAY2RGB)
        contours_img = cv2.cvtColor(contours_img, cv2.COLOR_GRAY2RGB)
        bboxes_image = image.copy()

        # draw the bboxes
        for (x, y, w, h) in bboxes:
            bboxes_image = cv2.rectangle(bboxes_image, (x, y), (x + w, y + h), (36, 255, 12), 3)

        images = np.hstack((gray_rgb, thresh_rgb, contours_img, opening_rgb, bboxes_image))
        cv2.imshow("res", images)
        cv2.waitKey()
