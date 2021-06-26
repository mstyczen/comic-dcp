import re

import cv2
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from textblob import Word


def cluster(bboxes, words, image, visualize=False):
    if not words:
        return ""
    if len(words) < 2:
        return ' '.join(words[0])

    clusters = aggl_cluster(bboxes)

    if visualize:
        visualize_clusters(clusters, bboxes, words, image)

    cluster_bboxes = [[] for _ in range(len(np.unique(clusters)))]
    cluster_words = [[] for _ in range(len(np.unique(clusters)))]

    for cluster_id, bbox, word in zip(clusters, bboxes, words):
        cluster_bboxes[cluster_id].append(bbox)
        cluster_words[cluster_id].append(word)

    centroids = []
    texts = []
    for i, (cluster_elems, cluster_word) in enumerate(zip(cluster_bboxes, cluster_words)):
        centroid = \
            (np.average([(x1 + x2) // 2 for (x1, _, x2, _) in cluster_elems])), \
            (np.average([(y1 + y2) // 2 for (_, y1, _, y2) in cluster_elems]))
        text = ' '.join(cluster_word)
        centroids.append(centroid)
        texts.append(text)

    output = []
    for text, _ in sorted(zip(texts, centroids), key=lambda x: bbox_key(x[1])):
        output.append(text)

    result = ' '.join(output)
    return result


def visualize_clusters(clusters, bboxes, words, image):
    if len(np.unique(clusters)) >= 1:
        standard_bboxes = image.copy()
        colors = [list(np.random.random(size=3) * 256) for _ in range(10)]
        for i, (bbox, word) in enumerate(zip(bboxes, words)):
            b_cluster = clusters[i]
            color = colors[b_cluster]
            x1, y1, x2, y2 = bbox
            cv2.rectangle(standard_bboxes, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        cv2.imshow("res", np.hstack((standard_bboxes, image)))
        cv2.waitKey()


def bbox_key(bbox):
    x, y = bbox
    return 5 * y + x


def aggl_cluster(bboxes):
    heights = [(y2 - y1) for (_, y1, _, y2) in bboxes]
    tolerance = 0.75 * np.average(heights)
    n = len(bboxes)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = distance(bboxes[i], bboxes[j])
    cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=tolerance, affinity='precomputed',
                                      linkage='single')
    cluster.fit(distances)
    return cluster.labels_


def dist_affinity(x):
    return pairwise_distances(x, metric=distance)


def distance(bbox1, bbox2):
    x11, y11, x12, y12 = bbox1
    x21, y21, x22, y22 = bbox2

    if overlaps(x11, x12, x21, x22):
        xdist = 0
    else:
        xdist = min(abs(x11 - x21), abs(x12 - x22), abs(x11 - x22), abs(x12 - x21))

    ydist = min(abs(y11 - y21), abs(y12 - y22), abs(y11 - y22), abs(y12 - y21))
    dist = max(xdist, ydist)

    return dist


def overlaps(a1, a2, b1, b2):
    return a1 < b1 < a2 or b1 < a1 < b2


def clear_text(text):
    text = text.lower()

    text = re.sub(r'[^a-zA-Z.,?!\s]+', '', text)
    _RE_COMBINE_WHITESPACE = re.compile(r"\s+")
    text = _RE_COMBINE_WHITESPACE.sub(" ", text).strip()

    return text


def autocorrect_text(text):
    words = text.split(' ')
    corrected = []
    for word in words:
        gfg = Word(word)
        spellcheck = gfg.spellcheck()
        if 0.75 < spellcheck[0][1] < 1:
            word = spellcheck[0][0]

        corrected.append(word)

    text = ' '.join(corrected)
    return clear_text(text)
