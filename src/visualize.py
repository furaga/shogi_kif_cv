import cv2
import matplotlib.pyplot as plt
import numpy as np


def display_cv2_img(img, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    plt.imshow(img)

def bounding_box_to_contour(bb):
    return np.array([
        (bb[0], bb[1]),
        (bb[0], bb[3]),
        (bb[2], bb[3]),
        (bb[2], bb[1]),
        (bb[0], bb[1])])

def draw_boards(img, bb_mb, bb_kd_sente, bb_kd_gote):
    rend_img = img.copy()
    rend_img = cv2.drawContours(rend_img, [bounding_box_to_contour(bb_mb)], -1, (0,255,0), 6)
    rend_img = cv2.drawContours(rend_img, [bounding_box_to_contour(bb_kd_sente)], -1, (255,255,0), 6)
    rend_img = cv2.drawContours(rend_img, [bounding_box_to_contour(bb_kd_gote)], -1, (0,0,255), 6)
    return rend_img

def draw_komadai(img, piece_names, piece_boxes):
    komadai_piece_colors = {
        "歩": (120, 120, 120),
        "飛": (0, 0, 255),
        "角": (255, 0, 0),
        "香": (0, 0, 100),
        "桂": (100, 0, 0),
        "銀": (200, 200, 200),
        "金": (0, 255, 255),
    }
    rend_img = img.copy()
    for name, box in zip(piece_names, piece_boxes):
        col = (255, 255, 255)
        if name in komadai_piece_colors.keys():
            col = komadai_piece_colors[name]
        rend_img = cv2.drawContours(rend_img, [bounding_box_to_contour(box)], -1, col, 3)
    return rend_img