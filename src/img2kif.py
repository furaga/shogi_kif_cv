import cv2
import numpy as np
import os
import pandas as pd
import time

# convert image to kif 
def convert(img):
    bb_mb, bb_kd_sente, bb_kd_gote = find_board_areas(img)
    main_board = img[bb_mb[1]:bb_mb[3], bb_mb[0]:bb_mb[2]]
    komadai_sente = img[bb_kd_sente[1]:bb_kd_sente[3], bb_kd_sente[0]:bb_kd_sente[2]]
    komadai_gote= img[bb_kd_gote[1]:bb_kd_gote[3], bb_kd_gote[0]:bb_kd_gote[2]]

    _, cell_values = parse_main_board(main_board)
    sente_komadai_pieces, _, _ = parse_sente_komadai(komadai_sente)
    gote_komadai_pieces, _, _ = parse_gote_komadai(komadai_gote)

    kif_text = create_kif(cell_values, sente_komadai_pieces, gote_komadai_pieces)
    return kif_text

# find main board, sente's komadai and gote's komadai
def find_board_areas(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_threshold = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(img_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    bbs = [get_bounding_box(c) for c in contours]    
    bb_mb = bounding_box_for_main_board(bbs)
    bb_kd_gote, bb_kd_sente = bounding_boxes_for_komadai(bbs)
    return bb_mb, bb_kd_sente, bb_kd_gote

def get_bounding_box(contour):
    contour = np.reshape(contour, (-1, 2))
    min_pt = np.min(contour, axis=0)
    max_pt = np.max(contour, axis=0)
    return min_pt[0], min_pt[1], max_pt[0], max_pt[1]
    
def error_bb_size(bb, gt_size):
    bb_w, bb_h = bb[2] - bb[0], bb[3] - bb[1]
    gt_w, gt_h = gt_size
    return abs(bb_w - gt_w) + abs(bb_h - gt_h)

def bounding_box_for_gt_size(bbs, gt_size, take_size=1):
    errors = [error_bb_size(bb, gt_size) for bb in bbs]
    bes = list(zip(enumerate(bbs), errors))
    sorted_bes = sorted(bes, key=lambda be: be[1])
    return [be[0][1] for be in sorted_bes[:take_size]]

def bounding_box_for_main_board(bbs, main_board_size=(648, 648)):
    return bounding_box_for_gt_size(bbs, main_board_size, take_size=1)[0]

def bounding_boxes_for_komadai(bbs, komadai_size=(101, 479)):
    ret_bbs = bounding_box_for_gt_size(bbs, komadai_size, take_size=2)
    if ret_bbs[0][0] < ret_bbs[1][0]:
        return ret_bbs[0], ret_bbs[1]
    return ret_bbs[1], ret_bbs[0]

# Split main baord image to cells and detect which piece is on each of the cells.
def parse_main_board(main_board, piece_info=None, cell_offset=(27,27), cell_size=(66,66)):
    piece_imgs, piece_info = load_reference_pieces(piece_info=piece_info)
    cell_imgs = main_board_to_cells(main_board, cell_offset, cell_size)
    results = [classify_piece(c, piece_imgs, piece_info) for c in cell_imgs]
    cell_values = [r[2][1] for r in results]
    return cell_imgs, cell_values

# Return abs path when path is seemed as relative path from the directory of this file
def as_relpath_from_this_dir(path):
    abspath = os.path.join(os.path.dirname(__file__ ), path)
    return abspath

def load_reference_pieces(piece_info = None):
    # [(img_path, piece name in .kif)]
    if piece_info is None:
        piece_info = [
            ("../data/denoban/hu.png",  " 歩"),
            ("../data/denoban/hi.png", " 飛"),
            ("../data/denoban/kaku.png", " 角"),
            ("../data/denoban/kyo.png", " 香"),
            ("../data/denoban/kei.png", " 桂"),
            ("../data/denoban/gin.png", " 銀"),
            ("../data/denoban/kin.png", " 金"),
            ("../data/denoban/gyoku.png", " 玉"),
            ("../data/denoban/to.png", " と"),
            ("../data/denoban/narikyo.png", " 杏"),
            ("../data/denoban/narikei.png", " 圭"),
            ("../data/denoban/ryu.png", " 龍"),
            ("../data/denoban/uma.png", " 馬"),
            ("../data/denoban/empty.png", " ・"),
            ("../data/denoban/hu.png", "v歩"),
            ("../data/denoban/hi.png", "v飛"),
            ("../data/denoban/kaku.png", "v角"),
            ("../data/denoban/kyo.png", "v香"),
            ("../data/denoban/kei.png", "v桂"),
            ("../data/denoban/gin.png", "v銀"),
            ("../data/denoban/kin.png", "v金"),
            ("../data/denoban/gyoku.png", "v玉"),
            ("../data/denoban/to.png", "vと"),
            ("../data/denoban/narikyo.png", "v杏"),
            ("../data/denoban/narikei.png", "v圭"),
            ("../data/denoban/ryu.png", "v龍"),
            ("../data/denoban/uma.png", "v馬"),
        ]

    piece_imgs = [cv2.imread(as_relpath_from_this_dir(path)) for path, _ in piece_info]

    # flip gotes'
    for i in range(len(piece_imgs)):
        if piece_info[i][1][0] == "v":
            piece_imgs[i] = cv2.flip(piece_imgs[i], -1)

    return piece_imgs, piece_info

def main_board_to_cells(main_board, main_board_offset, main_board_cell_size):
    cells = []
    for cell_y in range(9):
        for cell_x in range(9):
            c = picup_cell(main_board, cell_x, cell_y, main_board_offset, main_board_cell_size)
            cells.append(c)
    return cells

def picup_cell(main_board, cell_x, cell_y, offset, cell_size):
    offset_x, offset_y = offset
    cell_w, cell_h = cell_size
    x1 = offset_x + cell_w * cell_x
    x2 = offset_x + cell_w * (1 + cell_x)
    y1 = offset_y + cell_h * cell_y
    y2 = offset_y + cell_h * (1 + cell_y)
    if len(main_board.shape) == 3:
        return main_board[y1:y2,x1:x2,:]
    return main_board[y1:y2,x1:x2]

def classify_piece(cell, piece_imgs, piece_info):
    results = []
    for i, piece in enumerate(piece_imgs):
        res = cv2.matchTemplate(cell, piece, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        results.append((i, max_val, piece_info[i]))
    sorted_results = sorted(results, key=lambda sc: sc[1])
    return sorted_results[-1]


# Detect all pieces on komadai of sente
def parse_sente_komadai(komadai_sente, komadai_piece_info=None):
    return parse_komadai_common(komadai_sente, komadai_piece_info, False)

# Detect all pieces on komadai of gote
def parse_gote_komadai(komadai_gote, komadai_piece_info=None):
    return parse_komadai_common(komadai_gote, komadai_piece_info, True)

def parse_komadai_common(komadai, komadai_piece_info, flip_x):
    if komadai_piece_info is None:
        komadai_piece_info = [
            ("../data/denoban/hu.png", "歩"),
            ("../data/denoban/hi.png", "飛"),
            ("../data/denoban/kaku.png", "角"),
            ("../data/denoban/kyo.png", "香"),
            ("../data/denoban/kei.png", "桂"),
            ("../data/denoban/gin.png", "銀"),
            ("../data/denoban/kin.png", "金"),
        ]

    if flip_x:
        piece_imgs = [cv2.flip(cv2.imread(as_relpath_from_this_dir(path)), -1) for path, _ in komadai_piece_info]
    else:
        piece_imgs = [cv2.imread(as_relpath_from_this_dir(path)) for path, _ in komadai_piece_info]
    
    results = detect_pieces(komadai, piece_imgs, komadai_piece_info)
    values = [r[-1][1] for r in results]
    bounding_boxes = [r[2] for r in results]
    probabilities = [r[1] for r in results]
    return values, bounding_boxes, probabilities

def detect_pieces(komadai, piece_imgs, komadai_piece_info, threshold = 0.95):
    results = []
    for i, piece in enumerate(piece_imgs):
        res = cv2.matchTemplate(komadai, piece, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val >= threshold:
            bb = max_loc[0], max_loc[1], max_loc[0] + piece.shape[1], max_loc[1] + piece.shape[0]
            results.append((i, max_val, bb, komadai_piece_info[i]))
    return results

def create_kif(cell_values, sente_komadai_pieces, gote_komadai_pieces):
    lines = [
        "後手の持駒：", 
        "  ９ ８ ７ ６ ５ ４ ３ ２ １",
        "+---------------------------+",
        "|v香v桂v銀v金v玉v金v銀v桂v香|一",
        "| ・v飛 ・ ・ ・ ・ ・v角 ・|二",
        "|v歩 ・ ・v歩v歩v歩v歩v歩v歩|三",
        "| ・v歩 ・ ・ ・ ・ ・ ・ ・|四",
        "| ・ ・ ・ ・ ・ ・ ・ ・ ・|五",
        "| ・ ・ ・ ・ ・ ・ ・ ・ ・|六",
        "| 歩 歩 歩 歩 歩 歩 歩 歩 歩|七",
        "| ・ 角 ・ ・ ・ ・ ・ 飛 ・|八",
        "| 香 桂 銀 金 玉 金 銀 桂 香|九",
        "+---------------------------+",
        "先手の持駒：なし",
        "先手：",
        "後手：",
        "手数----指手---------消費時間--",
    ]

    if len(sente_komadai_pieces) >= 1:
        lines[13] = "先手の持駒：" + " ".join([name for name in sente_komadai_pieces])
    else:
        lines[13] = "先手の持駒：なし"

    if len(gote_komadai_pieces) >= 1:
        lines[0] = "後手の持駒：" + " ".join([name for name in gote_komadai_pieces])
    else:
        lines[0] = "後手の持駒：なし"
        
    for y in range(9):
        line = "|"
        for x in range(9):
            line += cell_values[y * 9 + x]
        line += lines[3+y][-2:]
        lines[3+y] = line

    kif_text = "\n".join(lines)
    return kif_text