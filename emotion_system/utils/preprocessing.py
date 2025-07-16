import cv2
import numpy as np
import os

# Haarカスケード：複数OpenCV標準の顔検出器（順番）
DETECTORS = [
    cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"),
    cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"),
    cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"),
    cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt_tree.xml")
]

def load_image(img_path):
    """
    指定されたパスの画像を読み込む。
    カラー画像とグレースケール画像を返す。

    参数:
        img_path (str): 画像ファイルのパス
    戻り値:
        img_color, img_gray
    """
    img_color = cv2.imread(img_path)
    if img_color is None:
        raise FileNotFoundError(f"画像ファイルが見つかりません: {img_path}")
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    return img_color, img_gray

def detect_faces(gray_img):
    """
    複数のHaar分類器を用いて顔検出を行う。
    最初に検出された結果を返す。

    参数:
        gray_img (np.array): グレースケール画像
    戻り値:
        faces (list): 検出された顔の位置（x, y, w, h）
    """
    for detector in DETECTORS:
        faces = detector.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
        if len(faces) > 0:
            return faces
    return []

def prepare_face(face_img, size=(48, 48)):
    """
    検出された顔画像をモデル入力用に前処理する。

    - リサイズ
    - 正規化（0〜1）
    - チャンネル次元追加（1ch）
    - バッチ次元追加（1枚分）

    参数:
        face_img (np.array): 検出された顔画像（グレースケール）
        size (tuple): リサイズ後の画像サイズ (height, width)

    戻り値:
        (1, height, width, 1)のNumPy配列
    """
    face_resized = cv2.resize(face_img, size)
    face_normalized = face_resized / 255.0
    face_expanded = np.reshape(face_normalized, (1, size[0], size[1], 1))
    return face_expanded
