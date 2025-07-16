# 表情認識システム / Emotion Recognition  System / 表情识别系统

---

## 📌 概要 / Overview / 概述

### 🇯🇵 日本語

このプロジェクトは、**FER-2013** データセットを用いて学習した **CNN モデル** により、人の顔から7種類の感情を分類する **表情認識システム** です。

* 使用技術：TensorFlow / Keras / OpenCV / NumPy / Matplotlib
* 対応感情：怒り、嫌悪、恐れ、幸福、悲しみ、驚き、無表情
* 顔検出：Haar Cascade（複数分類器に対応）
* 推論結果の可視化（顔枠＋感情ラベル）
* 結果は CSV に保存可能

---

### 🇬🇧 English

This is an **emotion recognition AI system** using a **CNN model** trained on the **FER-2013** dataset to classify facial expressions into 7 categories.

* Technologies: TensorFlow / Keras / OpenCV / NumPy / Matplotlib
* Emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
* Face detection via multiple Haar cascade classifiers
* Results are visualized (bounding box + emotion label)
* Results can be saved to CSV

---

### 🇨🇳 中文

本项目是一个  **表情识别 AI 系统** ，使用基于 **FER-2013 数据集** 训练的  **CNN 模型** ，可将人脸图像识别为7种基本情绪。

* 技术栈：TensorFlow / Keras / OpenCV / NumPy / Matplotlib
* 支持识别：愤怒、厌恶、恐惧、高兴、悲伤、惊讶、中性
* 人脸检测：支持多个 Haar 级联分类器
* 结果可视化（绘制人脸框 + 标签）
* 推论结果保存为 CSV 文件

---

## 📂 ディレクトリ構成 / Project Structure / 项目结构

emotion_system/
├── model/                             						# 学習・モデル保存関連
│   ├── emotion_model_fer2013.keras         		← 保存済みモデル
│   ├── train_emotion_model.ipynb                	 ← Colab用モデル学習Notebook
│   ├── CNN_model.py                        				← CNN構造定義（py版）
│   └── train_emotion_model.py              		← モデル学習スクリプト（py版）
│
├── predict/                           				# 推論関連
│   ├── predict_emotion.py                  	← 推論スクリプト（複数画像対応・CSV出力）
│   └── predict_emotion.ipynb              	 ← Colab用推論Notebook（アップロード対応）
│
├── utils/                             			# ユーティリティ関数モジュール
│   ├── preprocessing.py              	←  画像読み込み・顔検出・前処理関数
│   └── db_utils.py                      	←  SQLite DB操作用モジュール（初期化・挿入・取得）
│
├── data/                              				# データ置き場
│   ├── fer2013/                            		← FER - 2013学習データ（Colabで解凍）
│   └── test_images/                        	← 推論用画像フォルダ（jpg/jpeg/png可）
│       ├── test1.jpg
│       ├── test2.jpg
│       └── ...
│
├── result/                            				# 推論結果保存フォルダ
│   ├── emotion_predictions.csv          	← CSV形式の推論結果（自動生成・保存）
│   └── emotion_results.db               		← SQLite形式の推論結果（自動生成・保存）
│
└── README.md                         			# プロジェクト概要・実行手順説明書

---

## ⚙️ 実行手順 / How to Run / 运行方法

### 1. モデルの構築と学習（Google Colab推奨）

- `model/predict_emotion.ipynb` を Google Colab 上で実行
- `fer2013.zip` をアップロードして解凍
- モデルを学習し、`emotion_model_fer2013.keras` を保存（自動保存）

### 2. 感情分類の実行（Google Colab推奨）

#### ✔ Google Colab 版（複数画像＋CSV出力）

* `predict/predict_emotion.ipynb` を開く
* `.keras` モデルと画像をアップロード
* 結果はセル上に表示（CSV出力）

#### ✔ Pythonスクリプト版（複数画像＋CSV出力）

    python predict/predict_emotion.py

* `data/test_images/` 内の全画像を処理
* 結果は `result/emotion_predictions.csv` に自動保存
* Matplotlibで画像結果を表示

---

## 🧩 モジュール構成 / Key Modules / 核心模块

| ファイル                   | 概要                                          |
| -------------------------- | --------------------------------------------- |
| `CNN_model.py`           | CNNのモデル構造を定義                         |
| `train_emotion_model.py` | 学習データの前処理、モデル訓練、保存          |
| `predict_emotion.py`     | モデル読み込み、画像推論、結果可視化＆CSV保存 |
| `preprocessing.py`       | 顔検出、画像リサイズ、正規化、形式変換など    |

---

## 📝 備考 / Notes / 备注

* FER-2013 データセットは [Kaggle]() から取得可能
* 推論時に複数の顔が検出される場合、全ての顔について分類します
* Colab利用時は `.py` ファイルをアップロードしてからインポートしてください

---


    ーーーーーーーー***K.SK***
