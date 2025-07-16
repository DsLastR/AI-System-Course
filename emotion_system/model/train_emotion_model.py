import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from CNN_model import build_emotion_model

# ハイパーパラメータ設定
batch_size = 64            # バッチサイズ
epochs = 30                # 学習エポック数
learning_rate = 0.001      # 学習率
input_shape = (48, 48, 1)  # 画像サイズ（縦×横×チャンネル）
num_classes = 7            # 感情カテゴリ数（7種類）

# FER2013のtrain, test フォルダパス
train_dir = "../data/fer2013/train"
test_dir = "../data/fer2013/test"

# 訓練データジェネレータ（データ拡張付き）
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# テストデータジェネレータ（正規化のみ）
test_datagen = ImageDataGenerator(rescale=1./255)

# ジェネレータから画像読み込み
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# モデル構築
model = build_emotion_model(input_shape, num_classes, learning_rate)

# 学習実行
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator
)

# 最終評価
score = model.evaluate(test_generator)
print("✅ 評価完了：正解率 = {:.4f}, loss = {:.4f}".format(score[1], score[0]))

# モデル保存
model.save("emotion_model_fer2013.keras")
print("✅ モデル保存完了")
