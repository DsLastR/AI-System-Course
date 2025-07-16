import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import japanize_matplotlib
from utils.db_utils import init_db, insert_result

# ユーティリティのインポート
from utils.preprocessing import load_image, detect_faces, prepare_face

# 感情ラベル（FER2013）
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# モデル読み込み
model_path = "../model/emotion_model_fer2013.keras"
model = load_model(model_path)
print("✅ モデル読み込み成功")
model.summary()

# 出力用CSV初期化
results = []
# データベース初期化
init_db()

# テスト画像フォルダ
test_image_dir = "../data/test_images"
image_files = [f for f in os.listdir(test_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# 推論処理
for file_name in image_files:
    img_path = os.path.join(test_image_dir, file_name)
    print(f"\n📷 処理中: {file_name}")
    
    try:
        img_color, img_gray = load_image(img_path)
    except FileNotFoundError as e:
        print(f"⚠️ {e}")
        continue

    faces = detect_faces(img_gray)

    if len(faces) == 0:
        print(f"❌ 顔が検出されませんでした：{file_name}")
        results.append({
            "ファイル名": file_name,
            "検出顔数": 0,
            "表情ラベル": "検出なし",
            "信頼度": "-"
        })
        continue

    for (x, y, w, h) in faces:
        face_img = img_gray[y:y+h, x:x+w]
        face_input = prepare_face(face_img)

        prediction = model.predict(face_input)
        label_index = np.argmax(prediction)
        confidence = np.max(prediction)
        label = emotion_labels[label_index]

        print(f"🎭 {file_name} → {label}（信頼度: {confidence:.2f}）")

        # 推論結果をリストに保存
        results.append({
            "ファイル名": file_name,
            "検出顔数": len(faces),
            "表情ラベル": label,
            "信頼度": round(confidence, 4)
        })

        # ✅ 推論結果をデータベースにも保存
        insert_result(file_name, len(faces), label, float(confidence))

        # オプション：可視化
        cv2.rectangle(img_color, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(img_color, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    
    # 表示（複数顔対応済）
    img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title(f"{file_name} - 感情分類")
    plt.axis('off')
    plt.show()

# 結果CSV出力
os.makedirs("../result", exist_ok=True)
df = pd.DataFrame(results)
csv_path = "../result/emotion_predictions.csv"
df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\n✅ 感情分類結果をCSVに保存しました: {csv_path}")