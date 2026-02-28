import os
import json
import numpy as np
import cv2

# ===== 設定 =====
GT_JSON = "data/gt.json"
OUTPUT_DIR = "data/images"
# =================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(GT_JSON, "r") as f:
        gt = json.load(f)

    images = gt["images"]

    print(f"Generating {len(images)} dummy images...")

    for img_info in images:
        file_name = img_info["file_name"]
        width = img_info["width"]
        height = img_info["height"]

        save_path = os.path.join(OUTPUT_DIR, file_name)

        # フォルダが階層になっている場合に対応
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # ダミー画像生成（黒背景＋薄ノイズ）
        img = np.zeros((height, width, 3), dtype=np.uint8)
        noise = np.random.randint(0, 100, (height, width, 3), dtype=np.uint8)
        img = cv2.add(img, noise)

        cv2.imwrite(save_path, img)

    print("Done.")

if __name__ == "__main__":
    main()