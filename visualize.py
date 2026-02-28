import json
import os
import cv2
import numpy as np
from pycocotools.coco import COCO

# ====== 設定 ======
GT_JSON = "data/gt.json"
PRED_JSON = "data/pred.json"
IMAGE_ROOT = "data/images"
OUTPUT_DIR = "vis_output"
SCORE_THR = 0.3
# ==================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cocoGt = COCO(GT_JSON)

    # category_id -> name 辞書
    cats = cocoGt.loadCats(cocoGt.getCatIds())
    cat_id_to_name = {c["id"]: c["name"] for c in cats}

    with open(PRED_JSON, "r") as f:
        preds = json.load(f)

    img_ids = cocoGt.getImgIds()

    # 予測を image_id ごとに整理
    pred_by_img = {}
    for p in preds:
        pred_by_img.setdefault(p["image_id"], []).append(p)

    for img_id in img_ids:
        img_info = cocoGt.loadImgs(img_id)[0]
        img_path = os.path.join(IMAGE_ROOT, img_info["file_name"])

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        # ===== GT描画（緑）=====
        ann_ids = cocoGt.getAnnIds(imgIds=img_id)
        anns = cocoGt.loadAnns(ann_ids)

        for ann in anns:
            x, y, w, h = ann["bbox"]
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)

            cat_name = cat_id_to_name.get(ann["category_id"], str(ann["category_id"]))

            label = f"GT {cat_name} ({x1},{y1},{int(w)},{int(h)})"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, label, (x1, max(0, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # ===== 予測描画（赤）=====
        if img_id in pred_by_img:
            for p in pred_by_img[img_id]:
                if p.get("score", 1.0) < SCORE_THR:
                    continue

                x, y, w, h = p["bbox"]
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)

                cat_name = cat_id_to_name.get(p["category_id"], str(p["category_id"]))

                label = (
                    f"Pred {cat_name} "
                    f"{p.get('score',1.0):.2f} "
                    f"({x1},{y1},{int(w)},{int(h)})"
                )

                cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(img, label, (x1, max(0, y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,0,255), 1)

        # ===== 保存 =====
        save_path = os.path.join(
            OUTPUT_DIR,
            f"{os.path.splitext(img_info['file_name'])[0]}_vis.jpg"
        )
        cv2.imwrite(save_path, img)

        print(f"Saved: {save_path}")

    print("Done.")


if __name__ == "__main__":
    main()