from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.coco_pr_wrapper import CocoPRWrapper

ann_file = "data/gt.json"
res_file = "data/pred.json"

coco_gt = COCO(ann_file)
coco_dt = coco_gt.loadRes(res_file)

coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
coco_eval.evaluate()
coco_eval.accumulate()

coco_pr = CocoPRWrapper(coco_eval)

# PR 曲線を CSV に保存
coco_pr.save_pr(cat="person", iou=0.5, out="output/pr_cat_person_iou_0.5.csv")

# F1 が最大となる点のメトリクス（辞書で返す）
metrics = coco_pr.get_metrics_at_best_f1(cat="person", iou=0.5)
# -> {"precision": 0.85, "recall": 0.82, "f1": 0.83, "score": 0.45, "cat": "person", "iou": 0.5, ...}
print("Metrics at best F1(person):")
print(metrics)

# F1 が最大となる点のメトリクス（辞書で返す）
metrics = coco_pr.get_metrics_at_best_f1(cat="car", iou=0.5)
# -> {"precision": 0.85, "recall": 0.82, "f1": 0.83, "score": 0.45, "cat": "person", "iou": 0.5, ...}
print("Metrics at best F1(car):")
print(metrics)

metrics = coco_pr.get_metrics_at_best_f1(cat="mean", iou=0.5)
# -> {"precision": 0.85, "recall": 0.82, "f1": 0.83, "score": 0.45, "cat": "person", "iou": 0.5, ...}
print("Metrics at best F1(mean):")
print(metrics)

# 固定スコア時のメトリクス（cat="mean" で全カテゴリ平均）
metrics = coco_pr.get_metrics_at_fixed_score(0.5, cat="mean", iou=0.5)
print("Metrics at fixed score 0.5:")
print(metrics)

# 固定 recall 時のメトリクス
metrics = coco_pr.get_metrics_at_fixed_recall(0.9, cat="mean", iou=0.5)
print("Metrics at fixed recall 0.9:")
print(metrics)

# AP 取得
metrics = coco_pr.get_ap(cat="person", area="small")
print("AP for person (small area):", metrics["ap"])
print(metrics)
