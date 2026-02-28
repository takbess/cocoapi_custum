from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

cocoGt = COCO("data/gt.json")
cocoDt = cocoGt.loadRes("data/pred.json")


cocoEval = COCOeval(cocoGt, cocoDt, "bbox")

cocoEval.params.iouThrs = np.array([0.5, ])  # IoU閾値を指定

def filter_policy_upper(ann, imgId, catId):
    x, y, w, h = ann["bbox"]
    cx = x + w/2
    cy = y + h/2
    return cy < 480/2 # 例：画像下半分だけ評価

def filter_policy_lower(ann, imgId, catId):
    x, y, w, h = ann["bbox"]
    cx = x + w/2
    cy = y + h/2
    return cy > 480/2  # 例：画像下半分だけ評価

cocoEval.filter_policies = {
    "upper":  filter_policy_upper, 
    "lower":  filter_policy_lower, 
}

cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()