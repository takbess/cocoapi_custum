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

"""
実行結果
 Average Precision  (AP) @[ IoU=0.50:0.50 | area=   all | policy=   all | maxDets=100 ] = 0.752
 Average Precision  (AP) @[ IoU=0.50:0.50 | area=   all | policy=   all | maxDets=100 ] = 0.752
 Average Precision  (AP) @[ IoU=0.50      | area=   all | policy=   all | maxDets=100 ] = 0.752
 Average Precision  (AP) @[ IoU=0.75      | area=   all | policy=   all | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.50 | area= small | policy=   all | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.50 | area=medium | policy=   all | maxDets=100 ] = 0.500
 Average Precision  (AP) @[ IoU=0.50:0.50 | area= large | policy=   all | maxDets=100 ] = 1.000
 Average Recall     (AR) @[ IoU=0.50:0.50 | area=   all | policy=   all | maxDets=  1 ] = 0.750
 Average Recall     (AR) @[ IoU=0.50:0.50 | area=   all | policy=   all | maxDets= 10 ] = 0.750
 Average Recall     (AR) @[ IoU=0.50:0.50 | area=   all | policy=   all | maxDets=100 ] = 0.750
 Average Recall     (AR) @[ IoU=0.50:0.50 | area= small | policy=   all | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.50 | area=medium | policy=   all | maxDets=100 ] = 0.500
 Average Recall     (AR) @[ IoU=0.50:0.50 | area= large | policy=   all | maxDets=100 ] = 1.000
 Average Precision  (AP) @[ IoU=0.50:0.50 | area=   all | policy= upper | maxDets=100 ] = 1.000 # upper ポリシーでのAP
 Average Precision  (AP) @[ IoU=0.50:0.50 | area=   all | policy= lower | maxDets=100 ] = 0.000 # lower ポリシーでのAP
 Average Recall     (AR) @[ IoU=0.50:0.50 | area=   all | policy= upper | maxDets=100 ] = 1.000 # upper ポリシーでのAR 
 Average Recall     (AR) @[ IoU=0.50:0.50 | area=   all | policy= lower | maxDets=100 ] = 0.000 # lower ポリシーでのAR


"""