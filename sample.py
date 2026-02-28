from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

cocoGt = COCO("data/gt.json")
cocoDt = cocoGt.loadRes("data/pred.json")


cocoEval = COCOeval(cocoGt, cocoDt, "bbox")

cocoEval.params.iouThrs = np.array([0.5, ])  # IoU閾値を指定

"""
# policy の書き方
def your_filter_policy(ann, imgId, catId):
    # ann: アノテーション（GTまたは予測）の辞書
    # imgId: 画像ID
    # catId: カテゴリーID
    # 条件に基づいてTrue/Falseを返す処理
    return True / False

# policy の登録方法
cocoEval.filter_policies = {
    "your_policy_name1": your_filter_policy1,
    "your_policy_name2": your_filter_policy2,
    ...
}


# policy 登録時の挙動
if policy_fn(gt)==False: 
    gt["_ignore"]=True

# 割り当て処理。
#   _ignore=Falseのgtを優先して割り当て。
#   _ignore=True のgtに割り当たったら、pred["_ignore"]=Trueにする

if pred is not matched to any gt:
    if policy_fn(pred)==False:
        pred["_ignore"]=True

# TP,FP,FN のカウント時には、_ignore=Trueのものはカウントしない。

"""

# def filter_policy_upper(ann, imgId, catId):
#     x, y, w, h = ann["bbox"]
#     cx = x + w/2
#     cy = y + h/2
#     return cy < 480/2 # 例：画像の上半分だけ評価

# def filter_policy_lower(ann, imgId, catId):
#     x, y, w, h = ann["bbox"]
#     cx = x + w/2
#     cy = y + h/2
#     return cy > 480/2  # 例：画像の下半分だけ評価


# 例：画像中心から半径100px以内のオブジェクトだけ評価
def filter_policy_center_circle(ann, imgId, catId, radius=100):
    img_info = cocoGt.imgs[imgId]
    cx_img, cy_img = img_info["width"] / 2, img_info["height"] / 2
    x, y, w, h = ann["bbox"]
    cx_box, cy_box = x + w/2, y + h/2
    dist = np.sqrt((cx_box - cx_img)**2 + (cy_box - cy_img)**2)
    return dist <= radius


# 例：四角形ポリゴン
from shapely.geometry import Point, Polygon
polygon_coords = [(100, 100), (400, 100), (400, 300), (100, 300)]
poly = Polygon(polygon_coords)

def filter_policy_polygon(ann, imgId, catId, polygon=poly):
    x, y, w, h = ann["bbox"]
    cx, cy = x + w/2, y + h/2
    point = Point(cx, cy)
    return polygon.contains(point)


# 例：画像の四隅10%以内のオブジェクトだけ評価
def filter_policy_corners(ann, imgId, catId, ratio=0.1):
    img_info = cocoGt.imgs[imgId]
    H, W = img_info["height"], img_info["width"]
    x, y, w, h = ann["bbox"]
    cx, cy = x + w/2, y + h/2

    # 左上、右上、左下、右下の10%四隅
    left = cx < W*ratio
    right = cx > W*(1-ratio)
    top = cy < H*ratio
    bottom = cy > H*(1-ratio)

    return left or right or top or bottom






# # 例：visibilityフィルタ

# # 例：predのvisibilityが0.5以上のものだけ評価
# def filter_policy_pred_visibility(ann, imgId, catId, thresh=0.5):
#     # ann: GTまたはpredの辞書
#     # predにvisibilityを追加しておく必要があります
#     # predの場合、GTにはないフィールドなのでデフォルト0で安全
#     visibility = ann["visibility"]
#     return visibility > thresh

# # これだと割り当てが不自然、、、、gtにだけ属性がある場合は別の処理が良いか、、、
# """
# gt_only_filter_policy(ann, imgId, catId):
#     return True/False


# """


# """
# ### gt_policy ###
# 理念：
# gt_policy_fn(gt)=False となるものは難しいGT。
# そのGTを検出できたとしても、過検出としたくない。

# # gt_policy の書き方
# def your_gt_filter_policy(ann, imgId, catId):
#     # ann: アノテーション（GTまたは予測）の辞書
#     # imgId: 画像ID
#     # catId: カテゴリーID
#     # 条件に基づいてTrue/Falseを返す処理
#     return True / False

# # gt_policy の登録方法
# cocoEval.gt_filter_policies = {
#     "your_gt_policy_name1": your_gt_filter_policy1,
#     "your_gt_policy_name2": your_gt_filter_policy2,
#     ...
# }


# # gt_policy 登録時の挙動
# if gt_policy_fn(gt)==False: 
#     gt["_ignore"]=True

# # 割り当て処理。
# #   _ignore=True のgtに割り当たったら、pred["_ignore"]=Trueにする
# #   ※_ignore=Falseのgtを優先して割り当てることはしない。

# if pred is not matched to any gt:
#     if gt_policy_fn(pred)==False:
#         pred["_ignore"]=True

# # TP,FP,FN のカウント時には、_ignore=Trueのものはカウントしない。

# """









cocoEval.filter_policies = {
    # "upper":  filter_policy_upper, 
    # "lower":  filter_policy_lower, 
    # "center_circle":  filter_policy_center_circle,
    # "polygon": filter_policy_polygon,
    "corners": filter_policy_corners,
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