"""
CocoPRWrapper: COCOeval の precision, recall, scores をラップし、
F1 計算や各種メトリクス取得を便利にするクラス。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _compute_f1(precision: float, recall: float) -> float:
    """F1 = 2 * P * R / (P + R)"""
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


class CocoPRWrapper:
    """
    COCOeval の評価結果をラップし、F1 や各種メトリクスを辞書形式で取得するクラス。

    Usage:
        coco_gt = COCO(ann_file)
        coco_dt = coco_gt.loadRes(res_file)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()

        coco_pr = CocoPRWrapper(coco_eval)
        coco_pr.save_pr(cat="person", iou=0.5, out="pr.csv")
        metrics = coco_pr.get_metrics_at_best_f1(cat="person", iou=0.5)
    """

    def __init__(self, coco_eval: COCOeval):
        """
        Args:
            coco_eval: evaluate() と accumulate() を実行済みの COCOeval オブジェクト
        """
        if not coco_eval.eval:
            raise ValueError("coco_eval.evaluate() と coco_eval.accumulate() を先に実行してください")

        self._eval = coco_eval.eval
        self._params = coco_eval.params
        self._coco_gt = coco_eval.cocoGt

        self._precision: np.ndarray = self._eval["precision"]  # [T, R, K, A, Y, Z, M]
        self._recall: np.ndarray = self._eval["recall"]        # [T, K, A, Y, Z, M]
        self._scores: np.ndarray = self._eval["scores"]        # [T, R, K, A, Y, Z, M]
        self._rec_thrs: np.ndarray = self._params.recThrs
        self._iou_thrs: np.ndarray = self._params.iouThrs
        self._cat_ids: list = list(self._params.catIds)
        self._area_rng_lbl: list = list(self._params.areaRngLbl)
        self._gt_filter_policy_lbl: list = list(getattr(self._params, "gtFilterPolicyLbl", ["all"]))
        self._filter_policy_lbl: list = list(getattr(self._params, "filterPolicyLbl", ["all"]))
        self._max_dets: list = list(self._params.maxDets)

        # cat name -> id のマップ
        self._cat_name_to_id: dict[str, int] = {}
        if self._coco_gt is not None:
            for c in self._coco_gt.loadCats(self._coco_gt.getCatIds()):
                self._cat_name_to_id[c["name"]] = c["id"]

    def _resolve_cat(self, cat: Union[str, int]) -> list[int]:
        """cat をカテゴリ ID のリストに変換。'mean' の場合は全カテゴリ。"""
        if cat == "mean":
            return self._cat_ids
        if isinstance(cat, str):
            if cat not in self._cat_name_to_id:
                raise ValueError(f"Unknown category name: {cat}. Available: {list(self._cat_name_to_id.keys())}")
            cat_id = self._cat_name_to_id[cat]
        else:
            cat_id = int(cat)
        if cat_id not in self._cat_ids:
            raise ValueError(f"Category {cat_id} not in evaluation. Available: {self._cat_ids}")
        return [cat_id]

    def _resolve_indices(
        self,
        cat: Union[str, int],
        iou: float,
        area: str = "all",
        gt_policy_key: str = "all",
        policy_key: str = "all",
        maxdets: int = 100,
    ) -> tuple[list[int], int, int, int, int, int]:
        """
        cat, iou, area, gt_policy_key, policy_key, maxdets からインデックスを解決。
        Returns: (k_indices, t_idx, a_idx, y_idx, z_idx, m_idx)
        """
        cat_ids = self._resolve_cat(cat)
        k_indices = [self._cat_ids.index(cid) for cid in cat_ids]

        t_idx = int(np.argmin(np.abs(self._iou_thrs - iou)))
        a_idx = self._area_rng_lbl.index(area) if area in self._area_rng_lbl else 0
        y_idx = self._gt_filter_policy_lbl.index(gt_policy_key) if gt_policy_key in self._gt_filter_policy_lbl else 0
        z_idx = self._filter_policy_lbl.index(policy_key) if policy_key in self._filter_policy_lbl else 0
        m_idx = self._max_dets.index(maxdets) if maxdets in self._max_dets else len(self._max_dets) - 1

        return k_indices, t_idx, a_idx, y_idx, z_idx, m_idx

    def _get_pr_curve(
        self,
        k_indices: list[int],
        t_idx: int,
        a_idx: int,
        y_idx: int,
        z_idx: int,
        m_idx: int,
        mean_over_cats: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        precision, recall, scores の 1D 配列を返す。
        mean_over_cats=True のとき、カテゴリで平均する。
        """
        precisions_list = []
        recalls_list = []
        scores_list = []

        for k_idx in k_indices:
            prec = self._precision[t_idx, :, k_idx, a_idx, y_idx, z_idx, m_idx]
            rec = self._rec_thrs  # recall thresholds が x 軸
            sc = self._scores[t_idx, :, k_idx, a_idx, y_idx, z_idx, m_idx]

            valid = (prec >= 0) & (rec >= 0)
            prec = prec[valid]
            rec = rec[valid]
            sc = sc[valid]

            if len(prec) > 0:
                precisions_list.append(prec)
                recalls_list.append(rec)
                scores_list.append(sc)

        if not precisions_list:
            return np.array([]), np.array([]), np.array([])

        if mean_over_cats and len(precisions_list) > 1:
            # 共通の recall ポイントで補間して平均
            rec_common = self._rec_thrs
            prec_interp = []
            sc_interp = []
            for prec, rec, sc in zip(precisions_list, recalls_list, scores_list):
                if len(rec) < 2:
                    continue
                p_interp = np.interp(rec_common, rec, prec)
                s_interp = np.interp(rec_common, rec, sc)
                prec_interp.append(p_interp)
                sc_interp.append(s_interp)
            if not prec_interp:
                return np.array([]), np.array([]), np.array([])
            prec = np.mean(prec_interp, axis=0)
            sc = np.mean(sc_interp, axis=0)
            rec = rec_common
            valid = prec >= 0  # -1 は無効
            return prec[valid], rec[valid], sc[valid]
        else:
            # 単一カテゴリ or 連結
            prec = np.concatenate(precisions_list)
            rec = np.concatenate(recalls_list)
            sc = np.concatenate(scores_list)
            return prec, rec, sc

    def get_metrics_at_best_f1(
        self,
        cat: Union[str, int] = "person",
        iou: float = 0.5,
        area: str = "all",
        gt_policy_key: str = "all",
        policy_key: str = "all",
        maxdets: int = 100,
    ) -> dict[str, Any]:
        """
        F1 が最大となる点の precision, recall, f1, score などを辞書で返す。

        Returns:
            {
                "precision": float,
                "recall": float,
                "f1": float,
                "score": float,
                "cat": str|int,
                "iou": float,
                "area": str,
                "gt_policy_key": str,
                "policy_key": str,
                "maxdets": int,
            }
        """
        k_indices, t_idx, a_idx, y_idx, z_idx, m_idx = self._resolve_indices(
            cat, iou, area, gt_policy_key, policy_key, maxdets
        )
        mean_over_cats = cat == "mean"
        prec, rec, sc = self._get_pr_curve(k_indices, t_idx, a_idx, y_idx, z_idx, m_idx, mean_over_cats)

        if len(prec) == 0 or len(rec) == 0:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "score": 0.0,
                "cat": cat,
                "iou": iou,
                "area": area,
                "gt_policy_key": gt_policy_key,
                "policy_key": policy_key,
                "maxdets": maxdets,
            }

        f1_scores = np.array([_compute_f1(p, r) for p, r in zip(prec, rec)])
        best_idx = int(np.argmax(f1_scores))

        return {
            "precision": float(prec[best_idx]),
            "recall": float(rec[best_idx]),
            "f1": float(f1_scores[best_idx]),
            "score": float(sc[best_idx]),
            "cat": cat,
            "iou": iou,
            "area": area,
            "gt_policy_key": gt_policy_key,
            "policy_key": policy_key,
            "maxdets": maxdets,
        }

    def get_metrics_at_fixed_score(
        self,
        score: float,
        cat: Union[str, int] = "mean",
        iou: float = 0.5,
        area: str = "all",
        gt_policy_key: str = "all",
        policy_key: str = "all",
        maxdets: int = 100,
    ) -> dict[str, Any]:
        """
        指定した confidence score 以上の点の中で、
        recall が最大となる idx の precision, recall, f1, score を辞書で返す。
        """
        k_indices, t_idx, a_idx, y_idx, z_idx, m_idx = self._resolve_indices(
            cat, iou, area, gt_policy_key, policy_key, maxdets
        )
        mean_over_cats = cat == "mean"
        prec, rec, sc = self._get_pr_curve(k_indices, t_idx, a_idx, y_idx, z_idx, m_idx, mean_over_cats)

        if len(prec) == 0 or len(sc) == 0:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "score": score,
                "cat": cat,
                "iou": iou,
                "area": area,
                "gt_policy_key": gt_policy_key,
                "policy_key": policy_key,
                "maxdets": maxdets,
            }

        # 指定スコア以上の点のみを対象にし、その中で recall が最大の idx を選ぶ
        mask = sc >= score
        if not np.any(mask):
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "score": score,
                "cat": cat,
                "iou": iou,
                "area": area,
                "gt_policy_key": gt_policy_key,
                "policy_key": policy_key,
                "maxdets": maxdets,
            }

        prec_valid = prec[mask]
        rec_valid = rec[mask]
        sc_valid = sc[mask]

        best_local_idx = int(np.argmax(rec_valid))
        p_sel = float(prec_valid[best_local_idx])
        r_sel = float(rec_valid[best_local_idx])
        s_sel = float(sc_valid[best_local_idx])

        return {
            "precision": p_sel,
            "recall": r_sel,
            "f1": float(_compute_f1(p_sel, r_sel)),
            "score": s_sel,
            "cat": cat,
            "iou": iou,
            "area": area,
            "gt_policy_key": gt_policy_key,
            "policy_key": policy_key,
            "maxdets": maxdets,
        }

    def get_metrics_at_fixed_recall(
        self,
        recall_target: float,
        cat: Union[str, int] = "mean",
        iou: float = 0.5,
        area: str = "all",
        gt_policy_key: str = "all",
        policy_key: str = "all",
        maxdets: int = 100,
    ) -> dict[str, Any]:
        """
        指定した recall における precision, f1, score を辞書で返す。
        """
        k_indices, t_idx, a_idx, y_idx, z_idx, m_idx = self._resolve_indices(
            cat, iou, area, gt_policy_key, policy_key, maxdets
        )
        mean_over_cats = cat == "mean"
        prec, rec, sc = self._get_pr_curve(k_indices, t_idx, a_idx, y_idx, z_idx, m_idx, mean_over_cats)

        if len(prec) == 0 or len(rec) == 0:
            return {
                "precision": 0.0,
                "recall": recall_target,
                "f1": 0.0,
                "score": 0.0,
                "cat": cat,
                "iou": iou,
                "area": area,
                "gt_policy_key": gt_policy_key,
                "policy_key": policy_key,
                "maxdets": maxdets,
            }

        p_interp = np.interp(recall_target, rec, prec)
        s_interp = np.interp(recall_target, rec, sc)

        return {
            "precision": float(p_interp),
            "recall": recall_target,
            "f1": float(_compute_f1(p_interp, recall_target)),
            "score": float(s_interp),
            "cat": cat,
            "iou": iou,
            "area": area,
            "gt_policy_key": gt_policy_key,
            "policy_key": policy_key,
            "maxdets": maxdets,
        }

    def get_ap(
        self,
        cat: Union[str, int] = "person",
        area: str = "all",
        iou: float = 0.5,
        gt_policy_key: str = "all",
        policy_key: str = "all",
        maxdets: int = 100,
    ) -> dict[str, Any]:
        """
        Average Precision (COCO 方式: recall 閾値での precision の平均) を辞書で返す。
        """
        k_indices, t_idx, a_idx, y_idx, z_idx, m_idx = self._resolve_indices(
            cat, iou, area, gt_policy_key, policy_key, maxdets
        )
        mean_over_cats = cat == "mean"
        prec, rec, _ = self._get_pr_curve(k_indices, t_idx, a_idx, y_idx, z_idx, m_idx, mean_over_cats)

        if len(prec) == 0:
            return {
                "ap": 0.0,
                "cat": cat,
                "area": area,
                "iou": iou,
                "gt_policy_key": gt_policy_key,
                "policy_key": policy_key,
                "maxdets": maxdets,
            }

        valid = prec >= 0
        ap = float(np.mean(prec[valid])) if np.any(valid) else 0.0

        return {
            "ap": ap,
            "cat": cat,
            "area": area,
            "iou": iou,
            "gt_policy_key": gt_policy_key,
            "policy_key": policy_key,
            "maxdets": maxdets,
        }

    def save_pr(
        self,
        cat: Union[str, int],
        iou: float,
        out: Union[str, Path],
        area: str = "all",
        gt_policy_key: str = "all",
        policy_key: str = "all",
        maxdets: int = 100,
    ) -> None:
        """
        Precision-Recall 曲線を CSV に保存。
        Columns: recall, precision, score, f1
        """
        k_indices, t_idx, a_idx, y_idx, z_idx, m_idx = self._resolve_indices(
            cat, iou, area, gt_policy_key, policy_key, maxdets
        )
        mean_over_cats = cat == "mean"
        prec, rec, sc = self._get_pr_curve(k_indices, t_idx, a_idx, y_idx, z_idx, m_idx, mean_over_cats)

        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        f1_arr = np.array([_compute_f1(p, r) for p, r in zip(prec, rec)])

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["recall", "precision", "f1", "score"])
            for r, p, f1, s in zip(rec, prec, f1_arr, sc):
                writer.writerow([r, p, f1, s])
