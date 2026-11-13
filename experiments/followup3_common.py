"""Frozen study-1 detectors for follow-up 3 (FOLLOWUP3_PREREG.md, section 1).

Recomputes the confirmation study's frozen calibration exactly as ``run_confirm.py`` does (z-standardisation over study-1 training
negatives, B3 reference gains G0 from study-1 training C0 negatives, 5%-FPR thresholds on study-1 validation negatives) and asserts that
the thresholds equal those stored in ``results/confirm/summary.json``. Nothing here is fitted on follow-up-3 data.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import score_detect as sd  # noqa: E402
from aptg.baselines import net_influence_B1, symmetric_mi_B2  # noqa: E402
from aptg.edges import predictive_gain, robust_z  # noqa: E402
from aptg.eval import threshold_at_fpr  # noqa: E402

PAIRS = [(i, j) for i in range(3) for j in range(3) if i != j]
SINGLES = ("B1", "B2", "B3")
HELD_OUT_MECH = ("alternating", "hidden_channel")


def _fin(M):
    v = np.asarray(M, float)
    v = v[np.isfinite(v)]
    return float(v.max()) if v.size else 0.0


def raw_features(rows):
    """Per-episode detector inputs (same calls as run_confirm.process, without the unused B4)."""
    ep = sorted(rows, key=lambda r: (r["t"], r["agent"]))
    return {"B1": _fin(net_influence_B1(ep)), "B2": _fin(symmetric_mi_B2(ep)),
            "G_def": [float(predictive_gain(ep, i, j)) for i, j in PAIRS]}


class Frozen:
    def __init__(self, root="."):
        root = Path(root)
        warnings.simplefilter("ignore")
        rows = sd.load(root / "results/detect/raw.jsonl")
        sd.attach_windows(rows, root / "results/detect/windows.jsonl")
        rows = sd.compute_scores(rows)
        b4 = json.load(open(root / "results/followup/b4.json"))
        rows = [r for r in rows if r["episode"] in b4]
        for r in rows:
            r["s"]["B4"] = b4[r["episode"]]
        train = [r for r in rows if r["split"] == "train"]
        ref = [r for r in train if not r["label"] and r["condition"] == "C0"]
        self.G0 = np.concatenate([r["G_def"] for r in ref])
        neg = [r for r in train if not r["label"]]
        self.zst = {}
        for k in ("B1", "B2", "B3", "B4"):
            v = np.array([r["s"][k] for r in neg]); med = np.median(v)
            self.zst[k] = (med, 1.4826 * np.median(np.abs(v - med)) or 1.0)
        for r in rows:
            z = {k: (r["s"][k] - self.zst[k][0]) / self.zst[k][1] for k in self.zst}
            r["s"]["U3"] = max(z["B1"], z["B2"], z["B3"])
            r["z"] = [z[k] for k in SINGLES]
        self.study1 = rows
        val = sd.split(rows, "val")
        self.thr = {d: threshold_at_fpr(*sd.arr(val, d), 0.05) for d in ("B1", "B2", "B3", "U3")}
        stored = json.load(open(root / "results/confirm/summary.json"))["thr"]
        for d, v in self.thr.items():
            if not np.isclose(v, stored[d], rtol=0, atol=1e-12):
                raise RuntimeError(f"frozen threshold mismatch for {d}: {v} vs stored {stored[d]}")
        # E3 combiners, fitted/thresholded on study-1 data only
        yv = np.array([int(r["label"]) for r in val])
        Zv = np.array([r["z"] for r in val])
        self.thr_bonf = {k: threshold_at_fpr(yv, Zv[:, n], 0.05 / 3) for n, k in enumerate(SINGLES)}
        from sklearn.linear_model import LogisticRegression
        tr = [r for r in train if r["mechanism"] not in HELD_OUT_MECH]
        self.lr = LogisticRegression().fit(np.array([r["z"] for r in tr]), np.array([int(r["label"]) for r in tr]))
        self.lr_n_train = (len(tr), int(sum(r["label"] for r in tr)))
        self.thr["LR"] = threshold_at_fpr(yv, self.lr.decision_function(Zv), 0.05)

    def scores(self, feat):
        """Detector scores from raw_features output: B1, B2, B3, U3, z-scores, LR score, Bonferroni-OR decision."""
        s = {"B1": feat["B1"], "B2": feat["B2"], "B3": max(robust_z(g, self.G0) for g in feat["G_def"])}
        z = [(s[k] - self.zst[k][0]) / self.zst[k][1] for k in SINGLES]
        s["U3"] = max(z)
        s["LR"] = float(self.lr.decision_function(np.array([z]))[0])
        s["BONF"] = float(any(zk > self.thr_bonf[k] for zk, k in zip(z, SINGLES)))
        s["z"] = z
        return s
