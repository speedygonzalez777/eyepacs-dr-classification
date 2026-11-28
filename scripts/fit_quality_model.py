"""Fit an image-quality classifier on EyeQ labels and score every image.

Input: DenseNet121 features from scripts/score_quality.py (QuickQual feature
extractor) and the EyeQ manual grades in metadata/eyeq/.

Model: standardized features -> logistic regression, target = EyeQ "Reject".
  - For EyeQ-graded images p_reject is the out-of-fold prediction from a
    5-fold patient-grouped cross-validation (never scored by a model that saw them).
  - For ungraded images p_reject comes from a model fitted on all graded images.

Output: <quality_dir>/quality_scores.csv
    image, eyeq_quality, p_reject, p_reject_source (cv | model), quickqual_meme_p_bad

Example:
    python scripts/fit_quality_model.py --labels ~/data/eyepacs/raw/trainLabels.csv \
        --quality-dir ~/data/eyepacs/quality
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EYEQ_CSV = PROJECT_ROOT / "metadata" / "eyeq" / "Label_EyeQ_train.csv"
EYEQ_NAMES = {0: "Good", 1: "Usable", 2: "Reject"}

C = 0.01
N_FOLDS = 5


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, required=True, help="original trainLabels.csv")
    parser.add_argument("--quality-dir", type=Path, required=True, help="output folder of score_quality.py")
    args = parser.parse_args()

    qdir = args.quality_dir.expanduser()
    features = np.load(qdir / "densenet121_features.npy").astype(np.float32)
    index = pd.read_csv(qdir / "densenet121_features_index.csv")
    meme = pd.read_csv(qdir / "quickqual_meme.csv")
    if not (index["image"].equals(meme["image"])):
        raise ValueError("feature index and quickqual_meme.csv are not aligned")

    labels = pd.read_csv(args.labels.expanduser())
    if set(labels["image"]) != set(index["image"]):
        raise ValueError("features do not cover exactly the images in --labels")

    eyeq = pd.read_csv(EYEQ_CSV, index_col=0)
    eyeq["image"] = eyeq["image"].str.replace(".jpeg", "", regex=False)
    df = index.merge(eyeq[["image", "quality"]], on="image", how="left")
    df["eyeq_quality"] = df["quality"].map(EYEQ_NAMES).fillna("")
    df["patient_id"] = df["image"].str.replace(r"_(left|right)$", "", regex=True)

    graded = (df["eyeq_quality"] != "").to_numpy()
    y = (df.loc[graded, "eyeq_quality"] == "Reject").astype(int).to_numpy()

    oof = cross_val_predict(make_model(), features[graded], y,
                            groups=df.loc[graded, "patient_id"], cv=GroupKFold(N_FOLDS),
                            method="predict_proba")[:, 1]
    full = make_model().fit(features[graded], y)

    p_reject = full.predict_proba(features)[:, 1]
    p_reject[graded] = oof
    df["p_reject"] = p_reject
    df["p_reject_source"] = np.where(graded, "cv", "model")
    df["quickqual_meme_p_bad"] = meme["quickqual_meme_p_bad"]

    out = qdir / "quality_scores.csv"
    df[["image", "eyeq_quality", "p_reject", "p_reject_source", "quickqual_meme_p_bad"]].to_csv(
        out, index=False, lineterminator="\n")

    print(f"EyeQ-graded images: {graded.sum()}, Reject: {y.sum()}")
    print(f"CV ROC-AUC Reject vs rest: logistic regression {roc_auc_score(y, oof):.4f}, "
          f"QuickQual-MEME {roc_auc_score(y, meme.loc[graded, 'quickqual_meme_p_bad']):.4f}")
    for t in (0.3, 0.5, 0.7):
        tn, fp, fn, tp = confusion_matrix(y, oof >= t).ravel()
        print(f"  threshold {t}: flags {(oof >= t).mean():.1%}, "
              f"recall {tp / (tp + fn):.2f}, precision {tp / (tp + fp):.2f}")
    print(f"ungraded images flagged at 0.5: {(p_reject[~graded] >= 0.5).mean():.1%}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
