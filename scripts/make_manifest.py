"""Build metadata/manifest.csv: one row per image with split and keep/exclude decision.

The manifest is the single source of truth for the cleaned dataset. It is small,
lives in git, and everyone builds the same cleaned dataset from it with
scripts/prepare_dataset.py.

- `split` is computed ONCE on the full dataset (src.splits.create_patient_level_split,
  seed 42), so the test set stays fixed no matter which images are later excluded.
  For the full dataset it is identical to the split the training scripts compute.
- `keep` / `exclude_reason` come from the rules below.

Rules (v2), applied per image in this order:
  1. EyeQ manual grade exists:      Reject -> exclude ("eyeq_reject"), Good/Usable -> keep
  2. no EyeQ grade, quality model p_reject < 0.5                     -> keep
  3. no EyeQ grade, p_reject >= 0.5, DR level 0-1                    -> exclude ("model_reject")
  4. no EyeQ grade, p_reject >= 0.5, DR level 2-4: manual review
     (metadata/manual_review.csv)    reject -> exclude ("manual_reject"), keep -> keep;
     not reviewed yet -> exclude ("model_reject_unreviewed") and a warning is printed

p_reject comes from scripts/fit_quality_model.py (logistic regression on QuickQual
DenseNet121 features, trained on EyeQ labels) and is stored in the manifest.

Example:
    python scripts/make_manifest.py --labels ~/data/eyepacs/raw/trainLabels.csv \
        --quality-scores ~/data/eyepacs/quality/quality_scores.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.splits import create_patient_level_split  # noqa: E402

MANIFEST_VERSION = "v2-eyeq-model-manual"
EYEQ_CSV = PROJECT_ROOT / "metadata" / "eyeq" / "Label_EyeQ_train.csv"
MANUAL_REVIEW_CSV = PROJECT_ROOT / "metadata" / "manual_review.csv"
DEFAULT_OUT = PROJECT_ROOT / "metadata" / "manifest.csv"
EYEQ_NAMES = {0: "Good", 1: "Usable", 2: "Reject"}

P_REJECT_THRESHOLD = 0.5
MANUAL_REVIEW_LEVELS = (2, 3, 4)  # model-flagged images in these levels are decided by a human

SPLIT_FRACTIONS = dict(train_size=0.70, val_size=0.15, test_size=0.15, random_seed=42)


def load_eyeq(path: Path) -> pd.DataFrame:
    eyeq = pd.read_csv(path, index_col=0)
    eyeq["image"] = eyeq["image"].str.replace(".jpeg", "", regex=False)
    eyeq["eyeq_quality"] = eyeq["quality"].map(EYEQ_NAMES)
    return eyeq[["image", "eyeq_quality", "DR_grade"]]


def load_manual_review(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["image", "manual_decision"])
    review = pd.read_csv(path, keep_default_na=False)
    if review["image"].duplicated().any():
        raise ValueError(f"duplicate images in {path}")
    if not review["decision"].isin(["keep", "reject"]).all():
        raise ValueError(f"unexpected decision values in {path}")
    return review[["image", "decision"]].rename(columns={"decision": "manual_decision"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, required=True, help="original trainLabels.csv")
    parser.add_argument("--quality-scores", type=Path, required=True,
                        help="quality_scores.csv from scripts/fit_quality_model.py")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    labels = pd.read_csv(args.labels.expanduser())
    df = create_patient_level_split(labels, **SPLIT_FRACTIONS)

    eyeq = load_eyeq(EYEQ_CSV)
    df = df.merge(eyeq, on="image", how="left", validate="one_to_one")
    graded = df["eyeq_quality"].notna()
    if (graded & (df["DR_grade"] != df["level"])).any():
        raise ValueError("EyeQ DR grade differs from trainLabels")
    df = df.drop(columns="DR_grade")
    df["eyeq_quality"] = df["eyeq_quality"].fillna("")

    scores = pd.read_csv(args.quality_scores.expanduser(), keep_default_na=False)[["image", "p_reject"]]
    df = df.merge(scores, on="image", how="left", validate="one_to_one")
    if df["p_reject"].isna().any():
        raise ValueError(f"{df['p_reject'].isna().sum()} images have no p_reject")

    review = load_manual_review(MANUAL_REVIEW_CSV)
    df = df.merge(review, on="image", how="left", validate="one_to_one")
    df["manual_decision"] = df["manual_decision"].fillna("")

    ungraded = df["eyeq_quality"] == ""
    flagged = ungraded & (df["p_reject"] >= P_REJECT_THRESHOLD)
    needs_review = flagged & df["level"].isin(MANUAL_REVIEW_LEVELS)

    df["exclude_reason"] = ""
    df.loc[df["eyeq_quality"] == "Reject", "exclude_reason"] = "eyeq_reject"
    df.loc[flagged & ~needs_review, "exclude_reason"] = "model_reject"
    df.loc[needs_review & (df["manual_decision"] == "reject"), "exclude_reason"] = "manual_reject"
    df.loc[needs_review & (df["manual_decision"] == ""), "exclude_reason"] = "model_reject_unreviewed"
    df["keep"] = (df["exclude_reason"] == "").astype(int)

    ignored = (df["manual_decision"] != "") & ~needs_review
    if ignored.any():
        print(f"note: {ignored.sum()} manual decisions are outside the review group and were ignored")

    df["p_reject"] = df["p_reject"].round(4)
    df = df[["image", "level", "binary_label", "patient_id", "side", "split",
             "eyeq_quality", "p_reject", "manual_decision", "keep", "exclude_reason"]]
    order = df["image"].str.extract(r"^(\d+)_")[0].astype(int) * 2 + df["image"].str.endswith("_right")
    df = df.iloc[order.argsort(kind="stable")].reset_index(drop=True)

    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, lineterminator="\n")

    print(f"manifest {MANIFEST_VERSION}: {out}")
    print(f"images: {len(df)}, kept: {df['keep'].sum()}, excluded: {(1 - df['keep']).sum()}\n")
    print(df["exclude_reason"].replace("", "(kept)").value_counts().to_string(), "\n")
    print(pd.crosstab(df["level"], df["keep"], margins=True), "\n")
    print(pd.crosstab(df["split"], [df["binary_label"], df["keep"]], margins=True))

    pending = (df["exclude_reason"] == "model_reject_unreviewed").sum()
    if pending:
        print(f"\nWARNING: {pending} model-flagged images in levels {MANUAL_REVIEW_LEVELS} are not reviewed yet "
              "and are excluded for now. Run scripts/review_quality.py.")


if __name__ == "__main__":
    main()
