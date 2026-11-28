"""Manual keep/reject review of images flagged by the quality model, in a simple window.

Shows images whose quality-model score p_reject falls in [--low, --high]
(by default: flagged by the model, p_reject >= 0.5, without an EyeQ grade,
DR levels 2-4), one at a time, in random order. Decisions are saved after every key press to
metadata/manual_review.csv (lives in git, used by make_manifest.py), so you
can quit any time and continue later.

By default the DR level and p_reject are hidden to keep the review blind;
use --show-info to display them.

Keys:
    K / Left arrow    keep
    R / Right arrow   reject
    U / Backspace     undo (go back one image)
    Q / Esc           save and quit

Example:
    python scripts/review_quality.py
"""

import argparse
import getpass
import os
import sys
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image, ImageTk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "metadata" / "manual_review.csv"
COLUMNS = ["image", "decision", "p_reject", "reviewer", "timestamp"]


def load_decisions(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, keep_default_na=False)
    return pd.DataFrame(columns=COLUMNS)


def save_decisions(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    df.sort_values("image").to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, path)


class ReviewApp:
    def __init__(self, root: tk.Tk, queue: pd.DataFrame, decisions: pd.DataFrame,
                 out_path: Path, images_dir: Path, reviewer: str, show_info: bool, size: int):
        self.root = root
        self.queue = queue.reset_index(drop=True)
        self.decisions = decisions
        self.out_path = out_path
        self.images_dir = images_dir
        self.reviewer = reviewer
        self.show_info = show_info
        self.size = size
        self.pos = 0
        self.history: list[int] = []

        root.title("EyePACS quality review")
        root.configure(bg="black")
        self.header = tk.Label(root, font=("Sans", 13), fg="white", bg="black")
        self.header.pack(pady=(8, 4))
        self.canvas = tk.Label(root, bg="black")
        self.canvas.pack()
        tk.Label(root, text="K / ←  keep      R / →  reject      U / Backspace  undo      Q / Esc  quit",
                 font=("Sans", 11), fg="#aaaaaa", bg="black").pack(pady=(4, 8))

        for key in ("k", "K", "<Left>"):
            root.bind(key, lambda e: self.decide("keep"))
        for key in ("r", "R", "<Right>"):
            root.bind(key, lambda e: self.decide("reject"))
        for key in ("u", "U", "<BackSpace>"):
            root.bind(key, lambda e: self.undo())
        for key in ("q", "Q", "<Escape>"):
            root.bind(key, lambda e: self.quit())
        root.protocol("WM_DELETE_WINDOW", self.quit)

        self.show()

    def show(self) -> None:
        if self.pos >= len(self.queue):
            self.header.config(text=f"Done — all {len(self.queue)} images reviewed. Press Q to quit.")
            self.canvas.config(image="")
            return
        row = self.queue.iloc[self.pos]
        image = Image.open(self.images_dir / f"{row['image']}.jpeg").convert("RGB")
        image = image.resize((self.size, self.size), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.config(image=self.photo)

        done = len(self.decisions[self.decisions["image"].isin(self.queue["image"])])
        text = f"{self.pos + 1} / {len(self.queue)}   (decided: {done})   {row['image']}"
        if self.show_info:
            text += f"   |   class {row['level']}   p_reject={row['p_reject']:.2f}"
        self.header.config(text=text)

    def decide(self, decision: str) -> None:
        if self.pos >= len(self.queue):
            return
        row = self.queue.iloc[self.pos]
        self.decisions = self.decisions[self.decisions["image"] != row["image"]]
        new = pd.DataFrame([{
            "image": row["image"],
            "decision": decision,
            "p_reject": round(float(row["p_reject"]), 4),
            "reviewer": self.reviewer,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }])
        self.decisions = pd.concat([self.decisions, new], ignore_index=True)
        save_decisions(self.decisions, self.out_path)
        self.history.append(self.pos)
        self.pos += 1
        self.show()

    def undo(self) -> None:
        if not self.history:
            return
        self.pos = self.history.pop()
        image = self.queue.iloc[self.pos]["image"]
        self.decisions = self.decisions[self.decisions["image"] != image]
        save_decisions(self.decisions, self.out_path)
        self.show()

    def quit(self) -> None:
        save_decisions(self.decisions, self.out_path)
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--low", type=float, default=0.5, help="min p_reject (inclusive)")
    parser.add_argument("--high", type=float, default=1.0, help="max p_reject (inclusive)")
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 3, 4], help="DR levels to review")
    parser.add_argument("--include-eyeq", action="store_true",
                        help="also review images that already have a manual EyeQ grade")
    parser.add_argument("--reviewer", default=getpass.getuser(),
                        help="your name, stored with each decision (default: system user name)")
    parser.add_argument("--show-info", action="store_true", help="show DR level and p_reject")
    parser.add_argument("--scores", type=Path, default=Path("~/data/eyepacs/quality/quality_scores.csv"))
    parser.add_argument("--images", type=Path, default=Path("~/data/eyepacs/cache_768/images"))
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "metadata" / "manifest.csv")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--size", type=int, default=720, help="displayed image size in px")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    scores = pd.read_csv(args.scores.expanduser(), keep_default_na=False)
    manifest = pd.read_csv(args.manifest, keep_default_na=False)[["image", "level"]]
    df = scores.merge(manifest, on="image", validate="one_to_one")

    mask = (df["p_reject"] >= args.low) & (df["p_reject"] <= args.high) & df["level"].isin(args.levels)
    if not args.include_eyeq:
        mask &= df["eyeq_quality"] == ""
    selected = df[mask].sample(frac=1, random_state=args.seed)

    out_path = args.out.expanduser()
    decisions = load_decisions(out_path)
    queue = selected[~selected["image"].isin(decisions["image"])]

    print(f"selected: {len(selected)}, already decided: {len(selected) - len(queue)}, to review: {len(queue)}")
    print(f"decisions file: {out_path}")
    if queue.empty:
        sys.exit("nothing to review")

    root = tk.Tk()
    ReviewApp(root, queue, decisions, out_path, args.images.expanduser(), args.reviewer, args.show_info, args.size)
    root.mainloop()

    final = load_decisions(out_path)
    final = final[final["image"].isin(selected["image"])]
    print(f"reviewed in this range: {len(final)} / {len(selected)}")
    print(final["decision"].value_counts().to_string())


if __name__ == "__main__":
    main()
