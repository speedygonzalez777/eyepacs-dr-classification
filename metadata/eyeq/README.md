# EyeQ quality labels

`Label_EyeQ_train.csv` — manual image quality grades for 12,543 images of the
Kaggle EyePACS **train** set (all of them are in our `trainLabels.csv`).

- Columns: `image` (with `.jpeg`), `quality` (0 = Good, 1 = Usable, 2 = Reject), `DR_grade` (same as EyePACS `level`).
- Source: https://github.com/HzFu/EyeQ, file `data/Label_EyeQ_train.csv`,
  commit `1dcfabb08cd31045de1a4a3ee82e48f6833697aa`.
- SHA-256: `34f39f8923f62a13fcaeccef7e5da7ec607b45805e7736042fa274b2731b9480`
- License: CC BY-NC-SA 4.0 (non-commercial use only).

Citation:

> H. Fu, B. Wang, J. Shen, S. Cui, Y. Xu, J. Liu, L. Shao,
> "Evaluation of Retinal Image Quality Assessment Networks in Different Color-spaces", MICCAI 2019.
> https://arxiv.org/abs/1907.05345
