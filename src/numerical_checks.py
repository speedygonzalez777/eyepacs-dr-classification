"""Batch-local diagnostics for full EfficientNet fine-tuning."""
import json
from pathlib import Path

import torch


class BatchChecks:
    def __init__(self, log_path: Path, phase: str, epoch: int | None, batch: int, images):
        self.log_path = log_path
        self.context = dict(phase=phase, epoch=epoch, batch=batch, images=list(images))

    def report(self, stage, *, fatal=True, **details):
        event = dict(self.context, stage=stage, **details)
        message = json.dumps(event, allow_nan=False)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")
        if fatal:
            raise FloatingPointError(message)
        print(message, flush=True)

    def tensor(self, stage, tensor, *, per_image=False):
        finite = torch.isfinite(tensor.detach())
        if finite.all():
            return
        bad_images = []
        if per_image:
            failed = ~finite.reshape(len(finite), -1).all(dim=1)
            bad_images = [self.context["images"][i] for i in failed.nonzero().flatten().tolist()]
        self.report(stage, affected_images=bad_images)

    def state(self, stage, named_tensors, *, fatal=True, **details):
        # Reduce each tensor on-device. Only fetch individual names on failure.
        items = [(name, value.detach()) for name, value in named_tensors if value is not None]
        if not items:
            return
        maxima = torch.stack(torch._foreach_norm([v for _, v in items], float("inf")))
        finite = torch.isfinite(maxima)
        if not finite.all():
            bad = [items[i] for i in (~finite).nonzero().flatten().tolist()]
            self.report(stage, nan_tensors=[k for k, v in bad if torch.isnan(v).any()],
                        inf_tensors=[k for k, v in bad if torch.isinf(v).any()],
                        fatal=fatal, **details)

    def batch_norm(self, model, stage="forward"):
        buffers = [(k, v) for k, v in model.named_buffers() if k.endswith(("running_mean", "running_var"))]
        self.state(f"{stage}.batch_norm_buffers", buffers)
        variances = [(k, v) for k, v in buffers if k.endswith("running_var")]
        negative = torch.stack([v.amin() for _, v in variances]) < 0
        if negative.any():
            bad = [variances[i][0] for i in negative.nonzero().flatten().tolist()]
            self.report(f"{stage}.negative_running_variance", tensors=bad)
