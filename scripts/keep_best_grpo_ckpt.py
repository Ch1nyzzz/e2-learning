#!/usr/bin/env python3
"""Keep only the best-by-validation HF checkpoint of a verl GRPO run.

verl itself rotates full checkpoints (max_actor_ckpt_to_keep=1), so this
daemon's only job is to preserve the best one before rotation destroys it:
it tails the training log for `step:N - ... - val/success_rate:V` lines,
and whenever the best-scoring validated step has a checkpoint on disk
(global_step_N/actor/huggingface), copies that HF directory to
<ckpt_dir>/best_hf (atomically, via a .tmp directory) together with a
best_info.json recording the step and score.

Usage: keep_best_grpo_ckpt.py <training_log> <ckpt_dir> [poll_seconds]
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
from pathlib import Path

VAL_LINE = re.compile(r"step:(\d+) .*?val/success_rate:([0-9.]+)")


def save_finished(hf_dir: Path) -> bool:
    """True once the HF export looks complete and quiescent (no write in 30s)."""
    if not (hf_dir / "config.json").exists():
        return False
    if not any(hf_dir.glob("*.safetensors")):
        return False
    newest = max(p.stat().st_mtime for p in hf_dir.iterdir())
    return time.time() - newest > 30


def scan_log(log_path: Path) -> dict[int, float]:
    scores: dict[int, float] = {}
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return scores
    for match in VAL_LINE.finditer(text):
        scores[int(match.group(1))] = float(match.group(2))
    return scores


def main() -> int:
    log_path = Path(sys.argv[1])
    ckpt_dir = Path(sys.argv[2])
    poll = int(sys.argv[3]) if len(sys.argv) > 3 else 60

    best_dir = ckpt_dir / "best_hf"
    info_path = ckpt_dir / "best_info.json"
    copied_step = None
    if info_path.exists():
        copied_step = json.loads(info_path.read_text()).get("step")

    while True:
        scores = scan_log(log_path)
        if scores:
            best_step, best_score = max(scores.items(), key=lambda kv: (kv[1], kv[0]))
            src = ckpt_dir / f"global_step_{best_step}" / "actor" / "huggingface"
            if best_step != copied_step and best_step > 0 and save_finished(src):
                tmp = ckpt_dir / "best_hf.tmp"
                shutil.rmtree(tmp, ignore_errors=True)
                shutil.copytree(src, tmp)
                shutil.rmtree(best_dir, ignore_errors=True)
                tmp.rename(best_dir)
                info_path.write_text(
                    json.dumps(
                        {
                            "step": best_step,
                            "val_success_rate": best_score,
                            "copied_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "all_val_scores": scores,
                        },
                        indent=2,
                    )
                )
                copied_step = best_step
                print(
                    f"[keep-best] copied step {best_step} (val {best_score:.3f}) -> {best_dir}",
                    flush=True,
                )
        time.sleep(poll)


if __name__ == "__main__":
    raise SystemExit(main())
