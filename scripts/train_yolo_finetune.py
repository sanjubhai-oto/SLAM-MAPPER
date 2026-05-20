#!/usr/bin/env python3
"""Fine-tune YOLOv11n on synthetic gz auto-labeled dataset.

Runs ~3h on M-series MPS. Output: state/weights/slamrover_v1.pt

Usage:
  /tmp/yoloenv/bin/python train_yolo_finetune.py
"""
import pathlib
import sys

from ultralytics import YOLO


ROOT = pathlib.Path.home() / "PX4-Autopilot/slam_rover"
DATA = ROOT / "datasets/gz_v1/data.yaml"
OUT  = ROOT / "state/weights"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    if not DATA.exists():
        print(f"[train] dataset missing: {DATA}")
        print(f"[train] run collect_yolo_dataset.py first")
        sys.exit(1)

    print(f"[train] loading base yolo11n.pt + fine-tuning on {DATA}")
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(DATA),
        epochs=20,
        imgsz=640,
        batch=8,
        device="mps",
        project=str(OUT),
        name="slamrover_v1",
        save=True,
        save_period=5,
        patience=8,
        lr0=0.001,
        warmup_epochs=2,
        freeze=10,           # freeze first 10 backbone layers
        exist_ok=True,
        verbose=True,
    )
    # Copy best.pt out
    best = OUT / "slamrover_v1/weights/best.pt"
    if best.exists():
        target = OUT / "slamrover_v1.pt"
        target.write_bytes(best.read_bytes())
        print(f"[train] DONE. best weights → {target}")
    else:
        print("[train] WARNING: best.pt missing")


if __name__ == "__main__":
    main()
