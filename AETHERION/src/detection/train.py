# ================================================================
# AETHERION
# Aerial Emergency Threat and Hazard Response
# Intelligence Operations Network
#
# "Guardian of the skies. Protector of lives."
#
# Author : P. Shiva Charan Reddy
# Type   : Personal Project
# Year   : 2026
# GitHub : github.com/ShivaCharanReddy/AETHERION
# ================================================================
"""
AETHERION — Multi-Threat YOLOv8/v11 Training
Trains model to detect all 11 beach threat classes.

Classes:
  0: normal_swim       5: rip_current
  1: panic_drowning    6: jellyfish_swarm
  2: unconscious       7: heatstroke
  3: submerged         8: fight_assault
  4: shark_attack      9: net_entrapment
                      10: monsoon_surge

Usage:
  python src/detection/train.py --config configs/model_config.yaml
"""

import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO


ALL_CLASSES = [
    'normal_swim', 'panic_drowning', 'unconscious', 'submerged',
    'shark_attack', 'rip_current', 'jellyfish_swarm', 'heatstroke',
    'fight_assault', 'net_entrapment', 'monsoon_surge',
]


def generate_dataset_yaml(dataset_path: str, output: str = 'configs/dataset.yaml'):
    config = {
        'path':  dataset_path,
        'train': 'train/images',
        'val':   'valid/images',
        'test':  'test/images',
        'nc':    len(ALL_CLASSES),
        'names': ALL_CLASSES,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Dataset YAML written to {output}")
    return output


def train(config: dict):
    model_name = config.get('model', 'yolov8s.pt')
    print(f"[AETHERION] Loading {model_name}")
    model = YOLO(model_name)

    results = model.train(
        data       = config.get('dataset_yaml', 'configs/dataset.yaml'),
        epochs     = config.get('epochs', 100),
        imgsz      = config.get('imgsz', 640),
        batch      = config.get('batch', 16),
        name       = config.get('run_name', 'aetherion'),
        project    = config.get('project_dir', 'runs'),
        patience   = config.get('patience', 20),
        optimizer  = 'AdamW',
        lr0        = 0.001,
        lrf        = 0.01,

        # Aquatic / beach augmentation
        hsv_h      = 0.02,    # water hue shifts
        hsv_s      = 0.75,
        hsv_v      = 0.45,
        fliplr     = 0.5,
        flipud     = 0.1,
        mosaic     = 1.0,
        degrees    = 12.0,    # drone angle variation
        translate  = 0.12,
        scale      = 0.55,
        shear      = 2.5,
        perspective= 0.0005,  # aerial perspective

        # Class weights — boost rare/critical classes
        # (set via loss weighting in custom callback if needed)

        device     = config.get('device', 0),
        workers    = config.get('workers', 4),
        save       = True,
        save_period= 10,
        verbose    = True,
    )

    best = Path(config.get('project_dir','runs')) / config.get('run_name','aetherion') / 'weights' / 'best.pt'
    print(f"\n[AETHERION] Training complete!")
    print(f"[AETHERION] Best model: {best}")
    try:
        metrics = results.results_dict
        print(f"[AETHERION] mAP@0.5:     {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
        print(f"[AETHERION] mAP@0.5:0.95:{metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
        print(f"[AETHERION] Precision:   {metrics.get('metrics/precision(B)', 'N/A'):.4f}")
        print(f"[AETHERION] Recall:      {metrics.get('metrics/recall(B)', 'N/A'):.4f}")
    except Exception:
        pass
    return results, str(best)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/model_config.yaml')
    p.add_argument('--model', default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--dataset', default=None)
    p.add_argument('--gen-dataset-yaml', default=None,
                   help='Auto-generate dataset.yaml from path')
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.model:   config['model'] = args.model
    if args.epochs:  config['epochs'] = args.epochs
    if args.dataset: config['dataset_yaml'] = args.dataset

    if args.gen_dataset_yaml:
        generate_dataset_yaml(args.gen_dataset_yaml)

    train(config)


if __name__ == '__main__':
    main()
