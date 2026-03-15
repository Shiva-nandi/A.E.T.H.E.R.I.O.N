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
AETHERION — Multi-Threat Real-time Inference Engine
Integrates YOLOv8/v11 detection with threat classification and smart alerts.

Usage:
  python src/detection/inference.py --source video.mp4 --model models/best.pt
"""

import argparse
import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from src.threats.classifier import (
    ThreatClassifier, ThreatClass, Detection, Severity, THREAT_CONFIG
)
from src.alerts.smart_alert_engine import SmartAlertEngine
from src.detection.preprocess import AquaticPreprocessor
from src.utils.logger import get_logger

logger = get_logger('inference')

# Visual style per threat class
THREAT_STYLES = {
    ThreatClass.NORMAL_SWIM:     {'color': (0, 200, 0),    'label': 'NORMAL'},
    ThreatClass.PANIC_DROWNING:  {'color': (0, 165, 255),  'label': 'DISTRESS'},
    ThreatClass.UNCONSCIOUS:     {'color': (0, 0, 255),    'label': 'UNCONSCIOUS'},
    ThreatClass.SUBMERGED:       {'color': (0, 0, 200),    'label': 'SUBMERGED'},
    ThreatClass.SHARK_ATTACK:    {'color': (0, 0, 180),    'label': 'SHARK'},
    ThreatClass.RIP_CURRENT:     {'color': (255, 140, 0),  'label': 'RIP CURRENT'},
    ThreatClass.JELLYFISH_SWARM: {'color': (255, 200, 0),  'label': 'JELLYFISH'},
    ThreatClass.HEATSTROKE:      {'color': (0, 120, 255),  'label': 'HEATSTROKE'},
    ThreatClass.FIGHT_ASSAULT:   {'color': (0, 80, 255),   'label': 'FIGHT'},
    ThreatClass.NET_ENTRAPMENT:  {'color': (180, 100, 0),  'label': 'NET TRAP'},
    ThreatClass.MONSOON_SURGE:   {'color': (255, 255, 0),  'label': 'WAVE SURGE'},
}

SEVERITY_COLORS = {
    Severity.LOW:      (0, 200, 0),
    Severity.MEDIUM:   (0, 165, 255),
    Severity.HIGH:     (0, 80, 255),
    Severity.CRITICAL: (0, 0, 255),
}

# Map YOLO class index → ThreatClass
# UPDATE THIS to match your trained model's class order
YOLO_CLASS_MAP = {
    0:  ThreatClass.NORMAL_SWIM,
    1:  ThreatClass.PANIC_DROWNING,
    2:  ThreatClass.UNCONSCIOUS,
    3:  ThreatClass.SUBMERGED,
    4:  ThreatClass.SHARK_ATTACK,
    5:  ThreatClass.RIP_CURRENT,
    6:  ThreatClass.JELLYFISH_SWARM,
    7:  ThreatClass.HEATSTROKE,
    8:  ThreatClass.FIGHT_ASSAULT,
    9:  ThreatClass.NET_ENTRAPMENT,
    10: ThreatClass.MONSOON_SURGE,
}


class MultiThreatDetector:
    """
    Full inference pipeline integrating:
    - YOLOv8/v11 multi-class detection
    - Aquatic preprocessing
    - Multi-frame threat confirmation
    - Smart escalating alert dispatch
    """

    def __init__(self, model_path: str, conf: float = 0.5,
                 alert_engine: SmartAlertEngine = None,
                 drone_gps: tuple = (17.4239, 78.4738)):
        self.model          = YOLO(model_path)
        self.conf           = conf
        self.alert_engine   = alert_engine
        self.drone_gps      = drone_gps
        self.preprocessor   = AquaticPreprocessor()
        self.classifier     = ThreatClassifier()
        self.frame_count    = 0
        self.latency_log    = []

    def process_frame(self, frame: np.ndarray,
                      victim_gps: tuple = None) -> dict:
        """Full pipeline on one frame. Returns detections + events."""
        t0 = time.time()

        preprocessed = self.preprocessor.process(frame)

        results = self.model(
            preprocessed, conf=self.conf,
            iou=0.45, verbose=False
        )[0]

        raw_detections = []
        gps = victim_gps or self.drone_gps

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            threat = YOLO_CLASS_MAP.get(cls_id, ThreatClass.UNKNOWN)
            raw_detections.append(Detection(
                threat_class=threat,
                confidence=conf,
                bbox=box.xyxy[0].tolist(),
                gps=gps,
                timestamp=time.time(),
                frame_id=self.frame_count,
            ))

        # Multi-frame confirmation
        confirmed_events = self.classifier.process_detections(
            raw_detections, drone_pos=self.drone_gps
        )

        # Dispatch alerts for confirmed events
        if self.alert_engine:
            for event in confirmed_events:
                is_beach = True  # Set from GPS zone logic
                self.alert_engine.dispatch(event, is_beach_zone=is_beach)

        latency = (time.time() - t0) * 1000
        self.latency_log.append(latency)
        self.frame_count += 1

        return {
            'frame_id':         self.frame_count,
            'latency_ms':       latency,
            'raw_detections':   raw_detections,
            'confirmed_events': confirmed_events,
            'active_events':    self.classifier.get_active_events(),
        }

    def annotate(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """Draw bounding boxes, severity bars, and HUD on frame."""
        out = frame.copy()
        active = self.classifier.get_active_events()
        highest = self.classifier.get_highest_severity()

        # Severity border flash
        if highest and highest.severity in (Severity.CRITICAL, Severity.HIGH):
            col = SEVERITY_COLORS[highest.severity]
            border_alpha = 0.5 + 0.5 * abs(
                np.sin(self.frame_count * 0.15)
            )
            overlay = out.copy()
            cv2.rectangle(overlay, (0, 0), (out.shape[1]-1, out.shape[0]-1), col, 8)
            cv2.addWeighted(overlay, border_alpha, out, 1 - border_alpha, 0, out)

        # Draw detections
        for det in result['raw_detections']:
            style = THREAT_STYLES.get(det.threat_class, {'color': (255,255,255), 'label': '?'})
            x1, y1, x2, y2 = map(int, det.bbox)
            color = style['color']

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Corner brackets
            cs = 10
            for cx, cy in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
                sx = 1 if cx == x1 else -1
                sy = 1 if cy == y1 else -1
                cv2.line(out, (cx, cy+sy*cs), (cx, cy), color, 2)
                cv2.line(out, (cx, cy), (cx+sx*cs, cy), color, 2)

            # Label bar
            label = f"{style['label']} {det.confidence:.0%}"
            lw, lh = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(out, (x1, y1-lh-8), (x1+lw+8, y1), color, -1)
            cv2.putText(out, label, (x1+4, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        # Active events panel (top-left)
        y_off = 30
        cv2.putText(out, f"AETHERION | {result['latency_ms']:.0f}ms | F:{self.frame_count}",
                    (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,212,255), 1)
        y_off += 18
        for ev in active[:5]:
            sev_col = SEVERITY_COLORS.get(ev.severity, (255,255,255))
            label = f"[{ev.severity.name}] {ev.threat_class.value} {ev.confidence:.0%} | {ev.duration_sec:.0f}s"
            cv2.putText(out, label, (10, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, sev_col, 1)
            y_off += 16

        # Critical alert banner
        if highest and highest.severity == Severity.CRITICAL:
            btext = f"CRITICAL: {highest.threat_class.value.upper().replace('_',' ')}"
            tw = cv2.getTextSize(btext, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0]
            bx = (out.shape[1] - tw) // 2
            cv2.rectangle(out, (bx-10, out.shape[0]-45), (bx+tw+10, out.shape[0]-15), (0,0,200), -1)
            cv2.putText(out, btext, (bx, out.shape[0]-22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        return out

    def get_stats(self) -> dict:
        if not self.latency_log:
            return {}
        return {
            'frames':      self.frame_count,
            'avg_lat_ms':  round(np.mean(self.latency_log), 1),
            'avg_fps':     round(1000 / np.mean(self.latency_log), 1),
            'alerts_sent': len(self.alert_engine.get_log()) if self.alert_engine else 0,
        }


def run(source: str, model_path: str, conf: float = 0.5,
        save: bool = True, alerts_config: str = 'configs/alerts_config.yaml'):

    alert_engine = SmartAlertEngine(config_path=alerts_config, hyderabad_mode=True)
    detector     = MultiThreatDetector(model_path, conf=conf, alert_engine=alert_engine)

    src = 0 if source == '0' else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise ValueError(f"Cannot open: {source}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = None
    if save:
        writer = cv2.VideoWriter('aetherion_output.mp4',
                                 cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    logger.info("AETHERION — Multi-Threat Inference running. Press q to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result    = detector.process_frame(frame)
            annotated = detector.annotate(frame, result)
            if writer:
                writer.write(annotated)
            cv2.imshow('AETHERION — Beach Safety AI', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        for k, v in detector.get_stats().items():
            logger.info(f"  {k}: {v}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--model', default='models/best.pt')
    p.add_argument('--conf', type=float, default=0.5)
    p.add_argument('--no-save', action='store_true')
    p.add_argument('--alerts-config', default='configs/alerts_config.yaml')
    args = p.parse_args()
    run(args.source, args.model, args.conf, not args.no_save, args.alerts_config)


if __name__ == '__main__':
    main()
