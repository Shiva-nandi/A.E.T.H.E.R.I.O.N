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
AETHERION — Multi-Threat Classifier & Severity Scorer
Classifies all detected threats and assigns severity scores
for smart alert escalation.

Threat Priority Matrix:
┌─────────────────────┬──────────────┬───────────┬──────────────────────────┐
│ Threat              │ Detection    │ Severity  │ Responders               │
├─────────────────────┼──────────────┼───────────┼──────────────────────────┤
│ Shark Attack        │ Blood+thrash │ CRITICAL  │ PS + Hospital + CoastGrd │
│ Unconscious Drown   │ Face-down    │ CRITICAL  │ PS + Hospital            │
│ Panic Drowning      │ Vertical+arm │ HIGH      │ Lifeguards               │
│ Rip Current Cluster │ Drift pattern│ MEDIUM    │ Lifeguards + Public      │
│ Heatstroke          │ Collapsed    │ MEDIUM    │ Lifeguards + Hospital    │
│ Fight/Assault       │ Stagger+conf │ MEDIUM    │ Lifeguards + PS          │
│ Net Entrapment      │ Irregular    │ HIGH      │ Lifeguards               │
│ Jellyfish Swarm     │ Cluster+react│ LOW       │ Public Alert             │
│ Monsoon Surge       │ Wave pattern │ MEDIUM    │ Public Alert + Lifeguard │
└─────────────────────┴──────────────┴───────────┴──────────────────────────┘
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict
import time
import numpy as np


class Severity(Enum):
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4


class ThreatClass(Enum):
    # Water / Drowning threats
    NORMAL_SWIM         = "normal_swim"
    PANIC_DROWNING      = "panic_drowning"
    UNCONSCIOUS         = "unconscious"
    RIP_CURRENT         = "rip_current"
    NET_ENTRAPMENT      = "net_entrapment"
    SUBMERGED           = "submerged"

    # Marine threats
    SHARK_ATTACK        = "shark_attack"
    JELLYFISH_SWARM     = "jellyfish_swarm"

    # Environmental
    MONSOON_SURGE       = "monsoon_surge"
    HIDDEN_HAZARD       = "hidden_hazard"

    # Human/Medical
    HEATSTROKE          = "heatstroke"
    FIGHT_ASSAULT       = "fight_assault"

    # Unknown
    UNKNOWN             = "unknown"


# Maps ThreatClass → (Severity, responders, alert_template)
THREAT_CONFIG: Dict[ThreatClass, dict] = {
    ThreatClass.SHARK_ATTACK: {
        "severity": Severity.CRITICAL,
        "responders": ["police_station", "hospital", "coast_guard", "lifeguard"],
        "template": "🦈 SHARK ATTACK CONFIRMED at {gps}. Blood in water, victim thrashing. IMMEDIATE response required.",
        "conf_threshold": 0.80,
        "multiframe_confirm_sec": 3.0,
    },
    ThreatClass.UNCONSCIOUS: {
        "severity": Severity.CRITICAL,
        "responders": ["police_station", "hospital", "coast_guard"],
        "template": "🚨 CRITICAL: Unconscious swimmer at {gps}. Face-down, motionless >15s. Drone arriving in {eta}s with buoy.",
        "conf_threshold": 0.85,
        "multiframe_confirm_sec": 15.0,
    },
    ThreatClass.SUBMERGED: {
        "severity": Severity.CRITICAL,
        "responders": ["police_station", "hospital"],
        "template": "🚨 CRITICAL: Swimmer submerged at {gps}. No head above water >15s. ETA needed.",
        "conf_threshold": 0.85,
        "multiframe_confirm_sec": 10.0,
    },
    ThreatClass.PANIC_DROWNING: {
        "severity": Severity.HIGH,
        "responders": ["lifeguard"],
        "template": "⚠ HIGH: Distress swimmer at {gps}. Vertical struggle, no forward progress. Approach now.",
        "conf_threshold": 0.75,
        "multiframe_confirm_sec": 5.0,
    },
    ThreatClass.NET_ENTRAPMENT: {
        "severity": Severity.HIGH,
        "responders": ["lifeguard", "coast_guard"],
        "template": "⚠ HIGH: Net entrapment at {gps}. Swimmer tangled, irregular struggling near boat/gear.",
        "conf_threshold": 0.75,
        "multiframe_confirm_sec": 5.0,
    },
    ThreatClass.HEATSTROKE: {
        "severity": Severity.MEDIUM,
        "responders": ["lifeguard", "hospital"],
        "template": "🌡 MEDIUM: Suspected heatstroke at {gps}. Collapsed motionless adult on beach. Medical attention needed.",
        "conf_threshold": 0.70,
        "multiframe_confirm_sec": 8.0,
    },
    ThreatClass.FIGHT_ASSAULT: {
        "severity": Severity.MEDIUM,
        "responders": ["lifeguard", "police_station"],
        "template": "⚠ MEDIUM: Altercation detected at {gps}. Staggering and confrontation observed.",
        "conf_threshold": 0.70,
        "multiframe_confirm_sec": 4.0,
    },
    ThreatClass.RIP_CURRENT: {
        "severity": Severity.MEDIUM,
        "responders": ["lifeguard", "public_alert"],
        "template": "🌊 MEDIUM: Rip current cluster at {gps}. Multiple swimmers drifting seaward. Public warning issued.",
        "conf_threshold": 0.65,
        "multiframe_confirm_sec": 10.0,
    },
    ThreatClass.MONSOON_SURGE: {
        "severity": Severity.MEDIUM,
        "responders": ["lifeguard", "public_alert"],
        "template": "🌧 MEDIUM: Dangerous wave surge detected at {gps}. Wave height exceeds safe limit. Beach evacuation advised.",
        "conf_threshold": 0.70,
        "multiframe_confirm_sec": 5.0,
    },
    ThreatClass.JELLYFISH_SWARM: {
        "severity": Severity.LOW,
        "responders": ["public_alert"],
        "template": "🪼 LOW: Jellyfish swarm detected at {gps}. Swimmers advised to exit water.",
        "conf_threshold": 0.60,
        "multiframe_confirm_sec": 0.0,
    },
    ThreatClass.NORMAL_SWIM: {
        "severity": Severity.LOW,
        "responders": [],
        "template": "",
        "conf_threshold": 0.50,
        "multiframe_confirm_sec": 0.0,
    },
}


@dataclass
class Detection:
    """Single YOLO detection from one frame."""
    threat_class: ThreatClass
    confidence: float
    bbox: List[float]           # [x1, y1, x2, y2]
    gps: tuple = (0.0, 0.0)
    timestamp: float = field(default_factory=time.time)
    frame_id: int = 0


@dataclass
class ThreatEvent:
    """A confirmed multi-frame threat event."""
    threat_class: ThreatClass
    severity: Severity
    confidence: float
    gps: tuple
    first_seen: float
    last_seen: float
    frame_count: int
    confirmed: bool = False
    alert_sent: bool = False
    responders: List[str] = field(default_factory=list)
    eta_drone_sec: float = 0.0

    @property
    def duration_sec(self) -> float:
        return self.last_seen - self.first_seen

    def to_dict(self) -> dict:
        return {
            "threat": self.threat_class.value,
            "severity": self.severity.name,
            "confidence": round(self.confidence, 3),
            "gps": self.gps,
            "duration_sec": round(self.duration_sec, 1),
            "confirmed": self.confirmed,
            "responders": self.responders,
        }


class ThreatClassifier:
    """
    Processes YOLO detections across frames.
    Applies multi-frame confirmation logic to reduce false positives.
    Scores severity and determines responder escalation.
    """

    def __init__(self, drone_speed_ms: float = 5.0):
        self.drone_speed_ms = drone_speed_ms
        self._active_events: Dict[str, ThreatEvent] = {}
        self._confirmed_events: List[ThreatEvent] = []

    def process_detections(self, detections: List[Detection],
                           drone_pos: tuple = (0.0, 0.0)) -> List[ThreatEvent]:
        """
        Process a batch of detections from one frame.
        Returns list of newly confirmed threat events.
        """
        newly_confirmed = []
        seen_keys = set()

        for det in detections:
            cfg = THREAT_CONFIG.get(det.threat_class)
            if not cfg or det.threat_class == ThreatClass.NORMAL_SWIM:
                continue
            if det.confidence < cfg["conf_threshold"]:
                continue

            key = f"{det.threat_class.value}_{int(det.gps[0]*1000)}_{int(det.gps[1]*1000)}"
            seen_keys.add(key)

            if key in self._active_events:
                event = self._active_events[key]
                event.last_seen = det.timestamp
                event.frame_count += 1
                event.confidence = max(event.confidence, det.confidence)

                # Check multi-frame confirmation
                if not event.confirmed:
                    if event.duration_sec >= cfg["multiframe_confirm_sec"]:
                        event.confirmed = True
                        event.eta_drone_sec = self._estimate_eta(drone_pos, det.gps)
                        newly_confirmed.append(event)
                        self._confirmed_events.append(event)
            else:
                # New detection
                self._active_events[key] = ThreatEvent(
                    threat_class=det.threat_class,
                    severity=cfg["severity"],
                    confidence=det.confidence,
                    gps=det.gps,
                    first_seen=det.timestamp,
                    last_seen=det.timestamp,
                    frame_count=1,
                    responders=cfg["responders"],
                )

        # Expire stale events (not seen for >5s)
        expired = [k for k, v in self._active_events.items()
                   if k not in seen_keys and (time.time() - v.last_seen) > 5.0]
        for k in expired:
            del self._active_events[k]

        return newly_confirmed

    def _estimate_eta(self, drone_pos: tuple, victim_pos: tuple) -> float:
        """Rough ETA in seconds at cruise speed."""
        import math
        R = 6371000.0
        dlat = math.radians(victim_pos[0] - drone_pos[0])
        dlon = math.radians(victim_pos[1] - drone_pos[1])
        dist = math.sqrt((dlat*R)**2 + (dlon*R*math.cos(math.radians(drone_pos[0])))**2)
        return round(dist / self.drone_speed_ms, 1)

    def get_active_events(self) -> List[ThreatEvent]:
        return list(self._active_events.values())

    def get_confirmed_history(self) -> List[ThreatEvent]:
        return self._confirmed_events.copy()

    def get_highest_severity(self) -> Optional[ThreatEvent]:
        active = [e for e in self._active_events.values() if e.confirmed]
        if not active:
            return None
        return max(active, key=lambda e: e.severity.value)
