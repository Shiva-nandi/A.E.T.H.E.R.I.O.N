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
AETHERION — Smart Escalating Alert Engine
Tiered alert system that routes to the right responders based on severity.

Escalation tiers:
  CRITICAL  → Police Station + Hospital (ambulance) + Coast Guard + Lifeguard
  HIGH      → Lifeguard App + Drone Siren
  MEDIUM    → Lifeguard App + Public Alert
  LOW       → Public Alert only

Hyderabad-specific responder database included as fallback.
"""

import os
import smtplib
import time
import yaml
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional, List, Dict
from src.threats.classifier import ThreatEvent, Severity, ThreatClass, THREAT_CONFIG
from src.geolocation.responder_lookup import ResponderLookup
from src.utils.logger import get_logger

logger = get_logger('alert_engine')


class SmartAlertEngine:
    """
    Tiered alert engine. Routes each confirmed ThreatEvent to the
    correct responders based on severity and beach zone.

    Responder tiers:
      - lifeguard:      Firebase push + audible drone siren
      - police_station: Twilio SMS + voice call
      - hospital:       Twilio SMS (ambulance dispatch)
      - coast_guard:    Twilio SMS (beach/sea zone only)
      - public_alert:   Broadcast (app / PA system)
    """

    COOLDOWN_SEC = 60  # Don't re-alert same threat within 60s

    def __init__(self, config_path: str = 'configs/alerts_config.yaml',
                 hyderabad_mode: bool = True):
        self.config = self._load_config(config_path)
        self.hyderabad_mode = hyderabad_mode
        self._twilio = None
        self._firebase = None
        self._cooldowns: Dict[str, float] = {}
        self.alert_log: List[dict] = []
        self.responder_lookup = ResponderLookup(hyderabad_mode=hyderabad_mode)
        self._init_twilio()
        self._init_firebase()

    def _load_config(self, path: str) -> dict:
        if not os.path.exists(path):
            logger.warning(f"Alert config not found: {path} — alerts in log-only mode.")
            return {}
        with open(path) as f:
            return yaml.safe_load(f)

    def _init_twilio(self):
        twilio_cfg = self.config.get('twilio', {})
        sid   = twilio_cfg.get('account_sid', '')
        token = twilio_cfg.get('auth_token', '')
        if sid and token and 'YOUR' not in sid:
            try:
                from twilio.rest import Client
                self._twilio = Client(sid, token)
                logger.info("Twilio initialized.")
            except ImportError:
                logger.warning("twilio not installed. pip install twilio")

    def _init_firebase(self):
        fb_cfg = self.config.get('firebase', {})
        if fb_cfg.get('enabled') and fb_cfg.get('server_key'):
            try:
                import firebase_admin
                from firebase_admin import credentials, messaging
                cred = credentials.Certificate(fb_cfg['service_account_json'])
                firebase_admin.initialize_app(cred)
                self._firebase = messaging
                logger.info("Firebase initialized.")
            except Exception as e:
                logger.warning(f"Firebase init failed: {e}")

    # ── PUBLIC API ─────────────────────────────────────────────

    def dispatch(self, event: ThreatEvent, is_beach_zone: bool = True):
        """
        Main entry point. Dispatches alerts for a confirmed ThreatEvent.
        Applies cooldown per threat class.
        """
        key = event.threat_class.value
        now = time.time()
        if now - self._cooldowns.get(key, 0) < self.COOLDOWN_SEC:
            logger.debug(f"Cooldown active for {key}, skipping.")
            return
        self._cooldowns[key] = now

        cfg = THREAT_CONFIG.get(event.threat_class, {})
        template = cfg.get("template", "")
        message  = template.format(
            gps=f"{event.gps[0]:.4f}°N, {event.gps[1]:.4f}°E",
            eta=int(event.eta_drone_sec),
        )

        # Lookup nearest responders
        nearest = self.responder_lookup.get_nearest(
            lat=event.gps[0], lon=event.gps[1],
            responder_types=event.responders
        )

        logger.warning(f"[DISPATCH] {event.severity.name} — {event.threat_class.value}")
        logger.warning(f"  Message: {message}")
        logger.warning(f"  Responders: {list(nearest.keys())}")

        self._log_alert(event, message, nearest)

        # Route by responder type
        for responder_type, contact in nearest.items():
            if responder_type == 'lifeguard':
                self._alert_lifeguard(message, contact, event)
            elif responder_type in ('police_station', 'coast_guard'):
                self._alert_ps(message, contact, event, responder_type)
            elif responder_type == 'hospital':
                self._alert_hospital(message, contact, event)
            elif responder_type == 'public_alert':
                self._public_alert(message, event)

    # ── RESPONDER-SPECIFIC METHODS ─────────────────────────────

    def _alert_lifeguard(self, message: str, contact: dict, event: ThreatEvent):
        """Firebase push to lifeguard app + trigger drone siren."""
        logger.info(f"[LIFEGUARD] Push notification → {contact.get('name')}")

        if self._firebase:
            try:
                push = self._firebase.Message(
                    notification=self._firebase.Notification(
                        title=f"⚠ {event.threat_class.value.upper().replace('_',' ')}",
                        body=message,
                    ),
                    data={
                        "lat":        str(event.gps[0]),
                        "lon":        str(event.gps[1]),
                        "severity":   event.severity.name,
                        "threat":     event.threat_class.value,
                        "eta":        str(event.eta_drone_sec),
                    },
                    topic="lifeguard_alerts",
                )
                self._firebase.send(push)
                logger.info(f"[LIFEGUARD] Firebase push sent.")
            except Exception as e:
                logger.error(f"Firebase push failed: {e}")

        # SMS fallback
        self._sms(contact.get('phone', ''), message)
        logger.info("[LIFEGUARD] 🔊 Drone siren activated")

    def _alert_ps(self, message: str, contact: dict, event: ThreatEvent, ps_type: str):
        """SMS + voice call to police station / coast guard."""
        label = "POLICE" if ps_type == 'police_station' else "COAST GUARD"
        logger.info(f"[{label}] Alerting {contact.get('name')} — {contact.get('phone')}")

        ps_message = (
            f"EMERGENCY — AETHERION Beach Safety System\n"
            f"{message}\n"
            f"Nearest Unit: {contact.get('name')}\n"
            f"GPS: {event.gps[0]:.6f}, {event.gps[1]:.6f}\n"
            f"Maps: https://maps.google.com/?q={event.gps[0]},{event.gps[1]}\n"
            f"Drone on scene in {int(event.eta_drone_sec)}s."
        )
        self._sms(contact.get('phone', ''), ps_message)

        # Voice call for CRITICAL
        if event.severity == Severity.CRITICAL:
            self._voice_call(contact.get('phone', ''), event)

    def _alert_hospital(self, message: str, contact: dict, event: ThreatEvent):
        """SMS to hospital for ambulance dispatch."""
        logger.info(f"[HOSPITAL] Ambulance dispatch → {contact.get('name')}")

        hosp_message = (
            f"AMBULANCE REQUIRED — AETHERION Beach Safety\n"
            f"{message}\n"
            f"Trauma centre: {contact.get('name')}\n"
            f"GPS: {event.gps[0]:.6f}, {event.gps[1]:.6f}\n"
            f"Maps: https://maps.google.com/?q={event.gps[0]},{event.gps[1]}\n"
            f"Severity: {event.severity.name}"
        )
        self._sms(contact.get('phone', ''), hosp_message)

    def _public_alert(self, message: str, event: ThreatEvent):
        """Broadcast to public app + drone PA system."""
        logger.info(f"[PUBLIC] Broadcasting: {message[:60]}...")
        if self._firebase:
            try:
                push = self._firebase.Message(
                    notification=self._firebase.Notification(
                        title="⚠ Beach Safety Alert",
                        body=message,
                    ),
                    topic="public_beach_alerts",
                )
                self._firebase.send(push)
            except Exception as e:
                logger.error(f"Public push failed: {e}")

    # ── TWILIO HELPERS ─────────────────────────────────────────

    def _sms(self, to: str, body: str):
        if not self._twilio or not to:
            return
        twilio_cfg = self.config.get('twilio', {})
        try:
            msg = self._twilio.messages.create(
                body=body,
                from_=twilio_cfg.get('from_number', ''),
                to=to,
            )
            logger.info(f"SMS sent to {to} | SID: {msg.sid}")
        except Exception as e:
            logger.error(f"SMS failed to {to}: {e}")

    def _voice_call(self, to: str, event: ThreatEvent):
        """Automated voice call for CRITICAL events."""
        if not self._twilio or not to:
            return
        twilio_cfg = self.config.get('twilio', {})
        twiml = (
            f"<Response><Say voice='alice'>"
            f"Emergency alert. {event.threat_class.value.replace('_',' ')} detected "
            f"at coordinates {event.gps[0]:.4f} north, {event.gps[1]:.4f} east. "
            f"Immediate response required. This is an automated message from AETHERION Beach Safety."
            f"</Say><Pause length='2'/><Say voice='alice'>Repeating.</Say>"
            f"<Say voice='alice'>Emergency alert. {event.threat_class.value.replace('_',' ')} detected."
            f"</Say></Response>"
        )
        try:
            call = self._twilio.calls.create(
                twiml=twiml,
                from_=twilio_cfg.get('from_number', ''),
                to=to,
            )
            logger.info(f"Voice call to {to} | SID: {call.sid}")
        except Exception as e:
            logger.error(f"Voice call failed: {e}")

    # ── LOGGING ────────────────────────────────────────────────

    def _log_alert(self, event: ThreatEvent, message: str, nearest: dict):
        self.alert_log.append({
            'time':       time.strftime('%Y-%m-%d %H:%M:%S'),
            'threat':     event.threat_class.value,
            'severity':   event.severity.name,
            'confidence': round(event.confidence, 3),
            'gps':        event.gps,
            'message':    message,
            'responders': list(nearest.keys()),
            'eta_drone':  event.eta_drone_sec,
        })

    def get_log(self) -> List[dict]:
        return self.alert_log.copy()
