# ⚡ AETHERION

<div align="center">

### Aerial Emergency Threat and Hazard Response Intelligence Operations Network

*"Guardian of the skies. Protector of lives."*

**Developed by P. Shiva Charan Reddy**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red)](https://ultralytics.com)
[![Type](https://img.shields.io/badge/Type-Personal%20Project-orange)]()
[![Author](https://img.shields.io/badge/Author-P.%20Shiva%20Charan%20Reddy-brightgreen)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 👤 About

| Field | Details |
|-------|---------|
| **Full Name** | AETHERION — Aerial Emergency Threat and Hazard Response Intelligence Operations Network |
| **Developer** | P. Shiva Charan Reddy |
| **Tagline** | *Guardian of the skies. Protector of lives.* |
| **Type** | Personal Project |
| **Year** | 2026 |
| **Domain** | AI · Computer Vision · Autonomous Drones · Emergency Response |
| **Location** | Hyderabad, Telangana, India |

---

## 🌊 Overview

**AETHERION** is a personal AI project by **P. Shiva Charan Reddy** that deploys autonomous drones over beaches, lakes, and pools to detect life-threatening situations in real time. The system uses a fine-tuned YOLOv8 model to identify 11 distinct threat classes and automatically escalates alerts to the correct emergency responders — from lifeguards to police stations to hospitals — based on severity.

Named after *Aether* — the ancient Greek concept of the pure luminous sky — AETHERION watches from above with an all-seeing eye, responding instantly to protect lives below.

Designed with Hyderabad's water bodies (Hussain Sagar, Necklace Road) in mind.

---

## 🎯 Key Features

- **11-class threat detection** — drowning, shark attack, rip current, heatstroke, fight, jellyfish, and more
- **Smart tiered alert escalation** — CRITICAL triggers Police + Hospital + Coast Guard simultaneously
- **Hyderabad responder database** — auto-locates nearest PS and hospital by GPS
- **Multi-frame confirmation** — prevents false positives (unconscious = 15s, shark = 3s)
- **Twilio SMS + voice call** for CRITICAL, Firebase push for lifeguard app
- **Edge deployment** on NVIDIA Jetson Orin (<18ms inference)
- **AirSim simulation** for drone rescue mission testing without hardware

---

## 🧠 Threat Priority Matrix

| Threat | Severity | Responders |
|--------|----------|------------|
| 🦈 Shark Attack | **CRITICAL** | PS + Hospital + Coast Guard |
| 😶 Unconscious | **CRITICAL** | PS + Hospital |
| 🌊 Submerged | **CRITICAL** | PS + Hospital |
| 🆘 Panic Drowning | **HIGH** | Lifeguards |
| 🪢 Net Entrapment | **HIGH** | Lifeguards + Coast Guard |
| 🌡️ Heatstroke | **MEDIUM** | Lifeguards + Hospital |
| 👊 Fight/Assault | **MEDIUM** | Lifeguards + PS |
| ↩️ Rip Current | **MEDIUM** | Lifeguards + Public |
| 🌧️ Monsoon Surge | **MEDIUM** | Lifeguards + Public |
| 🪼 Jellyfish Swarm | **LOW** | Public Alert |
| 🏊 Normal Swim | **SAFE** | — |

---

## 🗂️ Project Structure

```
AETHERION/
├── src/
│   ├── threats/
│   │   └── classifier.py           ← 11-class threat scorer + multi-frame logic
│   ├── alerts/
│   │   └── smart_alert_engine.py   ← Tiered alert dispatch (PS/Hospital/Firebase)
│   ├── geolocation/
│   │   └── responder_lookup.py     ← GPS → nearest Hyderabad PS + hospital
│   ├── detection/
│   │   ├── train.py                ← YOLOv8 training (11 classes)
│   │   ├── inference.py            ← Real-time multi-threat inference
│   │   └── preprocess.py           ← Glare removal + stabilization
│   └── utils/
│       └── logger.py
├── notebooks/
│   └── AETHERION_Training.ipynb    ← Full Colab training pipeline
├── configs/
│   ├── dataset.yaml                ← 11-class dataset config
│   ├── model_config.yaml
│   └── alerts_config.yaml.example
├── webpage/
│   └── index.html                  ← Interactive system explainer
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start (Google Colab)

```python
# 1. Upload AETHERION_complete.zip to Colab
from google.colab import files
files.upload()

# 2. Extract
!unzip AETHERION_complete.zip -d /content/
%cd /content/AETHERION

# 3. Install dependencies
!pip install -r requirements.txt -q

# 4. Open notebooks/AETHERION_Training.ipynb and run all cells
```

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| Detection | YOLOv8s — 11 threat classes |
| Preprocessing | OpenCV — CLAHE, glare removal, stabilization |
| Alerts | Twilio SMS/Voice + Firebase Cloud Messaging |
| Geolocation | OpenStreetMap Overpass API + Hyderabad DB |
| Edge Deploy | NVIDIA Jetson Orin — TensorRT FP16 |
| Simulation | AirSim + DroneKit/PX4 |
| Training | Google Colab T4 GPU |

---

## 📄 License

MIT License — Copyright (c) 2026 **P. Shiva Charan Reddy**

---

<div align="center">

**AETHERION**<br>
*Aerial Emergency Threat and Hazard Response Intelligence Operations Network*<br><br>
*"Guardian of the skies. Protector of lives."*<br><br>
**P. Shiva Charan Reddy** · Personal Project · Hyderabad 🇮🇳 · 2026

</div>
