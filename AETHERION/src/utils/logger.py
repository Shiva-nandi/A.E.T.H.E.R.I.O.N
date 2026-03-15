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
"""AETHERION Logger"""
import logging, sys

def get_logger(name):
    logger = logging.getLogger(f'AETHERION.{name}')
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s [%(name)s] %(message)s', '%H:%M:%S'))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger
