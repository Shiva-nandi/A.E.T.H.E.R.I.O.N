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
"""AETHERION — Aquatic Preprocessor (same as v1, glare+CLAHE+stabilize)"""
import cv2, numpy as np

class AquaticPreprocessor:
    def __init__(self, stabilize=True, glare_correction=True, contrast_enhance=True):
        self.stabilize = stabilize
        self.glare_correction = glare_correction
        self.contrast_enhance = contrast_enhance
        self._prev_gray = None

    def process(self, frame):
        r = frame.copy()
        if self.glare_correction:  r = self._remove_glare(r)
        if self.contrast_enhance:  r = self._enhance_contrast(r)
        if self.stabilize:         r = self._stabilize(r)
        return r

    def _remove_glare(self, f):
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        _, mask = cv2.threshold(l, 240, 255, cv2.THRESH_BINARY)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.dilate(mask, k, iterations=2)
        return cv2.inpaint(f, mask, 5, cv2.INPAINT_TELEA)

    def _enhance_contrast(self, f):
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)

    def _stabilize(self, f):
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray; return f
        try:
            pts = cv2.goodFeaturesToTrack(self._prev_gray, 200, 0.01, 30)
            if pts is None: self._prev_gray=gray; return f
            curr, st, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, pts, None)
            v = st.ravel()==1
            M, _ = cv2.estimateAffinePartial2D(pts[v], curr[v])
            if M is None: self._prev_gray=gray; return f
            h, w = f.shape[:2]
            out = cv2.warpAffine(f, M, (w,h))
            self._prev_gray = gray
            return out
        except:
            self._prev_gray = gray; return f
