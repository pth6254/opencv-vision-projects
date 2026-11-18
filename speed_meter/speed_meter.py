#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
speed_meter.py - 웹캠 기반 실시간 속도 측정기

책상 위에 테이프로 그은 두 기준선(A, B) 사이를 특정 색 물체(기본: 빨강)가
지나가는 시간을 재서 속도(cm/s)를 계산한다.

처리 파이프라인
  1. 프레임 입력 + 타임스탬프        FrameSource.read()
  2. Perspective Transform(Top-view) build_perspective() / warp_to_top_view()
  3. 기준선 검출 (Hough)             detect_reference_lines()
  4. 물체 검출 (HSV)                 detect_object()
  5. 통과 감지 (부호 변화 + 보간)    find_crossing()
  6. 속도 계산                       SpeedMeasurer._finish()
  7. 상태 머신                       SpeedMeasurer  (WAITING → TRACKING → DONE)

사용법
  python speed_meter.py --calibrate              # 측정 영역 보정 → config.json
  python speed_meter.py                          # 웹캠 0번으로 실시간 측정
  python speed_meter.py --source 1               # 웹캠 1번
  python speed_meter.py --source clip.mp4        # 동영상 파일로 측정

실행 중 키
  q / ESC : 종료            r : 측정 리셋           l : 기준선 재검출
  s : 현재 기준선 위치를 config.json에 저장          m : 마스크 창 토글
  SPACE : 일시정지 (동영상 파일만)

※ OpenCV의 putText는 한글을 그리지 못하므로 화면 표시는 영어, 콘솔 출력은 한글.
"""

import argparse
import copy
import csv
import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, "speed_log.csv")

WIN_CAMERA = "Camera"
WIN_TOP = "Top View"
WIN_MASK = "Mask"
WIN_CALIB = "Calibrate"
WIN_CALIB_PREVIEW = "Calibration Preview"

# 색상 (BGR)
COLOR_LINE_A = (255, 170, 0)     # 하늘색 계열
COLOR_LINE_B = (255, 0, 255)     # 자홍
COLOR_CENTER = (0, 255, 0)       # 초록 (물체 중심)
COLOR_TRAIL = (0, 220, 255)      # 주황/노랑 (이동 경로)
COLOR_AREA = (0, 255, 255)       # 노랑 (측정 영역 외곽선)
COLOR_TEXT = (235, 235, 235)
COLOR_DIM = (150, 150, 150)
COLOR_OK = (0, 220, 0)
COLOR_WARN = (0, 165, 255)
COLOR_ERR = (60, 60, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_W = 350           # Top View 창 오른쪽 정보 패널 폭
PANEL_MIN_H = 500
HISTORY_LEN = 5          # 화면에 표시할 최근 측정 개수
CAMERA_WARMUP_FRAMES = 10  # 웹캠 자동 노출이 안정될 때까지 기준선 검출을 미룸

CORNER_NAMES = ("Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left")
CORNER_SHORT = ("TL", "TR", "BR", "BL")

# 색상 프리셋 (--color). 빨강은 Hue가 0 부근과 179 부근으로 나뉘므로 두 범위를 OR.
HSV_PRESETS = {
    "red": [{"lower": [0, 120, 70], "upper": [10, 255, 255]},
            {"lower": [170, 120, 70], "upper": [179, 255, 255]}],
    "green": [{"lower": [40, 70, 70], "upper": [85, 255, 255]}],
    "blue": [{"lower": [95, 120, 70], "upper": [130, 255, 255]}],
    "yellow": [{"lower": [20, 100, 100], "upper": [35, 255, 255]}],
}

DEFAULT_CONFIG = {
    "frame_size": None,            # [w, h] 보정 당시 카메라 해상도
    "src_points": None,            # 카메라 영상의 측정 영역 꼭짓점 (TL, TR, BR, BL)
    "area_width_cm": None,         # 측정 영역 실제 가로 (TL→TR)
    "area_height_cm": None,        # 측정 영역 실제 세로 (TL→BL)
    "px_per_cm": 10.0,             # Top-view 해상도 (1cm = 10px)
    "line_distance_cm": None,      # 두 기준선 사이 실제 거리 d
    "reference_lines": None,       # {"A": y_px, "B": y_px} Top-view 좌표 (마지막 검출 값)
    "hsv_ranges": HSV_PRESETS["red"],
    "min_object_area_px": 150,     # Top-view 기준 최소 컨투어 면적
    "max_missing_frames": 5,       # 이 프레임 수까지는 물체가 안 보여도 추적 유지
    "done_reset_sec": 3.0,         # DONE 후 자동 리셋까지 시간
    "trail_length": 64,            # 이동 경로 점 개수
    "line_detection": {
        "canny_low": 50,
        "canny_high": 150,
        "hough_threshold": 40,
        "min_line_length_ratio": 0.25,  # HoughLinesP minLineLength = 영역 폭 × 비율
        "max_line_gap_cm": 2.0,
        "max_angle_deg": 10.0,          # 수평 ±10° 이내 선분만 사용
        "cluster_gap_cm": 3.0,          # 이 거리 이내 선분은 같은 테이프로 묶음
        "min_coverage_ratio": 0.5,      # 클러스터가 영역 폭의 50% 이상을 덮어야 기준선 후보
        "border_margin_cm": 1.0,        # Top-view 가장자리 선분은 무시
    },
}
REQUIRED_KEYS = ("src_points", "area_width_cm", "area_height_cm", "line_distance_cm")


# ════════════════════════════════════════════════════════════════════
# 설정 파일
# ════════════════════════════════════════════════════════════════════
def _merge(base, override):
    """dict 재귀 병합 (override가 우선)."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path, require_calibration=True):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cfg = _merge(cfg, json.load(f))
    elif require_calibration:
        raise SystemExit(f"[오류] 설정 파일이 없습니다: {path}\n"
                         f"       먼저 --calibrate 로 측정 영역을 보정하세요.")
    if require_calibration:
        missing = [k for k in REQUIRED_KEYS if not cfg.get(k)]
        if missing:
            raise SystemExit(f"[오류] config에 보정 값이 없습니다: {', '.join(missing)}\n"
                             f"       먼저 --calibrate 로 보정하세요.")
        if len(cfg["src_points"]) != 4:
            raise SystemExit("[오류] src_points는 꼭짓점 4개여야 합니다.")
    return cfg


def save_config(path, cfg):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


# ════════════════════════════════════════════════════════════════════
# 1. 프레임 입력 + 타임스탬프
# ════════════════════════════════════════════════════════════════════
@dataclass
class Frame:
    image: np.ndarray
    t: float     # 초. 웹캠: 캡처 직후 perf_counter (첫 프레임 기준), 파일: idx / fps
    idx: int     # 0부터 시작하는 프레임 번호


class FrameSource:
    """웹캠/동영상 파일을 같은 방식으로 읽고 프레임마다 타임스탬프를 붙인다."""

    def __init__(self, source, frame_size=None, fps_override=None, fallback_fps=30.0):
        self.is_camera = isinstance(source, int)
        self.name = f"camera{source}" if self.is_camera else os.path.basename(str(source))
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise SystemExit(f"[오류] 입력을 열 수 없습니다: {source}")
        if self.is_camera and frame_size:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(frame_size[0]))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(frame_size[1]))

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not fps or not math.isfinite(fps) or fps <= 0 or fps > 1000:
            fps = None
        if fps_override:
            fps = float(fps_override)
        if not self.is_camera and fps is None:
            print(f"[경고] 동영상 FPS를 읽을 수 없어 {fallback_fps} FPS로 가정합니다 (--fps 로 지정).")
            fps = fallback_fps
        self.fps = fps          # 웹캠은 표시용(참고값), 파일은 타임스탬프 계산에 사용
        self.idx = -1
        self._t0 = None

    def read(self) -> Optional[Frame]:
        if self.is_camera:
            # grab() 직후에 시각을 찍어야 디코딩(retrieve) 시간이 타임스탬프에 섞이지 않는다.
            if not self.cap.grab():
                return None
            t_cap = time.perf_counter()
            ok, image = self.cap.retrieve()
            if not ok or image is None:
                return None
            if self._t0 is None:
                self._t0 = t_cap
            self.idx += 1
            return Frame(image, t_cap - self._t0, self.idx)

        # 동영상 파일: 처리/재생 속도와 무관하게 프레임 번호 / FPS
        ok, image = self.cap.read()
        if not ok or image is None:
            return None
        self.idx += 1
        return Frame(image, self.idx / self.fps, self.idx)

    def release(self):
        self.cap.release()


def parse_source(s):
    return int(s) if str(s).isdigit() else s


# ════════════════════════════════════════════════════════════════════
# 2. Perspective Transform → Top-view
# ════════════════════════════════════════════════════════════════════
def order_points(pts):
    """임의 순서의 4점을 좌상 → 우상 → 우하 → 좌하 순서로 정렬."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)              # x + y : 좌상 최소, 우하 최대
    d = pts[:, 1] - pts[:, 0]        # y - x : 우상 최소, 좌하 최대
    idx = [int(np.argmin(s)), int(np.argmin(d)), int(np.argmax(s)), int(np.argmax(d))]
    if len(set(idx)) == 4:
        return pts[idx]
    # 많이 기울어진 사각형이라 위 방법이 겹치면: 무게중심 기준 각도 정렬로 대체
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    ordered = pts[np.argsort(ang)]   # 이미지 좌표계(y 아래)에서 시계방향 = TL, TR, BR, BL
    start = int(np.argmin(ordered.sum(axis=1)))
    return np.roll(ordered, -start, axis=0)


def build_perspective(cfg, frame_shape=None):
    """
    반환: (M, M_inv, top_size(w, h), src_pts)
    출력 크기는 영역 실제 크기 × px_per_cm → Top-view에서 1px = 1/px_per_cm cm.
    frame_shape가 보정 당시 해상도와 다르면 꼭짓점을 비례 조정한다.
    """
    ppc = float(cfg["px_per_cm"])
    w = float(cfg["area_width_cm"]) * ppc
    h = float(cfg["area_height_cm"]) * ppc

    src = order_points(cfg["src_points"])
    saved = cfg.get("frame_size")
    if frame_shape is not None and saved:
        fh, fw = frame_shape[:2]
        if (fw, fh) != (int(saved[0]), int(saved[1])):
            print(f"[경고] 카메라 해상도({fw}x{fh})가 보정 당시({saved[0]}x{saved[1]})와 다릅니다. "
                  f"꼭짓점을 비례 조정합니다 — 가능하면 다시 --calibrate 하세요.")
            src = src * np.float32([fw / saved[0], fh / saved[1]])

    # dst_pts도 src와 같은 순서(좌상 → 우상 → 우하 → 좌하).
    # (w-1이 아니라 w를 쓰면 축척이 정확히 px_per_cm가 된다)
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, dst)
    return M, np.linalg.inv(M), (int(round(w)), int(round(h))), src


def warp_to_top_view(frame, M, size):
    return cv2.warpPerspective(frame, M, size, flags=cv2.INTER_LINEAR)


# ════════════════════════════════════════════════════════════════════
# 3. 기준선 검출 (시작 시 / l 키에서만 호출 — 매 프레임 X)
# ════════════════════════════════════════════════════════════════════
def _x_coverage(intervals):
    """x 구간들의 합집합 길이. 테이프 위/아래 모서리가 겹쳐도 두 번 세지 않는다."""
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def detect_reference_lines(top_img, px_per_cm, params):
    """
    Top-view에서 수평 테이프 기준선 두 개를 찾는다.
    반환: (lines, info)
      lines : {"A": y_px(위), "B": y_px(아래)} 또는 실패 시 None
      info  : {"segments": [...], "clusters": [...], "message": str}
    """
    info = {"segments": [], "clusters": [], "message": ""}
    h, w = top_img.shape[:2]

    gray = cv2.cvtColor(top_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, params["canny_low"], params["canny_high"])
    raw = cv2.HoughLinesP(edges, 1, np.pi / 180, int(params["hough_threshold"]),
                          minLineLength=max(10.0, params["min_line_length_ratio"] * w),
                          maxLineGap=params["max_line_gap_cm"] * px_per_cm)
    if raw is None:
        info["message"] = "no Hough segments"
        return None, info

    # (1) 수평에 가까운 선분만 (±max_angle_deg)
    margin = params["border_margin_cm"] * px_per_cm
    segs = []
    for x1, y1, x2, y2 in raw[:, 0].astype(float):
        if x2 < x1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(angle) > params["max_angle_deg"]:
            continue
        y_mid = (y1 + y2) / 2.0
        if y_mid < margin or y_mid > h - margin:     # 워핑 경계/책상 모서리 제외
            continue
        segs.append({"y": y_mid, "x1": x1, "x2": x2,
                     "len": math.hypot(x2 - x1, y2 - y1),
                     "pts": (int(x1), int(y1), int(x2), int(y2))})
    info["segments"] = segs

    # (2) y 기준 클러스터링 — 테이프 하나의 위/아래 모서리를 하나로 묶고 길이 가중 평균 y
    gap = params["cluster_gap_cm"] * px_per_cm
    clusters = []
    for s in sorted(segs, key=lambda s: s["y"]):
        if clusters and abs(s["y"] - clusters[-1]["y"]) <= gap:
            c = clusters[-1]
            c["segs"].append(s)
            wsum = sum(t["len"] for t in c["segs"])
            c["y"] = sum(t["y"] * t["len"] for t in c["segs"]) / wsum
        else:
            clusters.append({"y": s["y"], "segs": [s]})
    for c in clusters:
        c["coverage"] = _x_coverage([(t["x1"], t["x2"]) for t in c["segs"]])
        c["total_len"] = sum(t["len"] for t in c["segs"])
    info["clusters"] = clusters

    # (3) 길이가 충분한 클러스터 중 가장 긴 두 개 → 위 = A, 아래 = B
    valid = [c for c in clusters if c["coverage"] >= params["min_coverage_ratio"] * w]
    if len(valid) < 2:
        info["message"] = f"found {len(valid)} line(s), need 2"
        return None, info
    best = sorted(valid, key=lambda c: (c["coverage"], c["total_len"]), reverse=True)[:2]
    best.sort(key=lambda c: c["y"])
    info["message"] = "ok" if len(valid) == 2 else f"{len(valid)} candidates, used 2 longest"
    return {"A": float(best[0]["y"]), "B": float(best[1]["y"])}, info


def line_distance_check(lines, px_per_cm, distance_cm):
    """검출된 두 선의 거리(cm)와 입력한 d 비교 → (측정 cm, 오차 %) 또는 None."""
    if not lines:
        return None
    measured = abs(lines["B"] - lines["A"]) / px_per_cm
    return measured, (measured - distance_cm) / distance_cm * 100.0


# ════════════════════════════════════════════════════════════════════
# 4. 물체 검출 (HSV)
# ════════════════════════════════════════════════════════════════════
@dataclass
class Detection:
    cx: float
    cy: float
    area: float
    contour: np.ndarray


MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


def build_color_mask(hsv, hsv_ranges):
    """여러 HSV 범위를 OR로 합친다 (빨강: Hue 0 부근 | 179 부근)."""
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for r in hsv_ranges:
        lo = np.array(r["lower"], dtype=np.uint8)
        hi = np.array(r["upper"], dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    return mask


def detect_object(top_img, hsv_ranges, min_area):
    """반환: (Detection 또는 None, mask)"""
    hsv = cv2.cvtColor(top_img, cv2.COLOR_BGR2HSV)
    mask = build_color_mask(hsv, hsv_ranges)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)    # 작은 잡음 제거
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)   # 내부 구멍 메우기

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if not contours:
        return None, mask
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < min_area:
        return None, mask                      # 물체 없음
    m = cv2.moments(c)
    if m["m00"] == 0:
        return None, mask
    # 중심점은 float 그대로 유지 (보간 정밀도)
    return Detection(m["m10"] / m["m00"], m["m01"] / m["m00"], area, c), mask


# ════════════════════════════════════════════════════════════════════
# 5. 통과 감지 (부호 변화 + 선형 보간)
# ════════════════════════════════════════════════════════════════════
@dataclass
class TrackPoint:
    t: float
    x: float
    y: float
    frame: int


@dataclass
class Crossing:
    line: str            # "A" / "B"
    t: float             # 보간된 통과 시각
    frame_interp: float  # 보간된 프레임 번호
    frame: int           # 통과를 감지한 (현재) 프레임 번호
    frame_t: float       # 그 프레임의 타임스탬프
    direction: str       # "down"(y 증가) / "up"(y 감소)


def find_crossing(prev: TrackPoint, curr: TrackPoint, line_y: float, line_name: str):
    """
    이전 점과 현재 점이 선의 서로 반대편에 있으면 통과.
      (prev.y - line_y) * (curr.y - line_y) < 0   또는   현재 점이 정확히 선 위
    (이전 점이 정확히 선 위인 경우는 그 전 프레임에 '도착'으로 이미 셌으므로 제외 → 이중 카운트 방지)
    통과 시각은 두 프레임 사이 선형 보간:
      t_cross = t_prev + (t_curr - t_prev) * (line_y - prev_y) / (curr_y - prev_y)
    """
    d0 = prev.y - line_y
    d1 = curr.y - line_y
    if not (d0 * d1 < 0 or (d1 == 0 and d0 != 0)):
        return None
    ratio = (line_y - prev.y) / (curr.y - prev.y)        # 0 < ratio <= 1
    return Crossing(
        line=line_name,
        t=prev.t + (curr.t - prev.t) * ratio,
        frame_interp=prev.frame + (curr.frame - prev.frame) * ratio,
        frame=curr.frame,
        frame_t=curr.t,
        direction="down" if curr.y > prev.y else "up",
    )


# ════════════════════════════════════════════════════════════════════
# 6~7. 속도 계산 + 상태 머신
# ════════════════════════════════════════════════════════════════════
class State(Enum):
    WAITING = "WAITING"
    TRACKING = "TRACKING"
    DONE = "DONE"


@dataclass
class Measurement:
    start_line: str
    end_line: str
    direction: str
    t1: float
    t2: float
    distance_cm: float
    speed_cm_s: float
    frame1: int
    frame2: int
    frame_diff: int
    frame_diff_interp: float
    fps_measured: Optional[float]
    wall_time: str

    @property
    def dt(self):
        return self.t2 - self.t1

    @property
    def label(self):
        return f"{self.start_line}->{self.end_line}"


class SpeedMeasurer:
    """
    WAITING  --(시작선 통과: 두 선 사이로 '들어오는' 방향)-->  TRACKING
    TRACKING --(다른 선 통과)-->                                DONE   (v = d / (t2 - t1))
    TRACKING --(물체 max_missing 프레임 초과 소실 / 시작선으로 되돌아감)--> WAITING
    DONE     --(done_reset_sec 경과 또는 r 키)-->               WAITING

    A가 위(y 작음), B가 아래이므로
      A를 아래로(down) 통과 → A->B 측정 시작,  B를 위로(up) 통과 → B->A 측정 시작.
    """

    def __init__(self, distance_cm, max_missing=5, done_reset_sec=3.0):
        self.distance_cm = float(distance_cm)
        self.max_missing = int(max_missing)
        self.done_reset_sec = float(done_reset_sec)
        self.state = State.WAITING
        self.prev: Optional[TrackPoint] = None
        self.missing = 0
        self.start: Optional[Crossing] = None
        self.done_t = None
        self.now = 0.0
        self.message = "Waiting for object"

    # ── 외부 API ──────────────────────────────────────────────
    def reset(self, message="Reset"):
        self.state = State.WAITING
        self.start = None
        self.done_t = None
        self.message = message

    def elapsed(self):
        return self.now - self.start.t if self.state is State.TRACKING else None

    def time_to_reset(self):
        if self.state is not State.DONE:
            return None
        return max(0.0, self.done_reset_sec - (self.now - self.done_t))

    def update(self, point: Optional[TrackPoint], t_now: float, lines) -> Optional[Measurement]:
        """프레임마다 호출. 측정이 끝난 프레임에서만 Measurement를 반환."""
        self.now = t_now
        if self.state is State.DONE and t_now - self.done_t >= self.done_reset_sec:
            self.reset("Ready")

        if point is None:
            self.missing += 1
            if self.missing > self.max_missing:
                if self.state is State.TRACKING:
                    self.reset(f"Cancelled: object lost > {self.max_missing} frames")
                    print(f"[취소] 물체가 {self.max_missing}프레임 넘게 보이지 않아 측정을 취소합니다.")
                self.prev = None        # 긴 공백을 가로질러 보간하지 않도록
            return None

        # 짧게 사라졌다 돌아온 경우: 마지막으로 본 점(prev)과 현재 점 사이로 보간
        self.missing = 0
        result = None
        if self.prev is not None and lines:
            events = [find_crossing(self.prev, point, lines[n], n) for n in ("A", "B")]
            # 한 프레임 사이에 두 선을 모두 지나는 아주 빠른 물체도 시간순으로 처리
            for ev in sorted((e for e in events if e), key=lambda e: e.t):
                r = self._on_crossing(ev)
                if r is not None:
                    result = r
        self.prev = point
        return result

    # ── 내부 ─────────────────────────────────────────────────
    @staticmethod
    def _is_entering(ev: Crossing):
        return (ev.line == "A" and ev.direction == "down") or (ev.line == "B" and ev.direction == "up")

    def _on_crossing(self, ev: Crossing):
        if self.state is State.WAITING:
            if not self._is_entering(ev):
                return None             # 두 선 사이에서 바깥으로 나가는 통과는 시작으로 보지 않음
            self.start = ev
            self.state = State.TRACKING
            self.message = f"Start: crossed {ev.line} ({ev.direction})"
            return None

        if self.state is State.TRACKING:
            if ev.line == self.start.line:
                self.reset(f"Cancelled: went back over {ev.line}")
                print(f"[취소] 물체가 시작선 {ev.line}로 되돌아갔습니다.")
                return None
            return self._finish(ev)

        return None                     # DONE: 리셋 전까지 통과 무시

    def _finish(self, end: Crossing):
        s = self.start
        dt = end.t - s.t
        if dt <= 0:
            self.reset("Invalid time difference")
            return None
        frame_diff = end.frame - s.frame
        frame_dt = end.frame_t - s.frame_t
        fps = frame_diff / frame_dt if frame_diff > 0 and frame_dt > 0 else None
        m = Measurement(
            start_line=s.line, end_line=end.line, direction=s.direction,
            t1=s.t, t2=end.t,
            distance_cm=self.distance_cm,
            speed_cm_s=self.distance_cm / dt,          # v = d / (t2 - t1)
            frame1=s.frame, frame2=end.frame,
            frame_diff=frame_diff,
            frame_diff_interp=end.frame_interp - s.frame_interp,
            fps_measured=fps,
            wall_time=datetime.now().isoformat(timespec="seconds"),
        )
        self.state = State.DONE
        self.done_t = self.now
        self.message = f"Done: {m.label} {m.speed_cm_s:.2f} cm/s"
        return m


# ════════════════════════════════════════════════════════════════════
# CSV 기록
# ════════════════════════════════════════════════════════════════════
class CsvLogger:
    FIELDS = ["wall_time", "source", "start_line", "end_line", "direction",
              "t1_s", "t2_s", "dt_s", "distance_cm", "speed_cm_s",
              "frame1", "frame2", "frame_diff", "frame_diff_interp", "fps_measured"]

    def __init__(self, path, source_name):
        self.path = path
        self.source_name = source_name

    def write(self, m: Measurement):
        new_file = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            if new_file:
                w.writeheader()
            w.writerow({
                "wall_time": m.wall_time, "source": self.source_name,
                "start_line": m.start_line, "end_line": m.end_line, "direction": m.direction,
                "t1_s": f"{m.t1:.6f}", "t2_s": f"{m.t2:.6f}", "dt_s": f"{m.dt:.6f}",
                "distance_cm": f"{m.distance_cm:.3f}", "speed_cm_s": f"{m.speed_cm_s:.4f}",
                "frame1": m.frame1, "frame2": m.frame2, "frame_diff": m.frame_diff,
                "frame_diff_interp": f"{m.frame_diff_interp:.3f}",
                "fps_measured": f"{m.fps_measured:.3f}" if m.fps_measured else "",
            })


def format_measurement(m: Measurement, n=None):
    head = f"[측정 #{n}]" if n is not None else "[측정]"
    fps = f"{m.fps_measured:.1f}" if m.fps_measured else "-"
    return (f"{head} {m.label} ({m.direction})  v = {m.speed_cm_s:.2f} cm/s  "
            f"(dt = {m.dt:.4f} s, d = {m.distance_cm:g} cm, "
            f"프레임 {m.frame1}→{m.frame2} = {m.frame_diff} (보간 {m.frame_diff_interp:.2f}), FPS {fps})")


# ════════════════════════════════════════════════════════════════════
# 화면 그리기 (항상 복사본에 그린다 — 감지용 이미지는 건드리지 않음)
# ════════════════════════════════════════════════════════════════════
def put_text(img, text, org, color=COLOR_TEXT, scale=0.5, thickness=1):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_camera_view(frame, src_pts, M_inv, top_size, lines):
    """원본 + 측정 영역 외곽선 (+ 기준선을 원근 역변환해서 표시)."""
    vis = frame.copy()
    poly = np.round(src_pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(vis, [poly], True, COLOR_AREA, 2, cv2.LINE_AA)
    for name, p in zip(CORNER_SHORT, src_pts):
        p = (int(round(p[0])), int(round(p[1])))
        cv2.circle(vis, p, 4, COLOR_AREA, -1, cv2.LINE_AA)
        put_text(vis, name, (p[0] + 6, p[1] - 6), COLOR_AREA)
    if lines:
        w = top_size[0]
        for name, color in (("A", COLOR_LINE_A), ("B", COLOR_LINE_B)):
            pts = np.float32([[[0, lines[name]]], [[w, lines[name]]]])
            p1, p2 = cv2.perspectiveTransform(pts, M_inv).reshape(-1, 2)
            p1, p2 = tuple(np.round(p1).astype(int)), tuple(np.round(p2).astype(int))
            cv2.line(vis, p1, p2, color, 1, cv2.LINE_AA)
            put_text(vis, name, (p2[0] + 6, p2[1] + 5), color, 0.55, 1)
    return vis


def draw_trail(img, trail):
    """오래된 점일수록 얇게. trail에는 (x, y) 또는 None(물체 없음 → 선 끊김)이 들어 있다."""
    pts = list(trail)
    n = len(pts)
    maxlen = trail.maxlen or max(n, 1)
    for i in range(1, n):
        if pts[i - 1] is None or pts[i] is None:
            continue
        age = n - 1 - i                         # 0 = 가장 최근
        thickness = max(1, int(math.sqrt(maxlen / float(age + 1)) * 1.2))
        cv2.line(img, pts[i - 1], pts[i], COLOR_TRAIL, thickness, cv2.LINE_AA)


def draw_top_view(top, lines, det, trail, meter):
    vis = top.copy()
    w = vis.shape[1]
    if lines:
        for name, color in (("A", COLOR_LINE_A), ("B", COLOR_LINE_B)):
            y = int(round(lines[name]))
            active = meter.state is State.TRACKING and meter.start and meter.start.line == name
            cv2.line(vis, (0, y), (w - 1, y), color, 3 if active else 2, cv2.LINE_AA)
            put_text(vis, f"Line {name}", (6, y - 7), color, 0.5, 1)
    draw_trail(vis, trail)
    if det is not None:
        c = (int(round(det.cx)), int(round(det.cy)))
        cv2.drawContours(vis, [det.contour], -1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(vis, c, 6, COLOR_CENTER, -1, cv2.LINE_AA)
        cv2.circle(vis, c, 7, (0, 0, 0), 1, cv2.LINE_AA)
    return vis


def draw_info_panel(height, meter, lines, check, distance_cm, last, history,
                    stream_fps, warning, det, paused):
    panel = np.full((height, PANEL_W, 3), 32, np.uint8)
    y = 26

    def row(text, color=COLOR_TEXT, scale=0.5, gap=21, thick=1):
        nonlocal y
        put_text(panel, text, (12, y), color, scale, thick)
        y += gap

    def sep():
        nonlocal y
        cv2.line(panel, (10, y - 10), (PANEL_W - 10, y - 10), (80, 80, 80), 1)
        y += 14

    state_color = {State.WAITING: COLOR_TEXT, State.TRACKING: COLOR_WARN, State.DONE: COLOR_OK}
    row(f"STATE: {meter.state.value}" + ("  [PAUSED]" if paused else ""),
        state_color[meter.state], 0.65, 26, 2)
    if meter.state is State.TRACKING:
        other = "B" if meter.start.line == "A" else "A"
        row(f"{meter.start.line} -> {other}   elapsed {meter.elapsed():.2f} s", COLOR_WARN)
    elif meter.state is State.DONE:
        row(f"auto reset in {meter.time_to_reset():.1f} s  (r: now)", COLOR_DIM)
    else:
        row("cross line A or B to start", COLOR_DIM)
    row(meter.message[:40], COLOR_DIM)
    row(f"Object: {'(%.0f, %.0f) area %.0f' % (det.cx, det.cy, det.area) if det else 'not found'}",
        COLOR_TEXT if det else COLOR_ERR)
    sep()

    if last is not None:
        row(f"v = {last.speed_cm_s:.2f} cm/s", COLOR_OK, 0.85, 32, 2)
        row(f"{last.label}   dt {last.dt:.4f} s   d {last.distance_cm:g} cm")
        row(f"frames {last.frame_diff}  (interp {last.frame_diff_interp:.2f})")
        row(f"measured FPS {last.fps_measured:.1f}" if last.fps_measured else "measured FPS -")
    else:
        row("v = --.-- cm/s", COLOR_DIM, 0.85, 32, 2)
        row("no measurement yet", COLOR_DIM)
    sep()

    if lines:
        row(f"Lines  A y={lines['A']:.1f}px  B y={lines['B']:.1f}px")
        meas, err = check
        color = COLOR_OK if abs(err) <= 3 else (COLOR_WARN if abs(err) <= 8 else COLOR_ERR)
        row(f"A-B {meas:.2f}cm / input d {distance_cm:g}cm ({err:+.1f}%)", color)
    else:
        row("Lines NOT SET - press 'l'", COLOR_ERR)
    if warning:
        row(warning[:42], COLOR_ERR)
    row(f"stream FPS {stream_fps:.1f}" if stream_fps else "stream FPS -", COLOR_DIM)
    sep()

    row(f"History (last {HISTORY_LEN})", COLOR_TEXT, 0.5, 22)
    if not history:
        row("  -", COLOR_DIM)
    for i, m in enumerate(history, 1):
        row(f"{i}. {m.label}  {m.speed_cm_s:7.2f} cm/s  {m.dt:.3f}s", COLOR_TEXT, 0.48, 20)

    put_text(panel, "q quit  r reset  l lines  s save", (12, height - 32), COLOR_DIM, 0.42)
    put_text(panel, "m mask  SPACE pause(video)", (12, height - 12), COLOR_DIM, 0.42)
    return panel


def compose_top_window(top_vis, panel):
    h = max(top_vis.shape[0], panel.shape[0])
    canvas = np.zeros((h, top_vis.shape[1] + panel.shape[1], 3), np.uint8)
    canvas[:top_vis.shape[0], :top_vis.shape[1]] = top_vis
    canvas[:panel.shape[0], top_vis.shape[1]:] = panel
    return canvas


# ════════════════════════════════════════════════════════════════════
# 실행 모드: 실시간 측정
# ════════════════════════════════════════════════════════════════════
def run_measurement(args):
    cfg = load_config(args.config)
    if args.color:
        cfg["hsv_ranges"] = HSV_PRESETS[args.color]

    src = FrameSource(parse_source(args.source), cfg.get("frame_size"), args.fps)
    ppc = float(cfg["px_per_cm"])
    distance_cm = float(cfg["line_distance_cm"])

    saved = cfg.get("reference_lines") or {}
    lines = ({"A": float(saved["A"]), "B": float(saved["B"])}
             if saved.get("A") is not None and saved.get("B") is not None else None)

    meter = SpeedMeasurer(distance_cm, cfg["max_missing_frames"], cfg["done_reset_sec"])
    history = deque(maxlen=HISTORY_LEN)
    trail = deque(maxlen=int(cfg["trail_length"]))
    logger = CsvLogger(args.csv, src.name)
    show = not args.no_display
    warmup = CAMERA_WARMUP_FRAMES if src.is_camera else 0
    frame_delay = 1 if (src.is_camera or args.fast) else max(1, int(round(1000.0 / src.fps)))

    M = M_inv = top_size = src_pts = None
    detect_pending = True
    warning = None
    show_mask = False
    paused = False
    stream_fps = None
    prev_t = None
    count = 0
    fr = top = det = mask = None

    def run_line_detection(top_img):
        nonlocal lines, warning
        new_lines, info = detect_reference_lines(top_img, ppc, cfg["line_detection"])
        if new_lines:
            lines = new_lines
            warning = None if info["message"] == "ok" else info["message"]
            meas, err = line_distance_check(lines, ppc, distance_cm)
            print(f"[기준선] A y={lines['A']:.1f}px, B y={lines['B']:.1f}px → 간격 {meas:.2f} cm "
                  f"(입력 d = {distance_cm:g} cm, 오차 {err:+.1f}%)  [{info['message']}]")
        else:
            keep = " (kept previous)" if lines else ""
            warning = f"Line detect failed: {info['message']}{keep}"
            print(f"[경고] 기준선 검출 실패: {info['message']}"
                  + (" — 이전 값을 유지합니다." if lines else " — 측정할 수 없습니다. l 키로 재시도."))

    print(f"[시작] 입력: {src.name}" + (f" (FPS {src.fps:g})" if src.fps else "")
          + f", 기준선 거리 d = {distance_cm:g} cm, CSV: {args.csv}")
    try:
        while True:
            if not paused:
                fr = src.read()
                if fr is None:
                    print("[종료] 더 이상 프레임이 없습니다." if not src.is_camera
                          else "[오류] 웹캠 프레임을 읽지 못했습니다.")
                    break
                if M is None:
                    M, M_inv, top_size, src_pts = build_perspective(cfg, fr.image.shape)

                if prev_t is not None and fr.t > prev_t:
                    inst = 1.0 / (fr.t - prev_t)
                    stream_fps = inst if stream_fps is None else 0.9 * stream_fps + 0.1 * inst
                prev_t = fr.t

                top = warp_to_top_view(fr.image, M, top_size)        # 감지용 원본 (그리지 않음)

                if detect_pending and fr.idx >= warmup:
                    run_line_detection(top)
                    detect_pending = False

                det, mask = detect_object(top, cfg["hsv_ranges"], cfg["min_object_area_px"])
                point = TrackPoint(fr.t, det.cx, det.cy, fr.idx) if det else None
                result = meter.update(point, fr.t, lines)
                if result is not None:
                    count += 1
                    history.appendleft(result)
                    logger.write(result)
                    print(format_measurement(result, count))
                trail.append((int(round(det.cx)), int(round(det.cy))) if det else None)

            if not show:
                continue

            check = line_distance_check(lines, ppc, distance_cm)
            top_vis = draw_top_view(top, lines, det, trail, meter)
            panel = draw_info_panel(max(top_vis.shape[0], PANEL_MIN_H), meter, lines, check,
                                    distance_cm, history[0] if history else None, history,
                                    stream_fps, warning, det, paused)
            cv2.imshow(WIN_CAMERA, draw_camera_view(fr.image, src_pts, M_inv, top_size, lines))
            cv2.imshow(WIN_TOP, compose_top_window(top_vis, panel))
            if show_mask:
                cv2.imshow(WIN_MASK, mask)

            key = cv2.waitKey(30 if paused else frame_delay) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("r"):
                meter.reset("Manual reset")
                trail.clear()
            elif key == ord("l"):
                run_line_detection(top)
            elif key == ord("s"):
                if lines:
                    disk = load_config(args.config, require_calibration=False)
                    disk["reference_lines"] = {"A": round(lines["A"], 2), "B": round(lines["B"], 2)}
                    save_config(args.config, disk)
                    print(f"[저장] 기준선 위치를 {args.config}에 저장했습니다.")
                else:
                    print("[경고] 저장할 기준선이 없습니다.")
            elif key == ord("m"):
                show_mask = not show_mask
                if not show_mask:
                    cv2.destroyWindow(WIN_MASK)
            elif key == ord(" ") and not src.is_camera:
                paused = not paused
    except KeyboardInterrupt:
        print("\n[종료] Ctrl+C")
    finally:
        src.release()
        if show:
            cv2.destroyAllWindows()

    if count:
        print(f"[요약] 측정 {count}건, CSV: {args.csv}")
    return 0


# ════════════════════════════════════════════════════════════════════
# 보정 모드: --calibrate
# ════════════════════════════════════════════════════════════════════
def prompt_float(message):
    while True:
        try:
            s = input(message).strip()
        except EOFError:
            raise SystemExit("\n[취소] 입력이 없습니다.")
        try:
            v = float(s)
        except ValueError:
            print("  숫자를 입력하세요.")
            continue
        if v <= 0:
            print("  0보다 큰 값을 입력하세요.")
            continue
        return v


def parse_points_arg(text):
    """'x1,y1 x2,y2 x3,y3 x4,y4' 형식."""
    try:
        pts = [tuple(float(v) for v in p.split(",")) for p in text.replace(";", " ").split()]
    except ValueError:
        pts = []
    if len(pts) != 4 or any(len(p) != 2 for p in pts):
        raise SystemExit("[오류] --points 형식: \"x1,y1 x2,y2 x3,y3 x4,y4\"")
    return pts


def select_points(src: FrameSource):
    """마우스로 꼭짓점 4개 선택. 반환: (선택 시점의 프레임, 점 리스트) 또는 (None, None)."""
    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            print(f"  {len(points)}. {CORNER_NAMES[len(points) - 1]:<12} ({x}, {y})")

    cv2.namedWindow(WIN_CALIB)
    cv2.setMouseCallback(WIN_CALIB, on_mouse)
    frame = None
    while True:
        if frame is None or src.is_camera:          # 웹캠은 실시간, 파일은 첫 프레임 고정
            fr = src.read()
            if fr is not None:
                frame = fr.image
            elif frame is None:
                raise SystemExit("[오류] 프레임을 읽을 수 없습니다.")

        vis = frame.copy()
        if len(points) == 4:
            ordered = order_points(points)
            cv2.polylines(vis, [np.round(ordered).astype(np.int32).reshape(-1, 1, 2)],
                          True, COLOR_AREA, 2, cv2.LINE_AA)
            for name, p in zip(CORNER_SHORT, ordered):
                put_text(vis, name, (int(p[0]) + 8, int(p[1]) - 8), COLOR_AREA, 0.55, 1)
        elif len(points) >= 2:
            cv2.polylines(vis, [np.int32(points).reshape(-1, 1, 2)], False, COLOR_AREA, 1, cv2.LINE_AA)
        for i, p in enumerate(points):
            cv2.circle(vis, p, 5, (0, 0, 255), -1, cv2.LINE_AA)
            put_text(vis, str(i + 1), (p[0] + 8, p[1] + 16), (0, 0, 255), 0.55, 1)

        if len(points) < 4:
            put_text(vis, f"Click {len(points) + 1}/4: {CORNER_NAMES[len(points)]}",
                     (10, 28), COLOR_AREA, 0.7, 2)
        else:
            put_text(vis, "ENTER: confirm", (10, 28), COLOR_OK, 0.7, 2)
        put_text(vis, "order: TL -> TR -> BR -> BL   u: undo  c: clear  q: cancel",
                 (10, vis.shape[0] - 12), COLOR_TEXT, 0.5, 1)
        cv2.imshow(WIN_CALIB, vis)

        key = cv2.waitKey(1 if src.is_camera else 30) & 0xFF
        if key in (13, 10, 32) and len(points) == 4:
            return frame, points
        if key == ord("u") and points:
            points.pop()
        elif key == ord("c"):
            points.clear()
        elif key in (ord("q"), 27):
            return None, None


def draw_calibration_preview(top, lines, info, check, distance_cm):
    vis = top.copy()
    for s in info["segments"]:
        x1, y1, x2, y2 = s["pts"]
        cv2.line(vis, (x1, y1), (x2, y2), (120, 120, 120), 1, cv2.LINE_AA)   # 모든 수평 선분
    if lines:
        for name, color in (("A", COLOR_LINE_A), ("B", COLOR_LINE_B)):
            y = int(round(lines[name]))
            cv2.line(vis, (0, y), (vis.shape[1] - 1, y), color, 2, cv2.LINE_AA)
            put_text(vis, f"Line {name}", (6, y - 7), color, 0.5, 1)
        meas, err = check
        put_text(vis, f"A-B {meas:.2f}cm / input {distance_cm:g}cm ({err:+.1f}%)", (6, 20), COLOR_OK)
    else:
        put_text(vis, f"Lines not found: {info['message']}", (6, 20), COLOR_ERR)
    put_text(vis, "ENTER/s: save   q/ESC: cancel", (6, vis.shape[0] - 10), COLOR_TEXT)
    return vis


def run_calibration(args):
    cfg = load_config(args.config, require_calibration=False)
    req_size = (args.frame_width, args.frame_height) if args.frame_width and args.frame_height else None
    src = FrameSource(parse_source(args.source), req_size, args.fps)

    print("━━━ 측정 영역 보정 ━━━")
    print("  · 측정 영역의 위/아래 변이 테이프 기준선과 평행하도록 꼭짓점을 고르세요.")
    print("    (물체는 Top-view에서 위↔아래로 움직이고, 기준선은 가로로 놓입니다.)")
    if args.points:
        frame = None
        for _ in range(CAMERA_WARMUP_FRAMES if src.is_camera else 1):
            fr = src.read()
            if fr is not None:
                frame = fr.image
        if frame is None:
            raise SystemExit("[오류] 프레임을 읽을 수 없습니다.")
        points = parse_points_arg(args.points)
    else:
        if args.no_display:
            raise SystemExit("[오류] --no-display 로 보정하려면 --points 를 지정하세요.")
        print("  · 클릭 순서: 1 좌상 → 2 우상 → 3 우하 → 4 좌하 (순서가 틀려도 자동 정렬)")
        print("  · u: 마지막 점 취소, c: 전체 지우기, ENTER: 확정, q: 취소")
        frame, points = select_points(src)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    src.release()
    if frame is None:
        print("[취소] 보정을 취소했습니다.")
        return 1

    ordered = order_points(points)
    if not cv2.isContourConvex(ordered.reshape(-1, 1, 2)):
        print("[경고] 선택한 4점이 볼록 사각형이 아닙니다. 결과가 왜곡될 수 있습니다.")
    for name, p in zip(CORNER_NAMES, ordered):
        print(f"  {name:<12} ({p[0]:.0f}, {p[1]:.0f})")

    width_cm = args.width_cm or prompt_float("측정 영역 실제 가로 길이 (좌상→우상, cm): ")
    height_cm = args.height_cm or prompt_float("측정 영역 실제 세로 길이 (좌상→좌하, cm): ")
    distance_cm = args.distance_cm or prompt_float("두 기준선 사이 실제 거리 d (테이프 중심 간, cm): ")
    if distance_cm >= height_cm:
        print("[경고] d가 영역 세로 길이 이상입니다. 두 기준선이 영역 안에 있어야 합니다.")

    cfg.update({
        "frame_size": [int(frame.shape[1]), int(frame.shape[0])],
        "src_points": [[round(float(x), 1), round(float(y), 1)] for x, y in ordered],
        "area_width_cm": width_cm,
        "area_height_cm": height_cm,
        "line_distance_cm": distance_cm,
    })
    if args.px_per_cm:
        cfg["px_per_cm"] = args.px_per_cm
    if args.color:
        cfg["hsv_ranges"] = HSV_PRESETS[args.color]

    ppc = float(cfg["px_per_cm"])
    M, _, top_size, _ = build_perspective(cfg)
    top = warp_to_top_view(frame, M, top_size)
    lines, info = detect_reference_lines(top, ppc, cfg["line_detection"])
    check = line_distance_check(lines, ppc, distance_cm)
    if lines:
        cfg["reference_lines"] = {"A": round(lines["A"], 2), "B": round(lines["B"], 2)}
        print(f"[기준선] A y={lines['A']:.1f}px, B y={lines['B']:.1f}px → 간격 {check[0]:.2f} cm "
              f"(입력 {distance_cm:g} cm, 오차 {check[1]:+.1f}%)")
        if abs(check[1]) > 5:
            print("[경고] 오차가 5%를 넘습니다. 꼭짓점 위치나 실제 길이 입력을 확인하세요.")
    else:
        cfg["reference_lines"] = None
        print(f"[경고] 기준선 2개를 찾지 못했습니다 ({info['message']}). "
              f"보정은 저장되며, 실행 중 l 키로 재검출할 수 있습니다.")

    if not args.no_display:
        cv2.imshow(WIN_CALIB_PREVIEW, draw_calibration_preview(top, lines, info, check, distance_cm))
        print("미리보기 창에서 ENTER/s: 저장, q/ESC: 취소")
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (13, 10, ord("s")):
                break
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                print("[취소] 저장하지 않았습니다.")
                return 1
        cv2.destroyAllWindows()

    save_config(args.config, cfg)
    print(f"[저장] {args.config}  (Top-view {top_size[0]}x{top_size[1]}px, 1cm = {ppc:g}px)")
    return 0


# ════════════════════════════════════════════════════════════════════
def build_arg_parser():
    p = argparse.ArgumentParser(
        description="OpenCV 웹캠 기반 실시간 속도 측정기 (두 테이프 기준선 통과 시간 → cm/s)")
    p.add_argument("--calibrate", action="store_true", help="측정 영역 보정 모드")
    p.add_argument("--source", default="0", help="웹캠 번호(0, 1, ...) 또는 동영상 파일 경로 (기본 0)")
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="설정 파일 (기본: 스크립트 옆 config.json)")
    p.add_argument("--csv", default=DEFAULT_CSV_PATH, help="측정 기록 CSV (기본: speed_log.csv)")
    p.add_argument("--color", choices=sorted(HSV_PRESETS), help="HSV 프리셋으로 config 색 범위 덮어쓰기")
    p.add_argument("--fps", type=float, help="동영상 파일 FPS 강제 지정 (메타데이터가 틀릴 때)")
    p.add_argument("--fast", action="store_true", help="동영상을 최대 속도로 처리 (측정값은 동일)")
    p.add_argument("--no-display", action="store_true", help="창 없이 콘솔/CSV만 출력")

    g = p.add_argument_group("보정 옵션 (생략하면 대화형으로 입력)")
    g.add_argument("--width-cm", type=float, help="측정 영역 실제 가로 (cm)")
    g.add_argument("--height-cm", type=float, help="측정 영역 실제 세로 (cm)")
    g.add_argument("--distance-cm", type=float, help="두 기준선 사이 실제 거리 d (cm)")
    g.add_argument("--px-per-cm", type=float, help="Top-view 축척 (기본 10 → 1cm = 10px)")
    g.add_argument("--points", help='마우스 대신 꼭짓점 지정: "x1,y1 x2,y2 x3,y3 x4,y4"')
    g.add_argument("--frame-width", type=int, help="웹캠 요청 해상도 (가로)")
    g.add_argument("--frame-height", type=int, help="웹캠 요청 해상도 (세로)")
    return p


def main():
    args = build_arg_parser().parse_args()
    return run_calibration(args) if args.calibrate else run_measurement(args)


if __name__ == "__main__":
    raise SystemExit(main())
