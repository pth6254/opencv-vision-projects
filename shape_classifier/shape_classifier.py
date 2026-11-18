#!/usr/bin/env python3
"""
웹캠 기반 도형 선별기 (Shape Classifier)

흰 종이 위의 검은 도형(삼각형/사각형/원)을 노이즈를 제거한 뒤 모두 검출하고,
도형 종류를 판별해 각 도형 중심에 이름을 표시한다.

파이프라인:
  1. 프레임 입력      cv2.VideoCapture(0)  또는  --image 정지 이미지
  2. 1차 노이즈 제거   그레이스케일 → GaussianBlur(5x5)
  3. 이진화          Otsu(기본) / 고정 임계값 127  (THRESH_BINARY_INV → 도형이 흰색)
  4. 2차 노이즈 제거   모폴로지 열기(점 노이즈 제거) → 닫기(끊김·구멍 메우기)
  5. 컨투어 검출      RETR_EXTERNAL, 최소 면적 / 화면 테두리 접촉 필터
  6. 도형 판별        approxPolyDP 꼭짓점 수 + 원형도(circularity)
  7. 중심점 계산      cv2.moments

실행:
  python shape_classifier.py                         # 웹캠 (카메라 0)
  python shape_classifier.py --camera 1              # 다른 카메라
  python shape_classifier.py --image test_image.png  # 정지 이미지 + 트랙바 GUI
  python shape_classifier.py --image test_image.png --no-gui   # 헤드리스 테스트
      → 같은 이름의 .json(정답 라벨)이 있으면 자동으로 비교하고, output/에 결과 저장

키:  q 종료 / s 현재 Shapes 화면 PNG 저장 / d 디버그 표시 토글
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────
# 기본 파라미터 / 상수
# ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "mode": 1,          # 이진화 방식: 0 = 고정 임계값, 1 = Otsu
    "thresh": 127,      # 고정 임계값
    "kernel": 5,        # 모폴로지 커널 크기 (홀수)
    "iterations": 1,    # 모폴로지 반복 횟수 (0이면 생략)
    "min_area": 1000,   # 최소 면적(px). 이보다 작은 컨투어는 노이즈로 간주
    "eps": 0.04,        # approxPolyDP epsilon 계수 (× 둘레)
    "square": 1,        # 1이면 정사각형을 "Square"로 따로 구분
}

CIRCULARITY_MIN = 0.80          # 이 값보다 원형도가 크면 원 후보
SQUARE_RATIO = (0.95, 1.05)     # minAreaRect 가로/세로 비율이 이 범위면 정사각형
BORDER_MARGIN = 2               # 화면 가장자리에서 이 픽셀 이내에 닿으면 무시
BLUR_KSIZE = (5, 5)             # 1차 노이즈 제거용 가우시안 커널

SHAPE_NAMES = ["Triangle", "Rectangle", "Square", "Circle", "Unidentified"]
SHAPE_COLORS = {                # 도형 종류별 외곽선 색 (BGR)
    "Triangle": (0, 200, 0),
    "Rectangle": (255, 120, 0),
    "Square": (255, 200, 0),
    "Circle": (0, 0, 255),
    "Unidentified": (160, 160, 160),
}

WIN_SHAPES = "Shapes"
WIN_RAW = "Binary (raw)"
WIN_CLEAN = "Binary (clean)"
WIN_CTRL = "Controls"

# 트랙바 이름 (OpenCV 창은 한글이 깨질 수 있어 영문 사용)
TB_MODE = "Mode 0:Fix 1:Otsu"
TB_THRESH = "Threshold"
TB_KERNEL = "Kernel size"
TB_ITER = "Iterations"
TB_AREA = "Min area"
TB_EPS = "Epsilon x100"
TB_SQUARE = "Square 0/1"


# ─────────────────────────────────────────────────────────────
# 1. 프레임 입력
# ─────────────────────────────────────────────────────────────
def open_camera(index=0):
    """카메라를 연다. 실패하면 None 반환"""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def load_image(path):
    """정지 이미지를 BGR로 읽는다. 실패하면 None"""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


# ─────────────────────────────────────────────────────────────
# 2. 1차 노이즈 제거: 그레이스케일 → 가우시안 블러
# ─────────────────────────────────────────────────────────────
def preprocess(frame):
    """BGR 프레임 → 블러 처리된 그레이스케일"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    # 5x5 가우시안 블러: 소금-후추 같은 고주파 노이즈와 센서 노이즈를 뭉개 줌
    return cv2.GaussianBlur(gray, BLUR_KSIZE, 0)


# ─────────────────────────────────────────────────────────────
# 3. 이진화
# ─────────────────────────────────────────────────────────────
def binarize(blurred, mode=1, thresh=127):
    """
    흰 종이 위 검은 도형 → INV 이진화로 도형을 흰색(255)으로 만든다.
    mode 1: Otsu가 히스토그램을 보고 임계값을 자동 결정 (조명 변화에 강함)
    mode 0: 고정 임계값 (비교용)
    반환: (이진 영상, 실제 사용된 임계값)
    """
    if mode == 1:
        t, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        t, binary = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY_INV)
    return binary, t


# ─────────────────────────────────────────────────────────────
# 4. 2차 노이즈 제거: 모폴로지 열기 → 닫기
# ─────────────────────────────────────────────────────────────
def clean_binary(binary, ksize=5, iterations=1):
    """
    열기(침식→팽창): 커널보다 작은 흰 점 노이즈를 제거
    닫기(팽창→침식): 도형 외곽선의 끊김과 내부의 작은 검은 구멍을 메움
    """
    if iterations <= 0 or ksize <= 1:
        return binary.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return closed


# ─────────────────────────────────────────────────────────────
# 5. 컨투어 검출 + 필터링
# ─────────────────────────────────────────────────────────────
def touches_border(cnt, width, height, margin=BORDER_MARGIN):
    """컨투어의 바운딩 박스가 화면 가장자리에 닿는지 검사"""
    x, y, w, h = cv2.boundingRect(cnt)
    return x <= margin or y <= margin or x + w >= width - margin or y + h >= height - margin


def find_shape_contours(clean, min_area=1000):
    """
    바깥 컨투어만 검출(RETR_EXTERNAL)한 뒤
      - 면적이 min_area 미만인 것(잔여 노이즈)
      - 화면 테두리에 닿은 것(종이 가장자리, 배경)
    을 제외한 모든 컨투어를 반환한다. (가장 큰 것 하나만 고르지 않음)
    """
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = clean.shape[:2]
    kept = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        if touches_border(cnt, w, h):
            continue
        kept.append(cnt)
    return kept


# ─────────────────────────────────────────────────────────────
# 6. 도형 판별
# ─────────────────────────────────────────────────────────────
def classify_shape(cnt, eps_ratio=0.04, detect_square=True):
    """
    컨투어 하나를 받아 도형 종류를 판별한다.

    - 꼭짓점 수: approxPolyDP(epsilon = eps_ratio × 둘레)로 다각형 근사
    - 원형도   : 4π × 면적 / 둘레²  (원 1.0, 정사각형 ≈ 0.785, 정삼각형 ≈ 0.6)
      ※ 꼭짓점 수만으로 원을 판별하면 오각형·육각형도 원이 되므로 원형도를 함께 사용

    반환: dict(label, approx, vertices, circularity, aspect)
    """
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    info = {"label": "Unidentified", "approx": cnt, "vertices": 0,
            "circularity": 0.0, "aspect": 0.0}
    if perimeter == 0:
        return info

    # 다각형 근사 → 꼭짓점 수
    approx = cv2.approxPolyDP(cnt, eps_ratio * perimeter, True)
    vertices = len(approx)
    # 원형도
    circularity = 4.0 * math.pi * area / (perimeter * perimeter)
    info.update(approx=approx, vertices=vertices, circularity=circularity)

    if vertices == 3:
        info["label"] = "Triangle"
    elif vertices == 4:
        # 회전된 사각형도 올바르게 재기 위해 boundingRect 대신 minAreaRect 사용
        # (boundingRect는 40° 돌린 직사각형의 비율을 ~1.05로 재서 정사각형으로 오인함)
        (_, _), (w, h), _ = cv2.minAreaRect(cnt)
        aspect = w / h if h > 0 else 0.0
        info["aspect"] = aspect
        if detect_square and SQUARE_RATIO[0] <= aspect <= SQUARE_RATIO[1]:
            info["label"] = "Square"
        else:
            info["label"] = "Rectangle"
    elif vertices >= 5 and circularity > CIRCULARITY_MIN:
        info["label"] = "Circle"
    # 그 외(오각형 등 원형도가 낮은 다각형) → Unidentified
    return info


# ─────────────────────────────────────────────────────────────
# 7. 중심점 계산
# ─────────────────────────────────────────────────────────────
def compute_center(cnt):
    """모멘트로 무게중심 계산. m00 == 0(면적 0)이면 None"""
    m = cv2.moments(cnt)
    if m["m00"] == 0:
        return None
    return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])


# ─────────────────────────────────────────────────────────────
# 전체 파이프라인
# ─────────────────────────────────────────────────────────────
def detect_shapes(frame, params):
    """
    프레임 하나에 대해 2~7단계를 모두 수행한다.
    반환: (shapes 리스트, 이진화 직후 영상, 모폴로지 후 영상, 사용된 임계값)
    """
    blurred = preprocess(frame)
    binary_raw, used_t = binarize(blurred, params["mode"], params["thresh"])
    binary_clean = clean_binary(binary_raw, params["kernel"], params["iterations"])
    contours = find_shape_contours(binary_clean, params["min_area"])

    shapes = []
    for cnt in contours:
        center = compute_center(cnt)
        if center is None:          # 면적 0인 퇴화 컨투어는 건너뜀
            continue
        info = classify_shape(cnt, params["eps"], bool(params["square"]))
        info.update(contour=cnt, center=center, area=cv2.contourArea(cnt))
        shapes.append(info)
    return shapes, binary_raw, binary_clean, used_t


# ─────────────────────────────────────────────────────────────
# 화면 출력
# ─────────────────────────────────────────────────────────────
def draw_outlined_text(img, text, org, scale=0.6, thickness=2):
    """검은 테두리 + 흰 글씨로 두 번 그려 어떤 배경에서도 보이게 함"""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_centered_text(img, text, center, scale=0.6, thickness=2):
    """텍스트의 가운데가 center에 오도록 정렬해서 그림"""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    org = (int(center[0] - tw / 2), int(center[1] + th / 2))
    draw_outlined_text(img, text, org, scale, thickness)


def count_shapes(shapes):
    """종류별 개수 집계 (0개인 종류도 포함)"""
    counts = {name: 0 for name in SHAPE_NAMES}
    for s in shapes:
        counts[s["label"]] += 1
    return counts


def draw_counts(img, counts, status=""):
    """
    화면 좌상단에 종류별 개수를 한 줄로 표시 (예: Triangle: 2, Rectangle: 1, Circle: 3)
    그 아래 줄에 현재 파라미터 상태를 작게 표시. 도형을 덜 가리도록 반투명 띠 2줄만 사용.
    """
    font, scale1, scale2 = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 0.45
    parts = [f"{name}: {n}" for name, n in counts.items()]
    sep = ", "
    sep_w = cv2.getTextSize(sep, font, scale1, 1)[0][0]
    widths = [cv2.getTextSize(p, font, scale1, 1)[0][0] for p in parts]
    line1_w = sum(widths) + sep_w * (len(parts) - 1)
    line2_w = cv2.getTextSize(status, font, scale2, 1)[0][0] if status else 0
    box_w = max(line1_w, line2_w) + 16
    box_h = 46 if status else 26

    overlay = img.copy()
    cv2.rectangle(overlay, (4, 4), (4 + box_w, 4 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    # 1줄: 종류별 개수 (도형 색과 같은 색으로)
    x = 12
    for i, (p, w) in enumerate(zip(parts, widths)):
        name = p.split(":")[0]
        cv2.putText(img, p, (x, 22), font, scale1, SHAPE_COLORS[name], 1, cv2.LINE_AA)
        x += w
        if i < len(parts) - 1:
            cv2.putText(img, sep, (x, 22), font, scale1, (255, 255, 255), 1, cv2.LINE_AA)
            x += sep_w
    # 2줄: 파라미터 상태
    if status:
        cv2.putText(img, status, (12, 42), font, scale2, (220, 220, 220), 1, cv2.LINE_AA)


def draw_results(frame, shapes, debug=False, status=None):
    """원본 위에 컨투어·꼭짓점·이름·개수를 그린 결과 영상 반환"""
    out = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for s in shapes:
        color = SHAPE_COLORS[s["label"]]
        # 도형 종류별 색으로 외곽선
        cv2.drawContours(out, [s["contour"]], -1, color, 3)
        # approx 꼭짓점 위치에 작은 원
        for p in s["approx"].reshape(-1, 2):
            cv2.circle(out, (int(p[0]), int(p[1])), 5, color, -1)
            cv2.circle(out, (int(p[0]), int(p[1])), 5, (0, 0, 0), 1)
        # 중심에 도형 이름 (중앙 정렬)
        draw_centered_text(out, s["label"], s["center"])
        # 디버그: 도형 옆(바운딩 박스 오른쪽)에 꼭짓점 수·원형도 표시
        if debug:
            x, y, w, h = cv2.boundingRect(s["contour"])
            info = [f"V={s['vertices']}", f"C={s['circularity']:.3f}"]
            if s["label"] in ("Rectangle", "Square"):
                info.append(f"AR={s['aspect']:.2f}")
            text_w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for t in info)
            tx = x + w + 4
            if tx + text_w > out.shape[1]:        # 오른쪽이 잘리면 도형 왼쪽에 표시
                tx = max(0, x - text_w - 4)
            for i, t in enumerate(info):
                draw_outlined_text(out, t, (tx, y + 16 + i * 18), 0.5, 1)
    draw_counts(out, count_shapes(shapes), status or "")
    return out


# ─────────────────────────────────────────────────────────────
# 트랙바 ("Controls" 창)
# ─────────────────────────────────────────────────────────────
def create_controls(params):
    """트랙바 창 생성. 초기값은 params에서 가져옴"""
    noop = lambda _v: None
    cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CTRL, 420, 330)
    cv2.createTrackbar(TB_MODE, WIN_CTRL, params["mode"], 1, noop)
    cv2.createTrackbar(TB_THRESH, WIN_CTRL, params["thresh"], 255, noop)
    cv2.createTrackbar(TB_KERNEL, WIN_CTRL, params["kernel"], 21, noop)
    cv2.createTrackbar(TB_ITER, WIN_CTRL, params["iterations"], 5, noop)
    cv2.createTrackbar(TB_AREA, WIN_CTRL, params["min_area"], 20000, noop)
    cv2.createTrackbar(TB_EPS, WIN_CTRL, int(round(params["eps"] * 100)), 10, noop)
    cv2.createTrackbar(TB_SQUARE, WIN_CTRL, params["square"], 1, noop)


def read_controls():
    """트랙바 값을 읽어 유효 범위로 보정한 파라미터 dict 반환"""
    kernel = max(1, cv2.getTrackbarPos(TB_KERNEL, WIN_CTRL))
    if kernel % 2 == 0:                       # 커널은 홀수로 (중심이 있어야 함)
        kernel += 1
    eps = max(1, cv2.getTrackbarPos(TB_EPS, WIN_CTRL)) / 100.0   # 0.01 ~ 0.10
    return {
        "mode": cv2.getTrackbarPos(TB_MODE, WIN_CTRL),
        "thresh": cv2.getTrackbarPos(TB_THRESH, WIN_CTRL),
        "kernel": kernel,
        "iterations": cv2.getTrackbarPos(TB_ITER, WIN_CTRL),
        "min_area": cv2.getTrackbarPos(TB_AREA, WIN_CTRL),
        "eps": eps,
        "square": cv2.getTrackbarPos(TB_SQUARE, WIN_CTRL),
    }


def status_text(params, used_t, debug):
    """현재 파라미터를 한 줄 문자열로 (화면 2번째 줄에 표시)"""
    mode = f"Otsu T={used_t:.0f}" if params["mode"] == 1 else f"Fixed T={params['thresh']}"
    text = (f"{mode} | morph k={params['kernel']} it={params['iterations']}"
            f" | minArea {params['min_area']} | eps {params['eps']:.2f}")
    return text + (" | DEBUG" if debug else "")


# ─────────────────────────────────────────────────────────────
# GUI 루프 (웹캠 / 정지 이미지 공용)
# ─────────────────────────────────────────────────────────────
def run_gui(cap, still, params, out_dir, debug=False):
    """cap이 있으면 웹캠, 없으면 still 이미지를 매 프레임 반복 처리"""
    create_controls(params)
    cv2.namedWindow(WIN_SHAPES)
    cv2.namedWindow(WIN_RAW)
    cv2.namedWindow(WIN_CLEAN)


    while True:
        if cap is not None:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[ERROR] 카메라 프레임을 읽지 못했습니다. 종료합니다.", file=sys.stderr)
                break
        else:
            frame = still

        params = read_controls()
        shapes, raw, clean, used_t = detect_shapes(frame, params)
        vis = draw_results(frame, shapes, debug, status_text(params, used_t, debug))

        cv2.imshow(WIN_SHAPES, vis)
        cv2.imshow(WIN_RAW, raw)
        cv2.imshow(WIN_CLEAN, clean)

        key = cv2.waitKey(1 if cap is not None else 30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"shapes_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(str(path), vis)
            print(f"[SAVE] {path}")
        elif key == ord("d"):
            debug = not debug
            print(f"[DEBUG] {'ON' if debug else 'OFF'}")

        # 창의 X 버튼으로 닫은 경우도 종료
        if cv2.getWindowProperty(WIN_SHAPES, cv2.WND_PROP_VISIBLE) < 1:
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────
# 헤드리스 테스트 (--image + --no-gui)
# ─────────────────────────────────────────────────────────────
def evaluate(shapes, gt_shapes):
    """
    검출 결과와 정답 라벨을 중심 거리로 매칭해 비교한다.
      - 정답 하나당 가장 가까운 검출 1개 (허용 거리: 도형 크기의 35%, 최소 20px)
      - 매칭되지 않은 검출 = 오검출(노이즈 등), 매칭되지 않은 정답 = 미검출
    """
    pairs = []
    for gi, g in enumerate(gt_shapes):
        tol = max(20.0, 0.35 * math.sqrt(g.get("area", 1600)))
        for di, s in enumerate(shapes):
            d = math.dist(g["center"], s["center"])
            if d <= tol:
                pairs.append((d, gi, di))
    pairs.sort()
    gt_to_det, used_det = {}, set()
    for d, gi, di in pairs:              # 가까운 쌍부터 1:1 그리디 매칭
        if gi in gt_to_det or di in used_det:
            continue
        gt_to_det[gi] = (di, d)
        used_det.add(di)

    rows = []
    for gi, g in enumerate(gt_shapes):
        if gi in gt_to_det:
            di, d = gt_to_det[gi]
            s = shapes[di]
            rows.append({"id": g.get("id", gi + 1), "expected": g["label"], "predicted": s["label"],
                         "dist": d, "vertices": s["vertices"], "circularity": s["circularity"],
                         "ok": s["label"] == g["label"]})
        else:
            rows.append({"id": g.get("id", gi + 1), "expected": g["label"], "predicted": None,
                         "dist": None, "vertices": None, "circularity": None, "ok": False})
    false_pos = [shapes[di] for di in range(len(shapes)) if di not in used_det]
    return rows, false_pos


def make_montage(frame, raw, clean, vis):
    """원본 / 이진화 / 모폴로지 / 결과 2x2 비교 이미지"""
    def tile(img, title):
        img = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.rectangle(img, (0, 0), (img.shape[1], 28), (40, 40, 40), -1)
        cv2.putText(img, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return img
    top = np.hstack([tile(frame, "Input"), tile(raw, "Binary (raw)")])
    bottom = np.hstack([tile(clean, "Binary (clean)"), tile(vis, "Shapes")])
    return np.vstack([top, bottom])


def run_headless(image_path, frame, params, labels_path, out_dir, debug):
    """GUI 없이 한 번 처리 → 결과 저장 → 정답 JSON이 있으면 비교. 반환: 종료 코드"""
    shapes, raw, clean, used_t = detect_shapes(frame, params)
    vis = draw_results(frame, shapes, debug, status_text(params, used_t, debug))

    # 결과 이미지 저장
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    paths = {
        "shapes": out_dir / f"{stem}_shapes.png",
        "raw": out_dir / f"{stem}_binary_raw.png",
        "clean": out_dir / f"{stem}_binary_clean.png",
        "compare": out_dir / f"{stem}_compare.png",
    }
    cv2.imwrite(str(paths["shapes"]), vis)
    cv2.imwrite(str(paths["raw"]), raw)
    cv2.imwrite(str(paths["clean"]), clean)
    cv2.imwrite(str(paths["compare"]), make_montage(frame, raw, clean, vis))

    # 검출 결과 요약
    mode = "Otsu" if params["mode"] == 1 else "Fixed"
    print(f"[INFO] 이미지: {image_path}  ({frame.shape[1]}x{frame.shape[0]})")
    print(f"[INFO] 이진화: {mode} (T={used_t:.0f}) | 모폴로지 k={params['kernel']} it={params['iterations']}"
          f" | 최소 면적 {params['min_area']} | eps {params['eps']:.2f}")
    raw_n = len(cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
    clean_n = len(cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
    print(f"[INFO] 컨투어 수: 이진화 직후 {raw_n}개 → 모폴로지 후 {clean_n}개 → 필터 후 도형 {len(shapes)}개")
    print("[INFO] 검출: " + ", ".join(f"{k}: {v}" for k, v in count_shapes(shapes).items()))
    for s in sorted(shapes, key=lambda s: (s["center"][1], s["center"][0])):
        print(f"   - {s['label']:<12} center={s['center']}  area={s['area']:.0f}"
              f"  V={s['vertices']}  C={s['circularity']:.3f}"
              + (f"  AR={s['aspect']:.3f}" if s["label"] in ("Rectangle", "Square") else ""))

    # 정답 비교
    exit_code = 0
    if labels_path is not None:
        with open(labels_path, encoding="utf-8") as f:
            gt = json.load(f)["shapes"]
        rows, false_pos = evaluate(shapes, gt)

        print(f"\n[TEST] 정답 라벨: {labels_path}")
        print(f"{'id':>3}  {'expected':<12} {'predicted':<12} {'V':>2} {'circ':>6} {'dist':>6}  result")
        for r in rows:
            pred = r["predicted"] or "(missed)"
            v = f"{r['vertices']:>2}" if r["vertices"] is not None else " -"
            c = f"{r['circularity']:.3f}" if r["circularity"] is not None else "     -"
            d = f"{r['dist']:.1f}" if r["dist"] is not None else "     -"
            print(f"{r['id']:>3}  {r['expected']:<12} {pred:<12} {v} {c:>6} {d:>6}  {'PASS' if r['ok'] else 'FAIL'}")

        n_ok = sum(r["ok"] for r in rows)
        print(f"\n  1) 노이즈 오검출: {len(false_pos)}개 → {'PASS' if not false_pos else 'FAIL'}")
        for s in false_pos:
            print(f"     ! {s['label']} at {s['center']} area={s['area']:.0f}")
        print(f"  2) 분류 정확도  : {n_ok}/{len(rows)} → {'PASS' if n_ok == len(rows) else 'FAIL'}")
        all_pass = not false_pos and n_ok == len(rows)
        print(f"\n[RESULT] {'ALL PASS' if all_pass else 'FAILED'}")
        exit_code = 0 if all_pass else 1
    else:
        print("[INFO] 정답 JSON이 없어 비교는 생략했습니다. (--labels로 지정 가능)")

    print("\n[SAVE] " + "\n[SAVE] ".join(str(p) for p in paths.values()))
    return exit_code


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(description="OpenCV 웹캠 도형 선별기")
    ap.add_argument("--camera", type=int, default=0, help="카메라 인덱스 (기본 0)")
    ap.add_argument("--image", help="웹캠 대신 정지 이미지로 실행")
    ap.add_argument("--no-gui", action="store_true", help="창 없이 한 번 처리하고 결과만 저장 (--image 필요)")
    ap.add_argument("--labels", help="정답 JSON 경로 (기본: 이미지와 같은 이름의 .json)")
    ap.add_argument("--out-dir", default=None, help="결과 저장 폴더 (기본: 스크립트 옆 output/)")
    ap.add_argument("--debug", action="store_true", help="디버그 표시로 시작")
    # 파라미터 초기값 (GUI에서는 트랙바 초기값, 헤드리스에서는 그대로 사용)
    ap.add_argument("--mode", choices=["otsu", "fixed"], default="otsu", help="이진화 방식")
    ap.add_argument("--thresh", type=int, default=DEFAULTS["thresh"], help="고정 임계값")
    ap.add_argument("--kernel", type=int, default=DEFAULTS["kernel"], help="모폴로지 커널 크기")
    ap.add_argument("--iterations", type=int, default=DEFAULTS["iterations"], help="모폴로지 반복 횟수")
    ap.add_argument("--min-area", type=int, default=DEFAULTS["min_area"], help="최소 면적(px)")
    ap.add_argument("--eps", type=float, default=DEFAULTS["eps"], help="approxPolyDP epsilon 계수")
    ap.add_argument("--no-square", action="store_true", help="정사각형을 Rectangle로 합쳐서 표시")
    return ap.parse_args()


def main():
    args = parse_args()
    params = {
        "mode": 1 if args.mode == "otsu" else 0,
        "thresh": int(np.clip(args.thresh, 0, 255)),
        "kernel": max(1, args.kernel | 1),          # 홀수로 보정
        "iterations": max(0, args.iterations),
        "min_area": max(0, args.min_area),
        "eps": float(np.clip(args.eps, 0.01, 0.10)),
        "square": 0 if args.no_square else 1,
    }
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent / "output"

    # ── 정지 이미지 모드 ──
    if args.image:
        frame = load_image(args.image)
        if frame is None:
            print(f"[ERROR] 이미지를 읽을 수 없습니다: {args.image}", file=sys.stderr)
            return 1
        if args.no_gui:
            labels = args.labels
            if labels is None:                       # 같은 이름의 .json 자동 탐색
                cand = Path(args.image).with_suffix(".json")
                labels = str(cand) if cand.exists() else None
            return run_headless(args.image, frame, params, labels, out_dir, args.debug)
        run_gui(None, frame, params, out_dir, args.debug)
        return 0

    if args.no_gui:
        print("[ERROR] --no-gui는 --image와 함께 사용해야 합니다.", file=sys.stderr)
        return 1

    # ── 웹캠 모드 ──
    cap = open_camera(args.camera)
    if cap is None:
        print(f"[ERROR] 카메라 {args.camera}번을 열 수 없습니다. 연결 상태나 다른 프로그램 사용 여부를 확인하세요.",
              file=sys.stderr)
        return 1
    run_gui(cap, None, params, out_dir, args.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
