"""
웹캠 기반 실시간 주차 공간 감지 시스템

책상 위 테이프로 표시한 주차장을 비스듬히 찍은 영상을
Perspective Transform으로 Top-view로 펴고,
각 주차 칸에 빨간 블록(차량)이 있는지 HSV 색 검출로 판정한다.

실행 모드
  python parking.py --calibrate-area            # 1) 주차장 꼭짓점 4개 클릭
  python parking.py --calibrate-spots           # 2) Top-view에서 칸 드래그
  python parking.py --calibrate-spots --grid 2x2  #    (또는 자동 균등 분할)
  python parking.py                             # 3) 실시간 감지

공통 옵션
  --image 경로   : 웹캠 대신 정지 이미지 사용 (테스트용)
  --config 경로  : 보정 파일 경로 지정 (기본: 스크립트 폴더의 config.json)
  --no-gui       : 창 없이 1회 처리 후 결과를 파일/콘솔로 출력
  --expected 경로: (--no-gui) 정답 JSON과 비교해 PASS/FAIL 출력
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 기본 설정값 (config.json에 값이 없을 때 사용)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")

DEFAULTS = {
    "warp_size": [600, 400],          # Top-view 출력 크기 (가로, 세로)
    "hsv": {
        "h1_min": 0, "h1_max": 10,    # 빨강 구간 1 (Hue 0 부근)
        "h2_min": 170, "h2_max": 179,  # 빨강 구간 2 (Hue 179 부근)
        "s_min": 120, "s_max": 255,
        "v_min": 70, "v_max": 255,
    },
    "morph_kernel": 5,                # 모폴로지 커널 크기 (홀수)
    "occupied_ratio": 0.15,           # ROI 면적 대비 최대 컨투어 비율 임계값
    "debounce_frames": 10,            # 상태 변경에 필요한 연속 프레임 수
    "roi_margin": 8,                  # ROI를 안쪽으로 줄이는 픽셀 수
}

WIN_CAMERA = "Camera"
WIN_TOP = "Top View"
WIN_MASKS = "Masks"
WIN_CONTROLS = "Controls"

COLOR_GREEN = (0, 200, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 220, 255)
COLOR_WHITE = (255, 255, 255)

ENTER_KEYS = (13, 10)  # 운영체제마다 Enter 키 코드가 다를 수 있음
ESC_KEY = 27


# ---------------------------------------------------------------------------
# 설정 파일 입출력
# ---------------------------------------------------------------------------
def load_config(path):
    """config.json을 읽고 누락된 항목은 기본값으로 채운다. 파일이 없으면 None."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    merged = json.loads(json.dumps(DEFAULTS))  # 깊은 복사
    for key, value in cfg.items():
        if key == "hsv" and isinstance(value, dict):
            merged["hsv"].update(value)
        else:
            merged[key] = value
    return merged


def save_config(path, updates):
    """기존 config.json 내용을 유지하면서 updates 항목만 덮어써서 저장한다."""
    cfg = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    cfg.update(updates)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"[저장] {path}")


# ---------------------------------------------------------------------------
# 입력 소스 (웹캠 또는 정지 이미지)
# ---------------------------------------------------------------------------
class FrameSource:
    """웹캠과 정지 이미지를 같은 인터페이스(read)로 다루기 위한 래퍼."""

    def __init__(self, image_path=None, camera_index=0):
        self.image = None
        self.cap = None
        if image_path:
            self.image = cv2.imread(image_path)
            if self.image is None:
                print(f"[에러] 이미지를 읽을 수 없습니다: {image_path}")
                sys.exit(1)
        else:
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                print(f"[에러] 카메라({camera_index})를 열 수 없습니다. "
                      "연결 상태나 다른 프로그램의 사용 여부를 확인하세요.")
                sys.exit(1)

    @property
    def is_image(self):
        return self.image is not None

    def read(self):
        """프레임 1장을 반환. 실패하면 None."""
        if self.is_image:
            return self.image.copy()  # 원본이 오염되지 않도록 항상 복사본
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        if self.cap is not None:
            self.cap.release()


# ---------------------------------------------------------------------------
# 기하 변환 관련
# ---------------------------------------------------------------------------
def order_points(pts):
    """
    꼭짓점 4개를 (좌상, 우상, 우하, 좌하) 순서로 정렬한다.
      - x+y 최소 = 좌상, 최대 = 우하
      - y-x 최소 = 우상, 최대 = 좌하
    사용자가 클릭 순서를 틀려도 올바른 변환이 되도록 하기 위함.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)              # x + y
    d = pts[:, 1] - pts[:, 0]        # y - x
    rect[0] = pts[np.argmin(s)]      # 좌상
    rect[2] = pts[np.argmax(s)]      # 우하
    rect[1] = pts[np.argmin(d)]      # 우상
    rect[3] = pts[np.argmax(d)]      # 좌하
    return rect


def get_perspective_matrix(area_points, warp_size):
    """주차장 꼭짓점 → Top-view 직사각형으로 가는 변환 행렬을 구한다."""
    w, h = warp_size
    src = order_points(area_points)
    # dst도 반드시 src와 같은 순서(좌상, 우상, 우하, 좌하)로 정의
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                   dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def preprocess(frame):
    """1단계: GaussianBlur(5x5)로 센서 노이즈를 줄인다."""
    return cv2.GaussianBlur(frame, (5, 5), 0)


def warp_top_view(frame, M, warp_size):
    """2단계: Perspective Transform으로 Top-view 이미지를 만든다."""
    return cv2.warpPerspective(frame, M, tuple(warp_size))


def spot_inner_box(rect, margin, warp_size):
    """
    3-a단계: 주차 칸 사각형을 margin만큼 안쪽으로 줄이고,
    Top-view 범위를 벗어나는 좌표는 잘라낸다(clip).
    반환: (x0, y0, x1, y1)  — 비어 있으면 x1<=x0 또는 y1<=y0
    """
    W, H = warp_size
    x, y, w, h = [int(round(v)) for v in rect]
    x0, y0, x1, y1 = x + margin, y + margin, x + w - margin, y + h - margin
    # 줄였더니 뒤집혔으면(칸이 너무 작음) 원래 칸을 그대로 사용
    if x1 <= x0 or y1 <= y0:
        x0, y0, x1, y1 = x, y, x + w, y + h
    x0, x1 = np.clip([x0, x1], 0, W)
    y0, y1 = np.clip([y0, y1], 0, H)
    return int(x0), int(y0), int(x1), int(y1)


def clip_rect(rect, warp_size):
    """사각형 (x, y, w, h)를 Top-view 범위 안으로 잘라낸다."""
    W, H = warp_size
    x, y, w, h = rect
    x0, y0 = max(0, min(x, W)), max(0, min(y, H))
    x1, y1 = max(0, min(x + w, W)), max(0, min(y + h, H))
    return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]


# ---------------------------------------------------------------------------
# 색 검출 / 판정
# ---------------------------------------------------------------------------
def red_mask(roi_bgr, hsv_cfg):
    """
    3-b단계: HSV로 변환 후 빨간색 영역 마스크를 만든다.
    빨강은 Hue 0 부근과 179 부근 양쪽에 걸쳐 있으므로
    inRange를 두 번 하고 OR로 합친다.
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    lo1 = (hsv_cfg["h1_min"], hsv_cfg["s_min"], hsv_cfg["v_min"])
    hi1 = (hsv_cfg["h1_max"], hsv_cfg["s_max"], hsv_cfg["v_max"])
    lo2 = (hsv_cfg["h2_min"], hsv_cfg["s_min"], hsv_cfg["v_min"])
    hi2 = (hsv_cfg["h2_max"], hsv_cfg["s_max"], hsv_cfg["v_max"])
    m1 = cv2.inRange(hsv, lo1, hi1)
    m2 = cv2.inRange(hsv, lo2, hi2)
    return cv2.bitwise_or(m1, m2)


def clean_mask(mask, ksize):
    """
    3-c단계: 모폴로지 연산으로 마스크 정리.
      MORPH_OPEN  : 작은 점 노이즈 제거
      MORPH_CLOSE : 물체 내부의 작은 구멍 메우기
    """
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1  # 커널은 홀수로 맞춘다
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def largest_contour_ratio(mask):
    """
    3-d, 3-e단계: 컨투어를 찾아 가장 큰 것의 면적을 ROI 면적으로 나눈 비율을 구한다.
    반환: (비율 0~1, 가장 큰 컨투어 또는 None, 무게중심 또는 None)
    """
    roi_area = mask.shape[0] * mask.shape[1]
    if roi_area == 0:
        return 0.0, None, None
    # OpenCV 3.x / 4.x 반환값 개수 차이를 [-2]로 흡수
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
    if not contours:  # 빈 컨투어 목록이면 점유 아님
        return 0.0, None, None
    biggest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(biggest)
    # 무게중심: m00 == 0(면적 0 컨투어)이면 계산하지 않는다
    m = cv2.moments(biggest)
    center = None
    if m["m00"] != 0:
        center = (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))
    return area / roi_area, biggest, center


def analyze_spot(warped, spot, params, warp_size):
    """주차 칸 하나에 대해 3단계(a~e)를 수행하고 결과를 dict로 반환한다."""
    x0, y0, x1, y1 = spot_inner_box(spot["rect"], params["roi_margin"], warp_size)
    if x1 <= x0 or y1 <= y0:  # 범위를 완전히 벗어난 칸
        empty = np.zeros((10, 10), np.uint8)
        return {"box": (x0, y0, x1, y1), "mask": empty, "ratio": 0.0,
                "raw_occupied": False, "center": None}
    roi = warped[y0:y1, x0:x1]
    mask = red_mask(roi, params["hsv"])
    mask = clean_mask(mask, params["morph_kernel"])
    ratio, _, center = largest_contour_ratio(mask)
    if center is not None:  # ROI 좌표 → Top-view 좌표
        center = (center[0] + x0, center[1] + y0)
    return {"box": (x0, y0, x1, y1), "mask": mask, "ratio": ratio,
            "raw_occupied": ratio >= params["occupied_ratio"], "center": center}


# ---------------------------------------------------------------------------
# 상태 안정화 (디바운스)
# ---------------------------------------------------------------------------
class Debouncer:
    """
    4단계: 판정 결과가 연속 N프레임 같을 때만 상태를 바꾼다.
    손이 지나가거나 순간 노이즈로 상태가 깜빡이는 것을 막는다.
    첫 프레임의 판정은 바로 초기 상태로 사용한다.
    """

    def __init__(self, n):
        self.n = max(1, int(n))
        self.state = None
        self.count = 0  # 현재 상태와 '다른' 판정이 연속으로 나온 횟수

    def update(self, raw):
        if self.state is None:
            self.state = raw
            return self.state
        if raw == self.state:
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.n:
                self.state = raw
                self.count = 0
        return self.state


# ---------------------------------------------------------------------------
# 화면 그리기 (항상 복사본에 그린다)
# ---------------------------------------------------------------------------
def put_label(img, text, org, color, scale=0.55, thickness=1):
    """글자 뒤에 검은 배경 상자를 깔아 가독성을 높인다."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + base), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def draw_top_view(warped, spots, results, states):
    """
    5단계: Top-view 결과 화면.
    감지는 원본 warped에서 했으므로, 그리기는 반드시 복사본(display)에 한다.
    """
    display = warped.copy()
    overlay = display.copy()
    # OCCUPIED 칸은 반투명 빨간색으로 채운다
    for spot, occ in zip(spots, states):
        if occ:
            x, y, w, h = spot["rect"]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), COLOR_RED, -1)
    cv2.addWeighted(overlay, 0.35, display, 0.65, 0, dst=display)

    for spot, res, occ in zip(spots, results, states):
        x, y, w, h = spot["rect"]
        color = COLOR_RED if occ else COLOR_GREEN
        cv2.rectangle(display, (x, y), (x + w, y + h), color, 3)
        # 실제 검사 영역(안쪽으로 줄인 ROI)은 얇은 노란 선으로 표시
        x0, y0, x1, y1 = res["box"]
        cv2.rectangle(display, (x0, y0), (x1, y1), COLOR_YELLOW, 1)
        if res["center"] is not None:
            cv2.circle(display, res["center"], 4, COLOR_YELLOW, -1)
        text = f"P{spot['id']} {'OCCUPIED' if occ else 'AVAILABLE'}"
        put_label(display, text, (x + 10, y + 28), color)

    # 상단 요약 바를 별도로 붙여서 칸 라벨을 가리지 않게 한다
    available = sum(1 for s in states if not s)
    header = np.zeros((40, display.shape[1], 3), np.uint8)
    cv2.putText(header, f"Available: {available} / {len(spots)}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                COLOR_GREEN if available else COLOR_RED, 2, cv2.LINE_AA)
    return np.vstack([header, display])


def draw_camera_view(frame, area_points, spots, states, M):
    """원본 영상에 주차장 외곽선과 (역변환한) 칸 경계를 그린다."""
    display = frame.copy()
    rect = order_points(area_points).astype(np.int32)
    cv2.polylines(display, [rect], True, COLOR_YELLOW, 2)
    for label, p in zip(("TL", "TR", "BR", "BL"), rect):
        cv2.circle(display, tuple(int(v) for v in p), 5, COLOR_YELLOW, -1)
        put_label(display, label, (int(p[0]) + 6, int(p[1]) - 6), COLOR_YELLOW, 0.5)

    # Top-view 좌표의 칸을 역행렬로 원본 좌표에 투영해서 같이 표시
    if spots:
        Minv = np.linalg.inv(M)
        for spot, occ in zip(spots, states):
            x, y, w, h = spot["rect"]
            corners = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
            proj = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), Minv)
            cv2.polylines(display, [proj.astype(np.int32)], True,
                          COLOR_RED if occ else COLOR_GREEN, 2)
    return display


def build_mask_panel(spots, results, states, height=160):
    """각 ROI의 정리된 마스크를 가로로 이어 붙이고, 아래에 점유 비율(%)을 쓴다."""
    tiles = []
    for spot, res, occ in zip(spots, results, states):
        mask = res["mask"]
        mh, mw = mask.shape[:2]
        width = max(1, int(round(mw * height / max(mh, 1))))
        tile = cv2.cvtColor(cv2.resize(mask, (width, height),
                                       interpolation=cv2.INTER_NEAREST),
                            cv2.COLOR_GRAY2BGR)
        width = max(width, 120)  # 라벨이 들어갈 최소 폭
        canvas = np.zeros((height + 40, width, 3), np.uint8)
        canvas[:height, :tile.shape[1]] = tile
        cv2.putText(canvas, f"P{spot['id']} {res['ratio'] * 100:5.1f}%",
                    (6, height + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    COLOR_RED if occ else COLOR_GREEN, 2, cv2.LINE_AA)
        tiles.append(canvas)
        tiles.append(np.full((height + 40, 4, 3), 80, np.uint8))  # 구분선
    if not tiles:
        return np.zeros((height + 40, 200, 3), np.uint8)
    return np.hstack(tiles[:-1])


# ---------------------------------------------------------------------------
# 트랙바 ("Controls" 창)
# ---------------------------------------------------------------------------
def create_controls(params):
    """현재 설정값으로 트랙바를 만든다."""
    cv2.namedWindow(WIN_CONTROLS, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CONTROLS, 420, 330)
    noop = lambda _v: None
    hsv = params["hsv"]
    cv2.createTrackbar("H1 min", WIN_CONTROLS, hsv["h1_min"], 179, noop)
    cv2.createTrackbar("H1 max", WIN_CONTROLS, hsv["h1_max"], 179, noop)
    cv2.createTrackbar("H2 min", WIN_CONTROLS, hsv["h2_min"], 179, noop)
    cv2.createTrackbar("H2 max", WIN_CONTROLS, hsv["h2_max"], 179, noop)
    cv2.createTrackbar("S min", WIN_CONTROLS, hsv["s_min"], 255, noop)
    cv2.createTrackbar("V min", WIN_CONTROLS, hsv["v_min"], 255, noop)
    cv2.createTrackbar("Kernel", WIN_CONTROLS, params["morph_kernel"], 31, noop)
    cv2.createTrackbar("Ratio %", WIN_CONTROLS,
                       int(round(params["occupied_ratio"] * 100)), 100, noop)
    cv2.createTrackbar("Debounce", WIN_CONTROLS, params["debounce_frames"], 60, noop)


def read_controls(params):
    """트랙바 값을 읽어 params를 갱신한다."""
    g = lambda name: cv2.getTrackbarPos(name, WIN_CONTROLS)
    hsv = params["hsv"]
    hsv["h1_min"], hsv["h1_max"] = g("H1 min"), g("H1 max")
    hsv["h2_min"], hsv["h2_max"] = g("H2 min"), g("H2 max")
    hsv["s_min"], hsv["v_min"] = g("S min"), g("V min")
    params["morph_kernel"] = max(1, g("Kernel"))
    params["occupied_ratio"] = max(1, g("Ratio %")) / 100.0
    params["debounce_frames"] = max(1, g("Debounce"))


# ---------------------------------------------------------------------------
# 보정 모드 1: 주차장 꼭짓점 4개 클릭
# ---------------------------------------------------------------------------
def calibrate_area(source, config_path):
    if source is None:
        return
    points = []
    guide = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            print(f"  점 {len(points)}: ({x}, {y})")

    cv2.namedWindow(WIN_CAMERA)
    cv2.setMouseCallback(WIN_CAMERA, on_mouse)
    print("[보정 1] 주차장 꼭짓점을 좌상 → 우상 → 우하 → 좌하 순으로 클릭하세요.")
    print("        r: 다시 찍기 / Enter: 저장 / q·Esc: 취소")

    while True:
        frame = source.read()
        if frame is None:
            print("[에러] 프레임을 읽을 수 없습니다.")
            break
        disp = frame.copy()
        # 찍은 점과 번호 표시
        for i, p in enumerate(points):
            cv2.circle(disp, tuple(p), 6, COLOR_YELLOW, -1)
            put_label(disp, str(i + 1), (p[0] + 8, p[1] - 8), COLOR_YELLOW, 0.7, 2)
        if len(points) >= 2:
            cv2.polylines(disp, [np.int32(points)], len(points) == 4, COLOR_YELLOW, 2)
        if len(points) < 4:
            msg = f"Click {len(points) + 1}/4: {guide[len(points)]}"
        else:
            msg = "Enter: save  /  r: reset"
            # 자동 정렬 결과도 미리 보여준다
            for lbl, p in zip(("TL", "TR", "BR", "BL"), order_points(points)):
                put_label(disp, lbl, (int(p[0]) - 30, int(p[1]) + 25), COLOR_GREEN)
        put_label(disp, msg, (10, 30), COLOR_WHITE, 0.7, 2)
        cv2.imshow(WIN_CAMERA, disp)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("r"):
            points.clear()
            print("  다시 찍기")
        elif key in ENTER_KEYS:
            if len(points) == 4:
                ordered = order_points(points).round().astype(int).tolist()
                cfg = load_config(config_path) or {}
                updates = {"area_points": ordered}
                if "warp_size" not in cfg:
                    updates["warp_size"] = DEFAULTS["warp_size"]
                save_config(config_path, updates)
                print(f"  정렬된 꼭짓점(좌상,우상,우하,좌하): {ordered}")
                print("  다음 단계: python parking.py --calibrate-spots")
                break
            print("  점 4개를 모두 찍어야 저장할 수 있습니다.")
        elif key in (ord("q"), ESC_KEY):
            print("  취소했습니다. (저장 안 함)")
            break
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# 보정 모드 2: Top-view에서 주차 칸 지정
# ---------------------------------------------------------------------------
def make_grid_spots(grid, warp_size):
    """'2x2' 같은 문자열(열x행)로 Top-view를 균등 분할한다. id는 좌→우, 위→아래."""
    try:
        cols, rows = [int(v) for v in grid.lower().split("x")]
        assert cols > 0 and rows > 0
    except (ValueError, AssertionError):
        print(f"[에러] --grid 형식이 잘못되었습니다: {grid} (예: 2x2, 4x1)")
        sys.exit(1)
    W, H = warp_size
    spots = []
    for r in range(rows):
        for c in range(cols):
            x0, x1 = round(c * W / cols), round((c + 1) * W / cols)
            y0, y1 = round(r * H / rows), round((r + 1) * H / rows)
            spots.append({"id": len(spots) + 1, "rect": [x0, y0, x1 - x0, y1 - y0]})
    return spots


def calibrate_spots(source, config_path, cfg, grid=None):
    warp_size = cfg["warp_size"]

    # --grid 옵션: 마우스 없이 자동 분할 후 바로 저장
    if grid:
        spots = make_grid_spots(grid, warp_size)
        save_config(config_path, {"spots": spots, "warp_size": warp_size})
        print(f"[보정 2] {grid} 격자로 {len(spots)}개 칸을 생성했습니다.")
        for s in spots:
            print(f"  P{s['id']}: {s['rect']}")
        return

    if source is None:
        return
    M = get_perspective_matrix(cfg["area_points"], warp_size)
    spots = []
    drag = {"start": None, "cur": None}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drag["start"], drag["cur"] = (x, y), (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drag["start"] is not None:
            drag["cur"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drag["start"] is not None:
            (sx, sy), (ex, ey) = drag["start"], (x, y)
            drag["start"] = drag["cur"] = None
            # 어느 방향으로 드래그해도 되도록 정규화 + 범위 clip
            rect = clip_rect([min(sx, ex), min(sy, ey), abs(ex - sx), abs(ey - sy)],
                             warp_size)
            if rect[2] < 10 or rect[3] < 10:
                print("  너무 작은 사각형은 무시합니다.")
                return
            spots.append({"id": len(spots) + 1, "rect": rect})
            print(f"  P{len(spots)}: {rect}")

    cv2.namedWindow(WIN_TOP)
    cv2.setMouseCallback(WIN_TOP, on_mouse)
    print("[보정 2] Top-view에서 주차 칸을 드래그하세요.")
    print("        u: 마지막 칸 취소 / Enter: 저장 / q·Esc: 취소")

    while True:
        frame = source.read()
        if frame is None:
            print("[에러] 프레임을 읽을 수 없습니다.")
            break
        warped = warp_top_view(preprocess(frame), M, warp_size)
        disp = warped.copy()
        for s in spots:
            x, y, w, h = s["rect"]
            cv2.rectangle(disp, (x, y), (x + w, y + h), COLOR_GREEN, 2)
            put_label(disp, f"P{s['id']}", (x + 8, y + 26), COLOR_GREEN, 0.7, 2)
        if drag["start"] is not None:
            cv2.rectangle(disp, drag["start"], drag["cur"], COLOR_YELLOW, 1)
        put_label(disp, f"Drag spots ({len(spots)})  u: undo  Enter: save",
                  (10, warp_size[1] - 12), COLOR_WHITE, 0.55)
        cv2.imshow(WIN_TOP, disp)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("u"):
            if spots:
                removed = spots.pop()
                print(f"  P{removed['id']} 취소")
        elif key in ENTER_KEYS:
            if spots:
                save_config(config_path, {"spots": spots, "warp_size": warp_size})
                print("  다음 단계: python parking.py")
                break
            print("  최소 1개 이상의 칸을 지정해야 합니다.")
        elif key in (ord("q"), ESC_KEY):
            print("  취소했습니다. (저장 안 함)")
            break
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# 실행 모드
# ---------------------------------------------------------------------------
def process_frame(frame, M, cfg, params, debouncers):
    """프레임 1장에 대한 전체 파이프라인. 결과 dict를 반환한다."""
    warp_size = cfg["warp_size"]
    blurred = preprocess(frame)                       # 1. 블러
    warped = warp_top_view(blurred, M, warp_size)     # 2. Top-view
    results = [analyze_spot(warped, s, params, warp_size)  # 3. 칸별 판정
               for s in cfg["spots"]]
    states = []
    for res, deb in zip(results, debouncers):         # 4. 디바운스
        deb.n = max(1, int(params["debounce_frames"]))
        states.append(deb.update(res["raw_occupied"]))
    return {"warped": warped, "results": results, "states": states}


def render_all(frame, cfg, out, M):
    """5. 세 개의 출력 화면(Camera, Top View, Masks)을 만든다."""
    spots = cfg["spots"]
    cam = draw_camera_view(frame, cfg["area_points"], spots, out["states"], M)
    top = draw_top_view(out["warped"], spots, out["results"], out["states"])
    masks = build_mask_panel(spots, out["results"], out["states"])
    return cam, top, masks


def check_config_ready(cfg, config_path):
    """실행에 필요한 보정 값이 모두 있는지 확인한다."""
    if cfg is None:
        print(f"[안내] 보정 파일이 없습니다: {config_path}")
        print("  먼저 보정 모드를 실행하세요:")
        print("    1) python parking.py --calibrate-area")
        print("    2) python parking.py --calibrate-spots   (또는 --grid 2x2)")
        sys.exit(1)
    if "area_points" not in cfg or len(cfg["area_points"]) != 4:
        print("[안내] 주차장 꼭짓점이 없습니다. python parking.py --calibrate-area 를 먼저 실행하세요.")
        sys.exit(1)
    if not cfg.get("spots"):
        print("[안내] 주차 칸 정보가 없습니다. python parking.py --calibrate-spots 를 먼저 실행하세요.")
        sys.exit(1)
    # 저장된 칸 좌표가 Top-view 범위를 벗어나면 잘라서 사용
    for s in cfg["spots"]:
        clipped = clip_rect(s["rect"], cfg["warp_size"])
        if clipped != list(s["rect"]):
            print(f"[경고] P{s['id']} 좌표가 Top-view 범위를 벗어나 잘라냈습니다: "
                  f"{s['rect']} → {clipped}")
        s["rect"] = clipped


def detection_params(cfg):
    """config에서 감지 관련 파라미터만 꺼내 별도 dict로 만든다(트랙바가 수정)."""
    return {
        "hsv": dict(cfg["hsv"]),
        "morph_kernel": int(cfg["morph_kernel"]),
        "occupied_ratio": float(cfg["occupied_ratio"]),
        "debounce_frames": int(cfg["debounce_frames"]),
        "roi_margin": int(cfg["roi_margin"]),
    }


def run_live(source, cfg, config_path, shot_dir):
    """GUI 실시간 감지 루프."""
    M = get_perspective_matrix(cfg["area_points"], cfg["warp_size"])
    params = detection_params(cfg)
    debouncers = [Debouncer(params["debounce_frames"]) for _ in cfg["spots"]]
    create_controls(params)
    print("[실행] q: 종료 / s: 스크린샷 / p: 현재 HSV·판정 비율을 config.json에 저장")

    prev_states = None
    while True:
        frame = source.read()
        if frame is None:
            print("[에러] 프레임을 읽을 수 없습니다. 종료합니다.")
            break
        read_controls(params)
        out = process_frame(frame, M, cfg, params, debouncers)
        cam, top, masks = render_all(frame, cfg, out, M)
        cv2.imshow(WIN_CAMERA, cam)
        cv2.imshow(WIN_TOP, top)
        cv2.imshow(WIN_MASKS, masks)

        # 상태가 바뀌었을 때만 콘솔에 로그
        if out["states"] != prev_states:
            summary = ", ".join(f"P{s['id']}={'OCC' if o else 'FREE'}"
                                for s, o in zip(cfg["spots"], out["states"]))
            print(f"  [{time.strftime('%H:%M:%S')}] {summary}")
            prev_states = list(out["states"])

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == ESC_KEY:
            break
        if key == ord("s"):
            os.makedirs(shot_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            for name, img in (("camera", cam), ("topview", top), ("masks", masks)):
                cv2.imwrite(os.path.join(shot_dir, f"{stamp}_{name}.png"), img)
            print(f"  스크린샷 저장: {shot_dir}/{stamp}_*.png")
        if key == ord("p"):
            save_config(config_path, {
                "hsv": params["hsv"],
                "occupied_ratio": params["occupied_ratio"],
                "morph_kernel": params["morph_kernel"],
                "debounce_frames": params["debounce_frames"],
            })
    cv2.destroyAllWindows()


def run_headless(source, cfg, args):
    """
    --no-gui 모드: 창 없이 처리 후 결과 이미지를 저장하고 콘솔에 출력한다.
    정지 이미지에서는 디바운스를 1로 강제한다(같은 프레임 반복은 의미 없음).
    """
    M = get_perspective_matrix(cfg["area_points"], cfg["warp_size"])
    params = detection_params(cfg)
    if source.is_image:
        params["debounce_frames"] = 1
    n_frames = 1 if source.is_image else max(1, args.frames)
    debouncers = [Debouncer(params["debounce_frames"]) for _ in cfg["spots"]]

    out, frame = None, None
    for _ in range(n_frames):
        frame = source.read()
        if frame is None:
            print("[에러] 프레임을 읽을 수 없습니다.")
            sys.exit(1)
        out = process_frame(frame, M, cfg, params, debouncers)

    cam, top, masks = render_all(frame, cfg, out, M)
    os.makedirs(args.out, exist_ok=True)
    paths = {
        "camera": os.path.join(args.out, "camera.png"),
        "topview_raw": os.path.join(args.out, "topview_raw.png"),
        "topview": os.path.join(args.out, "topview.png"),
        "masks": os.path.join(args.out, "masks.png"),
    }
    cv2.imwrite(paths["camera"], cam)
    cv2.imwrite(paths["topview_raw"], out["warped"])
    cv2.imwrite(paths["topview"], top)
    cv2.imwrite(paths["masks"], masks)

    print("=" * 52)
    print(" 감지 결과")
    print("=" * 52)
    detected = {}
    for spot, res, occ in zip(cfg["spots"], out["results"], out["states"]):
        status = "OCCUPIED" if occ else "AVAILABLE"
        detected[str(spot["id"])] = status
        print(f"  P{spot['id']}: {status:<9}  (빨강 비율 {res['ratio'] * 100:5.1f}%, "
              f"임계 {params['occupied_ratio'] * 100:.0f}%)")
    available = sum(1 for s in out["states"] if not s)
    print(f"  Available: {available} / {len(out['states'])}")
    print(f"  결과 이미지: {args.out}/")

    if args.expected:
        ok = compare_with_expected(args.expected, detected, out["warped"], M, args.out)
        sys.exit(0 if ok else 1)


# ---------------------------------------------------------------------------
# 테스트: 정답 JSON과 비교
# ---------------------------------------------------------------------------
def compare_with_expected(expected_path, detected, warped, M, out_dir):
    with open(expected_path, "r", encoding="utf-8") as f:
        exp = json.load(f)
    base = os.path.dirname(os.path.abspath(expected_path))
    all_ok = True

    print("=" * 52)
    print(" 정답 비교")
    print("=" * 52)

    # (1) Top-view 복원 검사 ------------------------------------------------
    # (1-a) 기하: 왜곡에 쓴 알려진 행렬 H(top→camera)와 추정 M(camera→top)을
    #       합성하면 항등변환이어야 한다. Top-view 위 격자점으로 오차를 잰다.
    if "homography_top_to_camera" in exp:
        H = np.array(exp["homography_top_to_camera"], dtype=np.float64)
        W, Hh = exp.get("warp_size", warped.shape[1::-1])
        gx, gy = np.meshgrid(np.linspace(0, W - 1, 7), np.linspace(0, Hh - 1, 5))
        grid = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float64).reshape(-1, 1, 2)
        back = cv2.perspectiveTransform(cv2.perspectiveTransform(grid, H), M)
        err = float(np.abs(back - grid).max())
        ok = err < 1.0
        all_ok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] 기하 복원 오차 최대 {err:.4f}px (기준 < 1px)")

    # (1-b) 영상: 복원된 Top-view와 원래 Top-view 이미지의 차이
    ref_rel = exp.get("topview_reference")
    if ref_rel:
        ref = cv2.imread(os.path.join(base, ref_rel))
        if ref is None:
            print(f"  [SKIP] 기준 Top-view 이미지를 읽을 수 없음: {ref_rel}")
        else:
            ref = cv2.resize(ref, (warped.shape[1], warped.shape[0]))
            ref_b = preprocess(ref)
            diff = cv2.absdiff(ref_b, warped)
            mad = float(diff.mean())
            psnr = float(cv2.PSNR(ref_b, warped))
            ok = mad < 10.0
            all_ok &= ok
            compare = np.hstack([ref_b, warped,
                                 cv2.convertScaleAbs(diff, alpha=3.0)])
            for i, t in enumerate(("reference", "restored", "|diff| x3")):
                put_label(compare, t, (i * warped.shape[1] + 10, 25), COLOR_YELLOW, 0.7, 2)
            path = os.path.join(out_dir, "topview_compare.png")
            cv2.imwrite(path, compare)
            print(f"  [{'PASS' if ok else 'FAIL'}] 영상 복원: 평균 절대오차 {mad:.2f} "
                  f"(기준 < 10), PSNR {psnr:.1f}dB  → {path}")

    # (2) 칸별 점유 판정 -----------------------------------------------------
    expected_spots = exp.get("spots", {})
    correct = 0
    for sid, want in expected_spots.items():
        got = detected.get(str(sid), "MISSING")
        ok = got == want
        correct += ok
        all_ok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] P{sid}: 정답 {want:<9} / 감지 {got}")
    print("-" * 52)
    print(f"  점유 판정 정확도: {correct} / {len(expected_spots)}")
    print(f"  최종 결과: {'ALL PASS' if all_ok else 'FAILED'}")
    return all_ok


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="OpenCV 주차 공간 감지 시스템")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--calibrate-area", action="store_true",
                      help="보정 1: 주차장 꼭짓점 4개 클릭")
    mode.add_argument("--calibrate-spots", action="store_true",
                      help="보정 2: Top-view에서 주차 칸 지정")
    p.add_argument("--grid", help="--calibrate-spots와 함께: 균등 분할 (예: 2x2)")
    p.add_argument("--image", help="웹캠 대신 사용할 정지 이미지 경로")
    p.add_argument("--camera", type=int, default=0, help="웹캠 번호 (기본 0)")
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="보정 파일 경로")
    p.add_argument("--no-gui", action="store_true", help="창 없이 처리 후 결과 저장")
    p.add_argument("--frames", type=int, default=30,
                   help="--no-gui + 웹캠일 때 처리할 프레임 수")
    p.add_argument("--out", default=os.path.join(SCRIPT_DIR, "output"),
                   help="--no-gui 결과 이미지 저장 폴더")
    p.add_argument("--expected", help="--no-gui: 비교할 정답 JSON 경로")
    return p.parse_args()


def main():
    args = parse_args()
    if args.grid and not args.calibrate_spots:
        print("[에러] --grid는 --calibrate-spots와 함께 사용하세요.")
        sys.exit(1)

    # 보정 모드 1 -----------------------------------------------------------
    if args.calibrate_area:
        if args.no_gui:
            print("[에러] --calibrate-area는 마우스 클릭이 필요하므로 --no-gui와 함께 쓸 수 없습니다.")
            sys.exit(1)
        source = FrameSource(args.image, args.camera)
        try:
            calibrate_area(source, args.config)
        finally:
            source.release()
        return

    cfg = load_config(args.config)

    # 보정 모드 2 -----------------------------------------------------------
    if args.calibrate_spots:
        if cfg is None or len(cfg.get("area_points", [])) != 4:
            print("[안내] 주차장 꼭짓점이 없습니다. 먼저 --calibrate-area 를 실행하세요.")
            sys.exit(1)
        if args.grid:
            calibrate_spots(None, args.config, cfg, grid=args.grid)
            return
        if args.no_gui:
            print("[에러] 드래그 보정은 창이 필요합니다. --grid 옵션을 쓰거나 --no-gui를 빼세요.")
            sys.exit(1)
        source = FrameSource(args.image, args.camera)
        try:
            calibrate_spots(source, args.config, cfg)
        finally:
            source.release()
        return

    # 실행 모드 -------------------------------------------------------------
    check_config_ready(cfg, args.config)
    source = FrameSource(args.image, args.camera)
    try:
        if args.no_gui:
            run_headless(source, cfg, args)
        else:
            run_live(source, cfg, args.config,
                     os.path.join(SCRIPT_DIR, "screenshots"))
    finally:
        source.release()


if __name__ == "__main__":
    main()
