"""
가상 차선 이탈 경보 (Perspective Transform + Hough Line Transform)

책상 위에 붙인 두 줄의 테이프를 '도로'로 보고,
1) 카메라 영상의 ROI(사다리꼴 4점)를 Top-view(위에서 내려다본 시점)로 변환하고
2) Top-view에서 Canny 엣지를 검출한 뒤
3) HoughLinesP로 직선을 찾아
4) 찾은 직선을 Top-view 영상 위에 붉은색으로 그린다.
덤으로, 직선들을 좌/우 차선으로 묶어 화면 중앙(=차량 위치)이
차선 중앙에서 많이 벗어나면 'LANE DEPARTURE' 경보를 띄운다.

사용 예:
    python lane_detector.py                         # 웹캠 모드 (0번 카메라)
    python lane_detector.py --camera 1              # 다른 카메라
    python lane_detector.py --video road.avi        # 동영상 파일
    python lane_detector.py --image desk.png        # 정지 이미지 + GUI 창
    python lane_detector.py --image desk.png --no-gui --output-dir out
                                                    # GUI 없이 결과를 파일로 저장

키 조작 (GUI 모드):
    q : 종료
    s : 현재 Original / Edges / Top-view 화면을 PNG로 저장
    r : ROI를 기본값으로 되돌리기
    o : 현재 ROI를 roi.json으로 저장 (다음 실행 때 --roi roi.json으로 불러오기)
    마우스 : Original 창에서 ROI 꼭짓점(초록 원)을 드래그해서 옮기기
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 설정값 (튜닝 포인트)
# ---------------------------------------------------------------------------
# Top-view 출력 크기 (폭 x 높이). 도로를 세로로 길게 본다.
WARP_WIDTH = 400
WARP_HEIGHT = 600

# 기본 ROI: 프레임 크기에 대한 비율 (좌상, 우상, 우하, 좌하 순서)
# 카메라 앞쪽 바닥(아래)은 넓게, 먼 쪽(위)은 좁게 잡는 사다리꼴
DEFAULT_SRC_RATIOS = [(0.38, 0.50), (0.62, 0.50), (0.95, 0.95), (0.05, 0.95)]

# Canny / HoughLinesP 파라미터 (문제의 힌트 값이 기본값)
BLUR_KSIZE = 5            # Canny 전에 GaussianBlur 커널 크기 (책상 무늬 노이즈 제거)
CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_RHO = 1             # 거리 해상도 (px)
HOUGH_THETA = np.pi / 180  # 각도 해상도 (1도)
HOUGH_THRESHOLD = 50      # 누적 투표 수 임계값: 낮출수록 짧고 약한 직선까지 검출
MIN_LINE_LENGTH = 50      # 이보다 짧은 선분은 버림
MAX_LINE_GAP = 10         # 같은 선으로 이어 붙일 최대 끊김 간격

LINE_COLOR = (0, 0, 255)  # 검출된 직선 색 (BGR 붉은색)
LINE_THICKNESS = 3

# 차선 판정 (경보용)
MAX_ANGLE_FROM_VERTICAL = 25   # Top-view에서 차선은 거의 세로. 이 각도(도)보다 기울면 차선 후보에서 제외
LANE_MERGE_GAP = 40            # 아래쪽 x 좌표가 이 간격(px) 이내인 선들은 같은 차선(테이프 양쪽 엣지)으로 묶음
MIN_LANE_WIDTH = 80            # 두 차선 사이가 이보다 좁으면 차선 2개로 보지 않음
DEPARTURE_RATIO = 0.5          # (차량 x - 차선 중앙) / (차선 폭/2) 의 절댓값이 이보다 크면 경보
CAR_HALF_WIDTH_RATIO = 0.15    # 차선이 1개만 보일 때, 차량 중앙에서 Top-view 폭의 15% 안에 있으면 '선 밟음' 경보


# ---------------------------------------------------------------------------
# 1. ROI 설정 + Perspective Transform (Top-view 생성)
# ---------------------------------------------------------------------------
def default_src_points(frame_w, frame_h):
    """프레임 크기에 맞춘 기본 ROI 4점 (좌상, 우상, 우하, 좌하)을 반환한다."""
    return np.float32([[x * frame_w, y * frame_h] for x, y in DEFAULT_SRC_RATIOS])


def dst_points(width=WARP_WIDTH, height=WARP_HEIGHT):
    """Top-view 출력 화면의 4점. ROI 사다리꼴이 이 직사각형 전체로 펴진다."""
    return np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])


def get_transform(src_pts, width=WARP_WIDTH, height=WARP_HEIGHT):
    """원본→Top-view 변환 행렬 M을 구한다."""
    return cv2.getPerspectiveTransform(np.float32(src_pts), dst_points(width, height))


def warp_to_topview(frame, M, width=WARP_WIDTH, height=WARP_HEIGHT):
    """원본 프레임을 Top-view로 변환한다."""
    return cv2.warpPerspective(frame, M, (width, height))


# ---------------------------------------------------------------------------
# 2. Canny Edge Detection (Top-view 위에서)
# ---------------------------------------------------------------------------
def detect_edges(warped, low=CANNY_LOW, high=CANNY_HIGH):
    """Top-view 영상 → 그레이스케일 → GaussianBlur → Canny 엣지."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if warped.ndim == 3 else warped
    blurred = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)
    return cv2.Canny(blurred, low, high)


# ---------------------------------------------------------------------------
# 3. HoughLinesP 직선 검출
# ---------------------------------------------------------------------------
def detect_lines(edges, threshold=HOUGH_THRESHOLD, min_line_length=MIN_LINE_LENGTH,
                 max_line_gap=MAX_LINE_GAP):
    """확률적 허프 변환으로 선분을 찾는다. 결과는 (N, 4) int 배열 [x1, y1, x2, y2]."""
    lines = cv2.HoughLinesP(edges, HOUGH_RHO, HOUGH_THETA, threshold,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    if lines is None:
        return np.empty((0, 4), dtype=np.int32)
    return lines.reshape(-1, 4)


# ---------------------------------------------------------------------------
# 4. 직선 그리기 (Top-view 위에 붉은색)
# ---------------------------------------------------------------------------
def draw_lines(img, lines, color=LINE_COLOR, thickness=LINE_THICKNESS):
    """검출된 선분들을 img 위에 그린다 (img를 직접 수정)."""
    for x1, y1, x2, y2 in lines:
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return img


# ---------------------------------------------------------------------------
# 5. 차선 판정 (이탈 경보)
# ---------------------------------------------------------------------------
def line_angle_from_vertical(line):
    """선분이 세로축에서 몇 도 기울었는지 (0 = 완전 세로, 90 = 가로)."""
    x1, y1, x2, y2 = line
    return abs(np.degrees(np.arctan2(abs(x2 - x1), abs(y2 - y1) + 1e-9)))


def x_at_y(line, y):
    """선분을 연장했을 때 높이 y에서의 x 좌표."""
    x1, y1, x2, y2 = [float(v) for v in line]
    if abs(y2 - y1) < 1e-6:
        return (x1 + x2) / 2.0
    return x1 + (x2 - x1) * (y - y1) / (y2 - y1)


def group_lanes(lines, height=WARP_HEIGHT, max_angle=MAX_ANGLE_FROM_VERTICAL,
                merge_gap=LANE_MERGE_GAP):
    """세로에 가까운 선분만 골라, 화면 아래쪽(차량 위치)에서의 x로 묶는다.

    테이프 한 줄은 양쪽 경계에서 엣지가 2개 나오므로, 가까운 선들은 하나의 차선으로 합친다.
    반환: 차선 x 좌표 리스트 (왼쪽→오른쪽, 선분 길이로 가중 평균)
    """
    cands = []
    for ln in lines:
        if line_angle_from_vertical(ln) > max_angle:
            continue
        length = float(np.hypot(ln[2] - ln[0], ln[3] - ln[1]))
        cands.append((x_at_y(ln, height - 1), length))
    if not cands:
        return []

    cands.sort(key=lambda c: c[0])
    groups = [[cands[0]]]
    for c in cands[1:]:
        if c[0] - groups[-1][-1][0] <= merge_gap:
            groups[-1].append(c)
        else:
            groups.append([c])

    lanes = []
    for g in groups:
        xs = np.array([c[0] for c in g])
        ws = np.array([c[1] for c in g])
        lanes.append(float(np.sum(xs * ws) / np.sum(ws)))
    return lanes


def pick_lane_pair(lanes, car_x, min_width=MIN_LANE_WIDTH):
    """여러 차선 후보 중 차량(car_x)을 사이에 두는 좌/우 한 쌍을 고른다.

    차량을 감싸는 쌍이 없으면(이미 선을 넘은 경우 등) 차량에 가장 가까운 2개를 쓴다.
    """
    left = [x for x in lanes if x <= car_x]
    right = [x for x in lanes if x > car_x]
    if left and right and right[0] - left[-1] >= min_width:
        return left[-1], right[0]
    near = sorted(lanes, key=lambda x: abs(x - car_x))[:2]
    near.sort()
    if len(near) == 2 and near[1] - near[0] >= min_width:
        return near[0], near[1]
    return None


def assess_departure(lanes, width=WARP_WIDTH):
    """차선 x 목록으로 이탈 여부를 판단한다.

    차량 위치 = Top-view 화면 아래쪽 중앙 (카메라가 차량 정면에 달려 있다고 가정).
    반환 dict:
        status : "OK" | "LEFT" | "RIGHT" | "ONE_LANE" | "NO_LANE"
        warning: 경보 여부 (bool)
        offset : (차량 x - 차선 중앙) / (차선 폭/2). 음수=왼쪽으로 치우침. 판단 불가면 None
        left, right, center: 사용한 차선 x (없으면 None)
    """
    car_x = (width - 1) / 2.0
    result = {"status": "NO_LANE", "warning": False, "offset": None,
              "left": None, "right": None, "center": None, "car_x": car_x}
    if not lanes:
        return result

    pair = pick_lane_pair(lanes, car_x)
    if pair is not None:
        left, right = pair
        center = (left + right) / 2.0
        offset = (car_x - center) / ((right - left) / 2.0)
        result.update(left=left, right=right, center=center, offset=float(offset))
        if offset < -DEPARTURE_RATIO:
            result.update(status="LEFT", warning=True)
        elif offset > DEPARTURE_RATIO:
            result.update(status="RIGHT", warning=True)
        else:
            result["status"] = "OK"
        return result

    # 차선이 하나만 보이는 경우: 차량 바로 밑에 있으면 선을 밟고 있는 것
    lane = min(lanes, key=lambda x: abs(x - car_x))
    if abs(lane - car_x) < CAR_HALF_WIDTH_RATIO * width:
        # 선이 차량 중앙보다 왼쪽에 있으면 왼쪽 차선을 밟은 것
        result.update(status="LEFT" if lane <= car_x else "RIGHT", warning=True)
    else:
        result["status"] = "ONE_LANE"
    if lane <= car_x:
        result["left"] = lane
    else:
        result["right"] = lane
    return result


# ---------------------------------------------------------------------------
# 6. 한 프레임 전체 처리
# ---------------------------------------------------------------------------
def process_frame(frame, src_pts, canny_low=CANNY_LOW, canny_high=CANNY_HIGH,
                  hough_threshold=HOUGH_THRESHOLD, min_line_length=MIN_LINE_LENGTH,
                  max_line_gap=MAX_LINE_GAP):
    """원본 프레임 → Top-view → Canny → HoughLinesP → 차선 판정.

    반환 dict: warped(원본 Top-view), edges, lines, lanes, departure
    """
    M = get_transform(src_pts)
    warped = warp_to_topview(frame, M)
    edges = detect_edges(warped, canny_low, canny_high)
    lines = detect_lines(edges, hough_threshold, min_line_length, max_line_gap)
    lanes = group_lanes(lines)
    departure = assess_departure(lanes)
    return {"warped": warped, "edges": edges, "lines": lines,
            "lanes": lanes, "departure": departure}


# ---------------------------------------------------------------------------
# 7. 시각화
# ---------------------------------------------------------------------------
STATUS_TEXT = {
    "OK": ("LANE OK", (0, 200, 0)),
    "LEFT": ("LANE DEPARTURE: LEFT", (0, 0, 255)),
    "RIGHT": ("LANE DEPARTURE: RIGHT", (0, 0, 255)),
    "ONE_LANE": ("ONE LANE ONLY", (0, 200, 255)),
    "NO_LANE": ("NO LANE", (160, 160, 160)),
}


def draw_status(img, departure, blink=True):
    """이미지 위쪽에 상태 배너를 그린다. 경보 중이면 테두리도 빨갛게."""
    text, color = STATUS_TEXT[departure["status"]]
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    if departure["offset"] is not None:
        off = f"offset {departure['offset']:+.2f}"
        (tw, _), _ = cv2.getTextSize(off, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(img, off, (w - tw - 8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    if departure["warning"] and blink:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
    return img


def draw_topview(warped, lines, departure, blink=True):
    """Top-view 위에 검출 직선(붉은색) + 차선 위치/차량 위치 표시."""
    vis = warped.copy()
    h = vis.shape[0]
    draw_lines(vis, lines)                       # 요구사항: 검출 직선을 붉은색으로
    for key in ("left", "right"):                # 판정에 쓴 차선 위치 (아래쪽 노란 삼각형)
        x = departure[key]
        if x is not None:
            x = int(round(x))
            pts = np.array([[x, h - 22], [x - 9, h - 4], [x + 9, h - 4]], np.int32)
            cv2.fillPoly(vis, [pts], (0, 255, 255))
    if departure["center"] is not None:          # 차선 중앙 (초록 점선)
        cx = int(round(departure["center"]))
        for y in range(40, h, 20):
            cv2.line(vis, (cx, y), (cx, y + 10), (0, 255, 0), 2)
    car_x = int(round(departure["car_x"]))       # 차량 위치 (흰 막대)
    cv2.rectangle(vis, (car_x - 3, h - 60), (car_x + 3, h - 1), (255, 255, 255), -1)
    return draw_status(vis, departure, blink)


def draw_roi(frame, src_pts, departure=None, blink=True):
    """원본 영상 위에 ROI 사다리꼴과 꼭짓점 핸들을 그린다 (직선은 그리지 않음)."""
    vis = frame.copy()
    pts = np.int32(np.round(src_pts))
    cv2.polylines(vis, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
    for (x, y), name in zip(pts, ("TL", "TR", "BR", "BL")):
        cv2.circle(vis, (int(x), int(y)), 7, (0, 255, 0), -1)
        cv2.putText(vis, name, (int(x) + 9, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
    if departure is not None:
        draw_status(vis, departure, blink)
    return vis


# ---------------------------------------------------------------------------
# 8. ROI 저장/불러오기, 마우스 편집
# ---------------------------------------------------------------------------
def load_roi(path):
    with open(path, encoding="utf-8") as f:
        pts = np.float32(json.load(f)["src_pts"])
    if pts.shape != (4, 2):
        raise ValueError("roi 파일의 src_pts는 [[x,y]] 4개여야 합니다")
    return pts


def save_roi(path, src_pts):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"src_pts": np.round(src_pts, 1).tolist(),
                   "order": ["TL", "TR", "BR", "BL"]}, f, indent=2)


def parse_src(text):
    """'x1,y1,x2,y2,x3,y3,x4,y4' 문자열 → (4, 2) float32."""
    vals = [float(v) for v in text.replace(" ", "").split(",") if v]
    if len(vals) != 8:
        raise argparse.ArgumentTypeError("--src 는 숫자 8개(TL,TR,BR,BL의 x,y)여야 합니다")
    return np.float32(vals).reshape(4, 2)


class RoiEditor:
    """Original 창에서 가장 가까운 꼭짓점을 마우스로 드래그해 ROI를 고친다."""

    GRAB_RADIUS = 25

    def __init__(self, src_pts, frame_size):
        self.pts = np.float32(src_pts).copy()
        self.w, self.h = frame_size
        self.active = None

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            d = np.hypot(self.pts[:, 0] - x, self.pts[:, 1] - y)
            i = int(np.argmin(d))
            self.active = i if d[i] < self.GRAB_RADIUS else None
        elif event == cv2.EVENT_MOUSEMOVE and self.active is not None:
            self.pts[self.active] = [np.clip(x, 0, self.w - 1), np.clip(y, 0, self.h - 1)]
        elif event == cv2.EVENT_LBUTTONUP:
            self.active = None


# ---------------------------------------------------------------------------
# 9. 실행 모드
# ---------------------------------------------------------------------------
WIN_ORIG, WIN_EDGES, WIN_TOP, WIN_CTRL = "Original (ROI)", "Top-view Edges", "Top-view Lanes", "Controls"


def create_controls(args):
    """Canny/Hough 값을 실시간으로 바꿀 수 있는 트랙바 창."""
    cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CTRL, 420, 240)
    for name, val, mx in (("Canny low", args.canny_low, 255),
                          ("Canny high", args.canny_high, 255),
                          ("Hough thresh", args.hough_threshold, 300),
                          ("Min length", args.min_line_length, 300),
                          ("Max gap", args.max_line_gap, 100)):
        cv2.createTrackbar(name, WIN_CTRL, val, mx, lambda v: None)


def read_controls():
    return {"canny_low": cv2.getTrackbarPos("Canny low", WIN_CTRL),
            "canny_high": cv2.getTrackbarPos("Canny high", WIN_CTRL),
            "hough_threshold": max(1, cv2.getTrackbarPos("Hough thresh", WIN_CTRL)),
            "min_line_length": cv2.getTrackbarPos("Min length", WIN_CTRL),
            "max_line_gap": cv2.getTrackbarPos("Max gap", WIN_CTRL)}


def params_from_args(args):
    return {"canny_low": args.canny_low, "canny_high": args.canny_high,
            "hough_threshold": args.hough_threshold,
            "min_line_length": args.min_line_length, "max_line_gap": args.max_line_gap}


def initial_roi(args, frame_w, frame_h):
    if args.src is not None:
        return args.src
    if args.roi:
        return load_roi(args.roi)
    return default_src_points(frame_w, frame_h)


def save_views(output_dir, stem, views):
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for suffix, img in views.items():
        p = os.path.join(output_dir, f"{stem}_{suffix}.png")
        cv2.imwrite(p, img)
        paths.append(p)
    return paths


def print_result(res):
    dep = res["departure"]
    print(f"lines: {len(res['lines'])}")
    print("lanes: " + (", ".join(f"{x:.1f}" for x in res["lanes"]) or "-"))
    print(f"status: {dep['status']}")
    print(f"warning: {dep['warning']}")
    print("offset: " + ("-" if dep["offset"] is None else f"{dep['offset']:+.3f}"))


def run_image(args):
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"[ERROR] 이미지를 열 수 없습니다: {args.image}", file=sys.stderr)
        return 1
    h, w = frame.shape[:2]
    src = initial_roi(args, w, h)
    stem = os.path.splitext(os.path.basename(args.image))[0]

    if args.no_gui:
        res = process_frame(frame, src, **params_from_args(args))
        print_result(res)
        paths = save_views(args.output_dir, stem, {
            "original": draw_roi(frame, src, res["departure"], blink=False),
            "edges": res["edges"],
            "topview": draw_topview(res["warped"], res["lines"], res["departure"], blink=False),
        })
        for p in paths:
            print(f"saved: {p}")
        return 0 if len(res["lines"]) > 0 else 2

    # 정지 이미지도 ROI 드래그 / 트랙바 조절이 되도록 루프를 돈다
    return gui_loop(lambda: (True, frame), args, src, (w, h), stem, still=True)


def run_stream(args):
    source = args.video if args.video else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] 영상 소스를 열 수 없습니다: {source}", file=sys.stderr)
        return 1
    ok, frame = cap.read()
    if not ok:
        print("[ERROR] 첫 프레임을 읽지 못했습니다.", file=sys.stderr)
        cap.release()
        return 1
    h, w = frame.shape[:2]
    src = initial_roi(args, w, h)
    first = [frame]

    def read():
        if first:
            return True, first.pop()
        ok, f = cap.read()
        if not ok and args.video:                # 동영상은 끝나면 처음부터 반복
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, f = cap.read()
        return ok, f

    try:
        return gui_loop(read, args, src, (w, h), "lane")
    finally:
        cap.release()


def gui_loop(read, args, src, frame_size, stem, still=False):
    editor = RoiEditor(src, frame_size)
    cv2.namedWindow(WIN_ORIG)
    cv2.setMouseCallback(WIN_ORIG, editor.on_mouse)
    cv2.namedWindow(WIN_EDGES)
    cv2.namedWindow(WIN_TOP)
    create_controls(args)

    frame = None
    while True:
        if frame is None or not still:
            ok, frame = read()
            if not ok:
                print("[ERROR] 프레임을 읽지 못했습니다.", file=sys.stderr)
                return 1
        res = process_frame(frame, editor.pts, **read_controls())
        blink = int(time.time() * 4) % 2 == 0
        orig_vis = draw_roi(frame, editor.pts, res["departure"], blink)
        top_vis = draw_topview(res["warped"], res["lines"], res["departure"], blink)
        cv2.imshow(WIN_ORIG, orig_vis)
        cv2.imshow(WIN_EDGES, res["edges"])
        cv2.imshow(WIN_TOP, top_vis)

        key = cv2.waitKey(30 if still else 1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            ts = time.strftime("%Y%m%d_%H%M%S")
            for p in save_views("captures", f"{stem}_{ts}", {
                    "original": orig_vis, "edges": res["edges"], "topview": top_vis}):
                print(f"saved: {p}")
        elif key == ord("r"):
            editor.pts = default_src_points(*frame_size)
            print("ROI를 기본값으로 되돌렸습니다.")
        elif key == ord("o"):
            save_roi("roi.json", editor.pts)
            print("saved: roi.json  ->", np.round(editor.pts, 1).tolist())
    cv2.destroyAllWindows()
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="가상 차선 이탈 경보 (Perspective + HoughLinesP)")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=0, help="웹캠 번호 (기본 0)")
    src.add_argument("--video", help="웹캠 대신 사용할 동영상 파일")
    src.add_argument("--image", help="웹캠 대신 사용할 정지 이미지")
    p.add_argument("--no-gui", action="store_true", help="(--image 전용) 창 없이 결과를 파일로 저장")
    p.add_argument("--output-dir", default="output", help="--no-gui 결과 저장 폴더")
    p.add_argument("--roi", help="ROI 4점을 담은 json 파일 (키 o로 저장한 roi.json)")
    p.add_argument("--src", type=parse_src, help="ROI 4점 직접 지정: 'x1,y1,x2,y2,x3,y3,x4,y4' (TL,TR,BR,BL)")
    p.add_argument("--canny-low", type=int, default=CANNY_LOW)
    p.add_argument("--canny-high", type=int, default=CANNY_HIGH)
    p.add_argument("--hough-threshold", type=int, default=HOUGH_THRESHOLD)
    p.add_argument("--min-line-length", type=int, default=MIN_LINE_LENGTH)
    p.add_argument("--max-line-gap", type=int, default=MAX_LINE_GAP)
    args = p.parse_args(argv)
    if args.no_gui and not args.image:
        p.error("--no-gui 는 --image 와 함께 써야 합니다")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.image:
        return run_image(args)
    return run_stream(args)


if __name__ == "__main__":
    sys.exit(main())
