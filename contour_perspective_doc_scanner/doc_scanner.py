"""
웹캠 기반 문서 스캐너 (Contour 검출 + Perspective Transform)

어두운 배경 위의 흰 A4 용지 외곽선을 실시간으로 찾고,
정면에서 본 것처럼 펼친(warp) 결과를 보여준다.

사용 예:
    python doc_scanner.py                          # 웹캠 모드
    python doc_scanner.py --image test.png         # 정지 이미지 + GUI 창
    python doc_scanner.py --image test.png --no-gui --output-dir out
                                                   # GUI 없이 결과를 파일로 저장

키 조작 (GUI 모드):
    q : 종료
    s : 현재 Warped 이미지를 PNG로 저장
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 설정값 (튜닝 포인트)
# ---------------------------------------------------------------------------
CANNY_LOW = 75            # Canny 하단 임계값: 낮출수록 약한 엣지까지 검출
CANNY_HIGH = 200          # Canny 상단 임계값: 낮출수록 엣지가 많아짐(노이즈도 증가)
CANNY_RETRY_SCALES = (1.0, 0.5, 0.25)  # 못 찾으면 임계값에 이 배율을 곱해 재시도
MIN_AREA_RATIO = 0.10     # 화면 면적 대비 최소 컨투어 면적 비율 (10%)
TOP_N_CONTOURS = 10       # 면적 상위 몇 개의 컨투어를 순회할지
                          # (RETR_LIST는 테두리 하나당 바깥/안쪽 2개가 나오므로 5x2)
APPROX_EPS_RATIO = 0.02   # approxPolyDP epsilon = 0.02 * 둘레
HULL_EPS_RATIOS = (0.02, 0.03, 0.04, 0.05)  # 볼록 껍질 근사 시 점점 키워볼 epsilon
                                            # (손가락 등 튀어나온 부분을 흡수)
USE_THRESHOLD_PATH = True  # Canny 외에 밝기(Otsu) 이진화 경로도 함께 사용
DETECT_WIDTH = 640        # 이 폭으로 줄여서 검출 (0이면 원본 크기로 검출)
MIN_SIDE_SUPPORT = 0.75   # 사각형 각 변의 최소 몇 %가 실제 엣지 위에 있어야 하는지
BORDER_MARGIN = 4         # 화면 가장자리 몇 px는 변 검증에서 엣지로 치지 않음
SMOOTH_ALPHA = 0.5        # 웹캠 꼭짓점 EMA 계수 (1이면 보정 없음, 작을수록 부드러움)
SMOOTH_RESET_PX = 40      # 꼭짓점이 이만큼 넘게 튀면 보정을 초기화(새 문서로 간주)

# A4 비율(210:297)을 유지한 출력 크기
WARP_WIDTH = 420
WARP_HEIGHT = 594


# ---------------------------------------------------------------------------
# 1. 프레임 입력
# ---------------------------------------------------------------------------
def open_camera(index=0):
    """카메라를 연다. 실패하면 None을 반환한다."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    return cap


# ---------------------------------------------------------------------------
# 2. 전처리: 그레이스케일 → GaussianBlur(5x5) → Canny → dilate
# ---------------------------------------------------------------------------
def preprocess(frame, canny_low=CANNY_LOW, canny_high=CANNY_HIGH):
    """프레임을 받아 외곽선 검출용 엣지 이미지를 반환한다."""
    # 색 정보는 필요 없으므로 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 5x5 가우시안 블러로 노이즈(종이 질감, 센서 노이즈) 제거
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny로 엣지 검출
    edges = cv2.Canny(blurred, canny_low, canny_high)
    # 엣지가 중간에 끊겨 컨투어가 열리지 않도록 한 번 팽창(dilate)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    return edges


# ---------------------------------------------------------------------------
# 3. 컨투어 검출: 면적 상위 N개 중 꼭짓점 4개 + 검증을 통과한 가장 큰 것 선택
# ---------------------------------------------------------------------------
def approx_quad(cnt):
    """컨투어를 4각형으로 근사한다. 실패하면 None.
    1차: 컨투어 그대로 approxPolyDP(0.02 * 둘레)
    2차: 볼록 껍질(convex hull)로 오목한 부분(가려진 테두리)을 메우고,
         epsilon을 조금씩 키워 작게 튀어나온 부분(손가락 등)을 흡수"""
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, APPROX_EPS_RATIO * peri, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        return approx.reshape(4, 2).astype(np.float32)

    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    for eps in HULL_EPS_RATIOS:
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
        if len(approx) < 4:
            break  # 이미 4개 미만이면 epsilon을 더 키워도 소용없음
    return None


def side_support(edge_check, quad, samples=40):
    """사각형 네 변을 따라 점을 찍어, 실제 엣지 위에 있는 비율 중 최소값을 반환.
    볼록 껍질이 엉뚱한 곳을 가로질러 만든 '가짜 변'을 걸러내는 용도."""
    h, w = edge_check.shape[:2]
    t = np.linspace(0.1, 0.9, samples)[:, None]  # 모서리 근처는 제외
    worst = 1.0
    for i in range(4):
        p, q = quad[i], quad[(i + 1) % 4]
        pts = p + (q - p) * t
        xs = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
        ys = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
        worst = min(worst, float((edge_check[ys, xs] > 0).mean()))
    return worst


def find_document_contour(edges, min_area_ratio=MIN_AREA_RATIO,
                          top_n=TOP_N_CONTOURS, mode=cv2.RETR_LIST,
                          edge_check=None):
    """엣지(또는 이진 마스크) 이미지에서 문서로 보이는 사각형의
    4개 꼭짓점(4x2 float32)을 찾는다. 찾지 못하면 None을 반환한다."""
    # RETR_LIST: 바깥 컨투어뿐 아니라 '엣지 고리의 안쪽 경계'도 얻는다.
    # 종이 테두리가 다른 물체의 엣지와 붙어 바깥 컨투어가 망가져도
    # 안쪽 경계는 종이 모양 그대로 남기 때문에 검출이 훨씬 안정적이다.
    contours, _ = cv2.findContours(edges, mode, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None  # 컨투어가 하나도 없으면 다음 프레임으로

    # 화면 면적 기준 최소 면적 (이보다 작은 사각형은 문서가 아니라고 간주)
    h, w = edges.shape[:2]
    min_area = h * w * min_area_ratio

    # 변 검증용: 엣지를 조금 더 두껍게 만들어 1~2px 어긋남은 허용
    if edge_check is None:
        edge_check = cv2.dilate(edges, np.ones((5, 5), np.uint8))
    # 화면 가장자리는 '엣지 증거'로 인정하지 않는다.
    # (배경 무늬가 화면 전체를 덮으면 컨투어가 화면 테두리를 따라가며
    #  화면 자체를 문서로 오인하는 것을 방지)
    edge_check = edge_check.copy()
    m = BORDER_MARGIN
    edge_check[:m, :] = 0
    edge_check[-m:, :] = 0
    edge_check[:, :m] = 0
    edge_check[:, -m:] = 0

    # 볼록 껍질 면적 기준 내림차순 정렬 후 상위 N개만 순회
    # (볼록 껍질 기준이어야 테두리가 가려져 오목해진 컨투어도 순위에서 밀리지 않음)
    contours = sorted(contours, key=lambda c: cv2.contourArea(cv2.convexHull(c)),
                      reverse=True)[:top_n]

    for cnt in contours:
        if cv2.contourArea(cv2.convexHull(cnt)) < min_area:
            break  # 내림차순이므로 이후는 모두 더 작음 → 중단
        quad = approx_quad(cnt)
        if quad is None:
            continue
        if cv2.contourArea(quad) < min_area:
            continue
        # 꼭짓점이 화면 가장자리에 붙어 있으면 종이가 화면 밖으로 잘린 것 → 제외
        if (quad[:, 0].min() < m or quad[:, 1].min() < m or
                quad[:, 0].max() > w - 1 - m or quad[:, 1].max() > h - 1 - m):
            continue
        # 네 변이 모두 실제 엣지 위에 놓여 있는지
        # (볼록 껍질이 빈 공간을 가로질러 만든 가짜 변, 끊긴 엣지 조각을 걸러냄)
        if side_support(edge_check, order_points(quad)) < MIN_SIDE_SUPPORT:
            continue
        return quad

    return None  # 4각형을 찾지 못함


def find_by_threshold(small, min_area_ratio=MIN_AREA_RATIO):
    """보조 경로: '어두운 배경 위 흰 종이'라는 전제를 이용해 밝기로 종이 영역을 찾는다.
    종이 안의 글자/테두리, 테두리에 붙은 가는 선의 영향을 받지 않는다."""
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Otsu: 배경(어두움)과 종이(밝음)를 가르는 임계값을 자동으로 결정
    _, mask = cv2.threshold(blurred, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # 가는 선·잡티 제거
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 작은 틈 메우기
    # 변 검증은 마스크의 경계선 기준
    border = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    edge_check = cv2.dilate(border, kernel)
    return find_document_contour(mask, min_area_ratio, mode=cv2.RETR_EXTERNAL,
                                 edge_check=edge_check)


# ---------------------------------------------------------------------------
# 4. 꼭짓점 정렬: 좌상, 우상, 우하, 좌하
# ---------------------------------------------------------------------------
def order_points(pts):
    """임의 순서의 4점을 [좌상, 우상, 우하, 좌하] 순서로 정렬한다."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)

    # x + y 합: 최소 = 좌상(원점에 가장 가까움), 최대 = 우하
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # y - x 차이: 최소 = 우상(x 크고 y 작음), 최대 = 좌하(x 작고 y 큼)
    d = np.diff(pts, axis=1).ravel()  # 각 점의 (y - x)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


# ---------------------------------------------------------------------------
# 5. Perspective Transform
# ---------------------------------------------------------------------------
def warp_document(frame, pts, width=WARP_WIDTH, height=WARP_HEIGHT):
    """4개 꼭짓점 영역을 A4 비율 직사각형으로 펼친다."""
    # 원본 꼭짓점: 좌상, 우상, 우하, 좌하
    pts1 = order_points(pts)
    # 목적지 꼭짓점도 반드시 같은 순서(좌상, 우상, 우하, 좌하)
    pts2 = np.array([
        [0, 0],                    # 좌상
        [width - 1, 0],            # 우상
        [width - 1, height - 1],   # 우하
        [0, height - 1],           # 좌하
    ], dtype=np.float32)
    # 3x3 투시 변환 행렬 계산 후 적용
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(frame, matrix, (width, height))
    return warped


# ---------------------------------------------------------------------------
# 시각화 & 전체 파이프라인
# ---------------------------------------------------------------------------
def draw_detection(frame, pts):
    """원본 위에 검출된 사각형(초록 선)과 꼭짓점(원)을 그린다."""
    vis = frame.copy()
    if pts is not None:
        ordered = order_points(pts).astype(np.int32)
        cv2.polylines(vis, [ordered], True, (0, 255, 0), 2)
        # 꼭짓점 원 + 순서 라벨 (TL/TR/BR/BL) — 정렬이 맞는지 눈으로 확인용
        for (x, y), label in zip(ordered, ("TL", "TR", "BR", "BL")):
            cv2.circle(vis, (int(x), int(y)), 8, (0, 255, 0), -1)
            cv2.putText(vis, label, (int(x) + 10, int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(vis, "No document", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return vis


def detect_document(frame, canny_low=CANNY_LOW, canny_high=CANNY_HIGH,
                    min_area_ratio=MIN_AREA_RATIO, detect_width=DETECT_WIDTH):
    """프레임에서 문서 꼭짓점을 찾는다 (원본 좌표계).
    반환: (edges, pts 또는 None)  — edges는 검출 해상도 기준"""
    # 검출은 축소 이미지에서: 노이즈·흐림이 줄고 속도도 빨라진다
    scale = 1.0
    small = frame
    if detect_width and frame.shape[1] > detect_width:
        scale = detect_width / frame.shape[1]
        small = cv2.resize(frame, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)

    # 후보 수집:
    #  - Canny: 설정 임계값 + 낮춘 임계값(흐린 영상·저조도에서 엣지가 끊기는 경우)
    #  - 밝기(Otsu) 이진화: 종이 테두리가 주변 물체와 엉킨 경우
    candidates = []
    edges = None
    for s in CANNY_RETRY_SCALES:
        e = preprocess(small, canny_low * s, canny_high * s)
        if edges is None:
            edges = e  # Edges 창에는 설정한 임계값의 결과를 보여준다 (튜닝용)
        quad = find_document_contour(e, min_area_ratio)
        if quad is not None:
            candidates.append(quad)
    if USE_THRESHOLD_PATH:
        quad = find_by_threshold(small, min_area_ratio)
        if quad is not None:
            candidates.append(quad)

    # 모든 경로의 후보 중 가장 큰 사각형 선택
    # (종이를 놓치고 종이 안쪽의 표·테두리를 대신 잡는 것을 방지)
    pts = max(candidates, key=cv2.contourArea) if candidates else None

    if pts is not None:
        if scale != 1.0:
            # 축소 좌표 → 원본 좌표 (픽셀 중심 기준 보정 포함)
            pts = (pts + 0.5) / scale - 0.5
        # 원본 해상도에서 꼭짓점 위치를 서브픽셀 단위로 다듬는다
        # (축소 검출 + dilate로 생긴 몇 px 오차 보정)
        pts = refine_corners(frame, pts, win=max(3, int(round(3 / scale))))
    return edges, pts


def refine_corners(frame, pts, win=5):
    """cornerSubPix로 각 꼭짓점을 실제 종이 모서리에 맞춘다.
    보정량이 비정상적으로 크면(엉뚱한 모서리로 끌려감) 원래 값을 유지한다."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    # 창이 화면 밖으로 나가지 않는 꼭짓점만 보정
    inside = ((pts[:, 0] >= win + 1) & (pts[:, 0] < w - win - 1) &
              (pts[:, 1] >= win + 1) & (pts[:, 1] < h - win - 1))
    if not inside.any():
        return pts
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.05)
    refined = pts.copy()
    sub = pts[inside].reshape(-1, 1, 2).astype(np.float32).copy()
    sub = cv2.cornerSubPix(gray, sub, (win, win), (-1, -1), criteria)
    sub = sub.reshape(-1, 2)
    moved = np.linalg.norm(sub - pts[inside], axis=1)
    ok = moved <= win  # 창 크기 이상 움직였으면 신뢰하지 않음
    idx = np.flatnonzero(inside)
    refined[idx[ok]] = sub[ok]
    return refined


def process_frame(frame, canny_low=CANNY_LOW, canny_high=CANNY_HIGH,
                  min_area_ratio=MIN_AREA_RATIO, detect_width=DETECT_WIDTH):
    """한 프레임에 대해 전처리 → 컨투어 → warp를 수행한다.
    반환: (edges, pts 또는 None, warped 또는 None)"""
    edges, pts = detect_document(frame, canny_low, canny_high,
                                 min_area_ratio, detect_width)
    # warp는 원본 해상도에서 수행해 화질 손실이 없도록 한다
    warped = warp_document(frame, pts) if pts is not None else None
    return edges, pts, warped


class CornerSmoother:
    """웹캠 모드에서 프레임마다 꼭짓점이 떨리는 것을 줄이는 지수이동평균(EMA)."""

    def __init__(self, alpha=SMOOTH_ALPHA, reset_px=SMOOTH_RESET_PX):
        self.alpha = alpha
        self.reset_px = reset_px
        self.pts = None

    def update(self, pts):
        if pts is None:
            return None
        pts = order_points(pts)
        # 처음이거나 크게 움직였으면(문서를 옮김) 보정 없이 바로 따라간다
        if self.pts is None or \
                np.linalg.norm(pts - self.pts, axis=1).max() > self.reset_px:
            self.pts = pts
        else:
            self.pts = self.alpha * pts + (1 - self.alpha) * self.pts
        return self.pts


def placeholder_warp():
    """아직 성공 결과가 없을 때 Warped 창에 띄울 빈 화면."""
    img = np.zeros((WARP_HEIGHT, WARP_WIDTH, 3), dtype=np.uint8)
    cv2.putText(img, "Waiting...", (110, WARP_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    return img


def save_warped(warped, output_dir):
    """Warped 이미지를 타임스탬프 이름의 PNG로 저장한다."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir,
                        time.strftime("scan_%Y%m%d_%H%M%S.png"))
    cv2.imwrite(path, warped)
    print(f"[저장] {path}")
    return path


# ---------------------------------------------------------------------------
# 실행 모드
# ---------------------------------------------------------------------------
def run_image(args):
    """정지 이미지로 같은 파이프라인을 실행한다."""
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"[에러] 이미지를 읽을 수 없습니다: {args.image}", file=sys.stderr)
        return 1

    edges, pts, warped = process_frame(frame, args.canny_low, args.canny_high,
                                       args.min_area_ratio, args.detect_width)
    vis = draw_detection(frame, pts)

    if pts is None:
        print("[결과] 문서(4각형)를 찾지 못했습니다.")
    else:
        print("[결과] 검출된 꼭짓점 (좌상, 우상, 우하, 좌하):")
        for name, (x, y) in zip(("TL", "TR", "BR", "BL"), order_points(pts)):
            print(f"  {name}: ({x:.1f}, {y:.1f})")

    if args.no_gui:
        # GUI 없이 결과를 파일로 저장
        os.makedirs(args.output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.image))[0]
        cv2.imwrite(os.path.join(args.output_dir, f"{base}_original.png"), vis)
        cv2.imwrite(os.path.join(args.output_dir, f"{base}_edges.png"), edges)
        if warped is not None:
            out = os.path.join(args.output_dir, f"{base}_warped.png")
            cv2.imwrite(out, warped)
            print(f"[저장] {out}")
        return 0 if pts is not None else 2

    # GUI 모드: 창을 띄우고 q가 눌릴 때까지 대기
    cv2.imshow("Original", vis)
    cv2.imshow("Edges", edges)
    cv2.imshow("Warped", warped if warped is not None else placeholder_warp())
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s") and warped is not None:
            save_warped(warped, args.output_dir)
    cv2.destroyAllWindows()
    return 0


def run_camera(args):
    """웹캠 실시간 모드."""
    cap = open_camera(args.camera)
    if cap is None:
        # 카메라를 열지 못하면 에러 메시지 출력 후 종료
        print(f"[에러] 카메라({args.camera})를 열 수 없습니다.", file=sys.stderr)
        return 1

    last_warped = None  # 마지막으로 성공한 warp 결과 (못 찾으면 이걸 유지)
    smoother = CornerSmoother(alpha=args.smooth)
    print("q: 종료 / s: Warped 저장")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[에러] 프레임을 읽지 못했습니다.", file=sys.stderr)
                break

            edges, pts = detect_document(frame, args.canny_low,
                                         args.canny_high, args.min_area_ratio,
                                         args.detect_width)
            # 꼭짓점 떨림 보정 후 warp
            pts = smoother.update(pts)
            if pts is not None:
                last_warped = warp_document(frame, pts)

            cv2.imshow("Original", draw_detection(frame, pts))
            cv2.imshow("Edges", edges)
            cv2.imshow("Warped", last_warped if last_warped is not None
                       else placeholder_warp())

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                if last_warped is not None:
                    save_warped(last_warped, args.output_dir)
                else:
                    print("[알림] 저장할 Warped 이미지가 아직 없습니다.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="OpenCV 문서 스캐너")
    p.add_argument("--image", help="웹캠 대신 사용할 정지 이미지 경로")
    p.add_argument("--no-gui", action="store_true",
                   help="창을 띄우지 않고 결과를 파일로 저장 (--image 전용)")
    p.add_argument("--output-dir", default="scans",
                   help="저장 폴더 (기본: scans)")
    p.add_argument("--camera", type=int, default=0, help="카메라 인덱스")
    p.add_argument("--canny-low", type=int, default=CANNY_LOW)
    p.add_argument("--canny-high", type=int, default=CANNY_HIGH)
    p.add_argument("--min-area-ratio", type=float, default=MIN_AREA_RATIO)
    p.add_argument("--detect-width", type=int, default=DETECT_WIDTH,
                   help="검출용 축소 폭 (0이면 원본 크기로 검출)")
    p.add_argument("--smooth", type=float, default=SMOOTH_ALPHA,
                   help="웹캠 꼭짓점 EMA 계수 (1.0이면 보정 끔)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.no_gui and not args.image:
        print("[에러] --no-gui는 --image와 함께 사용해야 합니다.", file=sys.stderr)
        return 1
    return run_image(args) if args.image else run_camera(args)


if __name__ == "__main__":
    sys.exit(main())
