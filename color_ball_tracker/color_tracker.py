"""
웹캠 기반 컬러 볼 추적기 (OpenCV)

파이프라인:
    1. 프레임 입력   (cv2.VideoCapture 또는 --image 정지 이미지)
    2. 전처리        (GaussianBlur 5x5)
    3. 색상 검출     (BGR → HSV → cv2.inRange)
    4. 노이즈 제거   (모폴로지 열기 → 닫기)
    5. 컨투어 검출   (가장 큰 컨투어 1개, 최소 면적 미만이면 "물체 없음")
    6. 중심점 계산   (cv2.moments)

사용 예:
    python color_tracker.py                      # 웹캠
    python color_tracker.py --image test.png     # 정지 이미지 + GUI 튜닝
    python color_tracker.py --image test.png --no-gui --expect 400 260
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 기본 설정값
# OpenCV의 HSV는 H: 0~179 (실제 각도 0~360°를 2로 나눈 값), S/V: 0~255 이다.
# 오렌지는 실제 색상각 약 10°~50° → OpenCV 기준 약 5~25.
# 'p' 키로 출력한 값을 아래에 그대로 붙여 넣으면 기본값이 바뀐다.
# ---------------------------------------------------------------------------
LOWER_HSV = (5, 120, 120)
UPPER_HSV = (25, 255, 255)
OPEN_ITER = 2        # 열기(erode→dilate) 반복 횟수
CLOSE_ITER = 2       # 닫기(dilate→erode) 반복 횟수
MIN_AREA = 500       # 이 면적(px) 미만의 컨투어는 "물체 없음"으로 처리

BLUR_KSIZE = (5, 5)
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

WIN_TRACK = "Tracking"
WIN_RAW = "Mask (raw)"
WIN_CLEAN = "Mask (clean)"
WIN_CTRL = "Controls"

# 그리기 색상 (BGR)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)


# ===========================================================================
# 2. 전처리
# ===========================================================================
def preprocess(frame):
    """5x5 가우시안 블러로 센서 노이즈/미세한 픽셀 노이즈를 먼저 줄인다.
    고립된 1~2픽셀짜리 점은 주변 배경과 섞여 채도가 떨어지므로
    inRange 단계에서 상당수가 자연스럽게 걸러진다."""
    return cv2.GaussianBlur(frame, BLUR_KSIZE, 0)


# ===========================================================================
# 3. 색상 검출
# ===========================================================================
def detect_color(blurred, lower, upper):
    """BGR → HSV 변환 후 inRange로 이진 마스크(0 또는 255)를 만든다.

    HSV를 쓰는 이유: 조명 밝기가 바뀌어도 색상(H)은 비교적 일정하게 유지되므로
    BGR에서 범위를 잡는 것보다 훨씬 안정적이다.

    H_min > H_max 인 경우(예: 빨강 170~10)는 H축이 0/179에서 원형으로
    이어진다는 점을 이용해 두 구간 [H_min,179] ∪ [0,H_max] 를 합친다.
    """
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower = np.array(lower, dtype=np.uint8)
    upper = np.array(upper, dtype=np.uint8)

    if lower[0] <= upper[0]:
        return cv2.inRange(hsv, lower, upper)

    # 색상환 경계를 넘는 범위 처리
    lo1, up1 = lower.copy(), upper.copy()
    up1[0] = 179
    lo2, up2 = lower.copy(), upper.copy()
    lo2[0] = 0
    return cv2.bitwise_or(cv2.inRange(hsv, lo1, up1), cv2.inRange(hsv, lo2, up2))


# ===========================================================================
# 4. 노이즈 제거 (모폴로지)
# ===========================================================================
def clean_mask(mask, open_iter, close_iter, kernel=KERNEL):
    """erode / dilate를 직접 호출해 열기 → 닫기를 수행한다.

    - erode(침식): 커널이 '전부 흰색' 위에 있을 때만 흰색을 남긴다.
      → 흰 영역의 가장자리가 깎이고, 커널보다 작은 흰 점은 완전히 사라진다.
    - dilate(팽창): 커널 안에 흰색이 '하나라도' 있으면 흰색으로 만든다.
      → 흰 영역이 커지고, 커널보다 작은 검은 구멍은 메워진다.

    열기(Opening) = erode → dilate
      배경의 자잘한 흰 점(점 노이즈)을 먼저 침식으로 없앤 뒤,
      살아남은 큰 물체만 팽창으로 원래 크기로 되돌린다.

    닫기(Closing) = dilate → erode
      물체 내부의 작은 검은 구멍(반사광, 무늬 등)을 팽창으로 메운 뒤,
      부풀어 오른 외곽을 침식으로 원래 크기로 되돌린다.

    iterations=n 은 같은 연산을 n번 반복한다는 뜻이며, 5x5 커널 기준
    반복 1회당 약 2px씩 깎이거나/늘어난다. 횟수를 늘릴수록 더 큰 노이즈·구멍을
    처리하지만 물체 모서리도 둥글게 뭉개진다.
    (cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=n)과 동일한 결과)
    """
    out = mask
    if open_iter > 0:
        # 열기: 침식으로 점 노이즈 제거 → 팽창으로 물체 크기 복원
        out = cv2.erode(out, kernel, iterations=open_iter)
        out = cv2.dilate(out, kernel, iterations=open_iter)
    if close_iter > 0:
        # 닫기: 팽창으로 내부 구멍 메우기 → 침식으로 물체 크기 복원
        out = cv2.dilate(out, kernel, iterations=close_iter)
        out = cv2.erode(out, kernel, iterations=close_iter)
    return out


# ===========================================================================
# 5. 컨투어 검출
# ===========================================================================
def find_largest_contour(mask, min_area):
    """외곽 컨투어만(RETR_EXTERNAL) 꼭짓점 압축(CHAIN_APPROX_SIMPLE)으로 찾고,
    면적이 가장 큰 것 하나를 고른다. 최소 면적 미만이면 None(물체 없음)."""
    # OpenCV 3.x는 3개, 4.x는 2개를 반환하므로 마지막 두 값만 취한다.
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if not contours:
        return None, 0.0
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < min_area:
        return None, area
    return largest, area


# ===========================================================================
# 6. 중심점 계산
# ===========================================================================
def compute_center(contour):
    """영상 모멘트로 무게중심을 계산한다.
    m00 = 면적, m10 = Σx, m01 = Σy  →  cx = m10/m00, cy = m01/m00
    m00이 0이면(선처럼 납작한 컨투어) 0으로 나누지 않도록 None을 반환한다."""
    m = cv2.moments(contour)
    if m["m00"] == 0:
        return None
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


# ===========================================================================
# 화면 출력
# ===========================================================================
def draw_tracking(frame, contour, center):
    """원본 위에 컨투어(초록), 외접원(노랑), 중심점(파랑), 좌표 텍스트를 그린다."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    if contour is None or center is None:
        cv2.putText(vis, "No object", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, RED, 2, cv2.LINE_AA)
        return vis

    # 컨투어 외곽선 (초록)
    cv2.drawContours(vis, [contour], -1, GREEN, 2)

    # 최소 외접원 (노랑)
    (ex, ey), radius = cv2.minEnclosingCircle(contour)
    cv2.circle(vis, (int(round(ex)), int(round(ey))), int(round(radius)), YELLOW, 2)

    # 중심점 (파랑, 채움)
    cx, cy = int(round(center[0])), int(round(center[1]))
    cv2.circle(vis, (cx, cy), 6, BLUE, -1)

    # 좌표 텍스트: 중심점 오른쪽 위에 표시하되 화면 밖으로 나가지 않게 보정
    label = f"({cx}, {cy})"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    tx = min(max(cx + 12, 0), w - tw - 2)
    ty = min(max(cy - 12, th + 2), h - 2)
    cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 3, cv2.LINE_AA)
    cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLUE, 2, cv2.LINE_AA)
    return vis


def process_frame(frame, params):
    """2~6단계를 순서대로 실행하고 중간 결과를 모두 돌려준다."""
    blurred = preprocess(frame)
    mask_raw = detect_color(blurred, params["lower"], params["upper"])
    mask_clean = clean_mask(mask_raw, params["open_iter"], params["close_iter"])
    contour, area = find_largest_contour(mask_clean, params["min_area"])
    center = compute_center(contour) if contour is not None else None
    vis = draw_tracking(frame, contour, center)
    return {
        "mask_raw": mask_raw,
        "mask_clean": mask_clean,
        "contour": contour,
        "area": area,
        "center": center,
        "vis": vis,
    }


def count_blobs(mask):
    """마스크의 흰색 덩어리(연결 요소) 개수. 배경(라벨 0)은 제외."""
    n, _ = cv2.connectedComponents(mask)
    return n - 1


# ===========================================================================
# HSV 튜닝용 트랙바
# ===========================================================================
def _noop(_):
    pass


def create_controls():
    """Controls 창에 트랙바를 만든다. 값은 매 프레임 읽어오므로 즉시 반영된다."""
    cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CTRL, 420, 360)
    cv2.createTrackbar("H_min", WIN_CTRL, LOWER_HSV[0], 179, _noop)
    cv2.createTrackbar("H_max", WIN_CTRL, UPPER_HSV[0], 179, _noop)
    cv2.createTrackbar("S_min", WIN_CTRL, LOWER_HSV[1], 255, _noop)
    cv2.createTrackbar("S_max", WIN_CTRL, UPPER_HSV[1], 255, _noop)
    cv2.createTrackbar("V_min", WIN_CTRL, LOWER_HSV[2], 255, _noop)
    cv2.createTrackbar("V_max", WIN_CTRL, UPPER_HSV[2], 255, _noop)
    cv2.createTrackbar("Open_iter", WIN_CTRL, OPEN_ITER, 10, _noop)
    cv2.createTrackbar("Close_iter", WIN_CTRL, CLOSE_ITER, 10, _noop)
    cv2.createTrackbar("Min_area", WIN_CTRL, MIN_AREA, 20000, _noop)


def read_controls():
    """트랙바 현재 값을 파라미터 딕셔너리로 읽는다."""
    g = lambda name: cv2.getTrackbarPos(name, WIN_CTRL)
    return {
        "lower": (g("H_min"), g("S_min"), g("V_min")),
        "upper": (g("H_max"), g("S_max"), g("V_max")),
        "open_iter": g("Open_iter"),
        "close_iter": g("Close_iter"),
        "min_area": g("Min_area"),
    }


def default_params():
    return {
        "lower": LOWER_HSV,
        "upper": UPPER_HSV,
        "open_iter": OPEN_ITER,
        "close_iter": CLOSE_ITER,
        "min_area": MIN_AREA,
    }


def print_params(params):
    """'p' 키: 코드 상단 상수에 그대로 붙여 넣을 수 있는 형식으로 출력."""
    print("\n# ---- 현재 튜닝 값 (color_tracker.py 상단에 붙여 넣기) ----")
    print(f"LOWER_HSV = {tuple(params['lower'])}")
    print(f"UPPER_HSV = {tuple(params['upper'])}")
    print(f"OPEN_ITER = {params['open_iter']}")
    print(f"CLOSE_ITER = {params['close_iter']}")
    print(f"MIN_AREA = {params['min_area']}")
    sys.stdout.flush()


def save_snapshot(vis, out_dir):
    """'s' 키: 현재 Tracking 화면을 PNG로 저장."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, time.strftime("tracking_%Y%m%d_%H%M%S.png"))
    cv2.imwrite(path, vis)
    print(f"[저장] {path}")


# ===========================================================================
# 실행 모드
# ===========================================================================
def run_gui(get_frame, out_dir):
    """GUI 공통 루프. get_frame()이 None을 반환하면 종료한다."""
    create_controls()
    print("키 조작: q=종료, p=HSV 값 출력, s=Tracking 화면 저장")

    while True:
        # 1. 프레임 입력
        frame = get_frame()
        if frame is None:
            print("[오류] 프레임을 읽지 못했습니다. 종료합니다.", file=sys.stderr)
            break

        params = read_controls()
        result = process_frame(frame, params)

        cv2.imshow(WIN_TRACK, result["vis"])
        cv2.imshow(WIN_RAW, result["mask_raw"])
        cv2.imshow(WIN_CLEAN, result["mask_clean"])

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            print_params(params)
        elif key == ord("s"):
            save_snapshot(result["vis"], out_dir)

        # 사용자가 창의 X 버튼으로 Tracking 창을 닫은 경우에도 종료
        if cv2.getWindowProperty(WIN_TRACK, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


def run_camera(index, out_dir):
    """1. 프레임 입력: 웹캠"""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"[오류] 카메라 {index}번을 열 수 없습니다. "
              "연결 상태나 다른 프로그램의 사용 여부를 확인하세요.", file=sys.stderr)
        sys.exit(1)

    def get_frame():
        ok, frame = cap.read()
        return frame if ok else None

    try:
        run_gui(get_frame, out_dir)
    finally:
        cap.release()


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"[오류] 이미지를 읽을 수 없습니다: {path}", file=sys.stderr)
        sys.exit(1)
    return img


def run_headless(image_path, out_dir, expect, tol):
    """--no-gui: 기본 파라미터로 한 번 처리하고 결과를 파일로 저장 + 검증."""
    frame = load_image(image_path)
    params = default_params()
    result = process_frame(frame, params)

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    paths = {
        "tracking": os.path.join(out_dir, f"{stem}_tracking.png"),
        "mask_raw": os.path.join(out_dir, f"{stem}_mask_raw.png"),
        "mask_clean": os.path.join(out_dir, f"{stem}_mask_clean.png"),
        "compare": os.path.join(out_dir, f"{stem}_compare.png"),
    }
    cv2.imwrite(paths["tracking"], result["vis"])
    cv2.imwrite(paths["mask_raw"], result["mask_raw"])
    cv2.imwrite(paths["mask_clean"], result["mask_clean"])

    # 마스크 비교용 한 장짜리 이미지: [Tracking | Mask raw | Mask clean]
    def labeled(img, text):
        img = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = img.copy()
        cv2.putText(img, text, (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, RED, 2, cv2.LINE_AA)
        return img
    compare = np.hstack([labeled(result["vis"], "Tracking"),
                         labeled(result["mask_raw"], "Mask (raw)"),
                         labeled(result["mask_clean"], "Mask (clean)")])
    cv2.imwrite(paths["compare"], compare)

    raw_blobs = count_blobs(result["mask_raw"])
    clean_blobs = count_blobs(result["mask_clean"])

    print(f"입력 이미지     : {image_path}")
    print(f"파라미터        : {params}")
    print(f"raw 마스크 덩어리 수   : {raw_blobs}")
    print(f"clean 마스크 덩어리 수 : {clean_blobs}")
    if result["center"] is None:
        print("검출 결과       : No object")
    else:
        cx, cy = result["center"]
        print(f"검출 면적       : {result['area']:.0f} px")
        print(f"검출 중심       : ({cx:.2f}, {cy:.2f})")
    for name, p in paths.items():
        print(f"저장 [{name:10s}]: {p}")

    # --expect 가 주어지면 정답과 비교해 합격/불합격 판정
    if expect is None:
        return 0
    ok = True
    if clean_blobs != 1:
        print(f"[FAIL] clean 마스크에 물체 외 덩어리가 {clean_blobs - 1}개 남아 있음")
        ok = False
    else:
        print("[PASS] clean 마스크에 점 노이즈 없음 (덩어리 1개)")
    if result["center"] is None:
        print("[FAIL] 물체를 찾지 못함")
        ok = False
    else:
        err = float(np.hypot(cx - expect[0], cy - expect[1]))
        verdict = "PASS" if err <= tol else "FAIL"
        print(f"[{verdict}] 중심점 오차 {err:.3f}px (정답 {tuple(expect)}, 허용 {tol}px)")
        ok = ok and err <= tol
    return 0 if ok else 1


def parse_args():
    ap = argparse.ArgumentParser(description="OpenCV HSV 컬러 볼 추적기")
    ap.add_argument("--camera", type=int, default=0, help="웹캠 인덱스 (기본 0)")
    ap.add_argument("--image", help="웹캠 대신 사용할 정지 이미지 경로")
    ap.add_argument("--no-gui", action="store_true",
                    help="창 없이 한 번 처리 후 결과를 파일로 저장 (--image 필요)")
    ap.add_argument("--out-dir", default="output", help="결과/스냅샷 저장 폴더")
    ap.add_argument("--expect", type=float, nargs=2, metavar=("CX", "CY"),
                    help="--no-gui 검증용 실제 중심 좌표")
    ap.add_argument("--tol", type=float, default=3.0, help="중심점 허용 오차(px)")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.no_gui:
        if not args.image:
            print("[오류] --no-gui 는 --image 와 함께 사용해야 합니다.", file=sys.stderr)
            sys.exit(2)
        sys.exit(run_headless(args.image, args.out_dir, args.expect, args.tol))

    if args.image:
        # 정지 이미지를 매 프레임 같은 영상으로 넣어 트랙바 튜닝을 할 수 있게 한다.
        img = load_image(args.image)
        run_gui(lambda: img, args.out_dir)
    else:
        run_camera(args.camera, args.out_dir)


if __name__ == "__main__":
    main()
