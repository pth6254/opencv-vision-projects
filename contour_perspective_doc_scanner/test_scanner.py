"""
문서 스캐너 자동 테스트 (GUI 없이 실행)

1) make_test_image.py로 합성 이미지 생성
2) doc_scanner.py --image ... --no-gui 로 실제 CLI 실행
3) 검증
   - 검출 꼭짓점이 실제 꼭짓점과 몇 px 이내인지
   - warp 결과가 정답(평평한 용지)과 거의 같은지 (평균 픽셀 차이)
   - 빨간 마커가 좌상단, 파란 마커가 우하단에 있는지 (꼭짓점 순서 검증)
   - 용지 없음 / 너무 작은 용지는 크래시 없이 '미검출' 처리되는지
   - order_points 단위 테스트

사용: python test_scanner.py
"""

import json
import os
import subprocess
import sys

import cv2
import numpy as np

from doc_scanner import order_points

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(HERE, "test_images")
OUT_DIR = os.path.join(HERE, "test_output")

CORNER_TOL_PX = 6.0     # 꼭짓점 허용 오차 (dilate로 1~2px 바깥으로 밀림)
MEAN_DIFF_TOL = 20.0    # 정답 대비 평균 절대 픽셀 차이 허용치 (0~255)

failures = []


def check(cond, msg):
    print(("  [PASS] " if cond else "  [FAIL] ") + msg)
    if not cond:
        failures.append(msg)


def run_cli(image_path):
    """doc_scanner.py를 --no-gui 모드로 실행하고 (returncode, stdout) 반환."""
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, "doc_scanner.py"),
         "--image", image_path, "--no-gui", "--output-dir", OUT_DIR],
        capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def parse_corners(stdout):
    """CLI 출력에서 'TL: (x, y)' 형태의 꼭짓점 4개를 파싱한다."""
    pts = []
    for line in stdout.splitlines():
        line = line.strip()
        if line[:3] in ("TL:", "TR:", "BR:", "BL:"):
            x, y = line[4:].strip(" ()").split(",")
            pts.append([float(x), float(y)])
    return np.array(pts, dtype=np.float32)


def marker_center(img, color):
    """BGR 이미지에서 빨강/파랑 마커의 무게중심을 (x, y) 비율로 반환."""
    b, g, r = [img[:, :, i].astype(int) for i in range(3)]
    if color == "red":
        mask = (r > 150) & (g < 100) & (b < 100)
    else:
        mask = (b > 150) & (g < 100) & (r < 100)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    h, w = img.shape[:2]
    return xs.mean() / w, ys.mean() / h


def test_order_points():
    print("\n[단위 테스트] order_points")
    expected = np.array([[10, 10], [100, 15], [95, 200], [5, 190]],
                        dtype=np.float32)
    rng = np.random.default_rng(0)
    ok = True
    for _ in range(10):
        shuffled = expected[rng.permutation(4)]
        ok &= np.allclose(order_points(shuffled), expected)
    check(ok, "임의로 섞은 4점 → 좌상, 우상, 우하, 좌하 순서로 복원")


def test_positive_cases(meta, gt):
    for name, true_corners in meta.items():
        print(f"\n[케이스] {name}")
        code, out = run_cli(os.path.join(IMG_DIR, f"{name}.png"))
        check(code == 0, f"CLI 종료 코드 0 (실제: {code})")
        if code != 0:
            print(out)
            continue

        # 1) 꼭짓점 정확도
        found = parse_corners(out)
        err = np.linalg.norm(found - np.array(true_corners), axis=1)
        check(err.max() < CORNER_TOL_PX,
              f"꼭짓점 최대 오차 {err.max():.2f}px < {CORNER_TOL_PX}px")

        # 2) warp 결과 크기 및 정답과의 유사도
        warped = cv2.imread(os.path.join(OUT_DIR, f"{name}_warped.png"))
        check(warped is not None and warped.shape[:2] == (594, 420),
              f"Warped 크기 420x594 (A4 비율)")
        diff = np.abs(warped.astype(np.float32) - gt.astype(np.float32)).mean()
        check(diff < MEAN_DIFF_TOL,
              f"정답 대비 평균 픽셀 차이 {diff:.2f} < {MEAN_DIFF_TOL}")

        # 3) 방향 마커 위치 (뒤집히거나 회전되지 않았는지)
        red, blue = marker_center(warped, "red"), marker_center(warped, "blue")
        check(red is not None and red[0] < 0.3 and red[1] < 0.3,
              f"빨간 마커가 좌상단에 위치 {red}")
        check(blue is not None and blue[0] > 0.7 and blue[1] > 0.7,
              f"파란 마커가 우하단에 위치 {blue}")


def test_negative_cases():
    for name in ("no_document", "too_small"):
        print(f"\n[음성 케이스] {name}")
        code, out = run_cli(os.path.join(IMG_DIR, f"{name}.png"))
        check(code == 2 and "Traceback" not in out,
              f"크래시 없이 '미검출'(종료 코드 2) 처리 (실제: {code})")


def main():
    # 합성 이미지 생성
    subprocess.run([sys.executable, os.path.join(HERE, "make_test_image.py"),
                    "--out-dir", IMG_DIR], check=True)
    with open(os.path.join(IMG_DIR, "corners.json")) as f:
        meta = json.load(f)
    gt = cv2.imread(os.path.join(IMG_DIR, "ground_truth.png"))

    test_order_points()
    test_positive_cases(meta, gt)
    test_negative_cases()

    print("\n" + "=" * 50)
    if failures:
        print(f"실패 {len(failures)}건:")
        for f in failures:
            print("  -", f)
        return 1
    print("모든 테스트 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
