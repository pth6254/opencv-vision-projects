"""
차선 검출 자동 테스트 (GUI 없이 실행)

1) make_test_image.py로 합성 이미지 생성 (정답 ROI / 차선 위치 / 판정 포함)
2) lane_detector.py --image ... --no-gui 로 실제 CLI 실행
3) 검증
   - Top-view에서 찾은 차선 x가 실제 테이프 위치와 몇 px 이내인지
   - 판정(OK / LEFT / RIGHT / ONE_LANE / NO_LANE)이 정답과 같은지
   - 저장된 Top-view 이미지에 붉은 직선이 실제로 그려졌는지
   - 차선이 없으면 크래시 없이 종료 코드 2로 끝나는지
   - Perspective 변환, 차선 묶기, 이탈 판정 단위 테스트

사용: python test_lane.py
"""

import json
import os
import subprocess
import sys

import cv2
import numpy as np

from lane_detector import (assess_departure, default_src_points, detect_edges, detect_lines,
                           dst_points, get_transform, group_lanes, x_at_y)

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(HERE, "test_images")
OUT_DIR = os.path.join(HERE, "test_output")

LANE_TOL_PX = 6.0          # 차선 x 위치 허용 오차 (Top-view px)
MIN_RED_PIXELS = 500       # Top-view 결과에 최소 이만큼은 붉은 픽셀이 있어야 함

failures = []


def check(cond, msg):
    print(("  [PASS] " if cond else "  [FAIL] ") + msg)
    if not cond:
        failures.append(msg)


def run_cli(image_path, src=None):
    cmd = [sys.executable, os.path.join(HERE, "lane_detector.py"),
           "--image", image_path, "--no-gui", "--output-dir", OUT_DIR]
    if src is not None:
        cmd += ["--src", ",".join(f"{v:.1f}" for v in np.ravel(src))]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def parse_output(stdout):
    info = {}
    for line in stdout.splitlines():
        key, _, val = line.partition(":")
        info[key.strip()] = val.strip()
    lanes = [] if info.get("lanes", "-") == "-" else [float(v) for v in info["lanes"].split(",")]
    return lanes, info.get("status"), info.get("warning") == "True"


def red_pixel_count(img):
    b, g, r = cv2.split(img.astype(np.int16))
    return int(np.count_nonzero((r > 200) & (g < 50) & (b < 50)))


# ---------------------------------------------------------------------------
# 단위 테스트
# ---------------------------------------------------------------------------
def test_transform():
    print("[unit] Perspective Transform")
    src = default_src_points(640, 480)
    M = get_transform(src)
    mapped = cv2.perspectiveTransform(src.reshape(-1, 1, 2), M).reshape(-1, 2)
    check(np.allclose(mapped, dst_points(), atol=1e-3), "ROI 4점이 Top-view 네 모서리로 정확히 매핑됨")

    # 합성 Top-view: 세로 흰 줄 두 개 → Hough가 세로선을 찾는지
    top = np.full((600, 400, 3), 90, np.uint8)
    cv2.rectangle(top, (93, 0), (107, 599), (240, 240, 240), -1)
    cv2.rectangle(top, (293, 0), (307, 599), (240, 240, 240), -1)
    lines = detect_lines(detect_edges(top))
    check(len(lines) >= 2, f"직선 그림에서 HoughLinesP가 선분을 찾음 ({len(lines)}개)")
    lanes = group_lanes(lines)
    check(len(lanes) == 2 and abs(lanes[0] - 100) < 3 and abs(lanes[1] - 300) < 3,
          f"테이프 양쪽 엣지가 차선 하나로 묶임 -> {[round(x, 1) for x in lanes]}")


def test_grouping_and_departure():
    print("[unit] 차선 묶기 / 이탈 판정")
    check(abs(x_at_y([10, 0, 20, 100], 599) - 69.9) < 0.1, "x_at_y: 선분 연장 계산")

    horizontal = np.array([[50, 300, 350, 310]])
    check(group_lanes(horizontal) == [], "가로선(펜 등)은 차선 후보에서 제외")

    cases = [([100, 300], "OK", False), ([170, 370], "LEFT", True),
             ([30, 230], "RIGHT", True), ([190], "LEFT", True), ([215], "RIGHT", True),
             ([40], "ONE_LANE", False), ([], "NO_LANE", False),
             ([20, 100, 300], "OK", False),     # 옆 차선까지 보여도 차량을 감싼 쌍을 고름
             ([100, 120], "ONE_LANE", False)]   # 너무 붙은 두 선은 차선 쌍이 아님
    for lanes, status, warning in cases:
        d = assess_departure(lanes)
        check(d["status"] == status and d["warning"] == warning,
              f"lanes={lanes} -> {d['status']} (기대 {status})")


# ---------------------------------------------------------------------------
# CLI 통합 테스트 (합성 이미지)
# ---------------------------------------------------------------------------
def test_images(meta):
    print("[cli] 합성 이미지")
    for case in meta:
        path = os.path.join(IMG_DIR, case["file"])
        src = case["src"] if case["custom_src"] else None
        code, out = run_cli(path, src)
        lanes, status, warning = parse_output(out)
        name = case["name"]

        if not case["expect_lanes"]:
            check(code == 2 and status == "NO_LANE", f"{name}: 차선 없음 -> 종료 코드 2, NO_LANE (code={code})")
            continue

        check(code == 0, f"{name}: 종료 코드 0 (code={code})")
        check(status == case["expect_status"], f"{name}: 판정 {status} (기대 {case['expect_status']})")
        for gt in case["expect_lanes"]:
            err = min((abs(x - gt) for x in lanes), default=1e9)
            check(err < LANE_TOL_PX, f"{name}: 차선 x={gt:.0f} 검출 (오차 {err:.1f}px)")

        top = cv2.imread(os.path.join(OUT_DIR, f"{name}_topview.png"))
        reds = red_pixel_count(top) if top is not None else 0
        check(top is not None and top.shape[:2] == (600, 400),
              f"{name}: Top-view 결과 저장 (400x600)")
        check(reds > MIN_RED_PIXELS, f"{name}: Top-view 위에 붉은 직선이 그려짐 ({reds}px)")


def test_custom_roi_matters(meta):
    print("[cli] ROI 지정 효과")
    case = next(c for c in meta if c["custom_src"])
    path = os.path.join(IMG_DIR, case["file"])
    _, out_custom = run_cli(path, case["src"])
    _, out_default = run_cli(path)
    lc, _, _ = parse_output(out_custom)
    ld, _, _ = parse_output(out_default)
    err_c = max(min(abs(x - g) for x in lc) for g in case["expect_lanes"]) if lc else 1e9
    err_d = max(min(abs(x - g) for x in ld) for g in case["expect_lanes"]) if ld else 1e9
    check(err_c < err_d, f"올바른 --src를 주면 기본 ROI보다 차선 위치가 정확함 ({err_c:.1f}px < {err_d:.1f}px)")


def main():
    subprocess.run([sys.executable, os.path.join(HERE, "make_test_image.py")],
                   check=True, capture_output=True)
    with open(os.path.join(IMG_DIR, "cases.json"), encoding="utf-8") as f:
        meta = json.load(f)
    os.makedirs(OUT_DIR, exist_ok=True)

    test_transform()
    test_grouping_and_departure()
    test_images(meta)
    test_custom_roi_matters(meta)

    print()
    if failures:
        print(f"FAILED: {len(failures)}개")
        for f in failures:
            print("  - " + f)
        return 1
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
