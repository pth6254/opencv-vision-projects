#!/usr/bin/env python3
"""
속도 측정 자동 테스트 (GUI 없이 실행)

1) 정답 속도를 아는 합성 동영상을 만든다.
   - config.json의 측정 영역(Top-view 40x30cm, 10px/cm)에 테이프 기준선 두 개(간격 20cm)를 긋고
     빨간 공을 일정한 속도로 굴린 장면을 Top-view에서 그린 뒤,
     원근 역변환으로 '비스듬히 찍은 카메라 영상'(640x480, 30fps)으로 만든다.
2) speed_meter.py --source <영상> --no-display --fast 로 실제 CLI를 실행한다.
3) CSV에 기록된 측정 속도가 정답과 허용 오차 이내인지 검증한다.

사용: python test_speed.py            (결과 영상/CSV는 test_output/ 에 저장)
"""

import csv
import os
import subprocess
import sys

import cv2
import numpy as np

import speed_meter as sm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "test_output")
FPS = 30.0
TOL_PCT = 3.0                      # 허용 오차 (%)

# (이름, 속도 cm/s, 방향: +1 = 위→아래(A→B), -1 = 아래→위(B→A))
CASES = [
    ("slow_down", 8.0, +1),
    ("fast_down", 25.0, +1),
    ("medium_up", 15.0, -1),
]


def render_clip(path, speed_cm_s, direction, cfg):
    M, M_inv, (tw, th), _ = sm.build_perspective(cfg)
    ppc = cfg["px_per_cm"]
    fw, fh = cfg["frame_size"]
    line_a = 5.0 * ppc                                   # 위 기준선 y (5cm)
    line_b = line_a + cfg["line_distance_cm"] * ppc      # 아래 기준선 y (25cm)
    radius = int(1.5 * ppc)
    x = tw * 0.5

    y0, y1 = (radius + 2, th - radius - 2) if direction > 0 else (th - radius - 2, radius + 2)
    n = int(abs(y1 - y0) / (speed_cm_s * ppc) * FPS) + 1
    rng = np.random.default_rng(0)

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (fw, fh))
    for i in range(n + int(FPS)):                        # 끝난 뒤 1초 더 (DONE 상태 확인)
        top = np.full((th, tw, 3), (200, 205, 210), np.uint8)          # 밝은 책상
        for y in (line_a, line_b):                                       # 검정 테이프 (1.5cm 폭)
            cv2.rectangle(top, (int(1.5 * ppc), int(y - 0.75 * ppc)),
                          (int(tw - 1.5 * ppc), int(y + 0.75 * ppc)), (40, 40, 40), -1)
        if i < n:
            y = y0 + direction * speed_cm_s * ppc * (i / FPS)
            cv2.circle(top, (int(x), int(round(y))), radius, (30, 30, 210), -1)

        frame = np.full((fh, fw, 3), (90, 80, 70), np.uint8)            # 영역 밖 배경
        warped = cv2.warpPerspective(top, M_inv, (fw, fh))
        mask = cv2.warpPerspective(np.full((th, tw), 255, np.uint8), M_inv, (fw, fh))
        frame[mask > 0] = warped[mask > 0]
        noise = rng.normal(0, 3, frame.shape)                           # 센서 노이즈
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)
    writer.release()


def run_case(name, speed, direction, cfg):
    clip = os.path.join(OUT_DIR, f"{name}.avi")
    log = os.path.join(OUT_DIR, f"{name}.csv")
    if os.path.exists(log):
        os.remove(log)
    render_clip(clip, speed, direction, cfg)
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "speed_meter.py"), "--source", clip,
         "--no-display", "--fast", "--fps", str(FPS), "--csv", log],
        capture_output=True, text=True, timeout=120)
    if proc.returncode != 0 or not os.path.exists(log):
        print(proc.stdout[-800:], proc.stderr[-800:])
        return False, f"측정 기록 없음 (종료 코드 {proc.returncode})"
    with open(log, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        return False, f"측정 {len(rows)}회 (1회 예상)"
    measured = float(rows[0]["speed_cm_s"])
    err = abs(measured - speed) / speed * 100
    want_dir = "A->B" if direction > 0 else "B->A"
    got_dir = f"{rows[0]['start_line']}->{rows[0]['end_line']}"
    ok = err <= TOL_PCT and got_dir == want_dir
    return ok, (f"정답 {speed:5.1f} cm/s / 측정 {measured:6.2f} cm/s "
                f"(오차 {err:4.2f}%), 방향 {got_dir}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = sm.load_config(os.path.join(HERE, "config.json"))
    all_ok = True
    for name, speed, direction in CASES:
        ok, msg = run_case(name, speed, direction, cfg)
        all_ok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:10s} {msg}")
    print("\nALL TESTS PASSED" if all_ok else "\nSOME TESTS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
