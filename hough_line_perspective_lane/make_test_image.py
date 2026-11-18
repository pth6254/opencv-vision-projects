"""
차선 검출 테스트용 합성 이미지 생성기

책상(평면) 위에 흰 테이프 두 줄이 세로로 붙어 있는 장면을,
비스듬히 내려다보는 카메라 시점(640x480)으로 렌더링한다.

렌더링 방법:
    Top-view 좌표계(u, v)를 '책상 평면 좌표'로 그대로 사용한다.
    ROI 사다리꼴(src) ↔ Top-view 직사각형(dst)의 호모그래피 H를 알고 있으므로,
    카메라 픽셀마다 H^-1로 책상 좌표를 구해 그 위치의 색(책상/테이프/물건)을 칠한다.
    → lane_detector가 같은 ROI로 Top-view를 만들면 테이프가 u = lane_x 위치의 세로줄로 나와야 한다.

결과: test_images/*.png 와 정답 정보 test_images/cases.json
사용: python make_test_image.py
"""

import json
import os

import cv2
import numpy as np

from lane_detector import WARP_HEIGHT, WARP_WIDTH, default_src_points, dst_points

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "test_images")

FRAME_W, FRAME_H = 640, 480
SUPERSAMPLE = 2            # 2배로 렌더링 후 축소 (계단 현상 완화)
TAPE_HALF_WIDTH = 7        # 테이프 반폭 (Top-view px 단위, 실제 폭 14px)
DESK_FAR_V = -2500         # 이보다 먼 곳은 책상 끝(벽)
DESK_BGR = np.array([70, 95, 125], np.float32)    # 갈색 책상
TAPE_BGR = np.array([235, 238, 240], np.float32)  # 흰 테이프
WALL_BGR = np.array([150, 150, 145], np.float32)

# 테스트 케이스: lanes = 테이프 중심의 u 좌표 (Top-view 폭 400 기준, 범위 밖이면 화면에 안 보임)
#   expect_lanes  : Top-view에서 보여야 하는 차선 x
#   expect_status : lane_detector의 판정 결과
CASES = [
    {"name": "center", "lanes": [100, 300], "expect_status": "OK"},
    {"name": "drift_left", "lanes": [170, 370], "expect_status": "LEFT"},
    {"name": "drift_right", "lanes": [30, 230], "expect_status": "RIGHT"},
    {"name": "on_line", "lanes": [215, 515], "expect_status": "RIGHT"},
    {"name": "one_lane", "lanes": [40, 540], "expect_status": "ONE_LANE"},
    {"name": "no_lane", "lanes": [], "expect_status": "NO_LANE"},
    {"name": "noisy", "lanes": [110, 290], "expect_status": "OK", "clutter": True,
     "noise": 9.0, "gradient": True},
    {"name": "custom_roi", "lanes": [100, 300], "expect_status": "OK",
     "src": [[220, 200], [380, 215], [620, 440], [20, 420]]},   # 살짝 비뚤어진 카메라
]


def desk_texture(rng, size=256):
    """BORDER_WRAP으로 타일링할 나뭇결 느낌의 노이즈 텍스처 (-1~1)."""
    noise = rng.normal(0, 1, (size, size)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=6, sigmaY=1.5)   # 세로로 길쭉한 결
    return noise / (np.abs(noise).max() + 1e-6)


def render(case, rng):
    W, H = FRAME_W * SUPERSAMPLE, FRAME_H * SUPERSAMPLE
    src = np.float32(case.get("src", default_src_points(FRAME_W, FRAME_H).tolist()))
    # 카메라(슈퍼샘플 좌표) → 책상 평면(Top-view 좌표)
    H_top2cam = cv2.getPerspectiveTransform(dst_points(), src * SUPERSAMPLE)
    H_cam2top = np.linalg.inv(H_top2cam)

    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32) + 0.5,
                         np.arange(H, dtype=np.float32) + 0.5)
    pts = np.stack([xs, ys, np.ones_like(xs)], axis=-1) @ H_cam2top.T
    w = pts[..., 2]
    valid = w > 1e-6                      # 지평선 위쪽(w<=0)은 책상이 아님
    safe_w = np.where(valid, w, 1.0)
    u = pts[..., 0] / safe_w
    v = pts[..., 1] / safe_w
    on_desk = valid & (v > DESK_FAR_V)

    # 책상 색 + 나뭇결 텍스처
    tex = desk_texture(rng)
    tex_val = cv2.remap(tex, np.mod(u, 256).astype(np.float32), np.mod(v * 0.3, 256).astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    img = DESK_BGR[None, None, :] * (1.0 + 0.12 * tex_val[..., None])

    # 테이프
    for lx in case["lanes"]:
        mask = on_desk & (np.abs(u - lx) <= TAPE_HALF_WIDTH)
        img[mask] = TAPE_BGR

    # 방해물: 가로로 놓인 펜, 포스트잇 (차선 판정에 섞이면 안 됨)
    if case.get("clutter"):
        ang = np.radians(12)
        du, dv = u - 200, v - 250
        along = du * np.cos(ang) + dv * np.sin(ang)
        across = -du * np.sin(ang) + dv * np.cos(ang)
        pen = on_desk & (np.abs(along) < 120) & (np.abs(across) < 5)
        img[pen] = (40, 40, 40)
        note = on_desk & (u > 300) & (u < 360) & (v > 60) & (v < 120)
        img[note] = (90, 230, 240)

    img[~on_desk] = WALL_BGR

    if case.get("gradient"):              # 위쪽(먼 곳)이 어두운 조명
        g = np.linspace(0.65, 1.1, H, dtype=np.float32)[:, None, None]
        img = img * g

    img = cv2.resize(np.clip(img, 0, 255), (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
    img = img + rng.normal(0, case.get("noise", 3.0), img.shape)   # 센서 노이즈
    return np.clip(img, 0, 255).astype(np.uint8), src


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(7)
    meta = []
    for case in CASES:
        img, src = render(case, rng)
        path = os.path.join(OUT_DIR, f"{case['name']}.png")
        cv2.imwrite(path, img)
        visible = [float(x) for x in case["lanes"] if 0 <= x < WARP_WIDTH]
        meta.append({
            "name": case["name"],
            "file": os.path.basename(path),
            "src": src.tolist(),
            "custom_src": "src" in case,
            "expect_lanes": visible,
            "expect_status": case["expect_status"],
            "warp_size": [WARP_WIDTH, WARP_HEIGHT],
        })
        print(f"saved: {path}")
    with open(os.path.join(OUT_DIR, "cases.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"saved: {os.path.join(OUT_DIR, 'cases.json')}")


if __name__ == "__main__":
    main()
