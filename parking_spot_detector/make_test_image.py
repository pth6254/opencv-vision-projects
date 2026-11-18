"""
합성 테스트 데이터 생성기

1) 위에서 본 주차장(Top-view) 이미지를 그린다.
   - 회색 바닥 + 흰 테이프로 2x2 주차 칸
   - P1, P3: 빨간 차량 / P2: 파란 물체(빨강 아님) / P4: 비움(작은 빨간 점 노이즈만)
2) 알려진 호모그래피 행렬로 비스듬하게 왜곡해서 "카메라 시점" 이미지를 만든다.
3) 왜곡에 쓴 꼭짓점 좌표로 테스트용 config, 정답 상태로 expected JSON을 만든다.

사용법:  python make_test_image.py [--out test_data]
"""

import argparse
import json
import os

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TOP_W, TOP_H = 600, 400        # Top-view 크기 (parking.py의 warp_size와 동일)
CAM_W, CAM_H = 960, 720        # 카메라 이미지 크기

# 카메라 이미지 안에서 주차장 꼭짓점 (좌상, 우상, 우하, 좌하)
# 위쪽(먼 쪽)이 좁아 보이고 살짝 기울어진 사다리꼴
CAM_PTS = np.float32([[285, 160], [720, 190], [905, 610], [55, 575]])

RED = (30, 30, 200)            # BGR
DARK_RED = (25, 20, 150)
BLUE = (200, 90, 30)
TAPE = (235, 235, 235)


def draw_car(img, center, size, angle, body=RED):
    """회전된 직사각형 차량: 몸체 + 지붕 + 유리창 + 바퀴."""
    def box(c, s, a):
        return np.int32(cv2.boxPoints((c, s, a)))

    w, h = size
    # 바퀴 (몸체 바깥쪽으로 살짝 튀어나오게 먼저 그린다)
    rad = np.deg2rad(angle)
    ux, uy = np.cos(rad), np.sin(rad)     # 차량 길이 방향
    vx, vy = -np.sin(rad), np.cos(rad)    # 차량 폭 방향
    for sl in (-0.32, 0.32):
        for sw in (-0.5, 0.5):
            c = (center[0] + ux * w * sl + vx * h * sw,
                 center[1] + uy * w * sl + vy * h * sw)
            cv2.fillPoly(img, [box(c, (w * 0.18, h * 0.16), angle)], (20, 20, 20))
    cv2.fillPoly(img, [box(center, size, angle)], body)                     # 몸체
    cv2.fillPoly(img, [box(center, (w * 0.5, h * 0.75), angle)], DARK_RED)  # 지붕
    wc = (center[0] + ux * w * 0.18, center[1] + uy * w * 0.18)
    cv2.fillPoly(img, [box(wc, (w * 0.12, h * 0.7), angle)], (60, 50, 40))  # 앞유리


def make_top_view(rng):
    """정답(노이즈 없는) Top-view와 노이즈가 섞인 Top-view를 만든다."""
    # 회색 바닥 + 약한 조명 그라데이션 + 바닥 질감
    yy, xx = np.mgrid[0:TOP_H, 0:TOP_W].astype(np.float32)
    shade = 105 + 20 * (xx / TOP_W) - 10 * (yy / TOP_H)
    texture = cv2.GaussianBlur(rng.normal(0, 6, (TOP_H, TOP_W)).astype(np.float32),
                               (0, 0), 3)
    floor = np.dstack([shade, shade + 2, shade + 4]) + texture[..., None]
    img = np.clip(floor, 0, 255).astype(np.uint8)

    # 흰 테이프: 바깥 테두리 + 가운데 십자 (2x2 칸)
    t = 10
    cv2.rectangle(img, (t // 2, t // 2), (TOP_W - 1 - t // 2, TOP_H - 1 - t // 2), TAPE, t)
    cv2.line(img, (TOP_W // 2, 0), (TOP_W // 2, TOP_H - 1), TAPE, t)
    cv2.line(img, (0, TOP_H // 2), (TOP_W - 1, TOP_H // 2), TAPE, t)

    # P1(좌상): 빨간 차량
    draw_car(img, (150, 100), (185, 105), 0)
    # P2(우상): 파란 물체 (빨강이 아니므로 AVAILABLE이어야 함)
    cv2.rectangle(img, (370, 55), (530, 145), BLUE, -1)
    cv2.rectangle(img, (400, 75), (500, 125), (230, 140, 60), -1)
    # P3(좌하): 살짝 비스듬히 주차된 빨간 차량
    draw_car(img, (150, 300), (180, 100), 10)
    # P4(우하): 비어 있음. 단, 작은 빨간 물체/점 → 판정 비율로 걸러져야 함
    cv2.circle(img, (520, 340), 9, RED, -1)          # 병뚜껑 크기의 빨간 물체
    for _ in range(25):                              # 흩어진 작은 빨간 점
        x, y = rng.integers(320, 590), rng.integers(215, 390)
        cv2.rectangle(img, (int(x), int(y)), (int(x) + 1, int(y) + 1), RED, -1)

    clean = img.copy()
    noisy = np.clip(img.astype(np.float32) + rng.normal(0, 8, img.shape), 0, 255)
    return clean, noisy.astype(np.uint8)


def make_camera_view(top_noisy, rng):
    """알려진 행렬로 Top-view를 비스듬한 카메라 시점으로 왜곡한다."""
    src = np.float32([[0, 0], [TOP_W - 1, 0], [TOP_W - 1, TOP_H - 1], [0, TOP_H - 1]])
    H = cv2.getPerspectiveTransform(src, CAM_PTS)   # Top-view → 카메라

    # 책상(나무색) 배경
    yy, xx = np.mgrid[0:CAM_H, 0:CAM_W].astype(np.float32)
    wood = 0.5 + 0.5 * np.sin(xx / 23.0 + 3 * np.sin(yy / 90.0))
    bg = np.dstack([60 + 15 * wood, 105 + 20 * wood, 150 + 25 * wood])
    cam = np.clip(bg, 0, 255).astype(np.uint8)

    # 주차장을 배경 위에 투영해서 합성
    cv2.warpPerspective(top_noisy, H, (CAM_W, CAM_H), dst=cam,
                        borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.INTER_LINEAR)
    # 카메라 센서 노이즈
    cam = np.clip(cam.astype(np.float32) + rng.normal(0, 6, cam.shape), 0, 255)
    return cam.astype(np.uint8), H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "test_data"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    top_clean, top_noisy = make_top_view(rng)
    cam, H = make_camera_view(top_noisy, rng)

    cv2.imwrite(os.path.join(args.out, "topview_truth.png"), top_clean)
    cv2.imwrite(os.path.join(args.out, "topview_noisy.png"), top_noisy)
    cv2.imwrite(os.path.join(args.out, "camera_view.png"), cam)

    # 테스트용 config: 꼭짓점을 일부러 뒤섞인 순서로 저장해서
    # parking.py의 order_points가 제대로 정렬하는지도 함께 검증한다.
    shuffled = CAM_PTS[[2, 0, 3, 1]].astype(int).tolist()   # 우하, 좌상, 좌하, 우상
    half_w, half_h = TOP_W // 2, TOP_H // 2
    config = {
        "area_points": shuffled,
        "warp_size": [TOP_W, TOP_H],
        "spots": [
            {"id": 1, "rect": [0, 0, half_w, half_h]},
            {"id": 2, "rect": [half_w, 0, TOP_W - half_w, half_h]},
            {"id": 3, "rect": [0, half_h, half_w, TOP_H - half_h]},
            {"id": 4, "rect": [half_w, half_h, TOP_W - half_w, TOP_H - half_h]},
        ],
        "hsv": {"h1_min": 0, "h1_max": 10, "h2_min": 170, "h2_max": 179,
                "s_min": 120, "s_max": 255, "v_min": 70, "v_max": 255},
        "morph_kernel": 5,
        "occupied_ratio": 0.15,
        "debounce_frames": 1,
        "roi_margin": 8,
    }
    expected = {
        "spots": {"1": "OCCUPIED", "2": "AVAILABLE",
                  "3": "OCCUPIED", "4": "AVAILABLE"},
        "topview_reference": "topview_truth.png",
        "warp_size": [TOP_W, TOP_H],
        "homography_top_to_camera": H.tolist(),
        "camera_points_ordered": CAM_PTS.astype(int).tolist(),
    }
    with open(os.path.join(args.out, "test_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(args.out, "expected.json"), "w", encoding="utf-8") as f:
        json.dump(expected, f, indent=2)

    print(f"[생성 완료] {args.out}/")
    for name in ("camera_view.png", "topview_truth.png", "topview_noisy.png",
                 "test_config.json", "expected.json"):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
