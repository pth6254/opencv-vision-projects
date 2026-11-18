"""
합성 테스트 이미지 생성기

- 회색 배경 위에 위치를 알고 있는 오렌지색 원 1개
- 화면 곳곳에 오렌지색 소금-후추 노이즈 (1~3px 크기 점 수백 개)
- 원 내부에 검은 점(구멍) 몇 개

정답 중심 좌표는 콘솔과 <출력파일>.json 에 기록된다.

사용 예:
    python make_test_image.py                       # test_image.png 생성
    python make_test_image.py --cx 200 --cy 150 --seed 7
"""

import argparse
import json
import os

import cv2
import numpy as np

ORANGE_BGR = (0, 140, 255)   # HSV ≈ (16, 255, 255) → 기본 범위 5~25 안에 들어감
GRAY_BGR = (128, 128, 128)   # 채도 0인 회색 배경
BLACK_BGR = (0, 0, 0)


def make_image(width, height, cx, cy, radius, n_noise, n_holes, seed):
    rng = np.random.default_rng(seed)

    # 회색 배경
    img = np.full((height, width, 3), GRAY_BGR, dtype=np.uint8)

    # 추적 대상: 오렌지색 원 (안티에일리어싱 없이 그려 중심이 정확히 (cx, cy))
    cv2.circle(img, (cx, cy), radius, ORANGE_BGR, -1, lineType=cv2.LINE_8)

    # 원 내부의 검은 구멍: 닫기 연산이 메워야 하는 대상
    for _ in range(n_holes):
        ang = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(0, radius * 0.7)
        hx = int(cx + dist * np.cos(ang))
        hy = int(cy + dist * np.sin(ang))
        cv2.circle(img, (hx, hy), int(rng.integers(2, 4)), BLACK_BGR, -1)

    # 오렌지 소금-후추 노이즈: 열기 연산이 지워야 하는 대상
    # (원 주변은 피해서, 노이즈가 원에 붙어 중심을 왜곡하지 않게 함)
    placed = 0
    while placed < n_noise:
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        if np.hypot(x - cx, y - cy) < radius + 15:
            continue
        s = int(rng.integers(1, 4))           # 1x1 ~ 3x3 정사각형 점
        img[y:y + s, x:x + s] = ORANGE_BGR
        placed += 1

    # 소량의 검은 후추 노이즈도 배경에 섞음 (색 검출에는 영향 없어야 함)
    for _ in range(n_noise // 4):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        if np.hypot(x - cx, y - cy) < radius + 15:
            continue
        img[y, x] = BLACK_BGR

    return img


def main():
    ap = argparse.ArgumentParser(description="컬러 추적기용 합성 테스트 이미지 생성")
    ap.add_argument("--out", default="test_image.png")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--cx", type=int, default=400)
    ap.add_argument("--cy", type=int, default=260)
    ap.add_argument("--radius", type=int, default=60)
    ap.add_argument("--noise", type=int, default=400, help="오렌지 노이즈 점 개수")
    ap.add_argument("--holes", type=int, default=6, help="원 내부 검은 점 개수")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    img = make_image(args.width, args.height, args.cx, args.cy, args.radius,
                     args.noise, args.holes, args.seed)
    cv2.imwrite(args.out, img)

    truth = {"cx": args.cx, "cy": args.cy, "radius": args.radius,
             "noise": args.noise, "holes": args.holes, "seed": args.seed}
    json_path = os.path.splitext(args.out)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(truth, f, indent=2)

    print(f"생성: {args.out}  ({args.width}x{args.height})")
    print(f"정답: 중심=({args.cx}, {args.cy}), 반지름={args.radius}  → {json_path}")
    print(f"검증: python color_tracker.py --image {args.out} --no-gui "
          f"--expect {args.cx} {args.cy}")


if __name__ == "__main__":
    main()
