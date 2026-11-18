#!/usr/bin/env python3
"""
합성 테스트 이미지 생성기

흰 배경 위에 검은 도형을 그리고 노이즈를 섞어 test_image.png를 만들고,
정답 라벨(도형 종류 + 중심 좌표)을 test_image.json으로 함께 저장한다.

  - 삼각형 2개 (1개는 회전)
  - 사각형 2개: 정사각형 1개(회전) + 직사각형 1개(40° 회전)
      → 40° 회전된 직사각형은 boundingRect 기준 가로세로 비율이 ~1.05라
        정사각형으로 오인되기 쉽다. minAreaRect를 써야 하는 이유를 검증하는 용도.
  - 원 2개 (크기 다름)
  - 오각형 1개 (회전) → 정답은 "Unidentified"
  - 소금-후추 노이즈 + 가우시안 노이즈 + 작은 먼지 점들

사용법:
    python make_test_image.py                # test_image.png / test_image.json 생성
    python make_test_image.py --seed 7 --out my_test
"""
import argparse
import json
import math

import cv2
import numpy as np

WIDTH, HEIGHT = 1000, 700


def regular_polygon(center, radius, n, rot_deg):
    """중심/외접원 반지름/꼭짓점 수/회전각으로 정다각형 꼭짓점 좌표 생성"""
    cx, cy = center
    pts = []
    for i in range(n):
        # -90°에서 시작해야 회전 0일 때 꼭짓점이 위를 향한다
        a = math.radians(rot_deg - 90 + 360.0 * i / n)
        pts.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
    return np.round(np.array(pts)).astype(np.int32)


def rotated_rect(center, w, h, rot_deg):
    """회전된 사각형의 네 꼭짓점 좌표"""
    box = cv2.boxPoints(((float(center[0]), float(center[1])), (float(w), float(h)), float(rot_deg)))
    return np.round(box).astype(np.int32)


def mask_centroid(mask):
    """단일 도형 마스크의 무게중심 (정답 중심 좌표)"""
    m = cv2.moments(mask, binaryImage=True)
    return [round(m["m10"] / m["m00"], 1), round(m["m01"] / m["m00"], 1)]


def build_shapes():
    """(라벨, 그리기 함수) 목록. 각 함수는 주어진 마스크에 도형을 255로 칠한다."""
    shapes = [
        # 삼각형: 정삼각형(회전 없음) + 임의 삼각형(회전)
        ("Triangle", lambda m: cv2.fillPoly(m, [regular_polygon((150, 170), 95, 3, 0)], 255)),
        ("Triangle", lambda m: cv2.fillPoly(m, [np.array([[640, 520], [800, 600], [610, 640]], np.int32)], 255)),
        # 사각형: 정사각형(25° 회전) + 직사각형(40° 회전)
        ("Square", lambda m: cv2.fillPoly(m, [rotated_rect((420, 170), 140, 140, 25)], 255)),
        ("Rectangle", lambda m: cv2.fillPoly(m, [rotated_rect((170, 490), 200, 110, 40)], 255)),
        # 원: 큰 원 + 작은 원
        ("Circle", lambda m: cv2.circle(m, (700, 170), 85, 255, -1, lineType=cv2.LINE_AA)),
        ("Circle", lambda m: cv2.circle(m, (880, 420), 50, 255, -1, lineType=cv2.LINE_AA)),
        # 오각형(15° 회전) → 원도 사각형도 아니므로 Unidentified가 정답
        ("Unidentified", lambda m: cv2.fillPoly(m, [regular_polygon((430, 470), 90, 5, 15)], 255)),
    ]
    return shapes


def add_noise(img, rng, sp_amount, gauss_sigma, dust_count):
    """소금-후추 노이즈 + 가우시안 노이즈 + 작은 먼지 점 추가"""
    out = img.astype(np.float32)

    # 1) 가우시안 노이즈: 전체 화소에 약한 밝기 흔들림
    out += rng.normal(0, gauss_sigma, out.shape)

    # 2) 작은 먼지 점 (반지름 1~3px 검은 점) → 모폴로지 열기/최소 면적이 걸러내야 함
    for _ in range(dust_count):
        x, y = int(rng.integers(5, WIDTH - 5)), int(rng.integers(5, HEIGHT - 5))
        cv2.circle(out, (x, y), int(rng.integers(1, 4)), 0, -1)

    out = np.clip(out, 0, 255).astype(np.uint8)

    # 3) 소금-후추 노이즈: 일부 화소를 0(후추) 또는 255(소금)로
    r = rng.random(out.shape[:2])
    out[r < sp_amount / 2] = 0
    out[(r >= sp_amount / 2) & (r < sp_amount)] = 255
    return out


def main():
    ap = argparse.ArgumentParser(description="합성 도형 테스트 이미지 생성")
    ap.add_argument("--out", default="test_image", help="출력 파일 이름(확장자 제외)")
    ap.add_argument("--seed", type=int, default=42, help="노이즈 난수 시드")
    ap.add_argument("--sp", type=float, default=0.02, help="소금-후추 노이즈 비율 (기본 2%%)")
    ap.add_argument("--sigma", type=float, default=12.0, help="가우시안 노이즈 표준편차")
    ap.add_argument("--dust", type=int, default=60, help="먼지 점 개수")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    canvas = np.full((HEIGHT, WIDTH), 255, np.uint8)  # 흰 종이
    labels = []

    for idx, (label, draw) in enumerate(build_shapes(), start=1):
        # 도형별 마스크를 따로 그려서 정확한 정답 중심을 계산
        mask = np.zeros_like(canvas)
        draw(mask)
        canvas[mask > 127] = 0  # 흰 종이 위에 검은 도형
        labels.append({"id": idx, "label": label, "center": mask_centroid(mask),
                       "area": int(np.count_nonzero(mask > 127))})

    noisy = add_noise(canvas, rng, args.sp, args.sigma, args.dust)
    noisy_bgr = cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR)  # 웹캠 프레임과 같은 3채널로 저장

    img_path, json_path = f"{args.out}.png", f"{args.out}.json"
    cv2.imwrite(img_path, noisy_bgr)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"image": img_path, "width": WIDTH, "height": HEIGHT,
                   "noise": {"seed": args.seed, "salt_pepper": args.sp,
                             "gauss_sigma": args.sigma, "dust": args.dust},
                   "shapes": labels}, f, ensure_ascii=False, indent=2)

    print(f"[OK] {img_path} 저장 ({WIDTH}x{HEIGHT}, 도형 {len(labels)}개)")
    print(f"[OK] {json_path} 저장 (정답 라벨)")
    for s in labels:
        print(f"  #{s['id']} {s['label']:<12} center={s['center']}")


if __name__ == "__main__":
    main()
