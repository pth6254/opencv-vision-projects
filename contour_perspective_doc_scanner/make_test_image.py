"""
문서 스캐너 테스트용 합성 이미지 생성기 (numpy + OpenCV만 사용)

어두운 배경 위에 기울어진(회전 + 원근) 흰 A4 용지를 그린다.
용지 안에는 방향 확인용 마커를 넣는다:
    - 좌상단: 빨간 사각형 / 우하단: 파란 사각형
    - 검은 테두리, 격자선, 텍스트 줄
warp 결과를 정답(평평한 용지)과 비교할 수 있도록
정답 이미지와 실제 꼭짓점 좌표(JSON)도 함께 저장한다.

사용 예:
    python make_test_image.py                 # test_images/ 에 기본 세트 생성
    python make_test_image.py --out-dir imgs
"""

import argparse
import json
import os

import cv2
import numpy as np

# 정답 용지 크기 = 스캐너 출력 크기와 동일한 A4 비율(210:297)
PAPER_W, PAPER_H = 420, 594
CANVAS_W, CANVAS_H = 1280, 720   # 일반 웹캠 해상도


def make_flat_paper():
    """정면에서 본 평평한 A4 용지(정답 이미지)를 만든다."""
    paper = np.full((PAPER_H, PAPER_W, 3), 245, dtype=np.uint8)  # 약간 회색빛 흰색
    # 안쪽 검은 테두리 (펴졌을 때 직사각형인지 확인용)
    cv2.rectangle(paper, (20, 20), (PAPER_W - 21, PAPER_H - 21), (0, 0, 0), 3)
    # 격자선: warp 후 선이 수직/수평이면 원근이 제대로 보정된 것
    for x in range(60, PAPER_W - 20, 60):
        cv2.line(paper, (x, 20), (x, PAPER_H - 21), (180, 180, 180), 1)
    for y in range(60, PAPER_H - 20, 60):
        cv2.line(paper, (20, y), (PAPER_W - 21, y), (180, 180, 180), 1)
    # 방향 마커: 좌상=빨강, 우하=파랑 (꼭짓점 순서가 틀리면 위치가 바뀜)
    cv2.rectangle(paper, (35, 35), (95, 95), (0, 0, 255), -1)
    cv2.rectangle(paper, (PAPER_W - 96, PAPER_H - 96),
                  (PAPER_W - 36, PAPER_H - 36), (255, 0, 0), -1)
    # 텍스트 줄
    for i, text in enumerate(["DOCUMENT SCANNER", "Perspective Test",
                              "Line 3: 0123456789", "Line 4: ABCDEFG"]):
        cv2.putText(paper, text, (110, 150 + i * 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    return paper


def make_background(rng):
    """어두운 배경 (나무 책상 느낌의 약한 노이즈 + 그라데이션)."""
    grad = np.linspace(30, 60, CANVAS_W, dtype=np.float32)[None, :, None]
    bg = np.broadcast_to(grad, (CANVAS_H, CANVAS_W, 3)).copy()
    bg += rng.normal(0, 6, bg.shape).astype(np.float32)
    return np.clip(bg, 0, 255).astype(np.uint8)


def compose(corners, seed=0, noise=4.0):
    """평평한 용지를 corners(좌상, 우상, 우하, 좌하) 위치로 투영해 합성한다."""
    rng = np.random.default_rng(seed)
    paper = make_flat_paper()
    canvas = make_background(rng)

    # 정답 용지 → 화면 좌표로의 투시 변환
    src = np.array([[0, 0], [PAPER_W - 1, 0],
                    [PAPER_W - 1, PAPER_H - 1], [0, PAPER_H - 1]],
                   dtype=np.float32)
    dst = np.asarray(corners, dtype=np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(paper, m, (CANVAS_W, CANVAS_H))
    mask = cv2.warpPerspective(np.full((PAPER_H, PAPER_W), 255, np.uint8),
                               m, (CANVAS_W, CANVAS_H))
    # 용지 영역만 배경 위에 덮어쓰기
    canvas[mask > 0] = warped[mask > 0]

    # 카메라 센서 노이즈 흉내
    out = canvas.astype(np.float32) + rng.normal(0, noise, canvas.shape)
    return np.clip(out, 0, 255).astype(np.uint8), paper


def rotated_corners(cx, cy, w, h, angle_deg, persp=0.0):
    """중심(cx,cy), 크기(w,h)인 사각형을 angle만큼 회전하고
    persp 비율만큼 윗변을 좁혀 원근감을 준 꼭짓점(좌상, 우상, 우하, 좌하)."""
    shrink = w * persp / 2
    local = np.array([[-w / 2 + shrink, -h / 2], [w / 2 - shrink, -h / 2],
                      [w / 2, h / 2], [-w / 2, h / 2]], dtype=np.float32)
    t = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]],
                   dtype=np.float32)
    return local @ rot.T + np.array([cx, cy], dtype=np.float32)


# 기본 테스트 세트: (이름, 꼭짓점)
DEFAULT_CASES = {
    "tilted_cw":    rotated_corners(640, 360, 330, 467, 15),
    "tilted_ccw":   rotated_corners(600, 370, 320, 452, -20),
    "perspective":  rotated_corners(660, 360, 380, 520, 8, persp=0.25),
    "strong_persp": np.array([[520, 90], [830, 120],
                              [960, 650], [380, 610]], dtype=np.float32),
}


def main():
    p = argparse.ArgumentParser(description="합성 테스트 이미지 생성")
    p.add_argument("--out-dir", default="test_images")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    meta = {}
    for i, (name, corners) in enumerate(DEFAULT_CASES.items()):
        img, paper = compose(corners, seed=i)
        cv2.imwrite(os.path.join(args.out_dir, f"{name}.png"), img)
        meta[name] = corners.round(2).tolist()
        print(f"[생성] {name}.png  꼭짓점={meta[name]}")

    # 정답(평평한 용지) 이미지
    cv2.imwrite(os.path.join(args.out_dir, "ground_truth.png"),
                make_flat_paper())

    # 음성 케이스: 용지 없음 / 너무 작은 용지(면적 10% 미만)
    rng = np.random.default_rng(99)
    cv2.imwrite(os.path.join(args.out_dir, "no_document.png"),
                make_background(rng))
    small, _ = compose(rotated_corners(300, 300, 120, 170, 10), seed=7)
    cv2.imwrite(os.path.join(args.out_dir, "too_small.png"), small)
    print("[생성] no_document.png, too_small.png (검출되면 안 되는 케이스)")

    with open(os.path.join(args.out_dir, "corners.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[저장] {os.path.join(args.out_dir, 'corners.json')}")


if __name__ == "__main__":
    main()
