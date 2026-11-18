"""
실제 웹캠 환경에 가까운 어려운 장면에서 인식률과 속도를 측정한다.

각 시나리오마다 무작위 각도/위치로 여러 장을 생성해
- 검출 성공률 (꼭짓점 오차 10px 이내를 성공으로 간주)
- 프레임당 처리 시간(ms)
을 출력한다.

사용: python stress_test.py [--n 30] [--width 1280 --height 720]
"""

import argparse
import os
import time
import zlib

import cv2
import numpy as np

import doc_scanner as ds
from make_test_image import make_flat_paper, PAPER_W, PAPER_H

SUCCESS_TOL_PX = 10.0


def random_corners(rng, w, h, h_range=(0.55, 0.8)):
    """화면 안에 들어오는 무작위 회전 + 원근 용지 꼭짓점.
    h_range: 용지 높이 / 화면 높이 범위"""
    ph = h * rng.uniform(*h_range)
    pw = ph * 210 / 297
    cx = rng.uniform(w * 0.35, w * 0.65)
    cy = rng.uniform(h * 0.45, h * 0.55)
    ang = np.deg2rad(rng.uniform(-30, 30))
    shrink = pw * rng.uniform(0, 0.2) / 2
    local = np.array([[-pw / 2 + shrink, -ph / 2], [pw / 2 - shrink, -ph / 2],
                      [pw / 2, ph / 2], [-pw / 2, ph / 2]], np.float32)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]],
                   np.float32)
    c = local @ rot.T + np.array([cx, cy], np.float32)
    # 종이가 화면 밖으로 나가면 위치를 화면 안쪽(여백 15px)으로 밀어 넣는다
    margin = 15
    c[:, 0] += max(0, margin - c[:, 0].min()) - max(0, c[:, 0].max() - (w - 1 - margin))
    c[:, 1] += max(0, margin - c[:, 1].min()) - max(0, c[:, 1].max() - (h - 1 - margin))
    return c


def paste_paper(bg, corners, paper_level=245):
    paper = make_flat_paper().astype(np.float32) * (paper_level / 245.0)
    paper = np.clip(paper, 0, 255).astype(np.uint8)
    src = np.array([[0, 0], [PAPER_W - 1, 0], [PAPER_W - 1, PAPER_H - 1],
                    [0, PAPER_H - 1]], np.float32)
    m = cv2.getPerspectiveTransform(src, corners)
    h, w = bg.shape[:2]
    warped = cv2.warpPerspective(paper, m, (w, h))
    mask = cv2.warpPerspective(np.full((PAPER_H, PAPER_W), 255, np.uint8),
                               m, (w, h))
    out = bg.copy()
    out[mask > 0] = warped[mask > 0]
    return out


def dark_bg(rng, w, h, level=45):
    bg = np.full((h, w, 3), level, np.float32)
    bg += rng.normal(0, 5, bg.shape)
    return np.clip(bg, 0, 255).astype(np.uint8)


def textured_bg(rng, w, h):
    """나무결 같은 줄무늬 + 잡다한 물체(펜, 컵, 케이블)가 있는 책상."""
    bg = dark_bg(rng, w, h, 50).astype(np.float32)
    ys = np.arange(h)[:, None]
    grain = 12 * np.sin(ys / rng.uniform(4, 9) + rng.uniform(0, 6)) \
        + 6 * np.sin(np.arange(w)[None, :] / 40.0)
    bg += grain[..., None]
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    for _ in range(12):
        c = tuple(int(v) for v in rng.integers(60, 160, 3))
        p1 = tuple(int(v) for v in (rng.integers(0, w), rng.integers(0, h)))
        p2 = tuple(int(v) for v in (rng.integers(0, w), rng.integers(0, h)))
        cv2.line(bg, p1, p2, c, int(rng.integers(2, 8)))
    for _ in range(5):
        c = tuple(int(v) for v in rng.integers(60, 180, 3))
        center = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.circle(bg, center, int(rng.integers(20, 70)), c, -1)
    return bg


def add_noise(img, rng, sigma=4):
    out = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 시나리오: rng, w, h -> (image, true_corners)
# ---------------------------------------------------------------------------
def sc_clean(rng, w, h):
    c = random_corners(rng, w, h)
    return add_noise(paste_paper(dark_bg(rng, w, h), c), rng), c


def sc_cluttered_desk(rng, w, h):
    c = random_corners(rng, w, h)
    return add_noise(paste_paper(textured_bg(rng, w, h), c), rng), c


def sc_low_contrast(rng, w, h):
    # 회색 책상 위 약간 어두운 종이 (실내 저조도)
    c = random_corners(rng, w, h)
    img = paste_paper(dark_bg(rng, w, h, 110), c, paper_level=175)
    return add_noise(img, rng, 6), c


def sc_uneven_light(rng, w, h):
    # 한쪽에서 비치는 조명: 화면 전체에 강한 밝기 그라데이션
    c = random_corners(rng, w, h)
    img = paste_paper(dark_bg(rng, w, h, 60), c).astype(np.float32)
    gx = np.linspace(0.35, 1.15, w, dtype=np.float32)[None, :, None]
    gy = np.linspace(1.1, 0.7, h, dtype=np.float32)[:, None, None]
    img = np.clip(img * gx * gy, 0, 255).astype(np.uint8)
    return add_noise(img, rng), c


def sc_blur(rng, w, h):
    # 초점 흐림 + 손떨림
    c = random_corners(rng, w, h)
    img = paste_paper(dark_bg(rng, w, h), c)
    img = cv2.GaussianBlur(img, (0, 0), 3.0)
    k = np.zeros((15, 15), np.float32)
    k[7, :] = 1 / 15
    img = cv2.filter2D(img, -1, k)
    return add_noise(img, rng), c


def sc_heavy_noise(rng, w, h):
    # 어두운 방, 감도(ISO) 높은 웹캠 노이즈 + JPEG 압축
    c = random_corners(rng, w, h)
    img = add_noise(paste_paper(dark_bg(rng, w, h), c, 210), rng, 18)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 40])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), c


def sc_bright_object_touching(rng, w, h):
    # 종이 옆에 밝은 물체(키보드, 다른 종이 등)가 맞닿아 있음
    c = random_corners(rng, w, h)
    bg = dark_bg(rng, w, h)
    img = paste_paper(bg, c)
    # 종이의 오른쪽 변에 붙은 밝은 회색 직사각형
    edge_mid = (c[1] + c[2]) / 2
    pt = edge_mid.astype(int)
    cv2.rectangle(img, (int(pt[0]) - 5, int(pt[1]) - 60),
                  (int(pt[0]) + 160, int(pt[1]) + 60), (200, 200, 200), -1)
    img = paste_paper(img, c)  # 종이가 위에 놓이도록 다시 덮음
    return add_noise(img, rng), c


def sc_finger_occlusion(rng, w, h):
    # 손가락이 종이 아래 변을 가림
    c = random_corners(rng, w, h)
    img = paste_paper(dark_bg(rng, w, h), c)
    mid = ((c[2] + c[3]) / 2).astype(int)
    cv2.ellipse(img, (int(mid[0]), int(mid[1]) + 30), (28, 70), 0, 0, 360,
                (120, 150, 190), -1)
    return add_noise(img, rng), c


def sc_shadow_on_edge(rng, w, h):
    # 손/폰 그림자가 종이 한쪽 모서리를 덮음
    c = random_corners(rng, w, h)
    img = paste_paper(dark_bg(rng, w, h), c).astype(np.float32)
    mask = np.zeros(img.shape[:2], np.float32)
    corner = c[1].astype(int)
    cv2.circle(mask, (int(corner[0]), int(corner[1])), 140, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 40)[..., None]
    img = img * (1 - 0.55 * mask)
    return add_noise(np.clip(img, 0, 255).astype(np.uint8), rng), c


def sc_far_away(rng, w, h):
    # 멀리 놓인 용지 (16:9 화면 면적의 약 12~17%, 최소 면적 10%보다 조금 큼)
    c = random_corners(rng, w, h, h_range=(0.55, 0.65))
    return add_noise(paste_paper(dark_bg(rng, w, h), c), rng), c


# --- 음성 시나리오: 문서가 없으므로 '미검출'이 정답 (오검출 측정) ---
def neg_plain(rng, w, h):
    return add_noise(dark_bg(rng, w, h), rng), None


def neg_cluttered(rng, w, h):
    return add_noise(textured_bg(rng, w, h), rng), None


def neg_too_small(rng, w, h):
    # 화면 면적 4~6%의 작은 종이 → 최소 면적(10%) 미만이므로 무시해야 함
    c = random_corners(rng, w, h, h_range=(0.3, 0.38))
    return add_noise(paste_paper(textured_bg(rng, w, h), c), rng), None


SCENARIOS = {
    "clean": sc_clean,
    "cluttered_desk": sc_cluttered_desk,
    "low_contrast": sc_low_contrast,
    "uneven_light": sc_uneven_light,
    "blur": sc_blur,
    "heavy_noise_jpeg": sc_heavy_noise,
    "bright_obj_touch": sc_bright_object_touching,
    "finger_occlusion": sc_finger_occlusion,
    "shadow_on_edge": sc_shadow_on_edge,
    "far_away": sc_far_away,
}

NEGATIVE_SCENARIOS = {
    "neg_plain": neg_plain,
    "neg_cluttered": neg_cluttered,
    "neg_too_small": neg_too_small,
}


def evaluate(n, w, h, detect=None, save_fail_dir=None):
    """detect(frame) -> pts 또는 None. 기본은 doc_scanner.detect_document.
    반환: (양성 성공률, 음성 오검출 수)"""
    if detect is None:
        detect = lambda f: ds.detect_document(f)[1]
    results = {}
    times = []
    print(f"{'scenario':<20}{'correct':>10}{'ms/frame':>10}")
    for name, fn in {**SCENARIOS, **NEGATIVE_SCENARIOS}.items():
        # 시나리오 이름으로 고정 시드 (실행할 때마다 같은 이미지)
        rng = np.random.default_rng(zlib.crc32(name.encode()))
        ok, saved, sc_times = 0, 0, []
        for _ in range(n):
            img, true_c = fn(rng, w, h)
            t0 = time.perf_counter()
            pts = detect(img)
            sc_times.append((time.perf_counter() - t0) * 1000)
            if true_c is None:
                good = pts is None  # 음성: 아무것도 찾지 않아야 정답
            else:
                good = pts is not None and np.linalg.norm(
                    ds.order_points(pts) - true_c, axis=1).max() < SUCCESS_TOL_PX
            ok += good
            if not good and save_fail_dir and saved < 2:
                saved += 1
                os.makedirs(save_fail_dir, exist_ok=True)
                cv2.imwrite(f"{save_fail_dir}/{name}_{saved}.png",
                            ds.draw_detection(img, pts))
        results[name] = ok
        times += sc_times
        print(f"{name:<20}{ok:>6}/{n:<3}{np.median(sc_times):>10.1f}")

    pos = sum(results[k] for k in SCENARIOS)
    neg_fp = sum(n - results[k] for k in NEGATIVE_SCENARIOS)
    print(f"양성 검출 성공: {pos}/{n * len(SCENARIOS)} "
          f"({100 * pos / (n * len(SCENARIOS)):.0f}%)")
    print(f"음성 오검출:   {neg_fp}/{n * len(NEGATIVE_SCENARIOS)}")
    print(f"처리 시간 중앙값: {np.median(times):.1f} ms/frame")
    return pos / (n * len(SCENARIOS)), neg_fp


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--save-fail", default=None,
                   help="실패 사례 이미지를 저장할 폴더")
    a = p.parse_args()
    evaluate(a.n, a.width, a.height, save_fail_dir=a.save_fail)
