#!/usr/bin/env bash
# 합성 이미지 생성 → --image + --no-gui 로 감지 → 정답 JSON과 비교
set -e
cd "$(dirname "$0")"
python3 make_test_image.py
python3 parking.py --image test_data/camera_view.png \
                   --config test_data/test_config.json \
                   --no-gui --out test_output \
                   --expected test_data/expected.json
