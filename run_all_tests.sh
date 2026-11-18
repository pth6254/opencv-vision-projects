#!/usr/bin/env bash
# 6개 프로젝트의 자동 테스트를 순서대로 실행한다 (GUI 없음, 웹캠 불필요).
# 테스트용 합성 이미지/영상은 매번 새로 생성된다.
#   ./run_all_tests.sh            # 기본 테스트
#   ./run_all_tests.sh --stress   # 문서 스캐너 스트레스 테스트까지 (수십 초)
set -u
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
pass=0; fail=0; failed=()

run() {   # run <이름> <폴더> <명령...>
    local name="$1" dir="$2"; shift 2
    echo "════════ $name"
    if (cd "$dir" && "$@"); then pass=$((pass + 1)); else fail=$((fail + 1)); failed+=("$name"); fi
    echo
}

run "color_ball_tracker" color_ball_tracker \
    bash -c "$PY make_test_image.py >/dev/null && $PY color_tracker.py --image test_image.png --no-gui --expect 400 260"
run "shape_classifier" shape_classifier \
    bash -c "$PY make_test_image.py >/dev/null && $PY shape_classifier.py --image test_image.png --no-gui"
run "contour_perspective_doc_scanner" contour_perspective_doc_scanner "$PY" test_scanner.py
run "hough_line_perspective_lane" hough_line_perspective_lane \
    bash -c "$PY make_test_image.py >/dev/null && $PY test_lane.py"
run "parking_spot_detector" parking_spot_detector bash run_test.sh
run "speed_meter" speed_meter "$PY" test_speed.py
if [ "${1:-}" = "--stress" ]; then
    run "doc_scanner stress" contour_perspective_doc_scanner "$PY" stress_test.py --n 30
fi

echo "════════ 결과: 통과 $pass / 실패 $fail"
[ "$fail" -eq 0 ] || { printf '  실패: %s\n' "${failed[@]}"; exit 1; }
