"""
매장/공간 동선 분석 시스템
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
응용: 4위 추천 프로젝트 — 매장 동선 분석
기반 논문:
  📄 BoT-SORT   arXiv:2206.14651          (MOT — 궤적 추적)
  📄 ByteTrack  ECCV 2022 arXiv:2110.06864 (고신뢰 + 저신뢰 detection 통합)
  📄 CrowdSAM   ECCV 2024 arXiv:2408.01454 (밀도맵 기반 군중 분석)
  📄 CSRNet     CVPR 2018 arXiv:1802.10062 (Gaussian density map 표준)

━━ 핵심 기능 ━━
1. 구역(Zone) 설정  — 마우스로 최대 6개 사각형 구역 드래그
2. 구역별 실시간 체류 인원 + 과밀 경보
3. 누적 히트맵      — 어느 구역에 오래 머물렀는지 색상으로 표시
4. 동선 궤적        — ID별 색상 궤적 (최근 90프레임)
5. 시간대별 통계    — 구역별 총 방문자 수 + 현재 동시 체류 인원
6. CSV 저장         — s키로 현재 통계 snapshot 저장

━━ 단축키 ━━
  마우스 드래그 : 구역 추가 (최대 6개)
  z             : 마지막 구역 삭제
  c             : 구역 전체 초기화 + 히트맵 초기화
  h             : 히트맵 오버레이 ON/OFF
  t             : 궤적 ON/OFF
  s             : CSV 통계 저장 (stats_YYYYMMDD_HHMMSS.csv)
  q             : 종료
"""

from ultralytics import YOLO
import cv2
import numpy as np
import collections
import time
import csv
import os
from datetime import datetime

print("매장 동선 분석 시스템 로드 중...")
print("BoT-SORT (arXiv:2206.14651) | ByteTrack (ECCV 2022) | CSRNet (CVPR 2018)")

model = YOLO("yolo11n.pt")

# ─── 구역 정의 ───
# 각 zone: {"name": str, "rect": (x1,y1,x2,y2), "color": (B,G,R)}
ZONE_COLORS = [
    (0, 200, 255),   # 노랑-오렌지
    (255, 100, 0),   # 파랑
    (0, 255, 100),   # 초록
    (180, 0, 255),   # 보라
    (0, 180, 255),   # 황색
    (255, 0, 150),   # 핑크
]
zones = []
drawing = False
draw_start = (0, 0)
draw_current = (0, 0)

# ─── 추적 구조 ───
TRAJ_LEN = 90
trajectories = collections.defaultdict(lambda: collections.deque(maxlen=TRAJ_LEN))

def id_to_color(track_id):
    np.random.seed(track_id * 13 + 7)
    return tuple(int(c) for c in np.random.randint(80, 240, 3))

# ─── 구역 통계 ───
# zone_visit_counts[i] = 해당 구역을 방문한 총 ID 수 (중복 제거)
zone_visit_ids   = [set() for _ in range(6)]   # 전체 방문 ID 집합
zone_dwell_frames = [0] * 6                     # 누적 체류 프레임 수
zone_current      = [0] * 6                     # 현재 체류 인원

# ─── 히트맵 누적 ───
cap = cv2.VideoCapture(0)
ret, init_frame = cap.read()
h, w = init_frame.shape[:2]
heatmap_acc = np.zeros((h, w), dtype=np.float32)

show_heatmap  = True
show_traj     = True
fps_buf       = []
frame_idx     = 0
total_ids     = set()

# ─── 마우스 콜백 ───
def mouse_cb(event, x, y, flags, param):
    global drawing, draw_start, draw_current, zones
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(zones) < 6:
            drawing = True
            draw_start = (x, y)
            draw_current = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        draw_current = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        x1, y1 = min(draw_start[0], x), min(draw_start[1], y)
        x2, y2 = max(draw_start[0], x), max(draw_start[1], y)
        if x2 - x1 > 20 and y2 - y1 > 20:
            idx = len(zones)
            zones.append({
                "name":  f"Zone {idx+1}",
                "rect":  (x1, y1, x2, y2),
                "color": ZONE_COLORS[idx % len(ZONE_COLORS)],
            })
            print(f"구역 추가: Zone {idx+1} ({x1},{y1})-({x2},{y2})")

cv2.namedWindow("Store Traffic Analysis")
cv2.setMouseCallback("Store Traffic Analysis", mouse_cb)

print("마우스 드래그: 구역 추가 | z: 마지막 구역 삭제 | c: 초기화 | h: 히트맵 | t: 궤적 | s: CSV 저장 | q: 종료")
print("먼저 화면에서 분석할 구역을 마우스로 드래그해 지정하세요.")

# ─── CSV 저장 함수 ───
def save_csv():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"stats_{ts}.csv"
    fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
    with open(fpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["저장시각", ts])
        writer.writerow(["총 추적 ID 수", len(total_ids)])
        writer.writerow([])
        writer.writerow(["구역명", "총 방문자(ID)", "누적 체류 프레임", "현재 체류 인원"])
        for i, z in enumerate(zones):
            writer.writerow([
                z["name"],
                len(zone_visit_ids[i]),
                zone_dwell_frames[i],
                zone_current[i],
            ])
    print(f"CSV 저장 완료: {fpath}")

# ─── 메인 루프 ───
while True:
    t0 = time.perf_counter()
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # ── 추적 ──
    results = model.track(
        frame,
        device='mps',
        persist=True,
        tracker="botsort.yaml",
        classes=[0],          # 사람만
        verbose=False,
        conf=0.35,
    )

    fps_buf.append(1.0 / (time.perf_counter() - t0))
    if len(fps_buf) > 15:
        fps_buf.pop(0)
    fps = np.mean(fps_buf)

    # ── 현재 프레임 구역별 체류 초기화 ──
    zone_current = [0] * 6

    # ── 히트맵 감쇠 ──
    heatmap_acc *= 0.997

    # ── 검출 결과 처리 ──
    if results[0].boxes.id is not None:
        ids   = results[0].boxes.id.int().tolist()
        xyxys = results[0].boxes.xyxy.tolist()

        for box, track_id in zip(xyxys, ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            total_ids.add(track_id)
            trajectories[track_id].append((cx, cy))

            # 히트맵 누적
            if 0 <= cy < h and 0 <= cx < w:
                cv2.circle(heatmap_acc, (cx, cy), 18, 1.5, -1)

            # 구역 체류 판단 (발 위치 = 박스 하단 중심)
            foot_x, foot_y = cx, y2
            for i, z in enumerate(zones):
                zx1, zy1, zx2, zy2 = z["rect"]
                if zx1 <= foot_x <= zx2 and zy1 <= foot_y <= zy2:
                    zone_visit_ids[i].add(track_id)
                    zone_dwell_frames[i] += 1
                    zone_current[i] += 1

    # ─── 시각화 레이어 조합 ───
    out = frame.copy()

    # 히트맵 오버레이
    if show_heatmap and heatmap_acc.max() > 0:
        norm_h = cv2.normalize(heatmap_acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heat_col = cv2.applyColorMap(norm_h, cv2.COLORMAP_JET)
        # 히트맵이 있는 픽셀만 블렌드
        mask_heat = norm_h > 10
        out[mask_heat] = cv2.addWeighted(frame, 0.35, heat_col, 0.65, 0)[mask_heat]

    # 궤적 그리기
    if show_traj and results[0].boxes.id is not None:
        ids   = results[0].boxes.id.int().tolist()
        xyxys = results[0].boxes.xyxy.tolist()
        for box, track_id in zip(xyxys, ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            color = id_to_color(track_id)
            pts = list(trajectories[track_id])
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                c = tuple(int(v * alpha) for v in color)
                cv2.line(out, pts[i-1], pts[i], c, max(1, int(alpha * 2)), cv2.LINE_AA)
            cv2.circle(out, (cx, cy), 5, color, -1, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), 6, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(out, f"#{track_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # 드래그 중인 임시 구역
    if drawing:
        x1t = min(draw_start[0], draw_current[0])
        y1t = min(draw_start[1], draw_current[1])
        x2t = max(draw_start[0], draw_current[0])
        y2t = max(draw_start[1], draw_current[1])
        cv2.rectangle(out, (x1t, y1t), (x2t, y2t), (200, 200, 200), 1)

    # 구역 그리기 + 통계
    for i, z in enumerate(zones):
        zx1, zy1, zx2, zy2 = z["rect"]
        col = z["color"]
        cur = zone_current[i]

        # 과밀 판단 (3명 이상)
        alert = cur >= 3
        border_col = (0, 0, 255) if alert else col
        thickness  = 3 if alert else 1

        # 구역 반투명 채우기
        overlay = out.copy()
        fill_col = (0, 0, 200) if alert else col
        cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), fill_col, -1)
        cv2.addWeighted(overlay, 0.12, out, 0.88, 0, out)

        # 구역 테두리
        cv2.rectangle(out, (zx1, zy1), (zx2, zy2), border_col, thickness)

        # 구역 레이블 배경
        label_h = 44
        cv2.rectangle(out, (zx1, zy1), (zx1 + 160, zy1 + label_h), (15, 15, 15), -1)

        # 구역명 + 현재 인원
        status = f"!!! ALERT {cur}명" if alert else f"현재 {cur}명"
        label_col = (0, 80, 255) if alert else (255, 255, 255)
        cv2.putText(out, z["name"], (zx1 + 5, zy1 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1)
        cv2.putText(out, status, (zx1 + 5, zy1 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, label_col, 1)

    # 안내 문구 (구역 없을 때)
    if len(zones) == 0:
        msg = "마우스 드래그로 분석 구역을 지정하세요 (최대 6개)"
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        mx = (w - tw) // 2
        cv2.rectangle(out, (mx - 8, h//2 - 22), (mx + tw + 8, h//2 + 10), (30, 30, 30), -1)
        cv2.putText(out, msg, (mx, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1)

    # ─── 우측 통계 패널 ───
    panel_w = 200
    panel_x = w - panel_w - 8
    n_zones  = len(zones)
    panel_h  = 28 + max(n_zones, 1) * 56 + 10
    cv2.rectangle(out, (panel_x, 65), (panel_x + panel_w, 65 + panel_h), (18, 18, 18), -1)
    cv2.rectangle(out, (panel_x, 65), (panel_x + panel_w, 65 + panel_h), (80, 80, 80), 1)
    cv2.putText(out, f"Zone Statistics", (panel_x + 6, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1)

    for i in range(n_zones):
        z = zones[i]
        py = 65 + 28 + i * 56
        col = z["color"]
        cv2.putText(out, z["name"], (panel_x + 6, py + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1)
        cv2.putText(out, f"  방문자: {len(zone_visit_ids[i])}명",
                    (panel_x + 6, py + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        dwell_sec = zone_dwell_frames[i] / max(fps, 1)
        cv2.putText(out, f"  체류: {dwell_sec:.0f}초",
                    (panel_x + 6, py + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    if n_zones == 0:
        cv2.putText(out, "  구역 없음", (panel_x + 6, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    # ─── 상단 정보 패널 ───
    n_active = len(results[0].boxes.id) if results[0].boxes.id is not None else 0
    cv2.rectangle(out, (0, 0), (w, 58), (15, 15, 15), -1)
    cv2.putText(out, f"Store Traffic Analysis  |  Active: {n_active}  Total IDs: {len(total_ids)}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1)
    flags = []
    if show_heatmap: flags.append("HEATMAP")
    if show_traj:    flags.append("TRAJ")
    cv2.putText(out, f"FPS: {fps:.1f}  |  Zones: {n_zones}/6  |  {' + '.join(flags) if flags else 'PLAIN'}",
                (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1)

    # ─── 하단 논문 출처 ───
    cv2.putText(out, "BoT-SORT arXiv:2206.14651 | ByteTrack ECCV2022 | CSRNet CVPR2018:1802.10062",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (150, 150, 150), 1)

    cv2.imshow("Store Traffic Analysis", out)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('z'):
        if zones:
            removed = zones.pop()
            print(f"구역 삭제: {removed['name']}")
    elif key == ord('c'):
        zones.clear()
        heatmap_acc[:] = 0
        trajectories.clear()
        total_ids.clear()
        zone_visit_ids   = [set() for _ in range(6)]
        zone_dwell_frames = [0] * 6
        print("전체 초기화")
    elif key == ord('h'):
        show_heatmap = not show_heatmap
        print(f"히트맵: {'ON' if show_heatmap else 'OFF'}")
    elif key == ord('t'):
        show_traj = not show_traj
        print(f"궤적: {'ON' if show_traj else 'OFF'}")
    elif key == ord('s'):
        save_csv()

cap.release()
cv2.destroyAllWindows()
