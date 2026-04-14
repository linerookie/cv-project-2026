"""
ORB Feature Detection & Matching
Paper: "ORB: An efficient alternative to SIFT or SURF"
Authors: Rublee, E., Rabaud, V., Konolige, K., Bradski, G.
Conference: ICCV 2011 (IEEE International Conference on Computer Vision)
Citations: ~10,000+

핵심 아이디어:
- FAST keypoint detector + Harris corner measure로 키포인트 추출
- Rotation-invariant BRIEF descriptor 계산 (oFAST + rBRIEF)
- 이진 디스크립터(bit string)로 SIFT보다 100배 빠르고 특허 무료
- Hamming distance로 매칭 (XOR 연산 → 매우 빠름)

사용법:
- 'r': 현재 프레임을 참조 프레임으로 설정
- 'q': 종료
"""

import cv2
import numpy as np

# ─── Rublee et al. (ICCV 2011) ORB ───
orb = cv2.ORB_create(
    nfeatures=500,      # 최대 키포인트 수
    scaleFactor=1.2,    # 이미지 피라미드 축소 비율
    nlevels=8,          # 피라미드 레벨 수
    edgeThreshold=15,
    WTA_K=2             # descriptor 생성 시 비교 포인트 수
)

# BFMatcher with Hamming distance (이진 descriptor용)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

cap = cv2.VideoCapture(0)

ref_gray = None
ref_kp = None
ref_des = None
mode = "live"  # "live" or "matching"

print("=== ORB Feature Matching (ICCV 2011) ===")
print("논문: Rublee et al., ICCV 2011")
print("r: 현재 프레임을 레퍼런스로 저장 → 매칭 시작")
print("c: 레퍼런스 초기화 (라이브 모드)")
print("q: 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)

    if mode == "live" or ref_gray is None:
        # 라이브 모드: 현재 프레임의 키포인트만 표시
        out = cv2.drawKeypoints(
            frame, kp, None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS  # 방향/크기 표시
        )
        cv2.putText(out, f"ORB Keypoints: {len(kp)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(out, "Press 'r' to set reference frame", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
        cv2.putText(out, "ICCV 2011 - Rublee et al. ORB", (10, out.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.imshow("ORB Feature Matching (ICCV 2011)", out)

    else:
        # 매칭 모드: 레퍼런스 vs 현재 프레임
        if des is not None and ref_des is not None and len(des) > 10:
            matches = bf.match(ref_des, des)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches[:min(60, len(matches))]

            # 평균 매칭 거리 (낮을수록 유사)
            avg_dist = np.mean([m.distance for m in good_matches]) if good_matches else 999

            out = cv2.drawMatches(
                ref_gray, ref_kp,
                frame, kp,
                good_matches, None,
                matchColor=(0, 255, 0),
                singlePointColor=(0, 0, 255),
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            cv2.putText(out, f"Matches: {len(good_matches)} | Avg dist: {avg_dist:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(out, "ICCV 2011 - ORB Matching  [c: reset]", (10, out.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        else:
            out = frame.copy()
            cv2.putText(out, "Not enough keypoints", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("ORB Feature Matching (ICCV 2011)", out)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        ref_gray = frame.copy()
        ref_kp = kp
        ref_des = des
        mode = "matching"
        print(f"Reference set: {len(kp)} keypoints")
    elif key == ord('c'):
        ref_gray = None
        ref_kp = None
        ref_des = None
        mode = "live"
        print("Reference cleared.")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
