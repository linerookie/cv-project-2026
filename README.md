# Computer Vision Projects 2026
> CVPR / ICCV / ECCV / ICLR 탑티어 학회 논문 기반 실시간 OpenCV 프로젝트

## 프로젝트 목록

| # | 파일 | 논문 | 학회 | 기술 |
|---|------|------|------|------|
| 1 | `1_hog_pedestrian_cvpr2005.py` | HOG + SVM | CVPR 2005 | 보행자 검출 |
| 2 | `2_orb_matching_iccv2011.py` | ORB | ICCV 2011 | 특징점 매칭 |
| 3 | `3_optical_flow_eccv2020.py` | RAFT | ECCV 2020 | 광학 흐름 |
| 4 | `4_yolo12_arxiv2025.py` | YOLOv12 | arXiv 2025 | 객체 검출 비교 |
| 5 | `5_pose_estimation_2025.py` | RTMW | CVPR 2025 | 인체 자세 추정 |
| 6 | `6_depth_pro_iclr2025.py` | Depth Pro | ICLR 2025 | 단안 깊이 추정 |
| 7 | `7_yolo_world_open_vocab_2025.py` | YOLO-World | CVPR 2024/2025 | 오픈 어휘 검출 |
| 8 | `8_multi_object_tracking_2025.py` | BoT-SORT / ByteTrack | ECCV 2022 | 다중 객체 추적 |
| 9 | `9_instance_segmentation_2025.py` | EfficientSAM | CVPR 2024 | 인스턴스 분할 |
| 10 | `10_crowd_density_cvpr2025.py` | CrowdSAM / CSRNet | ECCV 2024 / CVPR 2018 | 군중 밀도 추정 |
| 11 | `11_store_traffic_analysis.py` | BoT-SORT + CSRNet 응용 | — | **매장 동선 분석** |

## 설치

```bash
pip install ultralytics opencv-python torch
```

> 모델 가중치(`.pt`, `.onnx`)는 첫 실행 시 ultralytics가 자동 다운로드합니다.

## 실행

```bash
python 1_hog_pedestrian_cvpr2005.py
# 원하는 번호의 파일 실행
```

## 응용 프로젝트 (11번)

`11_store_traffic_analysis.py` — 매장/공간 동선 분석 시스템

- 마우스 드래그로 구역(Zone) 설정 (최대 6개)
- 구역별 체류 인원 실시간 집계 + 과밀 경보
- 누적 히트맵으로 "어디에 오래 머물렀는지" 시각화
- `s` 키로 통계 CSV 저장

## 논문 참고

- [CrowdSAM](https://arxiv.org/abs/2408.01454) · [CSRNet](https://arxiv.org/abs/1802.10062)
- [BoT-SORT](https://arxiv.org/abs/2206.14651) · [ByteTrack](https://arxiv.org/abs/2110.06864)
- [Depth Pro](https://arxiv.org/abs/2410.02073) · [YOLO-World](https://arxiv.org/abs/2401.17270)
- [EfficientSAM](https://arxiv.org/abs/2312.00863) · [YOLOv12](https://arxiv.org/abs/2502.12524)
