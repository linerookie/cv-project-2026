# 객체 데이터셋 구조

각 객체 폴더명은 `{번호}_{재질}_{객체명}` 형식으로 지정

## 재질 5군 (폴더명에 포함)
- `matte`       — 무광 매트 (도자기, 석고상, 무광 도료)
- `glossy`      — 광택 (유리병, 금속 장식, 광택 도자기)
- `translucent` — 반투명 (유리컵, 아크릴 소품, 반투명 화병)
- `complex`     — 복잡 텍스처 (직물, 나무 소품, 패턴 쿠션)
- `fine`        — 미세 디테일 (피규어, 섬세한 조각품)

## 폴더 예시
```
objects/
  obj01_matte_vase/
    images/           ← 60~90장 원본
    images_masked/    ← rembg 배경 제거 결과 (자동 생성)
    database.db       ← COLMAP 특징점 DB (자동 생성)
    sparse/0/         ← COLMAP SfM 결과 (자동 생성)
    ns_data/          ← nerfstudio transforms.json (자동 생성)
    ns_data_masked/   ← 마스킹 버전 transforms.json (자동 생성)
    outputs/          ← 학습 결과 체크포인트 (자동 생성)
    eval_results.csv  ← 정량 평가 결과 (자동 생성)
  obj02_glossy_bottle/
    images/
    ...
```

## 촬영 체크리스트
- [ ] 배경: 단색 (흰색/검정) 또는 그린스크린 권장
- [ ] 조명: 균일한 확산광 (직사광원 피할 것)
- [ ] 초점: 객체에 맞춤, 흔들림 없이
- [ ] 범위: 수평 360도 + 앙각 30~60도 포함
- [ ] 수량: 최소 60장, 권장 75~90장
- [ ] 해상도: 1920×1080 이상
