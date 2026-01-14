# Bike-Sharing 데이터 기반 Tolerance Interval 실험

이 저장소는 **분포 가정이 없는 Tolerance Interval(TI)** 방법의 성능을
실제 데이터에서 평가하기 위한 코드 모음이다.  
특히 **근사적 조건부 타당성(approximate conditional validity)** 관점에서
서로 다른 TI 구성 방법을 비교한다.

실험은 **UCI Bike Sharing Dataset**을 사용하며,
다음 세 가지 방법을 비교한다.

- **Ours**  
  Hoeffding 부등식을 이용한 tolerance interval  
  (추정된 평균 + 분산, 전역 lambda 사용)
- **GY (Guan–Young)**  
  분산 표준화 spline 기반 tolerance interval
- **Normal**  
  고전적 정규분포 가정(homoscedastic) 기반 interval

각 방법에 대해 **전체 커버리지(marginal)** 와  
**조건부 커버리지(temperature bin별)** 성능을 평가한다.

---

## 1. 저장소 구조

```text
ti_realdata_bikeshare/
├── R/                      # 방법 정의 (함수만 포함)
│   ├── 01_data_bike.R      # Bike 데이터 로딩
│   ├── 10_scores_lambda.R  # 변환 및 Hoeffding lambda
│   ├── 20_intervals_ours.R # 제안 방법 (Ours)
│   ├── 30_intervals_gy.R   # GY 방법
│   ├── 40_intervals_normal.R # 정규분포 기반 TI
│   ├── 50_eval_conditional.R # 조건부 커버리지 계산
│   └── 90_one_run.R        # 단일 실험 실행 wrapper
│
├── scripts/                # 실행용 스크립트
│   ├── 01_download_bike.R  # 데이터 다운로드 및 확인
│   ├── 03_run_bike_manyseeds.R # 다중 seed 실험
│   └── 04_plot_summaries.R # 조건부 커버리지 시각화
│
├── data/
│   └── raw/                # 원본 데이터 (자동 다운로드)
│
├── results/
│   ├── figs/               # 그림 파일
│   └── tables/             # 결과 테이블
│
└── README.md

##2. 구현된 방법 설명
