# PiFi 실험 스크립트 매뉴얼

## 개요

PiFi(Plug-in Fine-tuning)는 SLM(Small Language Model)에 LLM(Large Language Model)의 특정 레이어를 플러그인하여 성능을 향상시키는 방법입니다.

이 문서는 통합 실험 스크립트 사용법을 설명합니다.

---

## 빠른 시작

```bash
# SST-2로 PiFi+ILM 실험
MODE=pifi_ilm DATASETS="sst2" ./scripts/run_classification.sh

# MNLI로 Base 모델 실험
MODE=base DATASETS="mnli" ./scripts/run_entailment.sh
```

---

## 실험 모드

| 모드 | 설명 |
|------|------|
| `base` | SLM만 사용 (LLM 플러그인 없음) |
| `pifi` | PiFi with 지정 레이어 (기본: 마지막 레이어) |
| `pifi_ilm` | PiFi with ILM 자동 레이어 선택 |

---

## 환경 변수

### 공통 설정

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MODE` | `pifi_ilm` | 실험 모드 (`base`, `pifi`, `pifi_ilm`) |
| `MODEL` | `bert` | SLM 모델 종류 |
| `DATASETS` | `sst2 imdb cola` (분류) / `mnli snli` (함의) | 공백으로 구분된 데이터셋 목록 |
| `EPOCHS` | `3` | 학습 에폭 수 |
| `BS` | `32` | 배치 크기 |
| `WORKERS` | `2` | DataLoader CPU 워커 수 |
| `SEED` | `2023` | 랜덤 시드 (재현성용) |

### PiFi 설정 (pifi, pifi_ilm 모드)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `LLM` | `llama3.1` | LLM 모델 종류 |
| `LAYER` | `-1` | LLM 레이어 번호 (-1: 마지막 레이어 또는 자동 선택) |

### ILM 선택 설정 (pifi_ilm 모드 전용)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `SEL_SAMPLES` | `400` | 레이어 선택에 사용할 샘플 수 |
| `SEL_PCS` | `16` | PCA 주성분 개수 |
| `SEL_TOP_PC` | `5` | 레이블 상관관계 상위 PC 개수 |

### 환경/경로 설정

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `DOCKER` | `false` | Docker 환경 여부 (`true`/`false`) |
| `USE_WANDB` | `true` | W&B 로깅 활성화 |
| `CUDA_VISIBLE_DEVICES` | (자동 선택) | 사용할 GPU 인덱스 |
| `CACHE_PATH` | `./cache` | HuggingFace 모델 캐시 경로 |
| `PREPROCESS_PATH` | `./preprocessed` | 전처리된 데이터 저장 경로 |
| `MODEL_PATH` | `./models` | 최종 모델 저장 경로 |
| `CHECKPOINT_PATH` | `./checkpoints` | 체크포인트 저장 경로 |
| `RESULT_PATH` | `./results` | 결과 저장 경로 |

---

## 사용 예시

### 1. SST-2로 Base 모델 학습

```bash
MODE=base DATASETS="sst2" ./scripts/run_classification.sh
```

### 2. BERT + Llama3.1 PiFi with ILM 선택

```bash
MODE=pifi_ilm MODEL=bert LLM=llama3.1 DATASETS="sst2 imdb" ./scripts/run_classification.sh
```

### 3. 특정 레이어로 PiFi 학습

```bash
MODE=pifi LLM=qwen2_0.5b LAYER=15 DATASETS="cola" ./scripts/run_classification.sh
```

### 4. 여러 데이터셋 순차 실행

```bash
MODE=pifi_ilm DATASETS="sst2 imdb cola trec" ./scripts/run_classification.sh
```

### 5. Docker 환경에서 실행

```bash
DOCKER=true MODE=pifi_ilm ./scripts/run_classification.sh
```

### 6. MNLI + RoBERTa + Qwen2 실험

```bash
MODE=pifi_ilm MODEL=roberta LLM=qwen2_1.5b DATASETS="mnli" ./scripts/run_entailment.sh
```

### 7. 커스텀 경로 지정

```bash
CACHE_PATH=/data/hf_cache MODEL_PATH=/data/models ./scripts/run_classification.sh
```

---

## 지원 모델

### SLM (--model_type / MODEL)

| 모델 | 설명 |
|------|------|
| `bert` | BERT base uncased |
| `bert_large` | BERT large uncased |
| `modern_bert` | ModernBERT base |
| `roberta` | RoBERTa base |
| `roberta-large` | RoBERTa large |
| `electra` | ELECTRA base discriminator |
| `albert` | ALBERT base v2 |
| `deberta` | DeBERTa base |
| `debertav3` | DeBERTa v3 base |
| `mbert` | Multilingual BERT |
| `kcbert` | Korean BERT |

### LLM (--llm / LLM)

| 모델 | 파라미터 | 설명 |
|------|----------|------|
| `llama3.1` | 8B | Meta Llama 3.1 8B |
| `llama3` | 8B | Meta Llama 3 8B |
| `llama2` | 7B | Meta Llama 2 7B |
| `qwen2_0.5b` | 0.5B | Qwen2 0.5B (경량) |
| `qwen2_1.5b` | 1.5B | Qwen2 1.5B |
| `qwen2_7b` | 7B | Qwen2 7B |
| `mistral0.3` | 7B | Mistral 7B v0.3 |
| `gemma2` | 9B | Google Gemma 2 9B |
| `falcon` | 7B | Falcon 7B |

---

## 지원 데이터셋

### Classification (분류)

| 데이터셋 | 설명 | 클래스 수 |
|----------|------|-----------|
| `sst2` | Stanford Sentiment Treebank (영화 리뷰) | 2 |
| `imdb` | IMDB 영화 리뷰 | 2 |
| `cola` | Corpus of Linguistic Acceptability | 2 |
| `trec` | TREC 질문 분류 | 6 |
| `subj` | Subjectivity 분류 | 2 |
| `agnews` | AG News 뉴스 분류 | 4 |
| `mr` | Movie Reviews | 2 |
| `cr` | Customer Reviews | 2 |
| `tweet_sentiment_binary` | 트위터 감정 분석 | 2 |
| `tweet_offensive` | 트위터 공격성 분류 | 2 |

### Entailment (함의)

| 데이터셋 | 설명 | 클래스 수 |
|----------|------|-----------|
| `mnli` | Multi-Genre Natural Language Inference | 3 |
| `snli` | Stanford Natural Language Inference | 3 |

---

## 출력 구조

### 체크포인트 (학습 중)

```
checkpoints/
└── {task}/
    └── {dataset}/
        └── {padding}/
            └── {model_type}/
                └── {method}/
                    └── {llm_model}/
                        └── {layer_num}/
                            └── checkpoint.pt
```

### 최종 모델

```
models/
└── {task}/
    └── {dataset}/
        └── ... (동일 구조)
            └── final_model.pt
```

### ILM 선택 결과

```
results/
└── layer_selection/
    └── {task}/
        └── {dataset}/
            └── {model_type}/
                └── {llm_model}/
                    ├── selection.json    # 선택 결과
                    └── layer_scores.csv  # 레이어별 점수
```

---

## ILM 레이어 선택 상세

ILM(Informative Layer Mining)은 LLM의 여러 레이어 중 태스크에 가장 적합한 레이어를 자동으로 선택합니다.

### 선택 알고리즘

1. **샘플 추출**: 학습 데이터에서 `SEL_SAMPLES`개 샘플 추출 (기본: 400)
2. **Hidden State 추출**: 각 레이어별 hidden state 추출
3. **PCA 적용**: `SEL_PCS`개 주성분으로 차원 축소 (기본: 16)
4. **점수 계산**: 각 레이어에 대해 세 가지 점수 계산
   - Probe Score: 선형 분류기 정확도
   - Fisher Score: 클래스 간 분리도
   - Silhouette Score: 클러스터 품질
5. **최적 레이어 선택**: 가중 평균 점수가 가장 높은 레이어 선택

### 관련 파라미터

```bash
# ILM 선택 상세 설정
python main.py \
  --auto_select_layer true \
  --selection_samples 400 \
  --selection_pcs 16 \
  --selection_top_pc 5 \
  --selection_pooling mean \
  --selection_dtype fp16 \
  --selection_score_mode mixed \
  --selection_score_alpha 0.4 \
  --selection_score_beta 0.3 \
  --selection_score_gamma 0.3 \
  --selection_depth_bias 0.3
```

---

## W&B 로깅

### 자동 로깅 항목

- 학습/검증 Loss, Accuracy, F1 Score
- 에폭별 메트릭 추이
- ILM 레이어 선택 결과 (테이블, 차트)
- 레이어별 PCA 시각화
- 모델 아키텍처

### W&B 비활성화

```bash
USE_WANDB=false ./scripts/run_classification.sh
```

---

## 재현성 (Reproducibility)

스크립트는 다음을 통해 재현성을 보장합니다:

1. **랜덤 시드 설정**: Python, NumPy, PyTorch, CUDA 모두 동일 시드 사용
2. **결정론적 알고리즘**: `cudnn.deterministic=True`, `cudnn.benchmark=False`
3. **DataLoader 워커 초기화**: 각 워커에 고유하고 결정론적인 시드 할당
4. **체크포인트 RNG 상태 저장**: 학습 재개 시 정확한 상태 복원

---

## 문제 해결

### GPU 메모리 부족

```bash
# 배치 크기 줄이기
BS=16 ./scripts/run_classification.sh

# 더 작은 LLM 사용
LLM=qwen2_0.5b ./scripts/run_classification.sh
```

### CUDA 오류

```bash
# 특정 GPU 지정
CUDA_VISIBLE_DEVICES=0 ./scripts/run_classification.sh
```

### W&B 로그인

```bash
wandb login
```

---

## 관련 파일

- `core/arguments.py`: 모든 명령줄 인자 정의
- `core/wandb_manager.py`: W&B 로깅 관리
- `core/checkpoint.py`: 체크포인트 저장/로드
- `selection/ilm_direct.py`: ILM 레이어 선택 알고리즘
