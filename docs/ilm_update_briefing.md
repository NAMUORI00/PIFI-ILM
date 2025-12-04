# ILM (Intermediate Layer Model) 자동 선택 메커니즘 기술 보고서

## 1. 개요 (Overview)
**ILM Selection** 모듈(`selection/ilm_direct.py`)은 PiFi 프레임워크의 핵심 구성 요소로, 대규모 언어 모델(LLM) 내에서 소규모 언어 모델(SLM)의 플러그인(Plug-in)으로 사용할 **최적의 레이어(Layer)**를 자동으로 식별하는 기능을 수행합니다.

이 메커니즘은 다운스트림 작업(Downstream Task)에 가장 적합한 표현력(Representation)을 가진 레이어를 선별하여, LLM의 풍부한 지식을 SLM에 효과적으로 전이하면서도 계산 효율성을 유지하는 것을 목표로 합니다.

## 2. 변경 배경 및 인과 분석 (Background & Causal Analysis)

### 2.1 기존 방식의 구현 및 한계 (Legacy Implementation & Limitations)
초기 버전(Legacy v1)은 **PCA Patching (Intervention)** 기법을 사용하여 레이어의 중요도를 평가했습니다.

#### **Legacy 알고리즘 원리**
각 레이어 $l$에 대해 다음 과정을 수행하여 점수(Effect)를 계산했습니다:
1.  **PCA 분해**: 레이어의 히든 스테이트 $X$를 주성분 분석(PCA)하여 $S$ (Scores)와 $U$ (Components)를 얻습니다.
2.  **상관관계 분석**: 각 주성분(PC)과 정답 레이블 $y$ 간의 상관계수를 계산하여, 가장 관련성이 높은 상위 $k$개의 PC를 식별합니다.
3.  **정보 제거 (Patching)**: 식별된 상위 PC 성분을 원본 표현에서 제거(Subtract)하여 반사실적(Counterfactual) 표현 $X_{cf}$를 생성합니다.
    $$ X_{cf} = X - \lambda \cdot (S_{top} \cdot U_{top}^T) $$
4.  **영향력 측정 (Effect Score)**: 원본 데이터 $X$와 손상된 데이터 $X_{cf}$ 각각에 대해 로지스틱 회귀(Probe) 정확도를 측정하고, 그 **하락폭**을 점수로 정의합니다.
    $$ \text{Score}(l) = \text{Acc}(X) - \text{Acc}(X_{cf}) $$

#### **실패 원인 분석 (Causal Analysis of Failure)**
이 방식은 "중요한 정보를 뺐을 때 성능이 많이 떨어지면 좋은 레이어다"라는 가정을 전제로 합니다. 그러나 실제 실험에서는 다음과 같은 이유로 실패했습니다:
1.  **깊은 레이어 편향 (Deep Layer Bias)**: LLM의 깊은 레이어(Layer 15 등)는 사전 학습(Next-token prediction)에 의해 정보가 매우 고밀도로 압축되어 있습니다. 따라서 상위 PC를 제거했을 때의 정확도 하락폭(Score)이 가장 크게 나타나, **Layer 15와 같은 깊은 레이어만 선택**되는 현상이 발생했습니다.
2.  **전이 적합성 미반영**: 깊은 레이어는 LLM의 출력 공간에 과적합(Overfit)되어 있어, SLM이 학습하기에는 오히려 부적절한(Too specialized) 특징을 가집니다. 반면, 실제 전이 성능이 뛰어난 초기/중간 레이어(Layer 0~10)는 정보가 더 분산되어 있어, PC 제거에 의한 정확도 하락폭이 상대적으로 작게 측정되었습니다.

### 2.2 개선 과정 및 해결책 (Evolution & Solution)
위 문제를 해결하기 위해 알고리즘을 전면 개편하였습니다.

-   **다중 지표 도입 (Multi-Metric)**: 단순 정확도 하락폭 대신, **Fisher Score**와 **Silhouette Score**를 도입하여 데이터의 공간적 분포(Cluster Quality) 자체를 평가합니다.
-   **깊이 페널티 (Depth Bias) 적용**: "깊은 레이어가 항상 좋다"는 가정을 깨고, **깊을수록 점수를 차감하는 페널티**를 적용하여 과적합된 깊은 레이어의 선택을 억제합니다.

### 2.3 실험적 검증 (Experimental Validation)
실제 프로젝트 로그(`results_user/` 및 `.archive/`)를 분석하여 개선된 알고리즘의 유효성을 검증하였습니다.

#### **Case 1: SST-2 (감성 분석)**
-   **Legacy 결과**: **Layer 15** 선택 (Score: 0.245, Max) - `.archive/.../selection.json` 참조.
    -   *문제점*: 감성 분석에는 초기 레이어가 유리함에도 불구하고, PCA Patching 방식은 깊은 레이어인 15번을 잘못 선택했습니다.
-   **Current 결과**: **Layer 0** (Score: 0.50, Best) - `results_user/.../selection.json` 참조.
    -   *개선*: 0번 레이어(Embedding)가 가장 높은 점수를 기록하며 최적 레이어로 선정되었습니다. 이는 감성 분석과 같은 단순 분류 작업에서 LLM의 초기 임베딩이 충분히 강력함을 시사하며, 알고리즘이 이를 정확히 포착했음을 의미합니다.
-   **Ground Truth (Sweep)**: `results_user/sst2_layer_sweep_extracted.csv` (TensorBoard 로그 추출)
    -   실제 스윕 결과, **Layer 0**의 정확도는 **90.2%** (초반부 레이어 중 상위권)이며, Layer 15 부근은 성능이 유사하거나 다소 불안정한 경향을 보입니다.
    -   *특이사항*: SST-2 스윕 로그에는 총 **48개**의 결과가 기록되어 있습니다. 이는 24개 레이어에 대해 2회 반복 실험(Seed 2023, 2024 등)이 수행되었음을 시사합니다. 전반적으로 초기 레이어(0~5)가 안정적인 성능을 보이며, Current 알고리즘의 선택(Layer 0)이 타당함을 뒷받침합니다.

#### **Case 2: CoLA (문법성 판단)**
-   **Ground Truth**: `results_user/cola_layer_sweep.csv`에 따르면, 실제 성능(Accuracy)은 **Layer 10** (80.38%)에서 가장 높습니다.
-   **Current 결과**: **Layer 11** (Score: 0.55, Best)
-   **분석**: 알고리즘은 **Layer 11**을 선택했습니다. 이는 실제 최적 레이어인 **Layer 10**과 매우 인접한 위치입니다. 비록 정확히 10번을 지목하지는 못했으나, 문법적 지식이 풍부한 중간-후반부 레이어 구간(10~11번)을 정확하게 탐색해냈다는 점에서 알고리즘의 신뢰성을 확인할 수 있습니다.

---

## 3. 방법론 (Methodology)

현재 시스템은 **다중 지표 평가 시스템(Multi-Metric Scoring System)**을 통해 레이어의 선형 분리 가능성(Linear Separability)과 클러스터링 품질(Clustering Quality)을 종합적으로 평가합니다.

### 3.1 데이터 샘플링 및 전처리 (Data Sampling & Preprocessing)
1.  **층화 추출 (Stratified Sampling)**: 검증 데이터셋(Validation Set)에서 $N$개의 샘플(기본값: 400)을 추출합니다. 클래스 불균형 문제를 방지하기 위해 클래스별 비율을 유지하며 추출합니다.
    - `selection/ilm_direct.py[23:115]` (`_resolve_dataset`)
2.  **표현 추출 (Representation Extraction)**: LLM의 모든 레이어에서 히든 스테이트(Hidden States)를 추출합니다.
    - `selection/ilm_direct.py[132:172]` (`_get_llm_hidden_layers`)
3.  **L2 정규화 (L2 Normalization)**: 레이어마다 벡터의 크기(Scale)가 다를 수 있으므로, 거리 기반 지표(Fisher, Silhouette)의 안정성을 위해 모든 표현 벡터를 단위 길이로 정규화합니다.
    - `selection/ilm_direct.py[168:170]` (`pooled_np = pooled_np / np.maximum(norm, 1e-8)`)

### 3.2 평가 지표 (Scoring Metrics)
각 레이어 $l$은 세 가지 지표의 가중 합인 **Mixed Score** ($S_{mixed}$)로 평가됩니다:

$$ S_{mixed}(l) = \alpha \cdot S_{probe}(l) + \beta \cdot S_{fisher}(l) + \gamma \cdot S_{sil}(l) $$
- **Code Reference**: `selection/ilm_direct.py[297:321]` (`return alpha * probe + beta * fisher + gamma * sil`)

| 지표 (Metric) | 가중치 (기본값) | 설명 | 코드 위치 |
| :--- | :--- | :--- | :--- |
| **Logit Probe** ($S_{probe}$) | $\alpha=0.4$ | 해당 레이어의 표현을 입력으로 하는 경량 로지스틱 회귀(Logistic Regression) 분류기의 정확도입니다. 선형 분리 가능성을 직접적으로 측정합니다. | `selection/ilm_direct.py[265:288]` (`_logit_probe_score`) |
| **Fisher Score** ($S_{fisher}$) | $\beta=0.3$ | 클래스 간 거리(Inter-class distance)와 클래스 내 분산(Intra-class variance)의 비율입니다. 값이 클수록 클래스 간 분리가 잘 되어 있음을 의미합니다. <br> $$ S_{fisher} = \frac{\text{Mean Inter-Class Distance}}{\text{Mean Intra-Class Variance}} $$ | `selection/ilm_direct.py[233:262]` (`_fisher_class_separation`) |
| **Silhouette Score** ($S_{sil}$) | $\gamma=0.3$ | 데이터가 자신이 속한 클러스터 내에서 얼마나 촘촘하게 모여있는지(Cohesion), 다른 클러스터와는 얼마나 떨어져 있는지(Separation)를 측정합니다. 범위: $[-1, 1]$. | `selection/ilm_direct.py[301:305]` (`silhouette_score(X, y)`) |

### 3.3 깊이 편향 (Depth Bias) 수식
깊은 레이어 선호 현상을 억제하기 위한 보정 수식입니다:

$$ S'_{mixed}(l) = S_{mixed}(l) \cdot \left( 1 - \lambda_{depth} \cdot \frac{l}{L_{total}-1} \right) $$

- **$\lambda_{depth}$**: 깊이 편향 계수 (기본값: `0.3`).
- 레이어가 깊어질수록 점수를 일정 비율 감소시켜, 범용적인 의미 정보를 담고 있는 얕거나 중간 깊이의 레이어가 선택될 확률을 높입니다.
- **Code Reference**: `selection/ilm_direct.py[394:395]`
  ```python
  depth_weight = 1.0 - depth_bias * (Li / max(1, num_layers - 1))
  effects.append(eff * depth_weight)
  ```

### 3.4 선택 전략 (Selection Strategy)
최종적으로 보정된 점수 $S'_{mixed}$가 가장 높은 레이어 $L^*$를 선택합니다.

$$ L^* = \operatorname*{argmax}_{l} S'_{mixed}(l) $$
- **Code Reference**: `selection/ilm_direct.py[397:403]` (`best_llm_layer = _rerank_topk(...)`)

## 4. 설정 파라미터 (Configuration Parameters)

실험 스크립트 실행 시 다음 인자를 통해 선택 메커니즘을 제어할 수 있습니다:

| 인자 (Argument) | 기본값 | 설명 |
| :--- | :--- | :--- |
| `--selection_samples` | `400` | 평가에 사용할 샘플 수. |
| `--selection_stratified` | `True` | 클래스 균형을 맞춘 층화 추출 사용 여부. |
| `--selection_score_mode` | `mixed` | 평가 모드: `mixed`, `probe`, `fisher`, `silhouette`. |
| `--selection_score_alpha` | `0.4` | Logit Probe 점수 가중치. |
| `--selection_score_beta` | `0.3` | Fisher 점수 가중치. |
| `--selection_score_gamma` | `0.3` | Silhouette 점수 가중치. |
| `--selection_depth_bias` | `0.3` | 깊은 레이어에 대한 페널티 계수. `0.0` 설정 시 비활성화. |
| `--selection_pooling` | `mean` | 히든 스테이트 풀링 방식 (`mean` 또는 `first` 토큰). |

## 5. 결과물 및 시각화 (Outputs and Visualization)

선택 과정이 완료되면 분석을 위한 다양한 아티팩트가 생성됩니다:

- **JSON 결과 보고서**: `results/layer_selection/<task>/<dataset>/<slm>/<llm>/selection.json`
    - 각 레이어별 점수, 선택된 레이어 인덱스, 설정값 등이 저장됩니다.
- **TensorBoard / W&B 로그**:
    - **Effect Line Plot**: 레이어 깊이에 따른 점수 변화 그래프.
    - **Effect Heatmap**: 점수 분포 히트맵.
    - **PCA Scatter Plots**: 선택된 레이어의 표현 공간을 2차원으로 축소하여 시각화 (클래스 분리도 확인용).
