---
title : "(Week6-2) 지도 학습과 텍스트 분류 : 편향을 잡으며 강세·약세 분류하기 (RF & LightGBM)"
date : 2026-09-29 00:57:10 +0900
categories : [Graduate School, (MGT6033) Analysis of Unstructured Data_'26 Fall]
tags : [MGT6033, NLP, Supervised Learning, Classification, Decision Tree, Naive Bayes, Random Forest, LightGBM, Ensemble]
math : true
---

1편에 이어, 이번 2편은 지도 학습의 나머지 절반인 **분류(Classification)** 입니다.

1편이 "**감성 점수라는 숫자**를 예측하는 회귀"였다면,
2편은 "트윗이 **강세(bullish=1)인가 약세(bearish=0)인가**를 맞히는 분류"입니다.

여기서는 다양한 **분류기(의사결정나무·나이브베이즈·SGD)** 와 **앙상블(Random Forest·LightGBM)**,
그리고 분류에서 가장 중요한 개념인 **편향(bias)과 클래스 불균형** 을 다룹니다.

## <span style="color:#FEB99C;">1. 분류 평가 지표: 정확도만 믿으면 안 된다</span>

분류는 회귀와 평가 방식이 완전히 다릅니다. **혼동 행렬(Confusion Matrix)** 에서 출발합니다.

| | 예측: 음성(0) | 예측: 양성(1) |
|---|---|---|
| **실제: 음성(0)** | TN (참음성) | FP (거짓양성) |
| **실제: 양성(1)** | FN (거짓음성) | TP (참양성) |

여기서 네 가지 핵심 지표가 나옵니다.

| 지표 | 공식 | 의미 |
|---|---|---|
| **정확도(Accuracy)** | (TP+TN)/전체 | 전체 중 맞힌 비율 |
| **정밀도(Precision)** | TP/(TP+FP) | "양성이라 예측한 것" 중 진짜 비율 |
| **재현율(Recall)** | TP/(TP+FN) | "실제 양성" 중 잡아낸 비율 |
| **F1 점수** | 2·(P·R)/(P+R) | 정밀도·재현율의 조화평균 |

> `classification_report`를 읽는 법:
> - **support**: 각 클래스의 실제 관측치 수
> - **macro avg**: 클래스별 지표의 **단순 평균** (소수 클래스도 동등 대우)
> - **weighted avg**: support로 **가중 평균** (다수 클래스에 유리)

### ⚠️ 헷갈리기 쉬운 포인트: "정확한 모델이 편향될 수 있다"

> **오해: "정확도가 99.5%면 좋은 모델 아닌가?"** → **전혀 아닐 수 있습니다.**

**사례** 🏭: 어떤 기계의 **고장을 예측**하는 모델. 고장은 1,000번 중 5번(0.5%) 일어납니다.
모델이 **무조건 "고장 안 남"** 이라고만 답해도 **정확도 99.5%** 입니다.
하지만 이 모델은 **정작 잡아야 할 고장을 하나도 못 잡습니다.** 완전히 쓸모없죠.

> 이것이 **편향(bias)** 입니다. 다수 클래스만 잘 맞히고 소수 클래스를 무시하는 것.
> **정확도(accuracy)만 보면 이 함정을 놓칩니다.** 그래서 **정밀도·재현율·F1** 을 함께 봐야 합니다.

### 🚒 소방관 비유: 정밀도 vs 재현율 트레이드오프

- **재현율 우선(FN 최소화)**: 조금이라도 연기가 나면 무조건 출동 → **실제 화재는 놓치지 않지만**, 헛출동(FP) 급증
- **정밀도 우선(FP 최소화)**: 확실할 때만 출동 → **헛출동은 없지만**, 진짜 화재(FN)를 놓칠 위험

> 무엇이 더 치명적인지는 **문제에 따라 다릅니다.** 화재·암 진단은 **FN(놓침)** 이 치명적이라 재현율을,
> 스팸 필터는 **FP(정상 메일 차단)** 가 성가시니 정밀도를 중시합니다.
> **이 트레이드오프를 조절하는 게 뒤에 나올 `class_weight`와 커스텀 비용함수입니다.**

## <span style="color:#FEB99C;">2. 텍스트 분류기 3종</span>

### 2-1. 로짓/프로빗 (기준선)

로지스틱 회귀(logit)와 프로빗은 **선형 분류의 출발점**입니다.
회귀식을 **확률(0~1)** 로 변환(로짓은 로지스틱 함수, 프로빗은 정규 CDF)하고, 임계값으로 분류합니다.
해석이 쉽지만, 1편의 OLS처럼 **선형·다중공선성·특성선택** 문제를 그대로 안습니다.

### 2-2. 의사결정나무 (Decision Tree)

> **개념**: "스무고개"처럼 특성에 대한 **질문(분기)** 을 반복해 데이터를 나눕니다.
> 각 분기는 **불순도(impurity)** 를 최대한 줄이는 방향으로 선택됩니다.

- **`criterion`**: 분기 품질 척도 — `gini`(지니 불순도), `entropy`, `log_loss`
- **`max_depth`**: 나무 깊이 (깊을수록 과적합 ↑)
- **`min_samples_split`**: 분기하려면 필요한 최소 샘플 수
- **`max_features`**: 각 분기에서 고려할 특성 수
- **`class_weight`**: ⭐ 클래스별 가중치 — **편향 대응의 핵심 무기**

✅ 해석 쉬움(규칙을 눈으로 봄), 비선형 가능 / ❌ **단독으로는 과적합에 매우 취약** → 그래서 앙상블로 발전

### 2-3. 나이브 베이즈 (Naive Bayes)

**베이즈 정리**에 기반한 확률 분류기입니다.

$$P(\text{class} \mid \text{words}) \propto P(\text{class}) \prod_j P(\text{word}_j \mid \text{class})$$

> **"나이브(순진한)"의 의미**: 모든 특성(단어)이 **서로 조건부 독립**이라고 **순진하게 가정**합니다.
> 실제 언어에서 단어는 당연히 서로 얽혀 있지만(1편의 다중공선성!), 이 가정 덕분에 계산이 매우 빠릅니다.

- **닫힌 형태 해**가 있어 훈련이 빠름 (튜닝할 게 거의 없음)
- **`var_smoothing`**: 유일한 주요 하이퍼파라미터 (0 확률 방지용 평활화)
- 텍스트 분류(스팸 필터 등)의 **고전적 기준선**

### 2-4. SGD (확률적 경사하강법)

> 🙋 **헷갈렸던 부분: "SGD는 모델인가?"**
> → **아닙니다. SGD는 모델이 아니라 "최적화 방법"입니다.**

SGD(Stochastic Gradient Descent)는 **손실함수를 무엇으로 두느냐에 따라 다른 모델**이 됩니다.
scikit-learn의 `SGDClassifier`는 `loss` 인자로 모델을 바꿉니다.

| `loss` | 실제 모델 |
|---|---|
| `hinge` | 선형 SVM |
| `log_loss` | 로지스틱 회귀 |
| `modified_huber` | 이상치에 강한 변형 |
| `squared_hinge` | 제곱 힌지 SVM |
| `perceptron` | 퍼셉트론 |

- **`penalty`**: `l1`, `l2`, **`elasticnet`**(L1+L2 혼합) — 정규화 방식
- **`alpha`**, **`l1_ratio`**: 정규화 강도와 L1/L2 비율
- 대용량 데이터에 효율적 (한 번에 하나씩 관측치로 경사 갱신)

## <span style="color:#FEB99C;">3. 앙상블(Ensemble): 약한 학습기를 모으다</span>

> **핵심 아이디어**: 성능이 그저 그런 **약한 학습기(weak learner)** 여러 개를 모으면,
> 하나의 강한 모델보다 나은 성능을 낼 수 있습니다. "집단 지성"이죠.

앙상블은 세 방식으로 나뉩니다.

| 방식 | 원리 | 대표 |
|---|---|---|
| **배깅(Bagging)** | 데이터를 **부트스트랩 샘플**로 나눠 **병렬** 학습 후 투표 | Random Forest |
| **부스팅(Boosting)** | 앞 모델의 **오차를 다음 모델이 보완**하며 **순차** 학습 | XGBoost, LightGBM, GBT |
| **스태킹(Stacking)** | 여러 모델의 예측을 **또 다른 모델(메타)** 이 종합 | — |

### 3-1. Random Forest (배깅)

> **의사결정나무 여러 개의 숲**. 각 나무를 **부트스트랩 샘플 + 무작위 특성 부분집합**으로 학습해 투표합니다.
> 나무들이 서로 달라지므로(decorrelate), 단일 나무의 과적합이 상쇄됩니다.

- 의사결정나무의 하이퍼파라미터 대부분을 공유 (`criterion`, `max_depth`, `max_features`...)
- **새 하이퍼파라미터**:
  - **`n_estimators`**: 나무 개수(숲의 크기), 기본 100
  - **`max_samples`**: 각 나무 학습에 쓸 관측치 비율 (`bootstrap=True` 필요)

### 3-2. XGBoost vs LightGBM (부스팅)

둘 다 **경사 부스팅 나무(GBT)** 계열의 최강자이지만, **나무를 키우는 방향**이 다릅니다.

| | **XGBoost** | **LightGBM** |
|---|---|---|
| 성장 방식 | **수준별(level-wise)** — 층을 균형있게 | **잎별(leaf-wise)** — 손실 큰 잎을 먼저 |
| 속도 | 상대적으로 느림 | **매우 빠름** |
| 특징 | 안정적·널리 쓰임 | 대용량·고차원에 강함 |

> 💡 **왜 LightGBM인가?** NLP 기반 분류 과제(Kaggle 등)에서 **사전학습 모델 없이 밑바닥부터** 시작할 때,
> LightGBM이 다른 모델을 압도하는 경우가 많습니다. scikit-learn 소속은 아니지만 **`LGBMClassifier`** 로 동일한 API를 제공합니다.

## <span style="color:#FEB99C;">4. 실습 데모: 텍스트 분류 (Demo B, StockTwits)</span>

이번엔 같은 StockTwits 데이터로 **강세(1)/약세(0)** 를 예측하는 **분류** 문제입니다.
그런데 이 데이터에는 **결정적 난관**이 있습니다.

### 4-1. 전처리 & 클래스 불균형

토큰화·DTM 과정은 1편(Demo A)과 유사하지만, 레이블을 만드는 부분이 다릅니다.

```python
import pandas as pd
# sentiment 컬럼(Bullish/Bearish)을 원-핫으로 → 한 컬럼을 라벨로
dummies = pd.get_dummies(df['sentiment'])
y = dummies['Bullish'].astype(int)   # 강세=1, 약세=0
```

> ⚠️ **결정적 난관: 약 79%가 강세(1), 21%만 약세(0)** — 심각한 클래스 불균형.
> 1장의 "고장 예측" 사례처럼, 모델이 **무조건 강세**라 답하면 정확도 79%가 그냥 나옵니다.
> **이 데모 전체의 목표는 "약세(소수 클래스)를 얼마나 잘 잡느냐"** 가 됩니다.

### 4-2. 여러 분류기 훈련 & 편향 확인

의사결정나무, 나이브베이즈, SGD(여러 loss), Random Forest, LightGBM을 차례로 훈련하며
**검증 데이터와 훈련 데이터의 classification_report를 함께** 확인했습니다.

발견된 패턴:

- 대부분의 모델이 **약세(0)를 거의 예측하지 못함** → "no predicted label" 경고 발생
- 훈련 데이터 F1은 0.96~0.98인데 검증은 형편없음 → **심각한 과적합**

**Random Forest 실험** (n_estimators × max_samples 격자):

```python
from sklearn.ensemble import RandomForestClassifier as RFC
for n in n_estimators:
    for m in max_samples:
        model = RFC(n_estimators=n, max_samples=m,
                    class_weight='balanced',   # ⭐ 편향 대응 시도
                    random_state=42)
        # class_weight로 균형을 줘도 여전히 강한 편향이 남음
```

> 🔎 **관찰**: `max_samples=1.0`(100%)이면 약세 예측이 **0개**. `0.75`로 줄이니 조금 예측하기 시작.
> 하지만 `0.25`로 더 줄이면 오히려 F1이 18%로 악화. **하이퍼파라미터 효과는 때로 예측 불가**합니다.

**LightGBM** (`reg_alpha=0.01`로 L1 희소성 유도):

```python
from lightgbm import LGBMClassifier
gbm = LGBMClassifier(reg_alpha=0.01, n_jobs=-1)
gbm.fit(X_train, y_train)
# 강세 F1 ≈ 0.87, 약세 F1 ≈ 0.13 — 여전히 심하게 편향
```

### 4-3. 최종 모델 선택: 대규모 튜닝

5개 분류기(의사결정나무·SGD·Random Forest·LightGBM, 나이브베이즈는 튜닝할 게 없어 제외)를 대상으로,
**넓은 하이퍼파라미터 공간 + 5-fold CV + macro-F1 최적화** 로 최고의 모델을 찾습니다.

> ⭐ **왜 macro-F1인가?** macro 평균은 **두 클래스를 동등하게** 다룹니다.
> 다수 클래스(강세)에 휘둘리지 않고 **소수 클래스(약세)도 잘 잡는** 모델을 찾기 위한 선택입니다.

특히 **`class_weight`를 하이퍼파라미터로** 광범위하게 탐색한 게 핵심입니다.

```python
from scipy.stats import uniform
import numpy as np
np.random.seed(42)

# 다수 클래스 가중치를 0~0.5 사이에서 100개 샘플링
majority = uniform(0, 0.5).rvs(100)
class_weights = [{0: 1 - w, 1: w} for w in majority]  # 다양한 가중 조합
class_weights += ['balanced', None]                    # 균형·기본값도 후보에

param_dist = {
    'class_weight': class_weights,
    'random_state': [42],   # 고정(튜닝 안 함)이지만 키워드로 넘겨야 인식됨
    # ... 모델별 파라미터(criterion, max_depth, n_estimators, loss 등)
}
```

> 📝 **팁**: `random_state`를 튜닝하진 않지만, scikit-learn이 이를 훈련 인자로 인식하게 하려면 **리스트로라도 grid에 넣어야** 합니다.

### 4-4. 커스텀 비용함수: F1을 넘어서

문제의 성격에 따라, 오분류의 **비용이 서로 다를 수 있습니다.**
(예: 약세를 강세로 잘못 봐서 물리는 손실 > 그 반대) 이럴 땐 **직접 점수 함수를 정의**합니다.

```python
from sklearn.metrics import confusion_matrix, make_scorer

def custom_score(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    # FP는 100배, FN은 10배 페널티, TP는 보상(-1)
    return cm[0, 1] * 100 + cm[1, 0] * 10 + cm[1, 1] * (-1)

# 비용이므로 낮을수록 좋음 → greater_is_better=False
scorer = make_scorer(custom_score, greater_is_better=False)
```

> ⭐ **`make_scorer(greater_is_better=False)`**: 이 함수가 **"낮을수록 좋은 비용"** 임을 scikit-learn에 알립니다.
> 그러면 내부적으로 부호를 뒤집어 최소화 문제로 다뤄, 다른 지표와 일관되게 최적화합니다.

이렇게 하면 F1처럼 정해진 지표가 아니라, **우리 문제의 실제 손익 구조**에 맞춰 모델을 고를 수 있습니다.

### 4-5. RandomizedSearchCV로 최종 탐색

```python
from sklearn.model_selection import RandomizedSearchCV
rs = RandomizedSearchCV(
    estimator, param_dist,
    n_iter=...,          # 무작위로 N개 조합만 시도 (Grid보다 효율적)
    scoring=scorer,      # 위의 커스텀 비용함수
    cv=5, n_jobs=-1, random_state=42
)
rs.fit(X_train, y_train)
best = rs.best_estimator_   # 최고 모델 저장
```

> 💡 **GridSearch vs RandomizedSearch**:
> - **Grid**: 모든 조합을 전수 탐색 → 정확하지만 조합 폭발
> - **Randomized**: 무작위로 N개만 → 넓은 공간을 **효율적으로** 탐색. 연속형(uniform) 분포와 궁합이 좋음

### 4-6. 최종 결론

> 아무리 튜닝해도 **약세(bearish) 트윗의 분류 성능은 크게 개선되지 않았습니다.**

이는 두 가지로 해석됩니다.

1. 데이터의 **심각한 불균형**(79:21)이 근본 원인
2. 어쩌면 **약세 투자자들이 일관된 언어를 쓰지 않아서** 학습할 패턴 자체가 약할 수도 있음

> 이것이 실무의 현실입니다. **모든 문제가 좋은 모델로 풀리는 것은 아닙니다.**
> 하지만 우리는 **정확도의 함정을 피하고(편향 인식), 올바른 지표(macro-F1·커스텀 비용)로 평가하며,
> 체계적으로 튜닝하는 방법**을 배웠습니다 — 이것이 진짜 자산입니다.

## <span style="color:#FEB99C;">5. 최종 정리</span>

Week 6 지도 학습을 두 편에 걸쳐 정리하면:

**1편 (회귀)**
- Y가 연속형 → 회귀 / 과적합은 홀드아웃·CV로 다스리고 **out-of-sample로 평가**
- OLS는 NLP에 부적합 → **Lasso**(L1·해석 가능) / **SVR**(커널·비선형)

**2편 (분류)**
- Y가 이산형 → 분류 / **정확도만 믿으면 안 됨** (편향의 함정)
- **혼동 행렬 → 정밀도·재현율·F1·macro avg** 로 소수 클래스까지 평가
- **분류기**: 로짓 → 의사결정나무(`class_weight`) → 나이브베이즈(조건부 독립 가정) → SGD(loss로 모델이 바뀜)
- **앙상블**: 배깅(Random Forest) / 부스팅(XGBoost·**LightGBM**, level-wise vs leaf-wise)
- **클래스 불균형** 대응: `class_weight`, macro-F1, **커스텀 비용함수 + `make_scorer`**
- **현실 인정**: 잘 안 되는 문제도 있다. 중요한 건 **올바른 진단과 평가 방법**

> 🎯 Week 6 한 줄 요약:
> **"회귀든 분류든, 지도 학습의 승부처는 화려한 모델이 아니라 — 과적합·편향을 제대로 진단하고, 문제에 맞는 지표로 정직하게 평가하는 것이다."**