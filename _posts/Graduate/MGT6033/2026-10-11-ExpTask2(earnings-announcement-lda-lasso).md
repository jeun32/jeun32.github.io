---
title : "(Experiential Task 2) 실적 발표문의 언어와 주가 반응 : LDA 토픽 모델링·Lasso·분류기 (Earnings Announcements)"
date : 2026-10-11 22:46:10 +0900
categories : [Graduate School, "(MGT6033) Analysis of Unstructured Data_'26 Fall"]
tags : [MGT6033, NLP, LDA, Topic Modeling, Coherence, Lasso, Logistic Regression, Random Forest, Macro-F1, Earnings Announcement]
math : true
---

상장 기업은 1년에 네 번 **실적을 발표(earnings announcement)** 합니다.
월가는 흔히 주당순이익(EPS)이 예상치를 **"충족했는가, 넘었는가(meet or beat)"** 에 주목하지만,
EPS는 발표문 속 숫자 하나에 불과합니다.

경영진은 실적과 함께 **긴 서술형 설명**을 공개합니다.
Experiential Task 2에서는 이 서술을 머신러닝으로 분석했습니다.

- **LDA**: 실적 발표문에는 어떤 종류의 이야기가 담겨 있는가?
- **Lasso**: 어떤 이야기(토픽)가 주가 반응과 관련이 있는가?
- **분류기**: 발표문으로 큰 폭의 주가 상승(3일 +5% 초과)을 예측할 수 있는가?

이번 글도 두 부분으로 나뉩니다.

- **1부 (개념)**: 실적 발표와 초과수익률, 텍스트 정규화, LDA, coherence, Lasso, 불균형 분류
- **2부 (실습)**: 12,500개 발표문으로 토픽을 만들고 주가 반응을 설명·예측

> 🎯 이번 글을 관통하는 질문은 이것입니다.
> **"경영진이 무엇을 이야기했는지가, 시장이 어떻게 반응했는지를 알려 줄까?"**

> 📝 **읽기 안내**
> 수치는 제가 과제를 수행하며 얻은 결과입니다.
> LDA와 모델 튜닝은 실행 환경·라이브러리 버전에 따라 결과가 조금 달라질 수 있습니다.

---

# 📘 1부. 개념

---

## <span style="color:#FEB99C;">1. 실적 발표와 시장 반응</span>

### 1-1. 데이터는 어디서 왔을까?

미국 기업은 실적을 발표할 때 보도자료를 SEC에 **8-K 공시의 첨부 문서(Exhibit 99.1)** 로 제출합니다.
과제의 `ea_text` 열이 `EX-99.1 ... htm PRESS RELEASE ...`로 시작하는 이유입니다.

| 열 | 의미 |
|---|---|
| `ea_text` | 실적 발표 보도자료 원문 |
| `datacqtr` | 회계 분기 (예: 2011Q3) |
| `announce_date` | 발표일 |
| `BHAR0_2` | 발표일(0일)부터 2일 후까지 3일간의 초과수익률 (%p) |

### 1-2. BHAR: 매수 후 보유 초과수익률

**BHAR(Buy-and-Hold Abnormal Return)** 은 "이 주식을 사서 보유했을 때, 비교 기준보다 얼마나 더 벌었는가"입니다.

$$
\text{BHAR}_{0,2}
=
\prod_{t=0}^{2}(1+R_{it})
-
\prod_{t=0}^{2}(1+R_{bt})
$$

- $R_{it}$: 기업 $i$의 일별 수익률
- $R_{bt}$: 시장지수 등 비교 기준의 일별 수익률

> 🙋 **왜 하루가 아니라 3일(0~2일)일까요?**
>
> 발표가 장 마감 후에 나오면 시장 반응은 **다음 날** 나타납니다.
> 또 정보가 가격에 완전히 반영되는 데 하루 이상 걸리기도 합니다.
> 그래서 발표일부터 이틀 뒤까지를 반응 기간(event window)으로 잡습니다.

> ⚠️ **단위 주의**: `BHAR0_2`는 **퍼센트 포인트**입니다. `-4.5`는 −4.5%입니다.
> 그래서 "5% 초과"는 `> 0.05`가 아니라 **`> 5`** 로 써야 합니다.

## <span style="color:#FEB99C;">2. 텍스트 정규화: 숫자를 지우지 않고 "치환"하기</span>

### 2-1. 왜 `financial_number`로 바꿀까?

실적 발표문은 숫자로 가득합니다.

```text
Revenue increased 12.5% to \$1,234.5 million.
```

숫자를 그대로 두면 `12.5`, `1,234.5`가 모두 **서로 다른 희귀 토큰**이 되어 DTM에 쓸모없는 열이 생깁니다.
그렇다고 지워 버리면 **"여기에 숫자가 있었다"는 정보까지** 사라집니다.

> 💡 **해결책: 모든 숫자를 하나의 토큰 `financial_number`로 치환합니다.**
> "매출이 [숫자]로 증가했다"는 구조는 남기고, 구체적인 값의 다양성만 없앱니다.

```python
re.sub(r"\$?\d+(?:[.,]\d+)*%?", " financial_number ", txt)
```

| 부분 | 의미 |
|---|---|
| `\$?` | 달러 기호가 있어도 되고 없어도 됨 |
| `\d+` | 숫자 1개 이상 |
| `(?:[.,]\d+)*` | 소수점·천 단위 쉼표와 뒤의 숫자 반복 |
| `%?` | 퍼센트 기호 선택 |

> 🙋 **숫자 치환을 문장 분리보다 먼저 하는 이유는?**
>
> `12.5`의 마침표 때문에 `sent_tokenize`가 문장을 잘못 자를 수 있습니다.
> 숫자를 먼저 치환하면 이런 오분할을 줄일 수 있습니다.

### 2-2. "나쁜 문장" 걸러내기

보도자료에는 서술 외에도 **재무제표 표, 머리글, 연락처**가 섞여 있습니다.
표를 텍스트로 풀면 숫자만 줄줄이 늘어선 "문장"이 됩니다.

과제의 기준은 두 가지입니다.

| 조건 | 걸러내는 대상 |
|---|---|
| 토큰 **5개 미만** | 머리글, "Thank you." 같은 조각 |
| `financial_number` 비율 **50% 초과** | 표에서 나온 숫자 나열 |

$$
\text{유지 조건}:\quad n \ge 5
\quad \text{and} \quad
\frac{n_{\text{fin}}}{n} \le 0.5
$$

> ⭐ **이 필터의 목적은 "경영진의 서술"만 남기는 것**입니다.
> 토픽 모델이 표의 숫자 패턴이 아니라 말의 내용을 학습하도록 만드는 전처리입니다.

> 🙋 **헷갈리기 쉬운 포인트: `"financial_number".isalpha()`는 `False`입니다.**
>
> 밑줄(`_`)이 알파벳이 아니기 때문입니다.
> "알파벳 토큰만 남긴다"는 조건만 쓰면 방금 만든 `financial_number`가 전부 사라집니다.
> 그래서 `word.isalpha() or word == "financial_number"`처럼 **예외를 명시**해야 합니다.

## <span style="color:#FEB99C;">3. DTM 설계: `max_df`로 "어디에나 있는 단어" 지우기</span>

LDA는 단어의 **등장 횟수**를 모델링하므로 TF-IDF가 아니라 `CountVectorizer`를 씁니다.

| 요구사항 | 설정 |
|---|---|
| 단어 + bigram | `ngram_range=(1, 2)` |
| alpha 3자 이상 + `financial_number` 허용 | `token_pattern=r"\b[a-zA-Z_]{3,}[a-zA-Z]*\b"` |
| 영어 불용어 제외 | `stop_words=stopwords.words("english")` |
| 문서의 50%를 넘게 등장하는 토큰 제외 | `max_df=0.5` |
| 상위 1,000개 | `max_features=1000` |

### 3-1. `max_df`는 어떤 단어를 지울까?

`max_df=0.5`는 **전체 문서의 50% 넘게 등장하는 토큰**을 제거합니다.

실적 발표문에는 `quarter`, `company`, `results` 같은 단어가 거의 모든 문서에 나옵니다.
이런 단어는 문서를 **구분하는 데 도움이 되지 않고**, 토픽마다 상위 단어로 끼어들어 토픽을 흐리게 만듭니다.

> 💡 **불용어 목록은 "언어 공통의" 흔한 단어를, `max_df`는 "이 말뭉치에서" 흔한 단어를 지웁니다.**
> 도메인마다 흔한 단어가 다르기 때문에 두 장치가 함께 필요합니다.

> ⚠️ `financial_number` 자체도 대부분의 문서에 등장한다면 `max_df`에 걸려 **단독 토큰으로는 제거될 수 있습니다.**
> 다만 `financial_number restructuring`처럼 **bigram 안에서는 살아남습니다.**
> 실제로 뒤의 토픽 단어에 `financial_number loss` 같은 bigram이 나타납니다.

### 3-2. `stop_words`에 `set`을 넘기면 안 되는 이유

최신 scikit-learn은 `stop_words`로 **`"english"` 문자열, 리스트, 또는 `None`** 만 받습니다.
`set(stopwords.words("english"))`를 넘기면 파라미터 검증 오류가 납니다.

## <span style="color:#FEB99C;">4. LDA: 문서는 토픽의 혼합, 토픽은 단어의 분포</span>

### 4-1. LDA의 생성 이야기

**잠재 디리클레 할당(Latent Dirichlet Allocation, LDA)** 은 문서가 이렇게 만들어졌다고 가정합니다.

1. 각 토픽 $k$는 단어에 대한 확률분포 $\beta_k$를 가진다
2. 각 문서 $d$는 토픽에 대한 비중 $\theta_d$를 가진다
3. 문서의 단어 하나를 만들 때마다:
   - $\theta_d$에서 토픽 $z$를 하나 뽑고
   - 그 토픽의 분포 $\beta_z$에서 단어를 하나 뽑는다

$$
\theta_d \sim \operatorname{Dirichlet}(\alpha),
\qquad
\beta_k \sim \operatorname{Dirichlet}(\eta)
$$

우리가 관찰하는 것은 **단어뿐**이고, LDA는 거꾸로 $\theta$와 $\beta$를 추정합니다.

> 💡 **한 실적 발표문은 "실적 요약 40% + 구조조정 25% + 배당 15% + …"처럼 여러 토픽의 혼합**으로 표현됩니다.
> 이 비중 벡터 $\theta_d$가 뒤의 회귀·분류에서 **설명변수**가 됩니다.

### 4-2. 두 개의 사전분포(prior)

| sklearn 파라미터 | 기호 | 의미 | 값이 작으면 |
|---|---|---|---|
| `doc_topic_prior` | $\alpha$ | 문서 안의 토픽 비중 분포 | 문서가 **소수의 토픽**에 집중 |
| `topic_word_prior` | $\eta$ | 토픽 안의 단어 분포 | 토픽이 **소수의 단어**에 집중 |

과제 설정은 `topic_word_prior=0.15`, `doc_topic_prior = 25 / K` 입니다.

> 🙋 **왜 $\alpha$를 토픽 수 $K$로 나눌까요?**
>
> 토픽이 많아질수록 각 토픽에 배정되는 사전 비중을 줄여,
> **$\alpha$의 전체 합($K \times \alpha = 25$)을 일정하게** 유지하려는 설계입니다.
> 토픽 수를 바꿔도 "문서가 얼마나 여러 토픽에 퍼지는가"에 대한 사전 가정이 비슷하게 유지됩니다.
> 예를 들어 $K=120$이면 $\alpha \approx 0.21$입니다.

### 4-3. sklearn LDA에서 무엇을 꺼낼까?

| 객체 | 모양 | 의미 |
|---|---|---|
| `lda.components_` | (토픽 수, 단어 수) | 토픽별 단어 가중치 → **토픽의 상위 단어** |
| `lda.transform(dtm)` | (문서 수, 토픽 수) | 문서별 토픽 비중 → **모델의 입력 특성** |

> ⭐ `transform`의 각 행은 **합이 1**입니다. 4-2의 Lasso 해석에서 중요한 성질입니다.

## <span style="color:#FEB99C;">5. 토픽 수는 어떻게 정할까? Coherence와 Word Intrusion</span>

LDA는 토픽 수 $K$를 **사람이 정해야** 합니다. 그 근거로 두 가지를 썼습니다.

### 5-1. u_mass coherence

u_mass는 **한 토픽의 상위 단어들이 실제 문서에서 함께 등장하는가**를 측정합니다.

$$
C_{\text{UMass}}
=
\frac{2}{N(N-1)}
\sum_{i=2}^{N}\sum_{j=1}^{i-1}
\log\frac{D(w_i,w_j)+1}{D(w_j)}
$$

- $D(w_j)$: 단어 $w_j$가 등장한 문서 수
- $D(w_i,w_j)$: 두 단어가 함께 등장한 문서 수
- $N$: 평가에 쓰는 토픽 상위 단어 수

> 💡 **값은 보통 음수이고, 0에 가까울수록(클수록) 좋습니다.**
> 상위 단어들이 늘 함께 나온다면 비율이 1에 가까워져 로그값이 0에 가까워집니다.

> ⚠️ **coherence가 가장 높은 $K$가 언제나 "정답"은 아닙니다.**
> 값 차이가 작다면 해석 가능성, 토픽 중복, 연구 목적도 함께 고려해야 합니다.

### 5-2. Word Intrusion: 사람이 토픽을 검증하는 방법

토픽 상위 단어 5개에 **무작위 단어(침입자, intruder) 하나**를 섞어 보여 줍니다.

```text
Topic 116: net loss, loss income, loss per, financial_number loss, operating loss | Intruder: ???
```

사람이 침입자를 쉽게 찾아낸다면, 나머지 5개 단어가 **하나의 의미로 잘 묶여 있다**는 뜻입니다.

> ⭐ **coherence는 통계적 검증, word intrusion은 사람의 해석 검증**입니다.
> 토픽 모델은 해석이 목적인 경우가 많아 둘을 함께 보는 것이 좋습니다.

## <span style="color:#FEB99C;">6. Lasso: 120개 토픽 중 정말 중요한 것만 남기기</span>

### 6-1. L1 벌점이 계수를 0으로 만든다

$$
\min_{w}\;
\frac{1}{2n}\lVert y - Xw \rVert_2^2
+
\alpha \lVert w \rVert_1
$$

- 첫 항: 예측 오차 (일반 회귀와 동일)
- 둘째 항: 계수 절댓값의 합에 대한 벌점

**L1 벌점은 쓸모없는 계수를 정확히 0으로** 만듭니다.
그래서 Lasso는 예측과 동시에 **변수 선택(feature selection)** 을 합니다.

| 모델 | 벌점 | 계수 |
|---|---|---|
| OLS | 없음 | 모두 0이 아님 |
| Ridge | L2 ($\sum w^2$) | 작아지지만 0은 드묾 |
| **Lasso** | L1 ($\sum \lvert w\rvert$) | **상당수가 정확히 0** |

> 💡 **`alpha`가 클수록 0이 되는 계수가 많아집니다.**
> 너무 작으면 잡음까지 학습하고, 너무 크면 중요한 토픽까지 사라집니다.
> 그래서 교차검증으로 `alpha`를 고릅니다.

### 6-2. RandomizedSearchCV

| 설정 | 값 |
|---|---|
| 후보 `alpha` | 0.00001 ~ 0.02 사이 무작위 100개 |
| 탐색 횟수 | 50번 (`n_iter=50`) |
| 교차검증 | 5-fold |
| 평가 지표 | `neg_root_mean_squared_error` |

> 🙋 **왜 RMSE 앞에 `neg`가 붙을까요?**
>
> scikit-learn의 탐색 도구는 **점수가 클수록 좋다**고 가정합니다.
> RMSE는 작을수록 좋으므로 음수로 바꿔 "클수록 좋은" 형태로 맞춘 것입니다.
> 보고할 때는 다시 `-`를 붙여 양수 RMSE로 해석합니다.

### 6-3. 토픽 비중 계수는 어떻게 읽을까?

토픽 비중은 0~1 사이 값이고, 계수 $w_k$는 이렇게 해석합니다.

> 토픽 $k$의 비중이 **10%p 늘면**, 수익률이 약 $0.1 \times w_k$ %p 변하는 것과 관련이 있다.

> ⚠️ **토픽 비중은 합이 1이라는 점을 기억해야 합니다.**
> 한 토픽의 비중이 늘면 **다른 토픽의 비중은 반드시 줄어듭니다.**
> 그래서 계수는 "다른 조건이 같을 때의 효과"라기보다 **"다른 토픽 대신 이 토픽을 이야기할 때"의 상대적 연관**으로 읽는 것이 정확합니다.
> 절편이 있는 회귀에서는 토픽 비중 전체가 완전한 공선성을 갖기 때문에, 어느 토픽을 0으로 둘지 Lasso의 선택에 영향을 줍니다.

## <span style="color:#FEB99C;">7. 불균형 분류: 정확도 대신 Macro-F1</span>

마지막 과제는 **3일 수익률이 +5%를 넘는가(1) 아닌가(0)** 를 예측하는 이진 분류입니다.
+5% 초과는 **소수 클래스**입니다.

### 7-1. 왜 정확도가 아니라 Macro-F1일까?

모든 문서를 0으로 예측해도 정확도는 높게 나옵니다. 하지만 이 모델은 쓸모가 없습니다.

$$
F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}},
\qquad
\text{Macro-F1} = \frac{F1_{0} + F1_{1}}{2}
$$

> ⭐ **Macro-F1은 클래스 크기와 관계없이 두 클래스의 F1을 똑같이 평균**합니다.
> 소수 클래스를 못 맞히면 점수가 크게 떨어지므로, 불균형 문제의 평가에 적합합니다.

### 7-2. 불균형을 다루는 장치

| 장치 | 원리 |
|---|---|
| **`class_weight="balanced"`** | 소수 클래스의 오분류에 더 큰 손실 가중치 |
| **`scoring="f1_macro"`** | 튜닝 기준을 채점 지표와 일치 |
| **`StratifiedKFold`** | 각 fold의 클래스 비율을 전체와 같게 유지 |
| **임계값 조정** | 0.5 대신 macro-F1이 최대가 되는 확률 기준 사용 |

### 7-3. 두 가지 모델, 두 가지 입력

| 모델 | 입력 | 튜닝한 하이퍼파라미터 |
|---|---|---|
| **TF-IDF + 로지스틱 회귀** | DTM (1,000개 단어) | `use_idf`, `sublinear_tf`, `penalty`(L1/L2), `C` |
| **랜덤 포레스트** | 토픽 비중 (120개) | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` |

> 🙋 **로지스틱 회귀의 `C`는 무엇인가요?**
>
> **규제 강도의 역수**입니다. Lasso의 `alpha`와 방향이 반대입니다.
> `C`가 작을수록 규제가 강하고, 클수록 데이터에 더 자유롭게 맞춥니다.

> 🙋 **TF-IDF는 왜 Pipeline 안에 넣었을까요?**
>
> IDF는 "몇 개 문서에 등장했는가"를 데이터에서 계산합니다.
> Pipeline 안에 두면 교차검증의 **각 fold마다 훈련 부분에서만 IDF를 계산**하므로 검증 fold의 정보가 새지 않습니다.

### 7-4. 최종 모델은 무엇으로 고를까?

> ⚠️ **검증 데이터 점수를 보고 모델을 고르면 안 됩니다.**
> 검증 데이터로 고른 모델의 검증 점수는 **실제보다 좋아 보이는 쪽으로 편향**됩니다.
> 모델 선택은 **훈련 데이터의 교차검증 점수**로 하고, 검증 데이터는 최종 보고에만 씁니다.

---

# 🧪 2부. 실습 — 12,500개 실적 발표문 분석

> 1부의 개념을 실제 데이터에 적용합니다.
> **전처리 → DTM → LDA → Lasso → 분류기** 순서로 진행합니다.

---

## <span style="color:#FEB99C;">8. 전처리와 DTM</span>

```python
import re
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

df = pd.read_csv("ExpTask2Data.csv.gz")
assert len(df) == 12500

def clean_sentences(txt):
    clean_txt = re.sub(r"\$?\d+(?:[.,]\d+)*%?", " financial_number ", txt)
    good_sents = []
    for sent in sent_tokenize(clean_txt):
        good_tokens = [w for w in word_tokenize(sent)
                       if w.isalpha() or w == "financial_number"]
        n = len(good_tokens)
        if n >= 5 and good_tokens.count("financial_number") / n <= 0.5:
            good_sents.append(" ".join(good_tokens))
    return " ".join(good_sents)

df["clean_sent"] = df["ea_text"].fillna("").astype(str).apply(clean_sentences)
```

간단한 예시로 동작을 확인할 수 있습니다.

```text
입력: Revenue increased 12.5% to \$1,234.5 million. EPS was \$0.45. Thank you.
출력: Revenue increased financial_number to financial_number million
```

- 첫 문장: 6토큰 중 숫자 2개(33%) → **유지**
- `EPS was financial_number`: 3토큰 → **제거**
- `Thank you`: 2토큰 → **제거**

> 📝 12,500개 문서 전처리는 15~20분이 걸립니다.
> 결과를 **CSV로 저장해 두면** 이후 커널을 재시작해도 다시 계산하지 않아도 됩니다.

```python
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer(
    token_pattern=r"\b[a-zA-Z_]{3,}[a-zA-Z]*\b",
    ngram_range=(1, 2),
    stop_words=stopwords.words("english"),
    max_df=0.5,
    max_features=1000
)
dtm = vec.fit_transform(df["clean_sent"])   # (12500, 1000)
```

## <span style="color:#FEB99C;">9. LDA 튜닝: 토픽 40~150개 비교</span>

시간을 줄이기 위해 **5,000개 표본**으로 토픽 수를 비교했습니다.

```python
from sklearn.decomposition import LatentDirichletAllocation as LDA
from tmtoolkit.topicmod.evaluate import metric_coherence_gensim
from scipy.sparse import csr_matrix

sample = pd.DataFrame(dtm.todense()).sample(5000, random_state=123)
vocab = vec.get_feature_names_out()

records = []
for K in range(40, 151, 10):
    lda = LDA(n_components=K,
              topic_word_prior=0.15,
              doc_topic_prior=25 / K,
              n_jobs=-1,
              random_state=123)
    lda.fit(sample)

    umass = metric_coherence_gensim(measure="u_mass",
                                    top_n=20,
                                    topic_word_distrib=lda.components_,
                                    dtm=csr_matrix(sample.values),
                                    vocab=np.array(vocab))
    records.append({"topics": K, "mean_umass": np.mean(umass)})
```

> 🙋 **`range(40, 151, 10)`에서 왜 151일까요?**
>
> `range`는 끝값을 포함하지 않습니다. 150까지 포함하려면 끝값을 151(또는 160)로 둬야 합니다.

### 9-1. 결과

<div style="display: flex; justify-content: center; gap: 2em; margin: 2em 0;">
  
  <div style="text-align: center;">
    <img 
      src="/assets/img/posts/2026-10-11-ExpTask2(earnings-announcement-lda-lasso)/umass_plot.jpg"
      alt="umass_plot"
      style="width:100%; max-width:600px; display:block;"
    >
    <div style="font-size: 0.9em; color: gray; margin-top: 0.4em;">
      <em>토픽 수별 평균 u_mass coherence (120개에서 0에 가장 가까움)</em>
    </div>
  </div>

</div>

| 토픽 수 | 평균 u_mass |
|---:|---:|
| 40 | −1.0298 |
| 50 | −1.0167 |
| 60 | −1.0160 |
| 70 | −1.0107 |
| 80 | −1.0143 |
| 90 | −1.0086 |
| 100 | −1.0089 |
| 110 | −1.0105 |
| **120** | **−1.0048** |
| 130 | −1.0231 |
| 140 | −1.0250 |
| 150 | −1.0092 |

- 토픽 수가 늘수록 대체로 좋아지다가 **120개에서 최고**(0에 가장 가까움)
- 130·140에서 떨어지고, 150은 120과 비슷하지만 조금 낮음
- 전체 차이는 약 −1.03 ~ −1.00으로 **크지 않음**

> 💡 차이가 작으므로 "120이 압도적으로 좋다"기보다는
> **"검토한 범위 안에서 120이 가장 좋고, 그 이상 늘려도 개선되지 않는다"** 정도로 읽는 것이 적절합니다.

### 9-2. 최종 모델

```python
final_lda = LDA(n_components=120,
                topic_word_prior=0.15,
                doc_topic_prior=25 / 120,
                n_jobs=-1,
                random_state=123)
topics = final_lda.fit_transform(dtm)   # (12500, 120)
```

최종 모델은 표본이 아닌 **12,500개 전체**로 학습합니다.
뒤의 회귀·분류에서 모든 문서의 토픽 비중이 필요하기 때문입니다.

## <span style="color:#FEB99C;">10. 토픽 단어와 침입자</span>

```python
import random
random.seed(123)

for i, topic in enumerate(final_lda.components_):
    words = [vocab[j] for j in topic.argsort()[::-1][:5]]
    intruder = random.choice([w for w in vocab if w not in words])
    print(f"Topic {i}: {', '.join(words)} | Intruder: {intruder}")
```

> 🙋 **침입자 후보에서 상위 5개 단어를 빼는 이유는?**
>
> 무작위로 뽑은 단어가 우연히 상위 단어와 겹치면 침입자 역할을 하지 못하기 때문입니다.

`argsort()[::-1][:5]`는 가중치를 **오름차순으로 정렬한 인덱스를 뒤집어** 큰 순서대로 5개를 고르는 관용구입니다.

## <span style="color:#FEB99C;">11. Lasso: 어떤 토픽이 주가 반응과 관련 있을까?</span>

```python
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import Lasso

X_tr, X_te, y_tr, y_te = train_test_split(
    topics, df["BHAR0_2"], test_size=0.2, random_state=123
)

np.random.seed(123)
param_grid = {"random_state": [123],
              "alpha": np.random.uniform(0.00001, 0.02, 100)}

rus = RandomizedSearchCV(Lasso(random_state=123), param_grid,
                         n_iter=50, cv=5,
                         scoring="neg_root_mean_squared_error",
                         random_state=123, n_jobs=-1)
rus.fit(X_tr, y_tr)

coefs = rus.best_estimator_.coef_
```

> 💡 `rus.best_estimator_`는 최적 `alpha`로 **훈련 데이터 전체를 다시 학습한 모델**입니다(`refit=True`가 기본값).

### 11-1. 결과

<div style="display: flex; justify-content: center; gap: 2em; margin: 2em 0;">
  
  <div style="text-align: center;">
    <img 
      src="/assets/img/posts/2026-10-11-ExpTask2(earnings-announcement-lda-lasso)/lasso_alpha_scatter.jpg"
      alt="lasso_alpha_scatter"
      style="width:100%; max-width:600px; display:block;"
    >
    <div style="font-size: 0.9em; color: gray; margin-top: 0.4em;">
      <em>alpha별 5-fold 검증 neg RMSE (빨간 점선: 최적 alpha ≈ 0.00573)</em>
    </div>
  </div>

</div>

> 💡 y축은 neg RMSE라서 **위쪽일수록 좋은 모델**입니다.
> alpha가 너무 작으면 잡음까지 학습하고, 너무 크면 중요한 토픽까지 0이 됩니다. 그 사이에서 점수가 가장 높은 지점이 최적 alpha입니다.

| 항목 | 값 |
|---|---|
| 최적 `alpha` | 약 0.00573 |
| 교차검증 RMSE | 약 9.27 %p |
| 0이 아닌 계수 | **120개 중 7개** |
| 가장 큰 양(+)의 계수 | Topic 84, **+3.64** |
| 가장 큰 음(−)의 계수 | Topic 116, **−16.92** |

| 토픽 | 상위 5개 단어 |
|---|---|
| **Topic 84 (+)** | restructuring, financial_number restructuring, restructuring charges, charges financial_number, restructuring costs |
| **Topic 116 (−)** | net loss, loss income, loss per, financial_number loss, operating loss |

### 11-2. 해석

**① 손실 토픽(116)의 음(−)의 계수는 직관적이다**
순손실·영업손실을 보고하는 것은 분명한 나쁜 소식입니다.
6-3의 방식으로 읽으면, 이 토픽의 비중이 10%p 높을수록 3일 수익률이 약 1.7%p 낮은 것과 관련이 있습니다($-16.92 \times 0.1$).

**② 구조조정 토픽(84)의 양(+)의 계수는 처음엔 의외다**
구조조정 비용은 보고 이익을 줄이므로 음(−)을 예상하기 쉽습니다. 하지만 이렇게 설명할 수 있습니다.

- 구조조정 비용은 보통 **일회성**이라, 경영진은 이를 들어 부진한 GAAP 이익이 일시적이라고 설명합니다.
- 투자자는 구조조정을 **향후 비용 절감 신호**로 읽을 수 있습니다.

**③ 텍스트 토픽의 설명력은 제한적이다**
120개 중 7개만 남았고 RMSE가 약 9.3%p라는 것은 **토픽만으로 설명되는 수익률 변동이 작다**는 뜻입니다.
주가 반응의 상당 부분은 EPS 서프라이즈 같은 숫자 정보와 시장 상황에 달려 있습니다.

> ⭐ **이 계수들은 연관성이지 인과 효과가 아닙니다.**
> 손실을 "언급해서" 주가가 떨어진 것이 아니라, 손실이라는 사실과 그것을 설명하는 언어가 함께 나타난 것입니다.

## <span style="color:#FEB99C;">12. 분류기: +5% 이상 상승을 예측할 수 있을까?</span>

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from scipy.stats import loguniform

YLR = (df["BHAR0_2"] > 5).astype(int)
train_Xdtm, test_Xdtm, train_Xtopic, test_Xtopic, train_YLR, test_YLR = train_test_split(
    np.asarray(dtm.todense()), topics, YLR, train_size=0.80, random_state=123
)

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

lr_search = RandomizedSearchCV(
    Pipeline([("tfidf", TfidfTransformer()),
              ("clf", LogisticRegression(class_weight="balanced", solver="liblinear",
                                         max_iter=2000, random_state=123))]),
    {"tfidf__use_idf": [True, False],
     "tfidf__sublinear_tf": [True, False],
     "clf__penalty": ["l1", "l2"],
     "clf__C": loguniform(1e-3, 10)},
    n_iter=30, cv=cv5, scoring="f1_macro", random_state=123, n_jobs=-1
).fit(train_Xdtm, train_YLR)

rf_search = RandomizedSearchCV(
    RandomForestClassifier(class_weight="balanced_subsample", random_state=123),
    {"n_estimators": [200, 300, 500],
     "max_depth": [None, 5, 10, 20],
     "min_samples_leaf": [1, 5, 10, 20],
     "max_features": ["sqrt", "log2", 0.3]},
    n_iter=20, cv=cv5, scoring="f1_macro", random_state=123, n_jobs=-1
).fit(train_Xtopic, train_YLR)
```

> 🙋 **`loguniform(1e-3, 10)`은 왜 쓸까요?**
>
> `C`는 0.001과 10처럼 **자릿수(scale) 단위로** 효과가 달라집니다.
> 균등분포로 뽑으면 대부분의 후보가 1~10 사이에 몰리므로, 로그 척도에서 고르게 뽑습니다.

### 12-1. 결과

| 모델 | 최적 하이퍼파라미터 | CV Macro-F1 |
|---|---|---:|
| **TF-IDF + 로지스틱 회귀 (DTM)** | `C`≈4.74, L2, `use_idf=False`, `sublinear_tf=False` | **0.5296** |
| 랜덤 포레스트 (토픽) | 200 trees, `max_depth`=10, `min_samples_leaf`=5, `max_features`="log2" | 0.5262 |

CV 점수가 더 높은 **로지스틱 회귀를 최종 모델**로 선택하고, 검증 데이터로 classification report를 출력했습니다.

```text
              precision    recall  f1-score   support

           0       0.80      0.60      0.68      1904
           1       0.28      0.51      0.36       596

    accuracy                           0.58      2500
   macro avg       0.54      0.55      0.52      2500
weighted avg       0.67      0.58      0.61      2500
```

검증 데이터 2,500개 중 **596개(23.8%)** 가 +5% 초과 상승(클래스 1)입니다.
최종 **Macro-F1은 0.52**입니다.

### 12-2. 어떻게 읽을까?

**① 정확도 0.58은 "전부 0"보다 낮지만, 그래도 더 나은 모델이다**

| 기준 | Accuracy | Macro-F1 |
|---|---:|---:|
| 모든 문서를 0으로 예측 | 0.76 | 약 0.43 |
| **최종 모델** | **0.58** | **0.52** |

모든 문서를 "상승 아님"으로 예측하면 정확도는 76%(1,904 / 2,500)입니다.
하지만 상승 사례를 하나도 찾지 못해 클래스 1의 F1이 0이 되고, Macro-F1은 약 0.43에 그칩니다.
최종 모델은 정확도를 희생하는 대신 **상승 사례의 절반을 찾아내** Macro-F1을 0.52로 끌어올렸습니다.
**7-1에서 본 "정확도 함정"** 이 실제 숫자로 나타난 예시입니다.

**② 상승 클래스: 재현율 0.51, 정밀도 0.28**
- **재현율 0.51**: 실제로 +5% 넘게 오른 발표문 596개 중 약 절반을 찾아냈습니다.
- **정밀도 0.28**: 모델이 "오른다"고 예측한 발표문 중 실제로 오른 것은 28%입니다. 기본 비율(23.8%)보다 조금 높은 수준입니다.

`class_weight="balanced"`가 소수 클래스를 더 많이 예측하도록 밀어 주어, **재현율은 올리고 정밀도와 정확도는 낮춘** 결과입니다.

**③ 교차검증과 검증 결과가 비슷하다**
CV Macro-F1 0.53과 검증 Macro-F1 0.52가 거의 같습니다.
튜닝 과정에서 훈련 데이터에 **과적합되지 않았다**는 신호입니다.

**④ 예측력 자체는 제한적이다**
Macro-F1 0.52는 높은 수준이 아닙니다. 11절의 Lasso 결과와 같은 결론입니다.
**발표문의 단어만으로 큰 폭의 상승을 가려내기는 어렵습니다.**

**⑤ 단어(DTM)가 토픽보다 조금 나았다**
토픽은 1,000개 단어를 120개로 **요약한 정보**입니다.
요약 과정에서 예측에 유용한 세부 신호가 일부 사라졌을 수 있습니다. 다만 CV 점수 차이(0.530 vs 0.526)는 작습니다.

**⑥ `use_idf=False`가 선택되었다**
이 과제에서는 IDF 가중치보다 **문서 길이로 정규화한 단순 빈도**가 더 나았습니다.
이미 `max_df=0.5`로 흔한 단어를 걸렀기 때문에 IDF의 추가 이득이 작았을 가능성이 있습니다.

## <span style="color:#FEB99C;">13. 비판적으로 다시 보기</span>

| 점검 | 내용 |
|---|---|
| **전체 데이터로 학습한 DTM·LDA** | 어휘와 토픽이 검증 문서까지 보고 만들어졌습니다. 레이블은 쓰지 않아 심각한 누수는 아니지만, 엄밀하게는 훈련 데이터로만 학습해야 합니다. |
| **무작위 분할** | 데이터는 2006~2020년에 걸쳐 있습니다. 실제 예측 상황을 반영하려면 **과거로 학습하고 미래로 평가하는 시간 기반 분할**이 더 현실적입니다. |
| **숫자 정보 제거** | `financial_number` 치환으로 EPS 서프라이즈 같은 핵심 정보가 빠졌습니다. 텍스트는 숫자를 **보완**하는 정보로 보는 것이 맞습니다. |
| **연도 숫자** | 정규식이 연도(2023)도 `financial_number`로 바꿉니다. 의도된 결과인지 확인할 필요가 있습니다. |

> 🙋 **실행 중 `BrokenProcessPool`과 `numpy._core` 오류가 났다면?**
>
> 노트북 실행 중에 `tmtoolkit` 같은 패키지를 설치하면서 **numpy 버전이 바뀐 경우**입니다.
> 커널은 예전 numpy를 쓰고, `n_jobs=-1`로 띄운 작업 프로세스는 새 numpy를 불러와 충돌합니다.
> **커널을 재시작**하면 해결되고, 임시로는 `n_jobs=1`로 병렬 처리를 끌 수 있습니다.
> 오래 걸린 전처리 결과는 버전과 무관한 **CSV로 저장**해 두는 것이 안전합니다.

## <span style="color:#FEB99C;">14. 정리</span>

### 1부 개념

- **실적 발표문** = 8-K Exhibit 99.1의 보도자료, 숫자 + 경영진의 서술
- **BHAR(0,2)** = 발표 후 3일간의 매수 후 보유 초과수익률 (단위 %p)
- **`financial_number` 치환** = 숫자의 다양성은 없애고 숫자가 있었다는 구조는 유지
- **나쁜 문장 필터** = 5토큰 미만·숫자 50% 초과 문장 제거 → 표와 조각 제거
- **`max_df`** = 이 말뭉치에서 너무 흔한 단어 제거 (불용어 목록을 보완)
- **LDA** = 문서는 토픽의 혼합, 토픽은 단어의 분포. $\alpha$·$\eta$가 집중도를 조절
- **u_mass** = 상위 단어의 동시 등장 정도. 0에 가까울수록 좋음
- **Word intrusion** = 사람이 토픽의 일관성을 검증하는 방법
- **Lasso** = L1 벌점으로 계수를 0으로 만들어 변수 선택
- **토픽 비중 계수** = 합이 1인 구성 데이터 → 상대적 연관으로 해석
- **Macro-F1** = 클래스별 F1의 단순 평균. 불균형 분류에 적합
- **모델 선택** = 검증 데이터가 아니라 교차검증 점수로

### 2부 실습

- **토픽 수** = 120개 (u_mass 최고, 다만 차이는 작음)
- **Lasso** = 120개 중 7개 토픽만 선택, CV RMSE 약 9.27%p
- **음(−) 토픽** = 손실(net loss, operating loss), 계수 −16.92 → 직관적
- **양(+) 토픽** = 구조조정, 계수 +3.64 → 일회성 비용·비용 절감 신호로 설명 가능
- **분류** = 로지스틱 회귀(CV Macro-F1 0.530)가 랜덤 포레스트(0.526)보다 근소하게 우세
- **최종 검증 결과** = Macro-F1 0.52, 상승 클래스 재현율 0.51·정밀도 0.28 (전부 0으로 예측할 때의 Macro-F1 약 0.43보다 나음)
- **결론** = 발표문의 언어는 주가 반응과 관련이 있지만, 그것만으로 설명·예측되는 부분은 작음

> 🎯 Task 2 한 줄 요약:
> **"경영진이 무엇을 이야기했는지는 시장 반응의 단서가 되지만, 숫자를 대신하지는 못한다. 텍스트는 숫자를 보완하는 또 하나의 정보다."**