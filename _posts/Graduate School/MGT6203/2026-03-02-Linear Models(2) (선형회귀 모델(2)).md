---
title : Linear Models(2)-Predictive Modeling & Interaction (선형회귀 모델(2))
date : 2026-03-02 16:40:00 +0900
categories : [Graduate School, (FA25) Data Analytics in Business]
tags : [MGT6203, R 회귀분석, 다항회귀]
math : true
---


## 1. Predictive Linear Model — Used Cars Data

이번 포스팅에서는 중고차 데이터(`usedcars2.csv`)를 활용해 **Prediction 중심의 선형모형 확장**을 진행한다.

- 다중선형회귀 (Multiple Linear Regression)
- Interaction term (상호작용항: Age × KM)
- Polynomial regression (KM의 함수형태 비선형성 반영)
- 예측곡선 시각화

---

## 2. 데이터 불러오기 및 전처리

### 2.1 Raw 데이터 확인

```r
data = read.csv("usedcars2.csv", header = TRUE)
str(data)
```

원본 데이터는 **1264개 관측치, 13개 변수**로 구성되어 있었으며,  
`Id`, `Model`, `Metallic`, `CC`, `Doors` 등은 본 분석에서 제외하였다.


### 2.2 변수 선택 및 타입 정리

```r
data <- data[, c(-1, -2, -7, -9, -10)]   # Id, Model, Metallic, CC, Doors 제거
data$Color <- as.factor(data$Color)      # Color는 범주형이므로 factor 변환
str(data)
```

전처리 후 데이터는 아래 8개 변수로 구성된다.

- Price (종속변수)
- Age, KM, HP, Automatic, Gears, Weight (수치형 )
- Color (범주형 factor: 10개 레벨)


<div style="display: flex; justify-content: center; gap: 2em; margin: 2em 0;">
  
  <div style="text-align: center;">
    <img 
      src="/assets/img/posts/2026-03-02-Linear Models(2) (선형회귀 모델(2))/linear2_1.jpg"
      alt="data 구조 확인(전처리 전/후)"
      style="width:100%; max-width:600px; display:block;"
    >
    <div style="font-size: 0.9em; color: gray; margin-top: 0.4em;">
      <em>data 구조 확인(전처리 전/후)</em>
    </div>
  </div>
</div>

---

## 3. 다중선형회귀: Price ~ . (Color 포함)

```r
lm.res = lm(Price ~ ., data)
lm.sum = summary(lm.res)
lm.sum
```

모형:

$$
Price = \beta_0 + \beta_1 Age + \beta_2 KM + \beta_3 HP + \beta_4 Automatic + \beta_5 Gears + \beta_6 Weight + \sum_c \gamma_c Color_c + u
$$

<div style="display: flex; justify-content: center; gap: 2em; margin: 2em 0;">
  
  <div style="text-align: center;">
    <img 
      src="/assets/img/posts/2026-03-02-Linear Models(2) (선형회귀 모델(2))/linear2_2.jpg"
      alt="data 구조 확인"
      style="width:100%; max-width:600px; display:block;"
    >
    <div style="font-size: 0.9em; color: gray; margin-top: 0.4em;">
      <em>다중선형회귀 결과</em>
    </div>
  </div>
</div>

### 3.1 결과 요약 (핵심만)

`summary(lm.res)` 결과에서 특히 중요한 부분은 다음이다.

- **Age:** Estimate = -128.0 (t = -48.853, p < 2e-16)
- **KM:** Estimate = -0.01516 (t = -10.537, p < 2e-16)
- **HP:** Estimate = 26.32 (t = 7.985, p ≈ 3.17e-15)
- **Automatic:** Estimate = 503.6 (t = 3.422, p ≈ 0.000641)
- **Gears:** Estimate = 576.8 (t = 3.048, p ≈ 0.002355)
- **Weight:** Estimate = 14.96 (t = 12.456, p < 2e-16)

모델 적합도:

- Multiple R-squared: **0.8675**
- Adjusted R-squared: **0.8659**
- Residual standard error: **1218** (df = 1248)

### 3.2 해석 포인트

- **Age가 1개월 증가할 때 가격이 약 128 유로 감소** (다른 변수 고정)
- **KM이 1 증가할 때 가격이 약 0.015 유로 감소**  
  → 10,000km 증가 시 약 151.6 유로 감소(선형 근사)
- **Weight, HP의 양(+)의 계수**는 “무게/마력 높은 차량이 더 비싸다”는 직관과 부합
- `Color` 더미들은 유의하지 않은 항들이 많았고(대체로 p-value 큼), 설명력은 주로 **Age/KM/Weight/HP** 에 의해 결정되는 구조임

---

## 4. Interaction Model: Age × KM (Color 제외)

Color 변수는 통계적으로 유의하지 않았으며,
모형 단순화(parsimony) 및 상호작용 효과 해석에 집중하기 위해 제외하였고
**Age와 KM의 상호작용항을 포함한 모델**을 추정하였다.

```r
data <- data[, c(-8)]  # Color 제외

lm.int <- lm(Price ~ Age * KM + HP + Automatic + Gears + Weight, data = data)
summary(lm.int)
```

모형:

$$
Price = \beta_0 + \beta_1 Age + \beta_2 KM + \beta_3(Age \times KM) + \beta_4 HP + \beta_5 Automatic + \beta_6 Gears + \beta_7 Weight + u
$$


<div style="display: flex; justify-content: center; gap: 2em; margin: 2em 0;">
  
  <div style="text-align: center;">
    <img 
      src="/assets/img/posts/2026-03-02-Linear Models(2) (선형회귀 모델(2))/linear2_3.jpg"
      alt="data 구조 확인"
      style="width:100%; max-width:600px; display:block;"
    >
    <div style="font-size: 0.9em; color: gray; margin-top: 0.4em;">
      <em>상호작용항(Interaction) 포함 모델(Age × KM)</em>
    </div>
  </div>
</div>

### 4.1 결과 요약 (핵심)

- **Age:** -160.8 (t = -43.214, p < 2e-16)
- **KM:** -0.05961 (t = -13.996, p < 2e-16)
- **Age:KM:** 0.0007237 (t = 11.156, p < 2e-16)
- Residual standard error: **1170** (df = 1256)
- Multiple R-squared: **0.8770**
- Adjusted R-squared: **0.8763**

이는 KM의 효과가 Age에 따라 달라진다는 구조가 실제 데이터에 존재함을 시사한다.

### 4.2 Age:KM 상호작용 해석 (핵심 아이디어)

상호작용항이 존재하면 KM의 한계효과는 Age에 따라 달라진다.

$$
\frac{\partial Price}{\partial KM} = \beta_2 + \beta_3 \cdot Age
$$

예를 들어 Age = 20개월일 때,

$$
\frac{\partial Price}{\partial KM}
= -0.05961 + 0.0007237 \times 20
= -0.04514
$$

즉, 동일한 1km 증가라도 차량 연식 수준에 따라 가격 하락의 기울기가 달라진다.  
이는 단순 선형모형이 포착하지 못하는 **조건부 효과(conditional effect)** 를 반영한 결과이다.

---

## 5. KM vs Price 산점도 (비선형 패턴 확인)

```r
plot(data$KM, data$Price,
     pch = 16, cex = 0.6,
     xlab = "KM (kilometers driven)",
     ylab = "Price (€)",
     main = "Scatterplot: KM vs Price")
```

산점도를 보면,

- KM이 증가할수록 Price는 감소하는 경향을 보임
- 하지만 완전한 직선이라기보단 **초반 급락 후 완만해지는 곡선 형태**가 관찰되는데,

이는 KM과 Price 간 관계가 단순 선형이 아닐 가능성을 시사한다.

<div style="display: flex; justify-content: center; gap: 2em; margin: 2em 0;">
  
  <div style="text-align: center;">
    <img 
      src="/assets/img/posts/2026-03-02-Linear Models(2) (선형회귀 모델(2))/linear2_4.jpg"
      alt="data 구조 확인"
      style="width:100%; max-width:600px; display:block;"
    >
    <div style="font-size: 0.9em; color: gray; margin-top: 0.4em;">
      <em>KM과 Price간 관계</em>
    </div>
  </div>
</div>
---

## 6. Polynomial Regression: KM의 4차 다항식 + Automatic

산점도에서 확인된 곡선형 패턴(초반 급락 후 완만한 감소)을 반영하기 위해 KM의 4차 다항항을 포함하였다.

```r
lm3 <- lm(Price ~ poly(KM, 4, raw = TRUE) + Automatic, data = data)
summary(lm3)
```

모형:

$$
Price = \beta_0 + \beta_1 KM + \beta_2 KM^2 + \beta_3 KM^3 + \beta_4 KM^4 + \beta_5 Automatic + u
$$

이 모델은 **계수에 대해선 선형(linear in parameters)이지만,  
KM에 대해서는 비선형적 함수 형태를 허용**한다.

따라서 예측 목적이라면 Age, HP, Weight 등을 포함한 확장된 polynomial 모형과 비교가 필요하다.

<div style="display: flex; justify-content: center; gap: 2em; margin: 2em 0;">
  
  <div style="text-align: center;">
    <img 
      src="/assets/img/posts/2026-03-02-Linear Models(2) (선형회귀 모델(2))/linear2_5.jpg"
      alt="data 구조 확인"
      style="width:100%; max-width:600px; display:block;"
    >
    <div style="font-size: 0.9em; color: gray; margin-top: 0.4em;">
      <em>킬로수/가격간 비선형성 반영한 모델</em>
    </div>
  </div>
</div>
---

### 6.1 결과 요약 (핵심)

- Intercept: 19780 (t = 43.586, p < 2e-16)
- \(KM\): -0.295 (p < 2e-16)
- \(KM^2\): 2.82e-06 (p ≈ 8.02e-05)
- \(KM^3\): -1.11e-11 (p ≈ 0.0775)
- \(KM^4\): 1.19e-17 (p ≈ 0.516)
- Automatic: 173.7 (p ≈ 0.532)

적합도:

- Multiple R-squared: **0.4873**
- Adjusted R-squared: **0.4853**
- Residual standard error: **2387** (df = 1258)

### 6.2 중요한 해석 포인트

이 모델의 설명력이 낮은 이유는 다항회귀 자체의 문제가 아니라,

- Age, HP, Weight 등 주요 설명 변수를 포함하지 않은
- **KM 중심의 부분모형(partial model)** 이기 때문이다.

따라서 본 모형은

- “가격 전체 설명” 목적보다는
- “KM과 Price 간의 비선형적 함수 형태 확인” 목적에 적합하다.

또한 추정 결과를 보면 4차항은 통계적으로 유의하지 않았으며,
KM² 항까지가 주된 설명력을 가진다.
따라서 실제 적용 시에는 2차 또는 3차 모형으로 단순화하는 것이 바람직할 수 있다.

일반적으로 고차항이 포함될수록 추정 분산이 증가하고
과적합(overfitting) 가능성이 커질 수 있으므로,
차수 선택은 적합도와 해석가능성 간의 균형을 고려하여 결정해야 한다.

---

## 7. 예측 곡선 계산 및 추가

### 7.1 계수 개수 확인

```r
length(coef(lm3))
coef(lm3)
```

총 6개 계수(Intercept + KM 4개 항 + Automatic 1개)를 확인할 수 있다.

<div style="display: flex; justify-content: center; gap: 2em; margin: 2em 0;">
  
  <div style="text-align: center;">
    <img 
      src="/assets/img/posts/2026-03-02-Linear Models(2) (선형회귀 모델(2))/linear2_6.jpg"
      alt="6개 파라미터"
      style="width:100%; max-width:600px; display:block;"
    >
    <div style="font-size: 0.9em; color: gray; margin-top: 0.4em;">
      <em>비선형성 반영</em>
    </div>
  </div>
</div>
---

### 7.2 예측 곡선 생성 (Automatic은 평균으로 고정)

```r
km.grid <- seq(from = min(data$KM, na.rm = TRUE),
               to   = max(data$KM, na.rm = TRUE),
               by   = 1000)

auto_mean <- mean(data$Automatic, na.rm = TRUE)

preds <- predict(
  lm3,
  newdata = list(KM = km.grid,
                 Automatic = rep(auto_mean, length(km.grid)))
)

lines(km.grid, preds, lwd = 3)
```

Automatic을 평균으로 고정한 이유는  
**평균적인 차량 조건에서 KM 변화에 따른 가격 변화를 관찰하기 위함**이다.

<div style="display: flex; justify-content: center; gap: 2em; margin: 2em 0;">
  
  <div style="text-align: center;">
    <img 
      src="/assets/img/posts/2026-03-02-Linear Models(2) (선형회귀 모델(2))/linear2_7.jpg"
      alt="주행킬로수에 따른 가격예측(비선형성 반영)"
      style="width:100%; max-width:600px; display:block;"
    >
    <div style="font-size: 0.9em; color: gray; margin-top: 0.4em;">
      <em>주행킬로수에 따른 가격예측</em>
    </div>
  </div>
</div>
---

## 8. 모델 비교 요약

| 모델 | 포함 변수 | 핵심 목적 | Adj. R² (참고) |
|------|----------|----------|----------------|
| 다중회귀(전체) | Age, KM, HP, Automatic, Gears, Weight, Color | 전반적 설명 및 예측 | 0.8659 |
| Interaction | Age, KM, Age×KM, HP, Automatic, Gears, Weight | 조건부 효과(기울기 변화) 반영 | 0.8763 |
| Polynomial | KM(1~4차), Automatic | 함수형태 분석(KM-Price 비선형 패턴 시각화) | 0.4853 |

---

## 9. 결론

이번 실습에서 얻은 포인트는 다음과 같다.

1. **Age, Weight, HP**는 Price를 설명하는 핵심 변수이다.
2. **Age × KM 상호작용**은 KM의 효과가 연식에 따라 달라짐을 보여주며, 모델 적합도를 개선한다.
3. KM과 Price 관계는 단순 직선이 아닌 **비선형적 형태**를 보인다.
4. 다항회귀는 설명력보다는 **함수 형태 분석 및 시각화 목적**에 유용하다.

선형회귀는 단순한 모형이지만,  
상호작용과 고차항을 포함함으로써 훨씬 풍부한 구조를 표현할 수 있다.