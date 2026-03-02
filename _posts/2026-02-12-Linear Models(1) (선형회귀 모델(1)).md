---
title : Linear Models(1) (선형회귀 모델(1))
date : 2026-02-12 19:45:00 +0900
categories : [Graduate School, (FA25) Data Analytics in Business]
tags : [MGT6203, R 회귀분석, lm()]
math : true
---


## 1. [R] lm() Module — Beta, t-value, CI

R에서 선형회귀분석 모듈인 `lm()`의 `summary()` 결과를 직접 도출해내는 과정이 첫번째 과제였다.

- 회귀계수 (β̂)
- 표준오차 (Standard Error)
- t-value (t-statistic)
- Confidence Interval (신뢰구간)

을 직접 계산하고 `summary()` 결과와 비교해보며  
선형회귀의 내부 구조를 이해한다.

> [📎 전체 코드 Github Link](https://github.com/jeun32/data-analytics-in-business/blob/main/notebooks/02-linearmodels1.R)

---

## 2. 데이터 불러오기 및 전처리

### 2.1 광고 sales효과 선형분석하기 위한 raw 데이터

```r
mydata = read.csv("advertising.csv", header = TRUE)
head(mydata)
```

<div style="display: flex; justify-content: center; gap: 2em; margin: 2em 0;">
  
  <div style="text-align: center;">
    <img 
      src="/assets/img/posts/2026-02-12-Linear Models(1) (선형회귀 모델(1))/linear1_1.jpg"
      alt="data 구조 확인"
      style="width:100%; max-width:600px; display:block;"
    >
    <div style="font-size: 0.9em; color: gray; margin-top: 0.4em;">
      <em>data 구조 확인</em>
    </div>
  </div>
</div>


---

## 3. 선형회귀 실행

```r

lm.res = lm(sales ~ TV + radio + newspaper, mydata)
lm.sum=summary(lm.res)
lm.sum
```

$$
sales = \beta_0 + \beta_1 TV + \beta_2 radio + \beta_3 newspaper + u
$$

- 종속변수: 매출 (Sales)
- 독립변수: 광고채널 (TV, radio, newspaper)

`summary()`함수는 다음을 제공한다:

- Estimate (β̂)
- Std. Error
- t-value
- p-value
- R-squared
- F-statistic
<div style="display: flex; justify-content: center; gap: 2em; margin: 2em 0;">
  
  <div style="text-align: center;">
    <img 
      src="/assets/img/posts/2026-02-12-Linear Models(1) (선형회귀 모델(1))/linear1_2.jpg"
      alt="summary 결과"
      style="width:100%; max-width:400px; display:block;"
    >
    <div style="font-size: 0.9em; color: gray; margin-top: 0.4em;">
      <em>summary 결과</em>
    </div>
  </div>
</div>
---

### 3.1 예측값과 잔차

```r
yhat = fitted(lm.res)
uhat = resid(lm.res)
cbind(TV, radio, newspaper, sales, yhat, uhat)[1:10,]
```

* 회귀모형에서 계산된 예측값

$$
\hat{y}_i = x_i' \hat{\beta}
$$

* 잔차(residual)는 실제값과 예측값의 차이로 정의된다. (회귀식으로 설명되지 않는 부분)

$$
\hat{u}_i = y_i - \hat{y}_i
$$

<div style="display: flex; justify-content: center; gap: 2em; margin: 2em 0;">
  
  <div style="text-align: center;">
    <img 
      src="/assets/img/posts/2026-02-12-Linear Models(1) (선형회귀 모델(1))/linear1_3.jpg"
      alt="각 데이터 선형회귀 price 예측 (총 1000개 데이터)"
      style="width:100%; max-width:350px; display:block;"
    >
    <div style="font-size: 0.9em; color: gray; margin-top: 0.4em;">
      <em>각 데이터 선형회귀 price 예측 (총 200개 데이터)</em>
    </div>
  </div>
  <div style="text-align: center;">
    <img 
      src="/assets/img/posts/2026-02-12-Linear Models(1) (선형회귀 모델(1))/linear1_4.jpg"
      alt="예측값의 잔차"
      style="width:100%; max-width:350px; display:block;"
    >
    <div style="font-size: 0.9em; color: gray; margin-top: 0.4em;">
      <em>예측값의 잔차</em>
    </div>
  </div>
</div>
<div style="display: flex; justify-content: center; gap: 2em; margin: 2em 0;">
  <div style="text-align: center;">
    <img 
      src="/assets/img/posts/2026-02-12-Linear Models(1) (선형회귀 모델(1))/linear1_5.jpg"
      alt="실제 sales데이터와 예측값 비교(sales vs yhat)"
      style="width:100%; max-width:600px; display:block;"
    >
    <div style="font-size: 0.9em; color: gray; margin-top: 0.4em;">
      <em>실제 sales데이터와 예측값 비교(sales VS yhat)</em>
    </div>
  </div>
</div>
---

## 4. t-value 직접 계산

```r
bhat = lm.sum$coefficients[,1]
se   = lm.sum$coefficients[,2]

tstat = bhat / se
tstat

cbind(tstat, lm.sum$coefficients[,3])
```
bhat은 베타(추정된 회귀계수),
se는 해당 계수의 표준오차(SE)를 의미하고,

t value는 '베타/표준오차'로 정의된다.

$$
t = \frac{\hat{\beta}}{SE(\hat{\beta})}
$$

직접 계산한 값이 `summary()`의 t value와 일치함을 확인할 수 있다.
<div style="display: flex; justify-content: center; gap: 2em; margin: 2em 0;">
  
  <div style="text-align: center;">
    <img 
      src="/assets/img/posts/2026-02-12-Linear Models(1) (선형회귀 모델(1))/linear1_6.jpg"
      alt="t-value 비교"
      style="width:100%; max-width:400px; display:block;"
    >
    <div style="font-size: 0.9em; color: gray; margin-top: 0.4em;">
      <em>t-value 비교</em>
    </div>
  </div>
</div>
---

## 5. t-value의 임계값 계산(5% 유의수준)

```r
df = lm.res$df.residual
alpha = 1 - 0.95

qt(1 - alpha/2, df)
qnorm(1 - alpha/2)
```
회귀에서는 오차항의 모분산 σ²를 모르기 때문에, 표본으로 분산을 추정해서 사용한다.
따라서 정규분포상 5% 유의수준 임계값인 1.96을 쓰지 않고 자유도를 고려한 t분포상 임계값을 쓴다.

* t분포 임계값:

$$
t_{0.975, df}
$$

* cf) 정규분포 임계값 (표본이 충분히 크다면 t분포는 정규분포에 수렴함):

$$
z_{0.975} \approx 1.96
$$

<div style="display: flex; justify-content: center; gap: 2em; margin: 2em 0;">
  
  <div style="text-align: center;">
    <img 
      src="/assets/img/posts/2026-02-12-Linear Models(1) (선형회귀 모델(1))/linear1_7.jpg"
      alt="t분포 threshold VS 정규분포 threshold"
      style="width:100%; max-width:600px; display:block;"
    >
    <div style="font-size: 0.9em; color: gray; margin-top: 0.4em;">
      <em>t분포 threshold VS 정규분포 threshold</em>
    </div>
  </div>
</div>
---
