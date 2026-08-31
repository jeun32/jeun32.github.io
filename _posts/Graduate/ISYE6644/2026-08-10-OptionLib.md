---
title : "OptionLib: A Python Library for Monte Carlo Option Pricing"
date : 2026-08-10 20:00:00 +0900
categories : [Graduate School, (SU26) Simulation]
tags : [ISYE6644, Monte Carlo, Option Pricing]
math : true
---

[📎보고서링크](/assets/files/OptionLib_Final_Report.pdf) <br>
[📎코드링크](https://github.com/jeun32/simulation/tree/main/notebooks)

올해 여름 ISYE6644(Simulation) 팀 프로젝트로.. 현업에 바쁜 와중에도 승준이, 태호와 함께 개발한 **OptionLib**를 소개 및 정리해보겠습니다. (다들 고맙습니다) <br>
"옵션 가격을 몬테카를로 시뮬레이션으로 어떻게, 그리고 얼마나 정확하게 계산할 수 있는가"를 다룬 파이썬 라이브러리인데, 쉽게 비유를 들어 풀어보겠습니다.

## <span style="color:#FEB99C;">1. 옵션이 뭔데?</span>

**옵션**은 "미래에 정해진 가격(K)으로 주식을 사거나 팔 수 있는 권리"입니다.

예를 들어 "1년 뒤에 삼성전자를 10만원에 살 수 있는 권리"를 미리 사뒀다고 해보겠습니다.

- 1년 뒤 주가가 12만원 → 10만원에 사서 바로 2만원 이득 (권리 행사)
- 1년 뒤 주가가 8만원 → 그냥 권리를 안 쓰면 그만 (손해는 옵션 구매 비용으로 제한)

문제는 **이 권리 자체의 "적정 가격"이 얼마여야 하는가**입니다. 이게 이 프로젝트가 다루는 문제입니다.

옵션은 "언제 행사할 수 있는지"에 따라 난이도가 완전히 달라집니다.

| 종류 | 특징 | 가격 계산 난이도 |
|---|---|---|
| European | 만기일에만 행사 가능 | 쉬움 — 닫힌 공식(BSM)으로 즉시 계산 |
| Asian | 만기까지의 "평균 주가"로 정산 | 어려움 — 닫힌 공식이 없음 |
| American | 아무 때나 행사 가능 | 매우 어려움 — "언제 행사할지"까지 결정해야 함 |

European은 1973년에 나온 **블랙-숄즈(BSM) 공식**으로 계산기 두드리듯 값이 바로 나옵니다. 하지만 Asian·American은 그런 공식이 없어서 **몬테카를로 시뮬레이션(MC)**이 필요합니다.

## <span style="color:#FEB99C;">2. Why Monte Carlo?</span> 

공식이 없으면 "그냥 수만~수십만 개의 미래 시나리오를 랜덤하게 만들어서 평균을 내보자"는 게 몬테카를로의 아이디어입니다.

> 주사위를 굴려서 나올 평균값을 공식으로 못 구한다면 어떻게 할까요? 그냥 10만 번 굴려서 평균을 내면 진짜 평균에 가까워집니다(대수의 법칙). 원리는 똑같습니다.

우리는 일반적인 통념에 따라 주가가 **기하 브라운 운동(GBM)**이라는 random 프로세스를 따른다고 가정했습니다.

$$
S_t = S_0 \cdot \exp\left[ \left(r - \frac{1}{2}\sigma^2\right) t + \sigma \sqrt{t}\, Z \right], \quad Z \sim N(0,1)
$$

이 식으로 컴퓨터가 가상 주가 경로를 수만 개 만들어내고(아래 이미지가 그 예시입니다), 각 경로에서 옵션의 수익(payoff)을 계산해 할인한 뒤 평균을 냅니다.

$$
V = \frac{1}{n}\sum_i e^{-rT}\cdot \text{payoff}_i, \qquad SE = \frac{s}{\sqrt{n}}, \qquad \text{95\% CI} = V \pm 1.96 \cdot SE
$$

<div style="display: flex; justify-content: center; gap: 2em; margin: 2em 0;">
  <div style="text-align: center;">
    <img
      src="/assets/img/posts/2026-08-10-OptionLib/optionlib_1.jpg"
      alt="30개의 GBM 시뮬레이션 주가 경로"
      style="width:100%; max-width:600px; display:block;"
    >
    <div style="font-size: 0.9em; color: gray; margin-top: 0.4em;">
      <em>30개의 시뮬레이션 주가 경로 (S0=100, r=5%, σ=20%, T=1년) — 논문 Figure 2 캡처</em>
    </div>
  </div>
</div>

**문제는 이 랜덤 계산이 매번 살짝 다르게 나오고 오차(노이즈)가 있다는 것**입니다. 이 오차는 시뮬레이션 횟수(n)의 제곱근에 반비례해서 줄어드는데($1/\sqrt{n}$), 오차를 절반으로 줄이려면 시뮬레이션을 **4배**나 더 돌려야 합니다. "어떻게 하면 적은 시뮬레이션으로도 정확한 값을 얻을 수 있을까?"가 이 프로젝트 전체의 핵심 질문입니다.

## <span style="color:#FEB99C;">3. OptionLib 라이브러리 구조</span>

라이브러리는 역할별로 7개 모듈로 쪼개서 설계했습니다. 하나의 GBM 시뮬레이터와 RNG(난수생성기)를 모든 옵션 종류가 공유하는 구조입니다.

| 모듈 | 역할 |
|---|---|
| `analytics.py` | BSM 닫힌 공식, 그릭스, 기하평균 아시안(Kemna–Vorst) 공식 |
| `gbm.py` | GBM 경로/만기가격 시뮬레이터, antithetic 샘플링, 시드 제어 |
| `mc_european.py` | European 콜/풋 MC 프라이서 (신뢰구간 + 통제변량 옵션) |
| `mc_asian.py` | 경로기반 arithmetic-Asian 프라이서 (기하평균 통제변량 포함) |
| `american.py` | 이항모형(CRR) 기준값 + Longstaff–Schwartz(LSM) |
| `greeks.py` | 해석적 그릭스 + 공통난수(CRN) 기반 유한차분 그릭스 |
| `varred.py` | Antithetic / 통제변량 추정기 및 분산감소 진단 |

모든 프라이서가 공유하는 템플릿 함수는 다음과 같습니다 (European 옵션 예시입니다).

```python
def mc_european(S0,K,r,sigma,T,n,call=True,antithetic=False,control=False,rng=None):
    if antithetic:
        Z = rng.standard_normal(n//2); Z = np.concatenate([Z,-Z])
    else:
        Z = rng.standard_normal(n)
    ST   = S0*np.exp((r-0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    disc = np.exp(-r*T)*np.maximum(ST-K,0) if call else np.exp(-r*T)*np.maximum(K-ST,0)
    if control:                        # CV = discounted S_T, E_Q[e^{-rT}S_T]=S0
        cv = np.exp(-r*T)*ST
        b = np.cov(disc,cv)[0,1]/np.var(cv,ddof=1)
        disc = disc - b*(cv - S0)
    return disc.mean(), disc.std(ddof=1)/np.sqrt(len(disc))
```

---

## <span style="color:#FEB99C;">4. 실험 1 — 우리 시뮬레이션이 제대로 작동하는지 검증</span>

European 옵션은 정답(BSM 공식값)을 이미 알고 있으니, 몬테카를로로 계산한 값이 정답과 비슷한지 5개 행사가 × {콜, 풋} = 10개 케이스로 대조해봤습니다.

| K | Type | BSM (정답) | MC 추정값 | SE | 95% CI | 커버? |
|---|---|---|---|---|---|---|
| 80 | Call | 24.5888 | 24.6844 | 0.0607 | [24.5654, 24.8035] | Yes |
| 80 | Put | 0.6872 | 0.6655 | 0.0083 | [0.6493, 0.6818] | **No** |
| 100 | Call | 10.4506 | 10.4479 | 0.0465 | [10.3567, 10.5391] | Yes |
| 100 | Put | 5.5735 | 5.6001 | 0.0274 | [5.5464, 5.6539] | Yes |
| 120 | Put | 17.3950 | 17.3550 | 0.0470 | [17.2629, 17.4470] | Yes |

10개 중 9개가 신뢰구간 안에 정답을 포함했습니다. 딱 하나(K=80 풋)만 벗어났는데, **95% 신뢰구간이면 20번 중 1번은 벗어나는 게 오히려 정상**입니다. 즉 시뮬레이션이 잘못된 게 아니라 통계적으로 예상되는 결과라는 뜻입니다.

<div style="display: flex; justify-content: center; gap: 2em; margin: 2em 0;">
  <div style="text-align: center;">
    <img
      src="/assets/img/posts/2026-08-10-OptionLib/optionlib_2.jpg"
      alt="ATM 콜옵션의 MC 추정값이 BSM 정답으로 수렴하는 과정"
      style="width:100%; max-width:600px; display:block;"
    >
    <div style="font-size: 0.9em; color: gray; margin-top: 0.4em;">
      <em>시뮬레이션 횟수(n)가 늘어날수록 추정값이 BSM 정답(10.451)으로 수렴 — 논문 Figure 1 캡처</em>
    </div>
  </div>
</div>

여기서 한 번 더 확인해본 게 있습니다. "95% 신뢰구간"이라는 말 자체가 진짜 95%의 확률로 정답을 포함하는지 확인하기 위해, 1,000번 독립적으로 반복 실험(macro-replication)을 돌려봤습니다.

| K | 정답 | 평균 MC 추정값 | 커버리지 (목표 0.95) | 평균 SE | 실제 SE |
|---|---|---|---|---|---|
| 90 | 16.6994 | 16.7155 | 0.9500 | 0.2461 | 0.2441 |
| 100 | 10.4506 | 10.4421 | 0.9520 | 0.2081 | 0.2109 |
| 110 | 6.0401 | 6.0378 | 0.9530 | 0.1644 | 0.1645 |

실제 커버리지가 0.950~0.953으로 목표치(0.95)와 거의 일치합니다 → **우리가 계산하는 오차 범위(신뢰구간) 자체를 믿어도 된다**는 게 검증된 셈입니다.

---

## <span style="color:#FEB99C;">5. 실험 2 — 분산감소: 적게 굴리고도 정확하게</span>

이 프로젝트에서 가장 인상적인 부분입니다. 핵심 아이디어는 **통제변량(control variate)** 기법입니다.

> 저울이 정확한지 모를 때, 무게를 이미 아는 물건(1kg 표준추)을 같이 재본다고 생각해보세요. "어? 저울이 1.05kg으로 표시하네. 그럼 이 저울은 5% 더 크게 나오는구나" 하고 그만큼 보정하면 됩니다.

**Asian 옵션(평균가격 기준으로 정산하는 옵션)**은 정확한 공식이 없습니다. 하지만 사촌뻘인 **기하평균 Asian 옵션**은 Kemna–Vorst 공식으로 정답을 알고 있습니다. 이 "사촌의 오차"를 이용해서 진짜 Asian 옵션 계산을 보정하는 방식입니다.

European 콜(K=90)에서는 이 방법으로 분산이 약 **17배** 줄었습니다.

| 방법 | 추정값 | SE | 분산감소(×) |
|---|---|---|---|
| Naive | 16.7484 | 0.0552 | 1.0 |
| Antithetic | 16.7137 | 0.0550 | 1.0 |
| Control variate | 16.7137 | 0.0132 | 17.3 |
| Antithetic + Control | 16.7094 | 0.0132 | 17.4 |

그런데 진짜 효과는 Asian 옵션에서 나타났습니다. 기하평균 통제변량을 쓰니 표준오차가 0.0256 → 0.0007로, **분산감소 약 1,300배**가 나왔습니다. 이건 naive 방식으로 13억 개의 경로를 시뮬레이션한 정확도와 맞먹는 수준입니다.

| 방법 | 추정값 | SE | 분산감소(×) |
|---|---|---|---|
| Naive | 5.8204 | 0.0256 | 1.0 |
| Antithetic | 5.8502 | 0.0256 | 1.0 |
| Geometric control | 5.7522 | 0.0007 | **1313.3** |

<div style="display: flex; justify-content: center; gap: 2em; margin: 2em 0;">
  <div style="text-align: center;">
    <img
      src="/assets/img/posts/2026-08-10-OptionLib/optionlib_3.jpg"
      alt="분산감소 기법별 표준오차 비교 (European vs Asian)"
      style="width:100%; max-width:600px; display:block;"
    >
    <div style="font-size: 0.9em; color: gray; margin-top: 0.4em;">
      <em>기법별 표준오차 비교 — 왼쪽 European 콜(선형축), 오른쪽 Asian 콜(로그축) — 논문 Figure 3 캡처</em>
    </div>
  </div>
</div>

---

## <span style="color:#FEB99C;">6. 실험 3 — American 옵션: 언제 행사해야 할지 정하기 (LSM)</span>

American 옵션은 "지금 행사할까, 더 기다릴까?"를 매 순간 결정해야 하는 **최적 정지 문제(optimal stopping)**입니다. 여기서 쓴 방법이 **Longstaff–Schwartz(LSM)**입니다.

원리는 미래에서 과거로 거슬러 오면서, 매 시점마다

- "지금 행사했을 때 받는 돈(즉시 행사가치)"
- "계속 들고 있으면 나중에 받을 것으로 예상되는 돈(계속가치, 회귀분석으로 추정)"

을 비교해서 더 유리한 쪽을 선택하는 방식입니다.

정확도 검증은 훨씬 느리지만 사실상 정답에 가까운 **이항모형(3,000단계 binomial tree)**과 비교했습니다.

| K | European put (BSM) | American (이항모형) | American (LSM) | 조기행사 프리미엄 | \|LSM-이항\| |
|---|---|---|---|---|---|
| 90 | 2.3101 | 2.4726 | 2.4486 | 0.1625 | 0.0240 |
| 100 | 5.5735 | 6.0900 | 6.0463 | 0.5165 | 0.0437 |
| 110 | 10.6753 | 11.9732 | 11.9007 | 1.2979 | 0.0725 |

모든 행사가에서 **오차 0.7% 이내**로 거의 일치했고, "일찍 행사할 수 있어서 얻는 추가 가치"(조기행사 프리미엄)도 이론대로 옵션이 내가격(ITM)일수록 커지는 패턴을 정확히 잡아냈습니다. LSM 값이 이항모형보다 살짝 낮게 나오는 것도 이론과 일치하는데, 회귀로 추정한 행사 전략이 완벽할 수 없어서 살짝 보수적인(under-price) 쪽으로 편향되기 때문입니다.

<div style="display: flex; justify-content: center; gap: 2em; margin: 2em 0;">
  <div style="text-align: center;">
    <img
      src="/assets/img/posts/2026-08-10-OptionLib/optionlib_4.jpg"
      alt="LSM 가격이 이항모형 기준값으로 수렴하는 과정"
      style="width:100%; max-width:600px; display:block;"
    >
    <div style="font-size: 0.9em; color: gray; margin-top: 0.4em;">
      <em>경로 수(n)가 늘어날수록 LSM 가격이 이항모형 기준값(점선)에 수렴 — 논문 Figure 4 캡처</em>
    </div>
  </div>
</div>

---

## <span style="color:#FEB99C;">7. 실험 4·5 — 그릭스(Greeks)와 공통난수(CRN)</span>

**그릭스**는 옵션 가격이 다른 요인에 얼마나 민감한지를 나타냅니다.

- Delta: 주가가 조금 바뀌면 옵션가격이 얼마나 바뀌는지
- Gamma: Delta 자체가 얼마나 빨리 바뀌는지
- Vega / Theta / Rho: 변동성 / 시간 / 금리 변화에 대한 민감도

European은 이것도 미분해서 구하는 공식이 있지만, 공식이 없는 옵션은 **"값을 살짝 흔들어보고(bump) 그 차이"**로 추정합니다(유한차분법).

| 그릭스 | 해석적(정답) | 유한차분 MC (CRN) | 차이 |
|---|---|---|---|
| Delta | 0.63683 | 0.63674 | -0.00009 |
| Gamma | 0.01876 | 0.01921 | 0.00044 |
| Vega | 37.52403 | 37.53892 | 0.01488 |
| Theta | -6.41403 | -6.41802 | -0.00399 |
| Rho | 53.23248 | 53.22316 | -0.00932 |

유효숫자 3자리까지 거의 정확히 일치합니다. 그런데 여기엔 숨겨진 조건이 있습니다. 바로 **공통난수(CRN)**를 반드시 써야 한다는 것입니다.

> 같은 사람의 키를 두 번 잰다고 해보겠습니다. 한 번은 정확한 자로, 한 번은 눈대중으로 재고 그 차이를 비교하면 아무 의미가 없습니다. 대신 **같은 자, 같은 조건**으로 두 번 재야 진짜 성장한 만큼의 차이를 알 수 있습니다.

주가를 살짝 올려서 계산한 가격과 살짝 내려서 계산한 가격을 비교할 때, **서로 다른 난수**를 쓰면 "가격이 흔들려서 생긴 차이"보다 "난수가 달라서 생긴 잡음"이 훨씬 커져서 결과가 엉망이 됩니다. 같은 난수(같은 시나리오)를 재사용해야 진짜 신호만 남습니다.

| 추정 방식 | 평균 Delta | 추정치 표준편차 | 분산감소 |
|---|---|---|---|
| 독립적인 난수 사용 | 0.6338 | 0.1379 | 1× |
| 공통난수(CRN) 사용 | 0.6369 | 0.0042 | **1104×** |

독립 난수로는 표준편차가 0.138로 사실상 단일 추정치를 믿을 수 없는 수준인데, CRN을 쓰니 0.004로 뚝 떨어졌습니다. **분산감소 약 1,100배** — 공짜로 얻는 정확도라고 볼 수 있습니다.

<div style="display: flex; justify-content: center; gap: 2em; margin: 2em 0;">
  <div style="text-align: center;">
    <img
      src="/assets/img/posts/2026-08-10-OptionLib/optionlib_5.jpg"
      alt="주가에 따른 Delta, Gamma 곡선 (해석적 값 vs 유한차분 MC)"
      style="width:100%; max-width:600px; display:block;"
    >
    <div style="font-size: 0.9em; color: gray; margin-top: 0.4em;">
      <em>Delta(왼쪽), Gamma(오른쪽) — 해석적 곡선과 유한차분 MC 추정치가 거의 겹칩니다 — 논문 Figure 5 캡처</em>
    </div>
  </div>
  <div style="text-align: center;">
    <img
      src="/assets/img/posts/2026-08-10-OptionLib/optionlib_6.jpg"
      alt="독립 난수 vs 공통난수(CRN) Delta 추정치 분포 비교"
      style="width:100%; max-width:600px; display:block;"
    >
    <div style="font-size: 0.9em; color: gray; margin-top: 0.4em;">
      <em>200회 반복 실험에서 CRN(파랑)은 정답 근처에 좁게 몰려있지만, 독립 난수(빨강)는 넓게 흩어져 있습니다 — 논문 Figure 6 캡처</em>
    </div>
  </div>
</div>

---

## <span style="color:#FEB99C;">8. 결론</span>

> 몬테카를로는 그냥 돌리면 부정확하지만, 문제의 구조를 잘 활용하면(통제변량, 공통난수 재사용 등) **거의 공짜로 정확도를 수백~수천 배** 높일 수 있습니다.

이 프로젝트(OptionLib)를 수행하며 이론적으로만 알고 있던 내용을 실제 코드로 구현하면서, 매 단계마다 정답인 벤치마크(BSM, 이항모형)와 대조하며 검증이 될때 뿌듯함을 느꼈습니다. 실무와도 연관이 되어있어 피부에 와닿는 플젝이다보니 바빴음에도 나름 무척 재미있었던 과목이었습니다.