---
title : "(Experiential Task 1) 특허 초록으로 특허 가치 읽기 : XML 파싱부터 DTM·상관분석까지 (KPSS & USPTO)"
date : 2026-10-11 22:46:00 +0900
categories : [Graduate School, "(MGT6033) Analysis of Unstructured Data_'26 Fall"]
tags : [MGT6033, NLP, Patent, KPSS, USPTO, XML, lxml, CountVectorizer, DTM, Correlation]
math : true
---

Experiential Task 1에서는 **특허 초록(abstract)의 단어가 특허의 경제적 가치와 관련이 있는지** 살펴봤습니다.

강의 데모가 "텍스트를 어떻게 숫자로 바꾸는가?"를 다뤘다면,
이번 과제는 **"그 숫자를 실제 경제 변수와 연결하면 무엇이 보이는가?"** 를 묻습니다.

이번 글도 두 부분으로 나뉩니다.

- **1부 (개념)**: 특허 가치 측정, 데이터 결합, XML 구조, DTM 설계, 상관분석의 해석
- **2부 (실습)**: KPSS 데이터와 USPTO 특허 원문을 연결해 단어와 가치의 상관을 계산

> 🎯 이번 글을 관통하는 질문은 이것입니다.
> **"초록에 쓰인 단어로 그 특허가 얼마나 가치 있는지 알 수 있을까?"**

> 📝 **읽기 안내**
> 수치는 제가 과제를 수행하며 얻은 결과입니다.
> 코드는 과제 흐름을 학습용으로 정리한 것이고, 실행 환경과 데이터 버전에 따라 결과가 조금 다를 수 있습니다.

---

# 📘 1부. 개념

---

## <span style="color:#FEB99C;">1. 특허의 가치를 어떻게 숫자로 만들까?</span>

특허는 지식재산을 보호하는 핵심 제도이고, 기업 가치와도 밀접하게 연결됩니다.
그런데 **특허 하나의 가치는 시장에서 직접 거래되지 않습니다.** 그렇다면 어떻게 측정할까요?

### 1-1. KPSS의 아이디어: 주가 반응으로 가치를 역산

Kogan, Papanikolaou, Seru, Stoffman(*QJE* 2017, 이하 KPSS)은 이렇게 접근합니다.

> **특허가 승인(grant)되는 시점에 기업의 주가가 얼마나 움직였는가?**

1. 특허 승인일 전후 기업의 초과수익률을 관찰
2. 같은 기간의 시장 전체 잡음(noise)을 통계적으로 걸러냄
3. 남은 반응에 기업의 시가총액을 곱해 **금액 단위의 특허 가치**로 환산

과제에서 사용한 `xi_real`은 이렇게 추정한 특허 가치를 **물가 조정한 실질 금액**으로 나타낸 변수입니다.

> 💡 **이 측정치는 "기술적으로 대단한 특허"가 아니라
> "투자자가 그 기업에 가치를 더해 준다고 판단한 특허"를 잡아냅니다.**
> 그래서 같은 기술이라도 시가총액이 큰 기업의 특허는 가치가 크게 추정되는 경향이 있습니다.

### 1-2. 인용 수와 무엇이 다를까?

| 측정 방식 | 무엇을 반영하나 | 한계 |
|---|---|---|
| **인용 수(citations)** | 기술적 영향력 | 인용이 쌓이기까지 오래 걸림 |
| **KPSS 가치(xi)** | 승인 시점 시장의 경제적 평가 | 상장 기업 특허만 측정 가능, 시가총액 영향 |

> 🙋 **왜 상장 기업만 측정할 수 있을까요?**
>
> KPSS 방식은 **주가 반응**이 필요하기 때문입니다.
> 비상장 기업·대학·개인 발명가의 특허는 주가가 없어 이 방식으로 가치를 매길 수 없습니다.
> 그래서 KPSS 데이터에는 `permno`(CRSP의 주식 식별번호)가 붙어 있습니다.

## <span style="color:#FEB99C;">2. 데이터 결합: 키·중복·merge 방식</span>

KPSS 데이터에는 기업 이름이 없고 `permno`만 있습니다.
기업 이름은 CRSP의 이름 파일(`crsp_names.csv`)에서 가져와야 합니다.

### 2-1. 이름 파일에는 "이력"이 있다

기업은 이름을 바꿉니다. 그래서 이름 파일에는 **한 `PERMNO`에 여러 행**이 있을 수 있습니다.

```text
PERMNO   COMNAM
10001    GAS NATURAL INC
10001    ENERGY WEST INC      ← 이름 변경 이력
```

이 상태로 merge하면 특허 한 건이 이름 개수만큼 복제됩니다.

> ⚠️ **키가 중복된 표와 merge하면 행이 늘어납니다.**
> 특허 1건 × 이름 3개 = 3행이 되고, 이후 평균·합계가 모두 왜곡됩니다.

그래서 `drop_duplicates(subset="PERMNO", keep="last")`로 **가장 마지막(최신) 이름 하나만** 남깁니다.
데이터가 2022년 특허이므로 최신 이름이 적절합니다.

> 🙋 **`keep="last"`가 정말 최신 이름일까요?**
>
> 파일이 날짜순으로 정렬되어 있을 때만 그렇습니다.
> 엄밀하게 하려면 이름의 유효 기간 열을 기준으로 먼저 정렬한 뒤 중복을 제거하는 편이 안전합니다.

### 2-2. merge 방식의 차이

| 방식 | 남는 행 | 언제 쓰나 |
|---|---|---|
| **`how="left"`** | 왼쪽 표의 모든 행 | 특허는 모두 유지하고 이름만 붙일 때 |
| **`how="inner"`** | 양쪽에 모두 있는 행 | 두 데이터가 모두 필요한 분석 |
| **`how="outer"`** | 양쪽의 모든 행 | 결합되지 않는 행을 점검할 때 |

과제에서는 두 번 merge했습니다.

- **특허 + 이름**: `left` → 이름이 없어도 특허는 유지
- **초록 + 특허 가치**: `inner` → 초록과 가치가 **모두 있는** 특허만 분석 가능

> 💡 merge 후에는 **행 수가 예상과 맞는지** 꼭 확인합니다.
> `validate="many_to_one"` 옵션을 쓰면 오른쪽 키가 중복될 때 오류를 내 주어 실수를 막을 수 있습니다.

## <span style="color:#FEB99C;">3. 평균의 함정: 특허가 적은 기업이 상위를 차지한다</span>

과제 Step 2는 **특허당 평균 가치가 높은 기업 20곳**을 찾는 것입니다.

```python
sub2.groupby("COMNAM")["xi_real"].mean().sort_values(ascending=False)
```

그런데 평균에는 함정이 있습니다.

| 기업 | 특허 수 | 평균 가치 |
|---|---:|---:|
| A | 1 | 매우 높음 |
| B | 2,000 | 중간 |

특허가 1건뿐인 기업은 그 한 건의 값이 곧 평균입니다.
**표본이 작을수록 평균은 극단값이 되기 쉽습니다.**

> ⭐ **"평균 1위"를 보고할 때는 특허 수를 함께 보여 줘야 합니다.**
> `agg(["mean", "count"])`로 건수를 같이 확인하고, 필요하면 최소 건수 기준을 둡니다.

또한 1-1에서 본 것처럼 KPSS 가치는 **시가총액이 큰 기업일수록 커지는 구조**입니다.
평균 가치가 높다는 결과는 "혁신적이다"는 뜻일 수도 있지만,
"규모가 큰 기업이다"는 뜻이 섞여 있을 수도 있습니다.

## <span style="color:#FEB99C;">4. XML: 특허 원문은 어떤 구조일까?</span>

USPTO는 매주 승인된 특허의 전문을 **XML** 형식으로 공개합니다.
과제에서는 2022년 12월 13일 주간 파일(`ipg221213.zip`)을 사용했습니다.

### 4-1. XML은 태그로 된 트리

```xml
<us-patent-grant>
  <us-bibliographic-data-grant>
    <publication-reference>
      <document-id>
        <doc-number>11523456</doc-number>
      </document-id>
    </publication-reference>
  </us-bibliographic-data-grant>
  <abstract>
    <p>A method for ...</p>
  </abstract>
</us-patent-grant>
```

- **요소(element)**: `<abstract>...</abstract>`처럼 태그로 감싼 덩어리
- **트리(tree)**: 요소가 다른 요소를 품는 계층 구조
- **루트(root)**: 가장 바깥 요소(`us-patent-grant`)

### 4-2. USPTO 파일의 특이점: XML 문서 수천 개가 이어 붙어 있다

정상적인 XML 파일은 **루트가 하나**여야 합니다.
그런데 USPTO 주간 파일은 특허마다 완전한 XML 문서가 있고, 이것이 **그대로 이어 붙어** 있습니다.

```text
<?xml version="1.0" ...?>   ← 특허 1 시작
<us-patent-grant>...</us-patent-grant>
<?xml version="1.0" ...?>   ← 특허 2 시작
<us-patent-grant>...</us-patent-grant>
```

통째로 파싱하면 "루트가 여러 개"라는 오류가 납니다.
그래서 먼저 **`<?xml`이 나올 때마다 잘라** 특허별 문서로 나눠야 합니다.

```python
records = re.split(r"(?=<\?xml)", contents_str)
```

> 🙋 **`(?=...)`는 무엇인가요?**
>
> **전방탐색(lookahead)** 입니다. "`<?xml` 바로 앞 위치"에서 자르되 `<?xml` 자체는 지우지 않습니다.
> 그냥 `re.split(r"<\?xml", ...)`로 자르면 각 문서의 첫머리가 사라져 XML 선언이 깨집니다.

### 4-3. 원하는 요소 찾기: XPath

| 표현 | 의미 |
|---|---|
| `tree.find("abstract")` | 바로 아래 자식 중 `abstract` |
| `tree.find(".//abstract")` | **어느 깊이든** 첫 번째 `abstract` |
| `.//publication-reference//doc-number` | 발행 정보 안의 특허 번호 |

> ⚠️ **`.//doc-number`만 쓰면 엉뚱한 번호를 가져올 수 있습니다.**
> 특허 XML에는 출원 번호·인용한 특허 번호 등 `doc-number`가 여러 개 있습니다.
> 그래서 **`publication-reference` 아래**의 번호로 경로를 좁혀야 합니다.

### 4-4. 텍스트 꺼내기: `.text`와 `itertext()`

초록은 `<p>` 안에 다시 `<b>`, `<i>`, `<sub>` 같은 태그가 섞여 있습니다.

```xml
<abstract><p>A <i>novel</i> method for H<sub>2</sub>O ...</p></abstract>
```

- `.text`: **첫 번째 태그 앞의 텍스트만** 반환 → `"A "`
- `itertext()`: 하위 요소의 텍스트를 **모두 순서대로** 반환

> 💡 초록 전체를 얻으려면 `" ".join(el.itertext())`를 씁니다.

### 4-5. 깨진 문서와 빈 초록

- `XMLParser(recover=True)`: 일부가 깨진 문서도 최대한 복구해서 파싱
- 초록이 없는 특허(예: 디자인 특허)는 `None` → 분석 전에 제거

## <span style="color:#FEB99C;">5. DTM 설계: 초록을 숫자 행렬로</span>

**문서-단어 행렬(Document-Term Matrix, DTM)** 은 행이 문서, 열이 단어, 값이 등장 횟수인 행렬입니다.

| | method | device | first | ... |
|---|---:|---:|---:|---|
| 특허 1 | 2 | 0 | 3 | |
| 특허 2 | 0 | 1 | 1 | |

### 5-1. 과제의 전처리 기준과 `CountVectorizer` 설정

| 요구사항 | 설정 |
|---|---|
| 모두 소문자 | `lowercase=True` |
| 알파벳만, 3글자 이상 | `token_pattern=r"\b[a-zA-Z]{3,}\b"` |
| 단어 + bigram | `ngram_range=(1, 2)` |
| 불용어 제외 (NLTK) | `stop_words=stopwords.words("english")` |
| 상위 1,000개 | `max_features=1000` |

### 5-2. `token_pattern` 읽기

```text
\b [a-zA-Z]{3,} \b
│       │        └ 단어 경계
│       └ 알파벳이 3번 이상 연속
└ 단어 경계
```

- `CO2`, `5G`, `x-ray` 같은 토큰은 숫자나 기호 때문에 그대로 잡히지 않습니다.
- 특허 문서에서는 이런 기술 용어가 의미 있을 수 있으므로, **"alpha만"이라는 기준은 정보 손실을 감수한 선택**입니다.

### 5-3. Bigram은 어떻게 만들어질까?

> 🙋 **bigram은 불용어를 지우기 전에 만들까요, 지운 뒤에 만들까요?**
>
> `CountVectorizer`는 **토큰화 → 불용어 제거 → n-gram 생성** 순서로 처리합니다.
> 그래서 `"first and second"`는 `and`가 사라진 뒤 `"first second"`라는 bigram이 됩니다.
> 원문에 없던 단어 조합이 생길 수 있다는 점을 기억해 둡니다.

### 5-4. `max_features`는 "중요한" 단어가 아니라 "많이 나온" 단어

`max_features=1000`은 **전체 말뭉치에서 빈도가 높은 순서로** 1,000개를 고릅니다.

> ⚠️ 빈도가 높다고 정보가 많은 것은 아닙니다.
> 특허 초록에는 `first`, `second`, `configured`, `unit`처럼
> **청구항 작성 관행에서 나오는 상투어(boilerplate)** 가 매우 많습니다.
> 이 단어들은 상위 빈도를 차지하지만 특허의 내용을 거의 구분하지 못합니다.

## <span style="color:#FEB99C;">6. 단어와 가치의 상관: 무엇을 말해 주고, 무엇을 말해 주지 않을까?</span>

### 6-1. 단어 하나와 가치의 상관계수

각 단어 $j$에 대해, 문서별 등장 횟수 $x_{ij}$와 특허 가치 $y_i$의 **피어슨 상관계수**를 계산합니다.

$$
r_j=\frac{\sum_i (x_{ij}-\bar{x}_j)(y_i-\bar{y})}
{\sqrt{\sum_i (x_{ij}-\bar{x}_j)^2}\sqrt{\sum_i (y_i-\bar{y})^2}}
$$

- $r_j>0$: 그 단어가 많이 나오는 특허일수록 가치가 높은 경향
- $r_j<0$: 그 단어가 많이 나오는 특허일수록 가치가 낮은 경향

1,000개 단어마다 이 값을 계산해 정렬하면 **가장 양(+)·음(−)으로 상관된 단어**를 볼 수 있습니다.

### 6-2. 해석할 때 주의할 점

| 주의점 | 이유 |
|---|---|
| **상관 ≠ 인과** | "chip"이라는 단어를 쓴다고 특허 가치가 올라가지 않습니다. 반도체 산업과 기업 규모가 함께 움직일 뿐입니다. |
| **가치의 강한 왜도(skewness)** | 소수 특허의 가치가 매우 큽니다. 피어슨 상관은 극단값 몇 개에 크게 좌우됩니다. |
| **문서 길이 효과** | 등장 횟수를 그대로 쓰면 긴 초록일수록 모든 단어의 값이 커집니다. |
| **다중 비교** | 1,000개를 검정하면 우연히 큰 상관이 몇 개는 나옵니다. 상위 몇 개만 보고하면 착시가 생깁니다. |
| **작은 표본** | 한 주치 특허뿐이라 특정 기업의 특허 묶음이 결과를 좌우할 수 있습니다. |

> 💡 **개선 아이디어**
> - 가치에 로그를 취해 극단값 영향을 줄이기 (`np.log1p(xi_real)`)
> - 순위 기반 **스피어만 상관** 사용
> - 등장 횟수 대신 TF-IDF나 문서 길이로 나눈 비율 사용
> - 기업·산업을 통제한 회귀분석

> ⭐ **이 분석은 "어떤 단어가 가치를 만든다"가 아니라
> "가치 있는 특허들의 초록은 어떤 단어를 쓰는 경향이 있는가"를 보여 주는 탐색 분석입니다.**

---

# 🧪 2부. 실습 — KPSS와 USPTO 연결하기

> 1부의 개념을 실제 데이터에 적용합니다.
> **가치 데이터(KPSS) → 이름(CRSP) → 원문(USPTO) → DTM → 상관** 순서로 진행합니다.

---

## <span style="color:#FEB99C;">7. Step 1–2: 2022년 특허와 기업 이름</span>

```python
import pandas as pd
import numpy as np
import re

df = pd.read_csv("KPSS_2023.csv")

df["issue_date"] = pd.to_datetime(df["issue_date"], format="%Y%m%d")
df["filing_date"] = pd.to_datetime(df["filing_date"], format="%Y%m%d")

sub = df[df["issue_date"].dt.year == 2022].copy()
```

> 🙋 **왜 `format="%Y%m%d"`를 지정할까요?**
>
> 날짜가 `20221213` 같은 **8자리 정수**로 저장되어 있기 때문입니다.
> 형식을 알려 주지 않으면 pandas가 이 숫자를 날짜로 제대로 해석하지 못할 수 있습니다.

```python
names = pd.read_csv("crsp_names.csv")
names = names.drop_duplicates(subset="PERMNO", keep="last")

sub2 = pd.merge(
    sub,
    names[["PERMNO", "COMNAM"]],
    how="left",
    left_on="permno",
    right_on="PERMNO"
)

firm_value = (
    sub2.groupby("COMNAM")["xi_real"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
)
print(firm_value.head(20))
```

> 📝 과제에서는 평균만 출력했지만, 3절에서 본 이유로 **특허 수(`count`)를 함께 보는 편**이 해석에 안전합니다.

### 7-1. 결과: 특허당 평균 가치 상위 20개 기업

| 순위 | 기업 | 평균 `xi_real` |
|---:|---|---:|
| 1 | UNITEDHEALTH GROUP INC | 616.07 |
| 2 | PFIZER INC | 457.57 |
| 3 | LILLY ELI & CO | 417.13 |
| 4 | ABBVIE INC | 402.77 |
| 5 | NVIDIA CORP | 367.73 |
| 6 | DANAHER CORP | 355.07 |
| 7 | CHEVRON CORP NEW | 320.65 |
| 8 | THERMO FISHER SCIENTIFIC INC | 304.79 |
| 9 | NEXTERA ENERGY INC | 294.04 |
| 10 | SCHWAB CHARLES CORP NEW | 279.83 |
| 11 | NETFLIX INC | 273.99 |
| 12 | EOG RESOURCES INC | 241.63 |
| 13 | BRISTOL MYERS SQUIBB CO | 237.65 |
| 14 | COCA COLA CO | 235.12 |
| 15 | EXXON MOBIL CORP | 222.20 |
| 16 | BOOKING HOLDINGS INC | 215.23 |
| 17 | LINDE PLC | 212.06 |
| 18 | UNITED PARCEL SERVICE INC | 195.48 |
| 19 | STARBUCKS CORP | 182.11 |
| 20 | LOWES COMPANIES INC | 180.48 |

특허당 평균 가치 1위는 **UnitedHealth Group(616.07)** 입니다.

### 7-2. 이 순위를 어떻게 읽을까?

**① 1위가 건강보험·헬스케어 서비스 기업이다**
UnitedHealth는 특허를 중심으로 사업하는 기업이라고 보기 어렵습니다.
보유 특허가 적다면 몇 건의 값이 곧 평균이 되므로, **3절에서 본 "평균의 함정"** 이 작동했을 가능성이 있습니다.
그래서 특허 수(`count`)를 함께 확인해야 합니다.

**② 상위권은 대부분 초대형 기업이다**
제약(Pfizer, Eli Lilly, AbbVie, BMS), 반도체(NVIDIA), 에너지(Chevron, Exxon Mobil)처럼 시가총액이 매우 큰 기업이 상위를 차지합니다.
**1-1에서 본 것처럼 KPSS 가치는 주가 반응에 시가총액을 곱해 계산**하므로, 기업 규모가 그대로 반영된 결과입니다.

**③ 기술 기업이 아닌 곳도 상위에 있다**
Coca-Cola, Starbucks, Lowe's, Netflix, Booking처럼 기술 집약적이지 않은 기업도 순위에 올랐습니다.
이 순위는 **"가장 혁신적인 기업"이라기보다 "규모가 큰 기업의 특허"** 를 보여 준다고 읽는 편이 안전합니다.

## <span style="color:#FEB99C;">8. Step 3: XML 파싱</span>

```python
from zipfile import ZipFile
from lxml import etree

with ZipFile("ipg221213.zip") as z:
    with z.open(z.namelist()[0]) as f:
        contents = f.read()

contents_str = contents.decode("utf-8")
records = [r for r in re.split(r"(?=<\?xml)", contents_str) if r.strip()]

parser = etree.XMLParser(recover=True)

rows = []
for record in records:
    root = etree.fromstring(record.encode("utf-8"), parser=parser)
    if root is None:
        continue

    num_el = root.find(".//publication-reference//document-id//doc-number")
    abs_el = root.find(".//abstract")

    rows.append({
        "patent_num": num_el.text if num_el is not None else None,
        "abstract": " ".join(abs_el.itertext()).strip() if abs_el is not None else None
    })

patent_abstracts = (
    pd.DataFrame(rows)
      .dropna(subset=["patent_num", "abstract"])
      .query("abstract.str.strip() != ''", engine="python")
      .reset_index(drop=True)
)
```

> 🙋 **왜 `encode("utf-8")`로 다시 바이트로 바꿀까요?**
>
> 각 문서에 `<?xml version="1.0" encoding="UTF-8"?>` 선언이 있습니다.
> lxml은 **인코딩 선언이 있는 문자열(str)** 을 받으면 오류를 냅니다.
> 그래서 바이트로 다시 넘겨 선언에 맞게 해석하도록 합니다.

## <span style="color:#FEB99C;">9. Step 4: 특허 번호 맞추고 DTM 만들기</span>

### 9-1. 키의 형식부터 맞추기

USPTO의 번호는 문자열(`"11523456"`), KPSS의 번호는 정수(`11523456`)입니다.
**형식이 다르면 merge 결과가 0행**이 됩니다.

```python
patent_abstracts["patent_num_clean"] = (
    patent_abstracts["patent_num"].astype(str).str.replace(r"\D", "", regex=True)
)
patent_abstracts = patent_abstracts[patent_abstracts["patent_num_clean"] != ""].copy()
patent_abstracts["patent_num_clean"] = patent_abstracts["patent_num_clean"].astype(int)

merged = pd.merge(
    patent_abstracts,
    sub2,
    left_on="patent_num_clean",
    right_on="patent_num",
    how="inner"
).reset_index(drop=True)
```

> ⚠️ **숫자가 아닌 문자를 모두 지우면 생길 수 있는 일**
>
> 디자인 특허 번호(`D0971234`)에서 `D`를 지우면 `971234`가 되어,
> 이론적으로는 같은 숫자의 일반(utility) 특허와 섞일 수 있습니다.
> 2022년 KPSS 데이터의 일반 특허 번호는 1,100만 번대라 실제로 겹칠 가능성은 낮지만,
> 엄밀하게 하려면 **숫자로만 된 일반 특허 번호만 남기는 편**이 안전합니다.

> 💡 `reset_index(drop=True)`는 이후 **DTM의 i번째 행과 `merged`의 i번째 행이 같은 특허**가 되도록 맞추는 장치입니다.
> 상관을 계산할 때 이 정렬이 어긋나면 결과 전체가 의미를 잃습니다.

### 9-2. DTM과 빈도 상위 25개

```python
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer(
    lowercase=True,
    token_pattern=r"\b[a-zA-Z]{3,}\b",
    ngram_range=(1, 2),
    stop_words=stopwords.words("english"),
    max_features=1000
)
dtm = vec.fit_transform(merged["abstract"])

freq = pd.Series(
    np.asarray(dtm.sum(axis=0)).ravel(),
    index=vec.get_feature_names_out()
)
print(freq.sort_values(ascending=False).head(25))
```

> 🙋 **`dtm.sum(axis=0)`은 왜 `np.asarray(...).ravel()`로 감쌀까요?**
>
> 희소 행렬의 합은 `(1, 1000)` 모양의 **numpy matrix**로 나옵니다.
> 그대로는 Series의 값으로 쓰기 어려워서 1차원 배열로 펼칩니다.

### 9-3. 결과: 가장 많이 쓰인 25개 단어

초록과 가치가 모두 있는 특허는 **1,733건**이고, DTM은 `(1733, 1000)`입니다.

| 순위 | 단어 | 빈도 | 순위 | 단어 | 빈도 |
|---:|---|---:|---:|---|---:|
| 1 | first | 2,720 | 14 | layer | 723 |
| 2 | second | 1,922 | 15 | user | 707 |
| 3 | one | 1,385 | 16 | unit | 598 |
| 4 | includes | 1,353 | 17 | image | 583 |
| 5 | data | 1,337 | 18 | information | 575 |
| 6 | device | 1,323 | 19 | portion | 572 |
| 7 | may | 1,051 | 20 | include | 457 |
| 8 | system | 985 | 21 | surface | 450 |
| 9 | based | 914 | 22 | least one | 449 |
| 10 | method | 836 | 23 | including | 447 |
| 11 | plurality | 828 | 24 | network | 423 |
| 12 | configured | 757 | 25 | provided | 421 |
| 13 | least | 734 | | | |

### 9-4. 결과에서 보이는 것

**① 상위권은 특허 작성 관행의 상투어다**
`first`(2,720)와 `second`(1,922)가 1, 2위입니다. "a first layer and a second layer"처럼 구성요소를 번호로 구분하는 특허 문체 때문입니다.
`plurality`, `configured`, `portion`도 마찬가지로 **5-4에서 본 상투어(boilerplate)** 입니다.

**② `least one`은 불용어 제거가 만든 bigram이다**
원문은 "at least one"입니다. 불용어 `at`이 먼저 지워진 뒤 bigram이 만들어져 `least one`이 되었습니다.
**5-3에서 본 처리 순서**가 실제로 나타난 예시입니다.

**③ 같은 단어의 변형이 따로 집계된다**
`includes`, `include`, `including`이 각각 다른 열로 세어졌습니다.
어간 추출(stemming)이나 표제어 추출(lemmatization)을 하지 않았기 때문입니다.

**④ 기술 내용을 알려 주는 단어는 소수다**
`data`, `image`, `network`, `layer` 정도만 기술 분야를 짐작하게 합니다.
**빈도가 높은 단어일수록 특허를 서로 구분하는 힘은 약하다**는 점이 그대로 드러납니다.

## <span style="color:#FEB99C;">10. Step 5: 단어-가치 상관 결과</span>

```python
dtm_array = dtm.toarray()
value = merged["xi_real"].to_numpy()
terms = vec.get_feature_names_out()

corr_series = pd.Series(
    {term: np.corrcoef(dtm_array[:, i], value)[0, 1] for i, term in enumerate(terms)}
).sort_values()

print(corr_series.head(10))   # 가장 음(-)의 상관
print(corr_series.tail(10))   # 가장 양(+)의 상관
```

### 10-1. 결과 요약

| 구분 | 관찰 |
|---|---|
| **음(−)의 상관** | 약 −0.05 ~ −0.07, `first`·`second`·`unit`·`configured` 같은 상투어 |
| **양(+)의 상관** | 대부분 +0.10 ~ +0.20, `chip`·`processors`·`processes` 같은 반도체·컴퓨팅 용어 |
| **눈에 띄는 예외** | `test`가 약 0.37로 다른 단어보다 훨씬 큼 |

### 10-2. 어떻게 읽을까?

**① 전반적으로 상관이 약하다**
대부분의 계수가 작아서 **어떤 단어 하나로 가치를 예측하기는 어렵습니다.**
특허 가치는 초록의 문장보다 기술 자체와 그 기술을 가진 기업에 더 크게 좌우되기 때문입니다.

**② 양(+)의 단어는 직관적이다**
반도체·컴퓨팅 특허는 대형 기술 기업이 많이 보유하고, 1-1에서 본 것처럼 KPSS 가치는 시가총액이 클수록 커집니다.
**산업과 기업 규모가 단어와 가치를 동시에 움직였을 가능성**이 큽니다.

**③ 음(−)의 단어는 "정보가 없는" 단어다**
`first`, `second`, `unit`, `configured`는 **9-3의 빈도 상위권 단어와 겹칩니다.**
거의 모든 특허 초록에 나오는 단어라서 가치가 높은 특허와 낮은 특허를 구분하지 못합니다.
음의 상관이 아주 약하다는 것 자체가 **이 단어들에 가치에 대한 신호가 거의 없다**는 뜻으로 읽을 수 있습니다.

**④ `test` 0.37은 의심부터**
한 주치 표본에서 다른 단어와 동떨어진 값이 나왔다면, 특정 고가치 기업의 특허 몇 건에 `test`가 몰려 있었을 가능성이 큽니다.

> ⭐ **"가장 큰 상관"일수록 표본의 우연을 먼저 의심합니다.**
> 다른 주의 데이터로 다시 계산했을 때도 상위에 남는지 확인하는 것이 가장 확실한 점검입니다.

## <span style="color:#FEB99C;">11. 정리</span>

### 1부 개념

- **KPSS 특허 가치** = 특허 승인 시점의 주가 반응으로 추정한 경제적 가치 (상장 기업만, 시가총액 영향)
- **이름 이력 데이터** = 키가 중복되면 merge 후 행이 복제됨 → `drop_duplicates`
- **merge 방식** = left(왼쪽 유지)·inner(양쪽 공통)·outer(전체), merge 후 행 수 확인
- **평균의 함정** = 건수가 적은 그룹이 극단값을 차지함 → 건수를 함께 보고
- **USPTO XML** = 문서 수천 개가 이어 붙은 구조 → `<?xml` 기준 lookahead 분할
- **XPath** = `.//`는 깊이와 무관하게 탐색, 경로를 좁혀야 정확한 번호를 얻음
- **`itertext()`** = 하위 태그까지 포함한 전체 텍스트
- **DTM 설정** = 소문자·alpha 3자 이상·bigram·불용어·빈도 상위 1,000개
- **`max_features`** = 중요도가 아니라 빈도 기준
- **단어-가치 상관** = 탐색 분석. 인과·왜도·길이·다중 비교·표본 크기를 주의

### 2부 실습

- **평균 가치 1위** = UnitedHealth Group(616.07). 상위권은 시가총액이 큰 기업 → 규모 효과와 건수 확인 필요
- **분석 대상** = 초록과 가치가 모두 있는 특허 1,733건
- **빈도 상위 단어** = first·second·one·includes 등 특허 상투어가 차지
- **날짜 형식** = 8자리 정수는 `format="%Y%m%d"`로 변환
- **키 형식 통일** = 문자열 번호와 정수 번호를 맞춰야 merge됨
- **행 정렬 유지** = `reset_index(drop=True)`로 DTM과 데이터의 행 순서를 맞춤
- **결과** = 상관은 전반적으로 약함. 반도체 용어는 양(+), 상투어는 약한 음(−), `test`는 표본의 우연일 가능성

> 🎯 Task 1 한 줄 요약:
> **"텍스트를 경제 변수와 연결하는 첫걸음은 화려한 모델이 아니라, 정확한 데이터 결합과 신중한 상관 해석이다."**