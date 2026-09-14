---
title : "(Week2) 웹에서 데이터 긁어오기 : 마크업 언어, 웹 스크래핑, XML, 그리고 API"
date : 2026-09-07 21:53:02 +0900
categories : [Graduate School, (FA26) Analysis of Unstructured Data]
tags : [MGT6033, Web Scraping, HTML, XML, JSON, API, BeautifulSoup, lxml, Requests, SEC EDGAR]
math : true
---

이번 주차(Week 2)에서는 **웹에 흩어져 있는 비정형 데이터를 어떻게 긁어와서(scraping) 구조화하는가**를 다뤘습니다.
세상에 존재하는 비정형 데이터의 상당수는 웹 위에 **마크업 언어(markup language)** 형태로 올라와 있습니다.
이걸 컴퓨터가 다룰 수 있게 접근하고, 태그 구조를 이용해 원하는 정보를 뽑아내는 게 이번 주차의 핵심입니다.
강의(2.1~2.4)와 실습 데모(SEC EDGAR, 미국 특허 데이터)를 쉬운 비유를 곁들여 풀어보겠습니다.

## <span style="color:#FEB99C;">1. 마크업 언어(Markup Language)란?</span>

**마크업 언어**는 "텍스트 문서 안에 기호(태그)를 삽입해서 그 **구조·서식·요소 간 관계**를 제어하는 텍스트 인코딩 시스템"입니다.
쉽게 말하면, 글자 사이사이에 "여긴 제목이야", "여긴 표야", "여기부터 굵게" 같은 **꼬리표(tag)** 를 붙여두는 것이죠.

- **태그(tag)** 는 꺾쇠 괄호 `< >`(부등호) 안에 넣습니다.
- 대부분 **여는 태그** `<h2>` 와 **닫는 태그** `</h2>`(슬래시 포함)가 짝을 이룹니다.

마크업 언어가 제어하는 것은 크게 4가지입니다.

| 요소 | 설명 |
|---|---|
| **구조(Structure)** | 표·문단·리스트 등 데이터가 어떤 형태인지 |
| **정렬(Alignment)** | 중앙 정렬? 배너? 사이드바? (모바일 반응형도 포함) |
| **서식(Formatting)** | 폰트 색·크기·굵기·기울임 등 (Word처럼 스타일 지정) |
| **관계(Relationship)** | 태그끼리 어떻게 연결되는가 (부모-자식 등) |

주요 마크업 언어를 정리하면:

| 언어 | 특징 |
|---|---|
| **HTML** (HyperText Markup Language) | 웹 페이지 표준. 페이지끼리 서로 링크(hypertext) |
| **XML** (eXtensible Markup Language) | 겉모습은 비슷하나 훨씬 유연. **데이터 저장·전송**이 목적 |
| **XBRL** | XML의 재무보고용 커스텀 버전 (규제기관이 taxonomy 관리) |
| **기타** | MathML, SGML, KML, LaTeX(논문 조판용) 등 |

> 핵심 포인트: **마크업은 결국 그냥 텍스트(text)입니다.**
> `.html`, `.xml` 파일도 텍스트 편집기로 열면 태그와 내용이 그대로 보입니다.

Python에서는 **접근(HTTP accessor)과 해석(parser)을 분리**해서 처리합니다.

- **requests** : 웹에서 데이터를 **가져오는** 패키지 (해석은 못 함)
- **Beautiful Soup** : 가장 널리 쓰이는 **HTML 파서** (내부적으로 다른 파서를 감싸는 wrapper라 일관된 API 제공)

## <span style="color:#FEB99C;">2. 웹 스크래핑의 기초와 윤리</span>

**웹 스크래핑(Web Scraping)** 은 여러 웹 페이지에서 데이터를 **자동으로 수집**하는 것입니다.
예를 들어 Reddit에 수백만 개 달린 제품 언급을 사람이 다 읽을 순 없으니, 스크립트로 자동화하는 것이죠.

교수님이 권장하는 스크래핑 프로세스는 다음과 같습니다.

1. 쿼리할 **링크 목록**을 확보 (이것 자체가 스크래핑 작업일 수도 있음)
2. `get` 메서드로 링크를 **가져오기** (`post`는 수정이라 우린 쓰지 않음)
3. **상태 코드(status code)** 확인 → 제대로 받았는지 검증
4. **저장 또는 처리** (원본 저장 + 가공을 분리하는 게 실무 팁)
5. **일시정지(pause)** → 초당 요청 제한(rate limit)을 지켜 차단 방지

> 💡 팁: 원본(raw)과 가공을 **분리**하면, 나중에 다른 정보가 필요할 때 웹을 다시 긁지 않아도 됩니다.

### HTTP 응답 코드

| 코드 | 의미 |
|---|---|
| **200** | ✅ 성공! 우리가 원하는 코드 |
| **3xx** | 리다이렉트 (새 페이지가 의도한 것인지 확인 필요) |
| **4xx** | **클라이언트(=나) 오류** — `404`(페이지 없음), `403`(접근 금지) |
| **5xx** | 서버 오류 (내 잘못 아님) — `503`(서버 이용 불가, 나중에 재시도) |

### 스크래핑의 윤리 (Robbie 교수님식 3원칙)

1. **모든 사이트를 긁어도 되는 건 아니다** → 루트 주소의 `robots.txt`와 이용약관을 확인
2. **사이트 규칙(특히 rate limit)을 지켜라** → 안 그러면 DoS 공격처럼 보일 수 있음
3. **가능하면 API를 써라** → 있다면 마크업 스크래핑보다 대개 더 나은 선택

## <span style="color:#FEB99C;">3. HTML 파싱 (1): 트리 구조와 재귀</span>

HTML은 **트리(tree) 구조**로 되어 있습니다.

| 용어 | 설명 |
|---|---|
| **부모(Parent)** | 어떤 태그를 감싸고 있는 상위 태그 |
| **자식(Child)** | 다른 태그 안에 열린 태그 |
| **형제(Sibling)** | 같은 부모를 공유하는 태그들 |

> ⚠️ 주의: HTML의 계층은 **들여쓰기로 결정되지 않습니다**(Python과 다름).
> 태그를 **다른 태그 안에서 여느냐**로 부모-자식 관계가 정해집니다. 들여쓰기는 그저 눈으로 보기 좋으라고 하는 것.

브라우저에서 아무 곳이나 **우클릭 → 검사(Inspect)** 하면 해당 부분의 HTML로 바로 이동할 수 있어, 스크래핑할 때 필수 도구입니다.

### 재귀(Recursion)

**재귀**는 "함수가 자기 자신을 호출하는" 기법입니다. 팩토리얼($3! = 3 \times 2!$)이 대표적 예시죠.
HTML 트리가 매우 깊어질 수 있어서, 파서는 트리를 오르내릴 때 재귀를 사용합니다.

> ⚠️ 함정: Python에는 무한 루프 방지를 위한 **재귀 한도(recursion limit)** 가 있어서,
> 복잡한 페이지를 파싱할 때 이 한도 때문에 에러가 날 수 있습니다. (한도 재설정, 정규식, 다른 파서 등이 해결책)

## <span style="color:#FEB99C;">4. HTML 파싱 (2): 태그의 속성(Properties)</span>

하나의 태그는 세 부분으로 구성됩니다.

```
<h2 class="headline heading">   ← 여는 태그 + 속성(attribute)
    Elevate your career...       ← 시각적 텍스트(visual text)
</h2>                            ← 닫는 태그
```

- **태그 타입(tag type)** : `h2`, `div`, `p` 등
- **속성(attribute)** : 여는 태그 안의 `키="값"` (예: `class="..."`, `id`, `href`, `style`)
- **텍스트(text)** : 여는 태그와 닫는 태그 사이의 실제 내용 (없는 태그도 많음)

주요 태그 타입:

| 종류 | 태그 | 역할 |
|---|---|---|
| **구조 태그** | `html`, `body`, `head`, `main` | 텍스트 없이 페이지 구조를 잡음 |
| **데이터 태그** | `div`(division/구역), `span`, `h1~h6`(제목), `p`(문단), `a`(링크), `table/tr/td` | 우리가 관심 있는 데이터가 담김 |

유용한 속성들: `style`(서식), `href`(URL), **`id`(페이지 내 유일 → 스크래핑의 보물!)**, `rows/cols`(표).

Beautiful Soup 사용법을 정리하면:

| 메서드 | 반환 |
|---|---|
| **`find`** | 조건에 맞는 **첫 번째** 태그 (없으면 `None`) |
| **`find_all`** | 조건에 맞는 태그들의 **리스트** (없으면 빈 리스트) |

> 핵심: 파싱된 Soup 객체의 **모든 태그 자체가 또 하나의 Soup 객체**입니다.
> 그래서 `tag1.find(...)` 처럼 특정 태그 안에서 자식을 다시 탐색할 수 있습니다.

## <span style="color:#FEB99C;">5. 실습 A: SEC EDGAR에서 8-K 공시 긁기</span>

첫 데모에서는 미국 증권거래위원회(**SEC**)의 공시 인덱스를 긁습니다.

### 첫 관문: 403 → 200 만들기

`requests.get`으로 바로 접근하면 **403(Forbidden)** 이 뜹니다.
기본 user-agent가 "Python requests"라 SEC가 차단하기 때문이죠.
SEC는 **자기 자신을 밝히라(declare yourself)** 고 요구하므로, 헤더에 **이메일 주소**를 넣어주면 됩니다.

```python
headers = {"User-Agent": "your_email@example.com"}
page = requests.get(url, headers=headers)  # → 200!
```

### 응답 객체(Response) 뜯어보기

| 속성 | 타입 | 내용 |
|---|---|---|
| `status_code` | int | 응답 코드 (200 확인용) |
| `content` | bytes | **인코딩된** 바이트 데이터 |
| `text` | str | **디코딩된** 문자열 (보통 이걸 파싱) |

인덱스 파일은 HTML이 아니라 **파이프(`|`) 구분 텍스트**라, `io.StringIO`로 버퍼를 만들어 `pd.read_csv(..., sep="|")` 로 바로 DataFrame으로 읽을 수 있습니다.

### 특정 정보 콕 집어 추출하기

인덱스에서 개별 8-K 공시의 **랜딩 페이지**(`-index.htm`)로 이동해 4가지를 뽑습니다.

| 추출 대상 | 전략 |
|---|---|
| **접수번호(Accession No.)** | `id="secNUM"` 로 검색 (id는 유일하므로 안전) → 텍스트 `.split()` 후 마지막 요소 |
| **접수 타임스탬프** | 유일 id가 없음 → **CSS selector** 또는 "Accepted" 텍스트를 찾아 `.next` 로 이동 |
| **아이템(Items)** | "Items" 헤더 태그를 찾아 `.next` 로 이동 |
| **산업분류(SIC)** | `id="filer_div"` → `p.identInfo` → `acronym` 태그 → `.next` × 3 |

> 🎯 파싱의 본질: 이건 **"추측하고(guess) 확인하기(check)"** 의 반복입니다.
> `.next`를 한 번, 두 번, 세 번... 눌러가며 원하는 태그에 도달하는 과정이죠.
> 그리고 문자열은 `strptime` 또는 pandas `to_datetime`으로 **datetime 객체**로 바꿔 필터링·메모리 효율을 높입니다.

### 마무리 실습: 모듈화 + apply로 100개 긁기

`sample(100, random_state=...)` 으로 **재현 가능한** 표본을 뽑고, 함수를 **모듈화**(접근 함수 / 추출 함수 / 마스터 함수)합니다.
`time.sleep(0.1)` 로 SEC의 "초당 10건" 제한을 지키고, 결과 dict 리스트를 `.to_list()` → DataFrame으로 변환.

> 📊 재미있는 발견: 공시가 가장 많이 제출되는 시각은 **16시(오후 4시)** — **장 마감 직후**입니다.
> 시장이 정보를 소화할 시간을 준 뒤 다음날 거래하도록 하는 것이죠.

## <span style="color:#FEB99C;">6. HTML vs XML, 그리고 JSON</span>

HTML과 XML은 둘 다 태그 기반의 중첩(nested) 구조지만, 목적이 다릅니다.

| 구분 | HTML | XML |
|---|---|---|
| 목적 | **표현(presentation)** | **데이터 저장·전송** |
| 태그 | 표준 taxonomy | **사용자 정의(custom)** → 태그명이 곧 데이터 설명 |
| 엄격함 | 느슨함 (닫는 태그 빠져도 렌더링) | **엄격함** (모든 태그 닫아야 하고 **대소문자 구분**) |
| Python 파서 | Beautiful Soup | **lxml** (XML엔 이게 더 나음) |

> 아이러니하게도 XML의 **엄격함(rigidity)** 이 오히려 데이터 추출을 **더 쉽게** 만들어 줍니다.

### JSON

XML은 종종 **JSON**(JavaScript Object Notation)으로 변환됩니다.

> **JSON은 구조적으로 Python 딕셔너리와 거의 동일합니다.**
> 그래서 JSON을 dict로 변환한 뒤 key-value로 접근하면 됩니다.
> 단, **중첩 JSON(nested JSON)** — dict의 값이 또 다른 dict인 경우 — 을 조심해야 합니다.

## <span style="color:#FEB99C;">7. 실습 B: XML 특허 데이터 파싱 (lxml)</span>

미국 특허청(USPTO)의 주간 특허 등록 데이터(2016년, zip 압축)를 파싱합니다.

### 데이터 로딩 & 트리 만들기 (feat. 에러 디버깅)

- **zip 파일**을 풀지 않고 Python `zipfile`로 **직접 읽기** (디스크 절약). 압축 파일은 대개 **바이너리(bytes)**.
- `etree.fromstring()` 실행 → **XMLSyntaxError!**
  - 에러 메시지: *"XML declaration allowed only at the start of the document, line 507"*
  - 진단 결과: 이 파일은 **하나의 트리가 아니라, 수천 개의 미니 트리(각 특허 = 개별 XML 레코드)** 가 텍스트로 쌓여있던 것!
  - 해결: `<?xml` 선언 기준으로 문자열을 **쪼개서** 각각을 개별 트리로 만들고, 리스트(**forest**, 나무들의 숲)에 담음.

### 트리 걷기(Walking a Tree)

XML은 HTML처럼 자유롭게 `find`할 수 없습니다(기본적으로 **바로 아래 자식만** 탐색).
그래서 **XPath** 라는 "언어 속의 언어"를 씁니다.

| XPath 표현 | 의미 |
|---|---|
| `.` | 루트(root) |
| `//document-id` | **어느 계층에서든** `document-id` 태그를 찾아라 (와일드카드 `//`) |
| `tag/` | 그 태그의 **직접 자식들** |

이렇게 각 특허 트리에서 문서ID(국가·번호·종류·날짜), 인용 특허 수, 도면 수, 청구항(claims) 텍스트, 발명자 수, 조직명 등을 dict로 수집합니다.

> ⚠️ 실무 처리: 태그가 **없을 수도** 있으니 `if sub is not None:` 같은 방어 코드가 필수.
> 결측을 대비해 dict 값이 없으면 건너뛰거나 빈 값으로 채웁니다.

### DataFrame으로 변환 & 분석

수집한 dict 리스트(6,095개)를 DataFrame으로 만들면 결측치도 자연스럽게 처리됩니다.
`value_counts()` 로 분석하면:

- 가장 흔한 특허 종류: **B2 (실용특허, utility patent)**
- 가장 많이 등장한 조직: **Samsung** (표기 정규화 안 해서 Samsung이 둘로 나뉘어 있었음)

> 🤔 생각해볼 질문: 발명자가 많을수록 특허가 더 혁신적일까?
> → 이 데이터 안에서 **상관관계(correlation)** 로 어떤 통찰을 줄 수 있을지 고민해보기.

## <span style="color:#FEB99C;">8. API (Application Programming Interface)</span>

**API**는 "컴퓨터끼리 소통하는 방법"입니다. 우리(Python 스크립트)가 상대 컴퓨터(웹 서버)에게 정해진 방식으로 정보를 요청하는 것이죠.

| 장점 | 단점 |
|---|---|
| **구조화된 데이터** 반환 | 항상 공개는 아님 (키가 필요할 수 있음) |
| 웹 스크래핑보다 **빠르고** 전송량 적음 | **레코드 단위** 접근 (한 번에 하나씩) → 트레이드오프 |
| 높은 rate limit | 전용 Python 패키지 학습 필요 |

> 예시: Glassdoor는 웹에선 리뷰 25~50개를 한 번에, API에선 **한 번에 하나만** 줍니다.
> 구조화는 좋지만 "1 쿼리 = 1 결과"라는 대가가 있죠.

**REST(ful) API** 는 표준화된 문법을 따르며, 대표 사례가 **SEC EDGAR API**입니다.
`https://data.sec.gov/submissions/CIK##########.json` 형식 — CIK(회사 식별자)를 **10자리**로 채워 넣으면 JSON을 반환.

## <span style="color:#FEB99C;">9. 실습 C: SEC API로 대량 데이터 수집</span>

### 단순 쿼리 (Tesla)

- CIK는 **10자리**여야 함 → 7자리만 주면 **404**! Python `str.zfill(10)` 으로 앞을 0으로 채움.
- 이메일 헤더 없이 요청하면 **403**.
- 둘 다 해결하면 → **200** 🎉

### JSON 파싱

`json.loads()`(load string) 로 문자열을 dict로 변환.
Tesla 데이터를 파고들면 **여러 계층 중첩**을 볼 수 있습니다.

> `record['filings']['recent']` → 딕셔너리, `record['filings']['files']` → 리스트
> 이 **부모-자식 중첩 구조**는 XML의 트리와 정확히 대응됩니다.
> 예: `record['filings']['files'][0]['filingFrom']` 처럼 계층을 타고 내려가 접근.

### 회사 리스트로 대량 수집

- `company_tickers.json` 에서 11,000+ 회사 목록 확보. dict의 **`.values()`** 를 바로 DataFrame으로!
- 회사별로 CIK·업종(SIC)·설립 주(state)·과거 사명·**총 공시 수**(recent + files 합산) 등을 수집하는 함수를 **모듈화**.
- **`enumerate`** 로 진행 상황 모니터링(50건마다 출력), **`time.sleep(0.1)`** 로 rate limit 준수.

> ⚠️ 함정 하나: pandas의 `.loc[:499]` 는 **500개**를 반환합니다(끝 포함).
> 다른 Python 인덱싱과 달라서 헷갈리기 쉬우니 주의!

### 결과 분석 (운영 법인 기준)

- 공시 최다: **JP Morgan Chase (9만 건) > Morgan Stanley > Citigroup** → 금융권이 규제 공시가 많음
- 리스트형 컬럼(`former_names`)은 pandas **`.str` 메서드**로 길이 등을 뽑아 새 컬럼 생성

## <span style="color:#FEB99C;">10. 결론</span>

> 이번 주의 핵심은 **"웹의 태그 구조(마크업)를 이해하고, 접근(requests/API)과 해석(BeautifulSoup/lxml)을 나눠 원하는 데이터만 뽑아내는 것"** 입니다.
> 접근 → 상태 확인 → 파싱 → 구조화(DataFrame)로 이어지는 파이프라인이 그 뼈대입니다.

이번 주에 배운 것을 정리하면:

- **마크업 언어** = 태그로 구조·서식·관계를 제어하는 텍스트. HTML(표현)과 XML(저장·전송)이 양대 산맥
- **웹 스크래핑** = 수집(requests) + 구조화의 2단계. `200`을 확인하고, `robots.txt`·rate limit 등 **윤리**를 지킬 것
- **HTML 파싱**: 부모-자식-형제 **트리 구조**, `find`/`find_all`, `id` 같은 속성이 핵심 열쇠
- **XML 파싱**: 더 엄격하지만 그래서 추출이 쉬움. **lxml + XPath(`//` 와일드카드)** 로 계층 탐색
- **JSON ≈ Python 딕셔너리**, 중첩 구조 주의 (`json.loads`)
- **API**: 구조화·고속의 장점 ↔ 레코드 단위·키 필요의 단점. SEC EDGAR는 **키 없이(이메일 헤더만)** 쓰는 RESTful API
- 실무 팁: **모듈화(함수 분리)**, `time.sleep`으로 pause, `enumerate`로 모니터링, 방어적 결측 처리, `zfill`/`to_datetime` 같은 잔기술
