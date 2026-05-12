# 단위와 메모리 계산법

> kilo / mega / giga / tera 같은 SI prefix와 "100B 파라미터 모델을 FP16으로 저장하면 몇 GB?" 같은 기초 계산. 책 전체에서 반복 등장하므로 한 곳에 정리.

## SI prefix (K / M / G / T / P / E)

| 약자 | 풀네임 | 값 | 한국어 |
|---|---|---|---|
| K | kilo | 10^3 = 1,000 | 천 |
| M | mega | 10^6 = 1,000,000 | 백만 |
| G | giga | 10^9 = 1,000,000,000 | 십억 |
| T | tera | 10^12 | 일조 |
| P | peta | 10^15 | 천조 |
| E | exa | 10^18 | 백경 |

AI 분야에선 "십억(G)" 자리를 **B (billion)** 으로 자주 표기.

- 100B params = 100 × 10^9 = 1,000억 개 파라미터
- 1.8T params = 1.8 × 10^12 = 1조 8천억 개

처리율(FLOPS) 단위에는 G/T/P/E 사용:
- 1 TFLOPS = 10^12 FLOPs/s = 초당 일조 번 부동소수점 연산
- 1 EFLOPS = 10^18 FLOPs/s

## Bytes (메모리 단위)

- **1 byte = 8 bits**
- 메모리 / 스토리지 측정의 기본 단위
- KB / MB / GB / TB도 같은 prefix 적용 (1 GB = 10^9 bytes)

(엄밀히 1 KiB = 2^10 = 1024 bytes 같은 binary prefix도 있지만, GPU 카탈로그에선 보통 십진 prefix 사용)

## 정밀도별 1 파라미터 크기

bits를 8로 나누면 bytes:

| 정밀도 | bits | bytes |
|---|---:|---:|
| FP64 | 64 | 8 |
| FP32 | 32 | 4 |
| BF16 / FP16 | 16 | 2 |
| FP8 | 8 | 1 |
| FP4 | 4 | 0.5 |

(FP는 Floating Point, BF는 Brain Float)

## 모델 메모리 계산

```
메모리(bytes) = 파라미터 수 × bytes per parameter
```

### 예시 1: 100B (1,000억) 파라미터 모델 weights

| 정밀도 | 계산 | 결과 |
|---|---|---:|
| FP32 | 100×10^9 × 4 | **400 GB** |
| BF16 / FP16 | 100×10^9 × 2 | **200 GB** |
| FP8 | 100×10^9 × 1 | **100 GB** |
| FP4 | 100×10^9 × 0.5 | **50 GB** |

### 예시 2: 잘 알려진 모델들

| 모델 | 파라미터 | 정밀도 | 메모리 (weights만) |
|---|---|---|---|
| Llama-3 8B | 8 × 10^9 | BF16 | 16 GB |
| Llama-3 70B | 70 × 10^9 | BF16 | 140 GB |
| Llama-3 405B | 405 × 10^9 | BF16 | 810 GB |
| DeepSeek-V3 | 680 × 10^9 | FP8 | 680 GB |
| GPT-4 (추정 MoE) | ~1.8 × 10^12 | FP16 | ~3.6 TB |
| 100T (책의 aspirational 목표) | 10^14 | FP16 | 200 TB |

### 위는 weights만 - 학습 중엔 훨씬 큼

| 항목 | 대략 크기 |
|---|---|
| weights | W |
| gradient | W (같은 크기) |
| optimizer state (Adam) | 2W (1차/2차 모멘트) |
| activation | 배치 크기·시퀀스 길이에 비례, 클 수 있음 |
| KV cache (추론) | 시퀀스 길이·layer 수에 비례 |

학습 시 **GPU 메모리는 weights의 5-6배** 필요. 이래서 100B 모델조차 단일 GPU(192-288 GB HBM)에 안 들어가서 모델 분산 (FSDP / TP / PP / EP)이 필수.

## "Reduced precision" 정의

기본 FP32(전통 ML 학습 baseline) 대신 **더 적은 비트** 사용 (BF16 / FP16 / FP8 / FP4).

**Trade-off**:
- 장점: 메모리 ↓ / 속도 ↑ / 대역폭 ↑
- 단점: 수치 정확도 ↓

AI 학습/추론은 약간의 정확도 손실이 모델 품질에 크게 영향 안 주는 영역이라 잘 활용. 단 학습 시엔 **mixed precision** (계산만 low precision, master weights / optimizer state는 FP32 유지)이나 **quantization-aware training** 같은 안전 기법 필요.

## "정밀도"의 정확한 의미

소수점 자릿수와 비슷하지만 정확히는 **유효 숫자 자릿수 (significant digits)**:

| 정밀도 | mantissa bits | 유효 자릿수 | 예시 표현 |
|---|---:|---|---|
| FP32 | 23 | ~7자리 | 3.141593 |
| BF16 | 7 | ~2-3자리 | 3.14 (또는 3.13) |
| FP16 | 10 | ~3-4자리 | 3.141 |
| FP8 (E4M3) | 3 | ~1자리 | 3.1 (특정 값만 정확) |
| FP4 | 1 | 거의 없음 | 4 (가능한 값이 매우 적음) |

**소수점 위치는 exponent가, 정밀도는 mantissa가 결정**:
- exponent: 어디 자리수에 소수점이 있냐 (10진법으로 비유하면 "0.003" vs "3000")
- mantissa: 그 위치에서 유효 숫자를 몇 개 표현하나 ("3.14159" vs "3.14")

그래서 BF16은 exponent 8 bit로 FP32와 동일한 큰 범위를 표현하지만, mantissa 7 bit라 큰 숫자도 거친 단위로만 표현됨 (예: BF16에서 100과 100.5가 같은 값으로 표현될 수도).

## NVIDIA GPU 아키텍처 세대

GPU 카탈로그의 H100 / B200 같은 모델명은 **세대(아키텍처) 코드명** + 모델 번호 조합.

| 세대 | 출시 | 주요 GPU | AI 측면에서 새로 추가된 것 |
|---|---|---|---|
| Volta | 2017 | V100 | Tensor Core 최초 도입 |
| Turing | 2018 | T4 | INT8/INT4 가속 (추론용) |
| Ampere | 2020 | A100 | TF32 자동 가속, BF16 Tensor Core |
| **Hopper** | **2022** | **H100 / H200 / H800** | **FP8 Tensor Core, Transformer Engine** |
| Ada Lovelace | 2022 | RTX 40 series | (consumer / 게임 위주) |
| **Blackwell** | **2024** | **B200 / B300 / GB200** | **FP4 Tensor Core, NVLink 5, 더 큰 HBM** |
| Vera Rubin | 2026 (예정) | VR200 | 차기 세대 |
| Feynman | 2028 (예정) | - | 그 다음 |

세대 이름은 컴퓨터 과학/수학 분야의 위대한 인물에서 따옴 (Grace Hopper, David Blackwell 등).

### 왜 "Hopper는 FP8, Blackwell은 FP4"인가

세대마다 Tensor Core가 새로운 reduced precision을 **하드웨어로 지원**:
- 이전엔 SW로 흉내 가능했어도 가속 X
- 새 세대 GPU에서 그 정밀도 전용 회로 추가 -> 갑자기 빨라짐
- 그래서 "Hopper의 FP8", "Blackwell의 FP4"가 마케팅 키워드

같은 세대 안 모델 차이 (예: H100 vs H800)는 NVLink 대역폭 / FP64 throughput 같은 부수 차이. H800은 중국 수출용 제한판.

## GPU 모델 명명 규칙 (B vs GB vs NVL)

같은 Blackwell GPU도 세 가지 다른 패키징 단위로 등장:

| 약자 | 의미 | 구성 |
|---|---|---|
| **B** | Blackwell GPU **단독** | 칩 1개 (dual-die in package). 예: B100, B200, B300 |
| **GB** | **Grace** CPU + **B**lackwell GPU 결합 (Superchip) | 1 Grace CPU + 2 Blackwell GPU. 예: GB200, GB300 |
| **NVL** | **N**V**L**ink fabric으로 묶인 rack-scale 시스템 | GB Superchip 다수 + NVSwitch fabric. 예: NVL36, NVL72 |

스펙 비교 (Blackwell 기준):

| 항목 | B200 (GPU only) | GB200 (Superchip) | GB200 NVL72 (Rack) |
|---|---|---|---|
| CPU 포함 | ✗ (외부 x86 필요) | ✓ 1 Grace CPU | 36 Grace CPUs |
| GPU 수 | 1 | 2 | 72 |
| HBM 합계 | 192 GB | 384 GB | 13.5 TB |
| + CPU 메모리 | (외부 의존) | + 480 GB LPDDR | ~17 TB LPDDR |
| CPU↔GPU 연결 | PCIe (~64 GB/s) | NVLink-C2C (900 GB/s coherent) | + NVSwitch fabric |
| TDP | ~1000 W | ~2700 W | 120-132 kW per rack |
| FP4 peak | 9 PFLOPS | 18 PFLOPS | 1.44 EFLOPS (2:1 sparse) |
| 대표 form | HGX 보드 → x86 서버 | Superchip 단일 보드 | Full rack |

같은 패턴이 다른 세대에도:
- Hopper: H100 (GPU only), GH200 (Grace Hopper Superchip)
- Vera Rubin (예정): R200 (GPU), VR200 (Vera + Rubin Superchip)

### 왜 두 종류가 있나

- **B (HGX/DGX 보드)**: 기존 x86 데이터센터 표준. Intel/AMD CPU + 8× GPU 노드. 호환성 ↑.
- **GB (Superchip)**: NVIDIA ARM Grace CPU와 GPU 통합. CPU↔GPU 사이 PCIe 대비 14배 빠른 NVLink-C2C (900 vs 64 GB/s). Cache-coherent unified memory 효과. LLM 워크로드 (KV cache offloading 등)에 매우 유리.

## HGX vs DGX vs Superchip - 같은 GPU의 다른 패키징

같은 B200 GPU도 시장에 나오는 갈래가 여럿:

| 라인 | 정체 | 만든 곳 | 통합 책임 |
|---|---|---|---|
| **HGX 보드** | 8-GPU baseboard 모듈 | NVIDIA | OEM (Supermicro / Dell / HP / Lenovo) |
| **DGX 서버** | NVIDIA 완성 서버 (1U/4U) | NVIDIA | NVIDIA |
| **MGX** | 일반화된 modular reference platform | NVIDIA | OEM |
| **DGX SuperPOD** | DGX 다수 묶은 reference architecture | NVIDIA | 통합 어플라이언스 |
| **DGX Cloud** | 클라우드에서 운영되는 DGX 서비스 | NVIDIA | NVIDIA |
| **NVL72** | GB Superchip 18개 + NVSwitch rack | NVIDIA | NVIDIA |

### 비유

- **HGX = 자동차 엔진** (자동차 회사가 자기 차에 통합)
- **DGX = NVIDIA가 직접 만든 완성차**

### B200이 시장에 나오는 두 갈래

```
NVIDIA 본사
    │
    ├─ HGX B200 보드 (모듈) ──► Supermicro AS-A126GS-TNBR
    │                       ──► Dell PowerEdge XE9680
    │                       ──► HP / Lenovo / Inspur 등
    │
    └─ DGX B200 서버 (완제품) ──► NVIDIA 직판
```

GPU 자체 성능은 동일. 차이는 chassis 디자인, CPU 선택 (HGX는 OEM이 결정), NIC 슬롯 수, 가격, 지원 채널.

### 비교 한눈에

| | HGX | DGX |
|---|---|---|
| 만든 곳 | NVIDIA | NVIDIA |
| 판매 형태 | GPU baseboard만 | 완성된 서버 |
| 통합 책임 | OEM | NVIDIA |
| CPU 선택 | OEM (Intel/AMD) | NVIDIA가 결정 |
| 가격 | 보드 + OEM 마진 | 통합 가격 (더 비쌈) |
| 지원 | OEM 지원 | NVIDIA 직접 |
