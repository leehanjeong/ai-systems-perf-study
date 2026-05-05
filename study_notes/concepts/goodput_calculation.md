# Goodput 실제 계산법

> Ch 1에서 goodput이 가장 중요한 지표라고 했지만 "어떻게 계산하나"는 자세히 안 나옴. 이 노트는 학습/추론 각각의 실제 계산 공식과 worked example.

## 일반 정의

```
goodput = useful_work / wall_clock_time
efficiency = goodput / theoretical_peak
```

핵심은 **"useful"의 정의**. 무엇을 빼고 무엇을 포함할지가 모든 계산의 출발점.

## Training goodput

### 형태 1: Token throughput 기반 (가장 직관적)

```
goodput_tokens_per_s = useful_tokens_processed / wall_clock_time
```

useful_tokens 정의에서 빠져야 할 것:
- 실패 후 rollback된 step의 토큰 (체크포인트 이후 다시 학습한 분)
- 의도하지 않은 reprocessing

wall_clock_time에 들어가는 것 (즉 효율을 깎는 것):
- 데이터 로딩 stall
- 통신 stall (NCCL all-reduce 대기)
- 체크포인트 save/load 시간
- failure 후 재시작 + 노드 재할당 + 환경 셋업
- queue/scheduler 대기 시간

### 형태 2: Meta의 ETTR (Effective Training-Time Ratio)

책에서 인용된 Meta 2025 paper "Revisiting Reliability"의 정의:

```
ETTR = productive_training_time / total_wall_clock_time
```

`productive_training_time` = 실제 useful gradient update가 일어난 시간

빠지는 시간 (lost time):
- Job preemption / queue wait (제출했지만 실행 못 함)
- Hardware fault detection + restart
- Network congestion으로 인한 step 시간 늘어난 부분
- Failure 후 rollback으로 손실된 학습 진척 만큼의 시간
- Non-productive overhead (logging, validation 등은 grey zone)

Meta 보고: 대규모 클러스터에서 ETTR이 25-30%까지 떨어짐. 즉 nominal "100% utilized"여도 70-75% 시간이 wasted.

### 형태 3: MFU (Model FLOPs Utilization)

```
MFU = (useful_model_flops / wall_clock_time) / peak_flops_per_s
useful_model_flops = 6 * params * useful_tokens   (dense LLM 학습 근사)
```

업계 보고치:
- Llama-3 405B 학습 (16K H100): ~38-43% MFU
- DeepSeek-V3 학습 (H800): ~50% MFU
- 잘 튜닝된 small model: 60%+ MFU

MFU와 ETTR은 다름:
- MFU: 동작하는 동안 GPU를 얼마나 잘 활용하나 (kernel-level efficiency)
- ETTR: 그 "동작" 자체가 전체 시간 중 얼마나 되나 (system-level reliability)
- 진짜 효율 = MFU * ETTR (대략적으로)

## Worked example - 4시간 학습 중 GPU failure

### 시나리오 (hypothetical)

- 의도: 4시간 학습
- 8 GPU H100 (BF16) -> peak 8 * 989 TFLOPS = 7.9 PFLOPS
- 7B params dense model
- batch당 처리 토큰 1M tokens, step 시간 5s 가정 -> 초당 ~200K tokens 가능
- 진행 중 2.5시간 후 GPU failure 발생
- 마지막 체크포인트는 2:00에 저장됨

### 시간 분해

| 시점 | 누적 시간 | 이벤트 |
|---|---|---|
| 0:00 | 0 s | 시작 |
| 2:30 | 9000 s | GPU failure 발생 |
| 2:40 | 9600 s | torchrun timeout 종료 |
| 2:45 | 9900 s | 노드 재할당 + 환경 셋업 |
| 2:50 | 10200 s | 체크포인트 로드 (마지막 저장: 2:00 시각) |
| 4:00 | 14400 s | 의도된 종료 시각 |

### useful tokens 계산

- 2:00까지 저장된 학습량 = 7200 s * 200K tokens/s = 1.44e9 tokens
- 2:00 ~ 2:30 동안 학습한 1800s * 200K = 3.6e8 tokens는 rollback으로 손실
- 재시작 후 2:50 ~ 4:00 = 4200 s 동안 200K tokens/s로 학습 가정 = 8.4e8 tokens
- 단 처음 30분은 이미 했던 부분 다시 학습 (1800 s * 200K = 3.6e8 tokens, redundant)
- 4:00 시점에서 2:00 + 1:10 = 3:10 분량까지 학습 = 11400 s * 200K = 2.28e9 tokens

### Goodput 지표

```
useful_tokens = 2.28e9 (실제로 학습 진척에 기여)
wall_clock = 14400 s
goodput = 158K tokens/s
peak = 200K tokens/s   (이 시스템의 speed of light)
ETTR = 11400 / 14400 = 79%
```

여기서 21%가 잃어버린 시간:
- 600 s (timeout) + 600 s (재시작) + 1800 s (rollback) = 3000 s lost
- 3000 / 14400 = 20.8%

(나머지 1% 정도는 통신/IO stall 등 micro-overhead. MFU는 또 다른 별도 차원에서 깎임)

### 비용 환산

- 8 GPU H100 hourly cost ~$32 (cloud 추정)
- 4시간 = $128
- Lost 21% = $26.88 의 학습 비용이 wasted
- 클러스터에서 한 달 50 sessions, 10% failure rate 가정 -> 5 failures * $27 = $135/month per cluster
- 1000 노드 클러스터로 scale하면 한 달 수만 달러 손실

## Inference goodput

### 정의 (SLO-aware)

```
goodput_req_per_s = successful_requests_within_SLO / wall_clock_time
```

`successful_requests_within_SLO` 의 조건:
- 응답 완료
- TTFT (Time to First Token) < SLO threshold (예: 1초)
- 전체 응답 latency < SLO threshold

SLO를 어긴 요청은 "수행했어도 안 한 것과 같다"고 보는 것이 goodput 사고방식.

### 단순 throughput과의 차이

```
raw throughput = total_requests / time     (모든 요청 카운트)
goodput        = SLO-met requests / time   (SLO 안 어긴 것만)
```

Raw throughput을 쥐어짜다 latency가 망가지면 goodput은 오히려 떨어짐. 적정 동시 요청 수(concurrency) tuning의 출발점.

### Token-level inference goodput

```
output_tokens_per_s_useful = sum_over_SLO_met_requests(output_tokens) / time
```

Disaggregated prefill-decode 시스템(책 Ch 17)에서는 prefill goodput, decode goodput을 분리 측정.

## Theoretical peak 계산

### Training peak tokens/s

```
peak_tokens_per_s = peak_flops_per_s / flops_per_token
flops_per_token (dense) = 6 * params
```

예: 8 H100, 7B model, BF16
- peak = 8 * 989 TFLOPS = 7.9e15 FLOPs/s
- flops_per_token = 6 * 7e9 = 4.2e10
- peak tokens/s = 7.9e15 / 4.2e10 = 188K tokens/s

### Memory bandwidth bound 가능성

특정 워크로드(decode, attention)는 compute가 아닌 HBM bandwidth가 bottleneck. peak 대신:
```
peak_tokens_per_s = HBM_BW / bytes_per_token
```
H100 HBM = 3.35 TB/s. Llama-3 8B FP16 decode 시 토큰당 ~16GB 메모리 read -> 3.35TB/16GB = 209 tokens/s 정도가 한계.

이래서 책이 "compute-bound vs memory-bound 진단"을 강조함.

## 응용/적용 아이디어

- 학습 작업 보고서에 ETTR 컬럼 추가하면 "시간 손실"이 분야 표준 지표로 표현됨
- Failure 영향을 시간/토큰/달러/FLOPs 네 단위로 환산하면 청중별로 임팩트 전달 가능
- ETTR과 MFU 분리: 시스템 신뢰성과 GPU 활용도가 별개 차원이라는 인식이 디버깅 시 출발점이 다름
- 추론 시스템에선 raw throughput이 아니라 SLO-aware goodput으로 모니터링해야 진짜 사용자 만족도 반영

## 한 줄 요약

- Goodput = useful_work / wall_clock_time. **useful이 무엇인지 정의가 핵심**
- Training: ETTR = productive_time / total_time (Meta), MFU = useful_flops / peak_flops
- Inference: SLO-met requests / time (latency 어긴 건 카운트 안 함)
- 진짜 효율 = MFU * ETTR
