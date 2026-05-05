# FLOPS와 Device Utilization

> Ch 1에서 책이 "FLOPS와 device utilization은 misleading하다"고 한 이유. 두 지표가 정확히 무엇을 말하고, 왜 못 믿는가.

## FLOPS

### 정의

- **FLOP** = floating point operation. 부동소수점 연산 1회 (덧셈 1번, 곱셈 1번, FMA(fused multiply-add)는 보통 2 FLOPs로 카운트)
- **FLOPS** = FLOPs per second. 초당 부동소수점 연산 횟수
- 표기 혼란 주의: FLOP은 단위 작업, FLOPS는 처리율. 둘 다 대문자 S로 끝나서 헷갈림

### 단위 스케일

| 약자 | 값 |
|---|---|
| GFLOPS | 10^9 |
| TFLOPS | 10^12 |
| PFLOPS | 10^15 |
| EFLOPS | 10^18 |

### 정밀도(precision)에 따라 FLOPS가 다르다

같은 GPU도 데이터 타입별 peak가 다름. Tensor Core는 reduced precision일수록 빠름.

| GPU | FP64 | FP32 | FP16/BF16 | FP8 | FP4 |
|---|---:|---:|---:|---:|---:|
| H100 SXM | 67 TFLOPS | 67 TFLOPS | 989 TFLOPS | 1979 TFLOPS | - |
| B200 | 40 TFLOPS | 80 TFLOPS | 2.25 PFLOPS | 4.5 PFLOPS | 9 PFLOPS |
| GB200 NVL72 (72 GPU) | - | - | 180 PFLOPS | 360 PFLOPS | 1.44 EFLOPS (2:1 sparse) |

(2:1 structured sparsity는 가중치 중 절반이 0이라고 가정하고 곱셈을 절반만 하는 트릭. dense FLOPS의 2배로 보고됨)

### 왜 misleading한가

- 카탈로그 FLOPS는 "**이론적 peak**" - 모든 SM이 동시에, Tensor Core 100% 채워서, 메모리 stall 없이 작동했을 때
- 실제 코드는 거의 도달 못 함. 이유:
  - 메모리 bandwidth 한계 (HBM에서 데이터 못 갖고 옴)
  - kernel launch overhead
  - 동기화 / barrier
  - 통신 (NCCL all-reduce 등) 동안 GPU idle
  - branching, 비효율적 access pattern
- Llama 405B를 H100에서 학습 시 MFU 38-43%, DeepSeek-V3는 ~50% MFU 보고. 즉 카탈로그의 절반 안 됨

## Device Utilization

### `nvidia-smi`의 "GPU-Util" 또는 NVML/DCGM의 utilization

#### 정의 (가장 흔한 함정)

- 지난 sampling interval(보통 1초 또는 100ms) 동안 GPU에서 **하나 이상의 kernel이 실행 중이었던 시간 비율**
- 다시 말해: "GPU가 일을 하고 있었나 idle이었나"의 매우 거친 binary 합산

#### 왜 misleading한가

- 1ms 짜리 kernel을 100ms마다 launch하면? -> kernel은 1% 시간만 쓰지만 sampling interval 안에서 measure하면 100% util로 보고될 수 있음
- 더 나쁜 경우: kernel이 메모리 stall로 99% 대기 중이어도 "실행 중"으로 카운트됨 -> 실제 SM은 idle인데 util은 100%
- 책 본문 인용: "device utilization is misleadingly high, as much of the time is likely spent on stalled communication, idling computation, or failed job restarts"

### 더 신뢰할 만한 sub-metric (DCGM)

| DCGM 메트릭 | 의미 | 목표 값 |
|---|---|---|
| `DCGM_FI_DEV_GPU_UTIL` | nvidia-smi와 동일. **거친 지표** | - |
| `DCGM_FI_PROF_SM_ACTIVE` | SM이 활성 상태였던 비율 | 높을수록 좋음 |
| `DCGM_FI_PROF_SM_OCCUPANCY` | active SM에서 warp 점유율 | 50%+ |
| `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | Tensor Core 사용 비율 | 학습/추론에서 60%+ 이상이면 좋음 |
| `DCGM_FI_PROF_DRAM_ACTIVE` | HBM 메모리 컨트롤러 활성 비율 | 패턴 따라 다름. memory-bound면 90%+ |
| `DCGM_FI_PROF_PCIE_TX_BYTES` / `_RX_BYTES` | PCIe 트래픽 | - |
| `DCGM_FI_PROF_NVLINK_TX_BYTES` / `_RX_BYTES` | NVLink 트래픽 | - |

`DCGM_FI_PROF_*` 시리즈가 진짜 실행 패턴을 보여줌. `DEV_GPU_UTIL`만 보고 "GPU가 일하고 있다"고 결론 내면 안 됨.

## MFU vs HFU

추가로 학계/산업계에서 쓰는 더 honest한 두 지표:

- **MFU (Model FLOPs Utilization)** = 모델 정의로부터 계산한 "있어야 할" FLOPs / peak FLOPS
- **HFU (Hardware FLOPs Utilization)** = 실제로 실행된 FLOPs (recompute, masked attention 포함) / peak FLOPS
- 보통 MFU < HFU (HFU는 redundant compute 포함)
- 둘 다 보고하면 "이 모델이 시스템을 얼마나 잘 활용하나" + "구현 오버헤드가 얼마나 되나" 분리 가능

## 응용/적용 아이디어

- 클러스터 모니터링 대시보드에서 `DEV_GPU_UTIL` 단독 차트는 함정. `PROF_SM_ACTIVE`, `PROF_PIPE_TENSOR_ACTIVE` 추가 패널 필요.
- Failure 직전 상태 진단 시: util 100%인데 throughput이 낮으면 SM stall 의심. PROF metric 함께 보면 원인 분리 가능.
- 학습 작업 efficiency 보고 시 MFU와 ETTR을 함께 표기하면 "GPU 활용도"와 "시스템 신뢰성"이 분리되어 명확해짐.

## 한 줄 요약

- FLOPS = 이론적 처리율 상한선. 실제는 30-50%만 도달
- Device utilization (GPU-Util) = "kernel이 떠 있었나" 정도. SM이 진짜 일하는지 모름
- 진짜 알고 싶으면: `DCGM_FI_PROF_SM_ACTIVE`, `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`, MFU
