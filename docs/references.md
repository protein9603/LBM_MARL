# 선행연구 출처 및 활용 기록

규칙: 코드에 선행연구의 방법·모델·상수를 넣을 때마다 이 표에 **무엇을, 어디에, 왜** 썼는지 기록합니다.
"검증 상태"는 서지 정보를 실제로 확인했는지 표시합니다 — `확인(URL)` = 링크로 확인, `기억 기반` = 서지 재확인 필요.
논문 작성 시 이 표를 관련연구·방법 절의 출발점으로 씁니다.

| # | 방법 / 개념 | 출처 | 본 연구에서의 활용 (모듈) | 채택 이유 | 검증 상태 | 기록일 |
|---|---|---|---|---|---|---|
| R1 | Wendland C6 커널 W(q) ∝ (1−q)⁸(1+8q+25q²+32q³), 3D 정규화 1365/(64π) | Dehnen, W. & Aly, H. (2012), "Improving convergence in smoothed particle hydrodynamics simulations without pairing instability", MNRAS 425(2):1068–1082 | 데이터의 `concn` 필드가 이 커널(지지반경 H = 7.5 m = 3 격자셀, 격자 단위 정규화)로 계산됨을 데이터 검증으로 확인(보고서 2.1). `sensor/detector.py`·`preprocess/gridder.py`의 gather에 동일 커널 사용 | 시뮬레이터와 **동일한** 공간 평활을 재현해 센서 모델과 데이터의 정의를 일치시키기 위해 | 기억 기반 (커널 식은 데이터로 R² 0.999999995 재현) | 2026-09-29 |
| R2 | SPH 수밀도 추정 n_i = Σ_j W(r_ij, h) | Monaghan, J. J. (1992), "Smoothed particle hydrodynamics", Annu. Rev. Astron. Astrophys. 30:543–574; Price, D. J. (2012), J. Comput. Phys. 231(3):759–794 | `concn`을 "같은 소스 입자에 대한 커널 수밀도 × 셀 부피 (2.5 m)³"로 해석(보고서 2.1); 드론 센서의 입자/m³ 환산 | 라그랑지안 입자 출력에서 국소 농도를 얻는 표준 방법 | 기억 기반 | 2026-09-29 |
| R3 | 포아송 계수 센서 모델 y ~ Poisson((k·C + b)·T) 및 이를 우도로 쓰는 베이지안 STE | Hutchinson, M., Oh, H., Chen, W.-H. (2018), "Entrotaxis as a strategy for autonomous search and source reconstruction in turbulent conditions", Information Fusion 42:179–189; Ristic, B., Skvortsov, A., Gunatilaka, A. (2016), "A study of cognitive strategies for an autonomous search", Information Fusion 28:1–9 | `sensor/detector.py`(D3 예정), `pf/particle_filter.py` 우도 | 핵종과 무관한 검출 통계로, 핵종은 감도 k에만 들어감(보고서 6.1 Q1). STE 문헌의 표준 관측 모델 | 확인(URL: sciencedirect S1566253517301811; semanticscholar 01390d9e…) | 2026-09-29 |
| R4 | 검출 하한(결정 임계) L_C = b + k√(b·T)/T 형태의 임계값 | Currie, L. A. (1968), "Limits for qualitative detection and quantitative determination", Anal. Chem. 40(3):586–593 | 검출 플래그 임계값 자리표시자(보고서 6.1) | 방사선 계측의 표준 검출 판정 | 기억 기반 | 2026-09-29 |
| R5 | Rao-Blackwellised particle filter(일부 상태를 해석적으로 주변화) | Doucet, A., de Freitas, N., Murphy, K., Russell, S. (2000), "Rao-Blackwellised particle filtering for dynamic Bayesian networks", UAI 2000 | `pf/particle_filter.py`: 위치 입자마다 방출률·감도 곱 κ를 로그 격자(또는 Gamma 켤레)로 주변화, PF 상태를 2차원으로 유지(보고서 6.2 Q2) | κ는 추정 목표가 아닌 nuisance이므로 주변화해야 위치 사후가 편향되지 않음 | 기억 기반 | 2026-09-29 |
| R6 | Gamma–Poisson 켤레(음이항 주변우도) | Gelman, A. et al. (2013), Bayesian Data Analysis, 3rd ed., Ch. 2 | RB-PF의 b = 0 빠른 경로(단위 테스트 T1-1) | 폐형 주변우도로 로그 격자 경로를 검증 | 기억 기반 | 2026-09-29 |
| R7 | PF 신념의 GMM 요약(가중치 정렬 K×파라미터 벡터)을 DRL 상태로 사용 | Park, M., Ladosz, P., Oh, H. (2022), "Source Term Estimation Using Deep Reinforcement Learning With Gaussian Mixture Model Feature Extraction for Mobile Sensors", IEEE Robotics and Automation Letters | `pf/gmm_summary.py`, `env/source_env.py` 관측 벡터(K×6) | 본 연구의 핵심 설계(PF → GMM → DRL)의 직접 선행연구 | 확인(URL: dblp journals/ral/ParkLO22) | 2026-09-29 |
| R8 | GMM 기반 Infotaxis(정보이득 최대 행동 선택) | Park, M., An, S., Seo, J., Oh, H. (2021), "Autonomous source search for UAVs using Gaussian mixture model-based infotaxis: algorithm and flight experiments", IEEE Trans. Aerospace and Electronic Systems 57(6):4238–4254 | `baselines/gmm_infotaxis.py` — 코드 검증용 참조 정책(공통 규칙 3에 따라 비교가 목적이 아님) | 같은 PF·GMM 위에서 동작하는 비학습 정책이라 환경·PF 코드 검증에 적합 | 확인(학회 목록 검색) | 2026-09-29 |
| R9 | Infotaxis | Vergassola, M., Villermaux, E., Shraiman, B. I. (2007), "'Infotaxis' as a strategy for searching without gradients", Nature 445:406–409 | 정보이득 보상 shaping의 개념적 근거(보고서 6.4) | 엔트로피 감소를 탐색 목적함수로 쓰는 원류 | 확인(URL: nature.com/articles/nature05464) | 2026-09-29 |
| R10 | 자율 종료(stop 행동) 문제 설정 | Shi, Y. et al. (2025), "Autonomous Goal Detection and Cessation in Reinforcement Learning: A Case Study on Source Term Estimation", AAAI 2025 (arXiv:2409.09541) | 성공 판정·stop 행동 설계(보고서 6.4) | 국소화 종료 시점을 정책이 결정하는 문제의 최근 선행연구 | 확인(URL: arxiv.org/abs/2409.09541) | 2026-09-29 |
| R11 | Proximal Policy Optimization | Schulman, J. et al. (2017), "Proximal Policy Optimization Algorithms", arXiv:1707.06347 | `rl/ppo.py` | 이산 행동·CPU 학습에 안정적인 표준 on-policy 알고리즘 | 기억 기반 | 2026-09-29 |
| R12 | 파라미터 공유 다중 에이전트 정책 학습 | Gupta, J. K., Egorov, M., Kochenderfer, M. (2017), "Cooperative Multi-agent Control Using Deep Reinforcement Learning", AAMAS 2017 Workshops (LNAI 10642) | `env/multi_agent.py`, `rl/ppo.py` — 한 정책이 모든 드론의 경험을 학습 | 20일 내 구현 가능한 가장 단순한 MARL 형태 | 기억 기반 | 2026-09-29 |
| R13 | Gaussian plume(지면 반사 포함) 전방 모델 | 표준 대기확산 교과서, 예: Seinfeld, J. H. & Pandis, S. N., Atmospheric Chemistry and Physics (3rd ed.), Ch. 18 | `pf/forward_model.py` — PF 우도의 해석적 예측 | 연속 소스 가설을 빠르게 평가; STE 문헌(R3, R7)이 같은 계열 모델을 사용 | 기억 기반 | 2026-09-29 |
| R14 | Legacy VTK 파일 형식(BINARY big-endian, POLYDATA, FIELD) | Kitware, "VTK File Formats" (docs.vtk.org) | `io/ldm_reader.py` | 데이터 파일 형식 그 자체 | 확인(공식 문서) | 2026-09-29 |

## 후속 기록 규칙
- 새 방법을 코드에 넣을 때 행을 추가하고 커밋 메시지에 `refs: R#`을 적습니다.
- "기억 기반" 항목은 논문 작성 전 서지(권·호·쪽·DOI)를 재확인해 `확인(DOI)`로 바꿉니다.
