# ICRS-15 발표 자료 개요 (초안 2026-10-04) — "상세 속도장 없이 multi-agent로 하는 Source Term Estimation"

제목(안): Multi-drone source localisation in an urban plume with a particle filter and shared-parameter PPO: how much of the wind field must the estimator know?

## 슬라이드 구성 (15분 기준 12~14장)
1. **문제**: 도시에서 방사성 물질 방출 → 드론 여러 대가 공기 중 농도를 재며 소스 위치를 찾는다. 현장 제약: 상세 바람장 계산(CFD/SPH)은 며칠 → 실시간 불가. 질문: 추정기가 바람장을 얼마나 알아야 하는가? [그림: 도시 장면 + 소스 13개, `fig_wind_fields.png` W2 패널]
2. **데이터(테스트베드)**: 김도현 박사 SPH(확인 후 표기) 유동 + LDM 입자 확산, 13개 소스, 600 프레임, 15 m 비행 고도 슬랩, 시간 변화 플룸(T2). 측정 = 포아송 계수 검출기. [그림: 플룸 스냅샷 2~3장(`fig_episode`의 배경)]
3. **방법 파이프라인**: 측정 → Rao-Blackwell PF(음이항 우도, 소스 강도 주변화) → GMM(K=3) 요약 → 파라미터 공유 PPO(2대) 또는 기준 정책. 성공 = σ < 30 m & 오차 < 50 m. [도식]
4. **바람 정보 수준 W0/W1/W2**: W0 바람 1점 + 가우시안 플룸; W1 바람 1점 + 건물 지도 → 질량 보존 포텐셜 유동(1~2 s) → adjoint; W2 CFD. 같은 파이프라인, 같은 에피소드. [`fig_wind_fields.png`]
5. **W1 모델**: 2-D 포텐셜 유동(Laplace, 건물 무투과, 원거리 조건), adjoint 이류–확산(K 16 m²/s, λ 0.005 s⁻¹). 검증: 질량 보존, 거울 대칭, 합성 진실 PF < 15 m. 한계: 후류·재순환 없음. [소스 바람 방향 비교 표]
6. **학습 방법**: 커리큘럼 A → B → B2(시작 거리 60~120 → 120~200 → 120~250 m, 소스 4 → 8), 관측 v2(신념 방향·거리·최근 측정), PPO 설정; 3시드. [`fig_training` 학습 곡선]
7. **기준 방법**: random, cross-wind lawnmower, greedy-MAP, GMM-Infotaxis, 오라클(검증용 상한). 평가: 새 에피소드 390개(소스당 30), 2대, 150스텝, Wilson 구간.
8. **결과 1 — PF 해석과 바람장**: 오라클 W0 47 / W1 54 / W2 52 %. 건물 지도 + 바람 1점이면 CFD와 같다. [`fig_wind_levels_final.png` 오라클 막대]
9. **결과 2 — 탐색 방법 × 바람 수준**: greedy/Infotaxis 8~12.5 %, PPO W0 13.5 / W1 6.0 / W2 7.9 %. 전체에서는 PPO = 기준 방법 수준. [`fig_wind_levels_final.png` 왼쪽]
10. **결과 3 — 학습에 쓰지 않은 소스**: W0 정책 26 % 대 기준 13~17 %(부호 검정 p ≤ 0.002, 3시드 재현); 소스 103: 기준 0/30, PPO 20/90. W2 정책에는 없음. 가설: 관측에 상세 바람이 없으면 신념 기하에 더 의존 → 특화 덜함. [`fig_wind_levels_final.png` 오른쪽 + `fig_episode_W0_src103_ep74_*.png` PPO 대 greedy]
11. **민감도·1대**: 풍향 ±10° 구간 안; W0 풍속 민감(느리게 잡는 쪽이 안전), W1 둔감; 1대는 모든 방법이 절반 이하. [민감도 표]
12. **영상**: `video/ppo_W0_src103_ep74.mp4`(홀드아웃 소스, 15 s), `greedy_W0_src103_ep74_fail.mp4`(같은 에피소드 실패) — 발표 중 재생 또는 QR.
13. **결론**: (1) 추정기는 바람 1점 + 건물 지도면 충분(며칠짜리 CFD 불필요); (2) 학습 정책은 상세 바람 없이도 동작하고 새 소스 일반화에서 기준 방법보다 낫다(W0); (3) 남는 격차는 탐색 경로(오라클 50 % 대 10~15 %) → 향후: 더 나은 탐색, 2대 이상, 고도 가변, ROM/진단 바람 모델(QES-Winds·URock) 결합.
14. **한계·향후**: 풍향 1개의 계산 1건(일반화 미검증), 15 m 단일 고도(103·106·107·110 관측 불가), W1 정책의 약점 미해결(진단 진행 중), 시드 3개.

## 발표 전 확인
- SPH/LBM 용어(김도현 박사 확인) → 2장·14장.
- 그림 300 dpi 재생성(`--dpi 300`), 표의 수치는 `docs/wind_knowledge_levels.md` 8장과 `cache/eval/final_*` summary.json.
- 참고문헌: `docs/references.md` R1~R44 중 발표에 쓰는 것(R29~R38, R42~R43 등) 서지 재확인.
