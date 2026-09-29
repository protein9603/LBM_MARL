# 데이터 캐시 기록 (LDM → NPZ 캐시 + 슬랩)

생성일: 2026-09-29 (D1-6) · 생성 코드: `srcloc_env/scripts/convert_ldm.py` @ commit f6e784b · 실행 인터프리터: 시스템 Python 3.13.7 (numpy 2.4, scipy 1.16, vtk 9.6)

## 실행 명령
```
python -m srcloc_env.scripts.convert_ldm --index 400 599 --slab-z 15 12.5 17.5 --log cache/convert_400_599.log
python -m srcloc_env.scripts.convert_ldm --index 0 399 --slab-z 15 --log cache/convert_000_399.log
```

## 결과 (`F:\김도현박사님 자료\JH\cache\`)
| 항목 | 값 |
|---|---|
| 프레임 | 600 (index 0~599 = step 15025~30000), 오류 0 |
| `frames/frame_XXX.npz` | 공중 입자만(침적·x≥1315 제외): `xyz` float32 (N,3), `p_type` uint8 (= p_type − 100, 1~13); 합계 3.9 GB |
| `frames/meta_XXX.json` | 원본 개수(total/deposited/outflow/airborne, 소스별), 슬랩 통계, 소요 시간 |
| `slabs/slab_XXX.npz` | `density` float16 (13, n_z, 207, 200) 입자/m³, `z_levels`, `sources`, `grid`(x0 330, y0 −487.5, nx 200, ny 207, res 5); index 400~599는 z = 15/12.5/17.5 m, 0~399는 z = 15 m; 합계 1.1 GB |
| 공중 입자 수 | 1,709 (index 0) → 957,618 (index 599), 단조 증가 |
| 소요 | index 400~599: 1,488 s (약 7.4 s/프레임, 3레벨) · index 0~399: 558 s · 프레임당 중앙값 1.8 s, 최대 31.7 s(디스크 경합) |

## 사용 규칙
- 환경·센서는 `slabs/`를 이중선형 보간으로 읽는다(플랜 S0 T0-3에서 정확 gather와의 상관 ≥ 0.9 확인 예정).
- 정확 gather가 필요한 검증은 `frames/`(공중 입자)를 cKDTree로 다시 계산한다.
- T0-1(커널 재현)은 캐시가 아니라 원본 30000 프레임 전체 입자로 계산한다(마스크 없음).
- 캐시는 재생성 가능하므로 Git에 넣지 않는다(`.gitignore`의 `cache/`).