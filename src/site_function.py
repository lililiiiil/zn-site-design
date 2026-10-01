#!/usr/bin/env python3
"""
site_function.py — (E) 레이어: 배위 자리가 '촉매를 할 수 있는가'를 채점.

metal_site_score.py 의 (A)~(D)는 Zn을 얼마나 잘 '잡는가'만 본다.
이 모듈은 그 자리가 carbonic anhydrase처럼 '일을 할 수 있는가'를 본다.

  1. 열린 4번째 배위 자리 (촉매 물이 앉을 곳)  → water_site_open()
  2. 2차 배위권 (Thr199 등가물)                → second_shell()
  3. 소수성 기질 포켓 (CO2 도킹)               → substrate_pocket()
  4. 벌크 용매 통로 (양성자 배출)              → solvent_access()
  5. CO2 가 앉을 공간이 비었는가               → co2_open_frac()

사용:
  import site_function as sf
  sf.function_metrics(atoms, coords, metal_xyz)      # dict 반환

자기검증:
  python site_function.py <renum.pdb> ...
"""
import numpy as np

PROBE = 2.0          # Zn–촉매물 거리(Å)
# 물 자리 충돌 판정 — 원자 종류별로 다르게. 균일 3.2Å 를 쓰면 Thr199 OG1
# 같은 정상 수소결합 파트너(2.6~2.8Å)가 '차단원자'로 잡힌다. 그러면 그 모델은
# '막힌 자리'로 분류되어 열린 모델 집합에서 빠지고, 남은 모델들은 정의상
# 3.2Å 안에 극성 원자가 없으므로 shell_hbond_n 이 0 이 될 수밖에 없다.

HBOND_CUT = 3.4      # 2차 배위권 수소결합 상한(탄소기준)
RAY_CUT = 2.8        # 광선이 원자에 이만큼 가까워지면 막힘
SELF_TOL = 0.1       # 배위원자 자기 자신 판정 허용오차
HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "MET", "PRO"}
# DONOR_ATOMS 는 더 이상 쓰지 않는다. 원자 '이름' 목록은 공여체와 수용체를
# 구분하지 못하고, 새 잔기가 나올 때마다 손으로 늘려야 한다.
# 판정은 원소(N/O) + 기하(거리·각도)로만 한다. 참고용으로만 남긴다.

BACKBONE = {"N", "CA", "C", "O"}


# ── 새 상수 ───────────────────────────────────────────────────────────
# Bondi vdW 반지름(Å). 
VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}
R_WATER = 1.52          # 물 산소 자신의 vdW 반지름

# 수소결합 중원자 거리 창(Å). 아래는 진짜 충돌, 위는 너무 멀어 무관.
HB_LO, HB_HI = 2.4, 3.4
# Zn–Owat···X 각도 창(도). sp3 산소라 109° 근처가 정상.
HB_ANG_LO, HB_ANG_HI = 80.0, 140.0
# ※ 네 숫자 전부 미보정. 2CBA 에서 실측한 뒤 좁힐 것.


def steric_cut(elem):
    """
    물 산소와 원소 elem 원자 사이의 입체 충돌 하한(Å).
    두 원자의 vdW 반지름 합보다 가까우면 물리적으로 겹친 것이다.
    표에 없는 원소는 탄소로 취급(가장 큰 유기 원소 쪽으로 보수적).
    기대값: C → 3.22 · N → 3.07 · O → 3.04 · S → 3.32
    """
    # 정규화는 호출부(classify_neighbors)에서 한 번만 한다. 여기서 또 하면
    # 규칙이 두 군데로 갈라진다. 직접 부를 땐 대문자 원소기호를 넘길 것.
    return R_WATER + VDW.get(elem, VDW["C"])

def _label(a):
    """잔기 라벨. renumber_site.py 의 +1000 오프셋을 되돌려 표시."""
    r = a["resi"]
    n = a["name"]
    return f"{a['resn']}{r - 1000 if r > 1000 else r}"


def fourth_vertex(coords, metal_xyz):
    """배위원자 3개가 만든 사면체에서 비어 있는 네 번째 방향(단위벡터)."""
    V = coords - metal_xyz
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    w = -V.sum(0)
    n = np.linalg.norm(w)
    return None if n < 1e-6 else w / n


def _water_pos(coords, metal_xyz):
    """가상 촉매 물 좌표. 방향이 정의 안 되면 None."""
    w = fourth_vertex(coords, metal_xyz)
    return None if w is None else (metal_xyz + PROBE * w, w)


def _coord_atom_mask(P, coords):
    """P 각 행이 배위원자 자신인지 (N,) 불리언.
    SELF_TOL 매칭 규칙은 이 함수 하나에만 있다."""
    return np.linalg.norm(P[:, None, :] - coords[None, :, :], axis=2).min(1) <= SELF_TOL


def _protein(atoms, coords, drop_self=True):
    """단백질 원자만. drop_self면 배위원자 자신은 뺀다."""
    prot = [a for a in atoms if a["rec"] == "ATOM"]
    if not prot:
        return [], np.empty((0, 3))
    P = np.array([a["xyz"] for a in prot])
    if drop_self:
        keep = ~_coord_atom_mask(P, coords)
        prot = [a for a, k in zip(prot, keep) if k]
        P = P[keep]
    return prot, P


def coordinating_residues(atoms, coords):
    """배위원자가 속한 잔기 (chain, resi) 집합.

    왜 필요? 2차 배위권은 정의상 '배위에 관여하지 않는 별개 잔기'다.
    His96 이 NE2 로 배위하면 ND1 은 2.2Å 옆에 남아 가상 물에서 2.6~3.1Å 에
    앉는다. 원자 단위 제외로는 안 빠져서 '2차 배위권 파트너'로 세어진다.
    """
    prot, P = _protein(atoms, coords, drop_self=False)
    if not prot:
        return set()
    mask = _coord_atom_mask(P, coords)
    return {(a["chain"], a["resi"]) for a, k in zip(prot, mask) if k}

# ── 통합 분류 ─────────────────────────────────────────────────────────
def classify_neighbors(atoms, coords, metal_xyz):
    """
    물 자리 주변 단백질 원자를 '충돌 / 수소결합 / 무관' 중 정확히 하나로 배정.

    이 함수가 존재하는 이유: 예전엔 water_site_open 과 second_shell 이 각자
    다른 컷오프로 같은 원자를 따로 판정했다. 그래서 3.2~3.4Å 사이에만 수소결합이
    잡히는 구멍이 생겼고, Thr199 처럼 2.7Å 에 있는 정상 파트너는 '차단원자'가 됐다.
    분류를 한 곳에 모으면 두 지표가 다시 어긋나는 게 구조적으로 불가능해진다.

    반환: dict 또는 None
      None      물 자리 방향이 정의 안 됨(_water_pos 가 None) 또는 단백질 원자 0개
      clash     [(침범량, 거리, atom), ...]  침범량 = 거리 - steric_cut, 오름차순
                (거리순이 아니라 침범량순인 이유: 큰 원자가 살짝 닿은 것보다
                 작은 원자가 깊이 박힌 게 더 심각하다)
      hbond     [(거리, 각도, atom), ...]  거리 오름차순
      min_dist  물 자리에서 가장 가까운 단백질 원자까지 거리(float)

    불변식: 같은 atom 이 clash 와 hbond 에 동시에 들어가지 않는다.
            수소결합 자격을 통과한 원자는 충돌 판정에서 면제된다.

    분류 절차:
      1. _water_pos 로 가상 물 좌표 wat 를 얻는다. None 이면 None 반환.
      2. _protein(atoms, coords) 로 배위원자 자신을 뺀 단백질 원자를 얻는다.
      3. 원자마다:
         a. 원소를 뽑는다.  a.get("elem") or a["name"][0]
            수소는 건너뛴다 (AF3 출력엔 없지만 native PDB 엔 있을 수 있다)
         b. 극성(N/O)이고 HB_LO <= d <= HB_HI 이면 각도를 잰다.
            각도 = (metal_xyz - wat) 와 (P - wat) 사이의 각
            HB_ANG_LO <= 각도 <= HB_ANG_HI 이면 hbond 에 넣고 '다음 원자로 넘어간다'
         c. 여기까지 왔으면 d < steric_cut(원소) 인지 본다. 맞으면 clash.
      4. 각각 정렬해서 반환.

    주의: 공여체 원자 '이름' 으로 거르지 않는다(DONOR_ATOMS). 이름 목록은
          공여체와 수용체를 구분 못 하고, 원소 + 기하가 더 정직하다.
    """
    HEAVY_SKIP = {"H", "D"}

    if not any(a["rec"] == "ATOM" for a in atoms):
        return None                       # PDB 에 단백질 원자가 없음 = 파싱 실패 신호

    wp = _water_pos(coords, metal_xyz)
    if wp is None:
        return None                       # 4번째 꼭짓점 방향이 정의 안 됨
    wat, _ = wp  # wat — 가상 물 위치. 배위원자 3개 방향의 합을 뒤집어서 금속에서 2.0 Å 떨어진 지점. 4번째 배위 자리

    # drop_self=True 는 선택이 아니라 필수.
    # 이상적 사면체에서 N···Owat = 3.31 Å (코사인법칙: 2.05 / 2.00 / 109.47°)
    # → HB_HI(3.4) 안쪽이라 배위 N 자신이 수소결합 파트너로 세어진다.
    prot, P = _protein(atoms, coords, drop_self=True)

    u = metal_xyz - wat
    u = u / (np.linalg.norm(u) + 1e-9)    # Zn 방향 단위벡터. 각도의 기준축.
    coord_res = coordinating_residues(atoms, coords)

    clash, hbond, dists = [], [], []
    for a, x in zip(prot, P):
        is_coord = (a["chain"], a["resi"]) in coord_res
        elem = (a.get("elem") or a["name"][0]).strip().upper()
        if elem in HEAVY_SKIP:
            continue                      # 수소는 vdW 표에 없어 전부 탄소로 폴백된다
        # d = 그 단백질 원자 x에서 가상 물까지의 거리
        d = float(np.linalg.norm(x - wat))
        dists.append(d)

        # (b) 수소결합 자격 심사. 통과하면 충돌 판정에서 '면제'된다.
        #     면제는 원소가 주는 게 아니라 기하가 획득하는 것 — 이게 핵심이다.
        #     원소로 무조건 면제하면 175°(친핵 공격 축)에 박힌 산소가
        #     '막는 원자'가 아니라 '무관'으로 빠져나간다.
        #     not is_coord: 2차 배위권은 정의상 '배위에 관여하지 않는 별개 잔기'다.
        #     His96 이 NE2 로 배위하면 ND1 이 2.2Å 옆에 남아 파트너로 세어진다.
        #     ※ 이 조건은 hbond 에만 붙고 clash 에는 일부러 안 붙인다.
        #       배위 His 곁사슬이 물 자리를 막는 건 '배위 자리 붕괴'의 진짜 증거다.
        if elem in ("N", "O") and not is_coord and HB_LO <= d <= HB_HI:
            v = x - wat
            cos = float(u @ v / (np.linalg.norm(v) + 1e-9))
            ang = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
            if HB_ANG_LO <= ang <= HB_ANG_HI:
                hbond.append((d, ang, a))
                continue                  # ← 이 continue 가 상호배타를 보장한다

        # (c) 자격 미달이면 원소 그대로의 vdW 합으로 충돌 판정.
        # cut = 원소 기준 충돌 하한 
        cut = steric_cut(elem)
        if d < cut: # 겹침, 침범량
            clash.append((d - cut, d, a))  # 침범량은 음수. 작을수록 깊이 박힌 것

    clash.sort(key=lambda c: c[0])         # 침범량 오름차순 = 가장 깊은 것이 [0]
    hbond.sort(key=lambda h: h[0])         # 거리 오름차순
    return {"clash": clash, "hbond": hbond,
            "min_dist": min(dists) if dists else 99.0,
            "wat": wat}                    # 물 좌표를 밖으로. probe_distance 가 쓴다.


# ── 1. 촉매 물 자리 (재작성) ─────────────────────────────────────────
def water_site_open(nb):
    """
    nb: classify_neighbors 결과. None 이면 폴백.

    반환 키는 절대 바꾸지 말 것 — metal_site_score.py 의 passes() 와
    CSV allcols 가 이 이름을 그대로 쓴다.
      water_clash      충돌 원자 수 (int). None 폴백이면 -1
      water_min_dist   float. None 폴백이면 0.0
      water_blocker    "THR199/OG1" 꼴 문자열. 충돌 없으면 "". 폴백이면 "undef"
                       → 가장 깊이 박힌 원자 하나. _label(atom) 을 쓸 것

    (_label 은 renumber_site.py 의 +1000 오프셋을 되돌려준다)

    ※ water_min_dist 폴백을 0.0 으로 두는 것은 원래 계약을 그대로 이은 것이다.
      위험: 0.0 은 '가능한 최악의 충돌'과 값이 같아서, df[water_min_dist < X]
      필터에 측정불가 행이 섞여 든다. 반대 방향으로 거를 거면 99.0 이 낫다.
      필터 방향을 정하는 순간 여기도 같이 정할 것.
    """
    if nb is None:
        return {"water_clash": -1, "water_min_dist": 0.0, "water_blocker": "undef"}
    clash = nb["clash"]
    return {
        "water_clash": len(clash),
        "water_min_dist": nb["min_dist"],
        # clash 는 침범량 오름차순 → [0] 이 '가장 깊이 박힌' 원자.
        # '가장 가까운' 원자가 아니다. 큰 원자가 살짝 닿은 것보다
        # 작은 원자가 깊이 박힌 게 더 심각하기 때문.
        "water_blocker": f"{_label(clash[0][2])}/{clash[0][2]['name']}" if clash else "",
    }


# ── 2. 2차 배위권 (재작성) ───────────────────────────────────────────
def second_shell(nb):
    """
    nb: classify_neighbors 결과. 각도는 이미 계산돼 있으므로 metal_xyz 불필요.

    반환 키(폴백값):
      shell_hbond_n      수소결합 파트너 수 (0)
      shell_hbond_dist   최근접 파트너 거리 (99.0)
      shell_hbond_angle  그 파트너의 각도 (-1.0)
      shell_hbond_resid  그 파트너 라벨, _label 사용 ("")

    2CBA 에서 shell_hbond_resid 가 THR199 로 안 나오면 구현이 틀린 것이다.
    이 한 줄이 전체 수정의 판정 기준.
    """
    empty = {"shell_hbond_n": 0, "shell_hbond_dist": 99.0,
             "shell_hbond_angle": -1.0, "shell_hbond_resid": ""}
    if nb is None or not nb["hbond"]:
        return empty
    d0, ang0, a0 = nb["hbond"][0]          # 거리 오름차순 → 최근접 파트너
    #가상 물의 수소결합 파트너 중 가장 가까운 것의 잔기 라벨
    return {"shell_hbond_n": len(nb["hbond"]), "shell_hbond_dist": d0,
            "shell_hbond_angle": ang0, "shell_hbond_resid":f"{_label(a0)}/{a0['name']}"}

# ── 2b. 지정 잔기 ↔ 물 자리 거리 ────────────────────────────────────
def probe_distance(nb, atoms, site):
    """site["probe"] 로 지정한 원자에서 가상 물 자리까지의 거리. 창 조건 없이 무조건 잰다.

    second_shell 과 다른 점 — 이게 이 함수의 존재 이유다:
      second_shell 은 HB_LO~HB_HI(2.4~3.4Å) 창 안에 든 원자만 파트너로 센다.
      Thr112 가 4.83Å 로 밀려나면 창 밖이라 잡히지 않고 shell_hbond_dist 는
      폴백 99.0 을 준다. '얼마나 밀려났나'를 재려면 무조건 재는 값이 필요하다.
      (3라 과정.md: AF3 예측 45개 전부 이탈, 최선 4.83Å, 3Å 근처 0개)

    잔기 번호를 하드코딩하지 않는다 — native 199, 설계 112, 다음 라운드엔 또 다르다.
    원자 이름도 JSON 에서 읽는다. Thr 이면 OG1 이지만 Ser(OG)/Tyr(OH) 로 바뀔 수 있고,
    코드가 잔기명을 보고 추측하게 만들면 그 추측이 조용히 틀린다.

    반환 키(폴백값):
      probe_dist   float. 잴 수 없으면 "" (빈 문자열)
      probe_resid  "THR199/OG1" 꼴. probe 미설정이면 "", 잔기를 못 찾으면 "missing"

    ※ 폴백을 숫자 센티널로 두지 않는 이유: 08-28 에 shell_hbond_dist 폴백 99.0 이
      최대차 96.5 를 만들어 숫자 비교를 오염시킨 것이 실증됐다. 빈 문자열은
      pandas 에서 NaN 이 되어 숫자 연산에서 자동으로 빠진다.
      probe_resid 가 "" 인지 "missing" 인지로 '설정 안 함'과 '못 찾음'을 구분한다.
    """
    empty = {"probe_dist": "", "probe_resid": ""}
    p = site.get("probe") if isinstance(site, dict) else NameError
    if not p:
        return empty                       # probe 미설정 — 옛 site JSON 과 호환
    if nb is None:
        return {"probe_dist": "", "probe_resid": "undef"}   # 물 자리가 정의 안 됨
    want = (p["chain"], int(p["resi"]), p["atom"])
    for a in atoms:
        if a["rec"] == "ATOM" and (a["chain"], a["resi"], a["name"]) == want:
            return {"probe_dist": float(np.linalg.norm(a["xyz"] - nb["wat"])),
                    "probe_resid": f"{_label(a)}/{a['name']}"}
    return {"probe_dist": "", "probe_resid": "missing"}


# ── 3. 기질 포켓 ──────────────────────────────────────────────────────
def substrate_pocket(atoms, coords, metal_xyz, rmin=4.0, rmax=8.0):
    """물 자리 주변 껍질의 소수성/극성 비율. 어디까지나 대리지표."""
    empty = {"pocket_hydrophobic": 0, "pocket_polar": 0, "pocket_ratio": -1.0}
    wp = _water_pos(coords, metal_xyz)
    if wp is None:
        return empty
    wat, _ = wp
    prot, P = _protein(atoms, coords)
    if len(prot) == 0:
        return empty

    d = np.linalg.norm(P - wat, axis=1)
    shell = (d >= rmin) & (d <= rmax)
    nh = np_ = 0
    for a, s in zip(prot, shell):
        if not s:
            continue
        nm, rn = a["name"], a["resn"].upper()
        if rn in HYDROPHOBIC and nm.startswith("C") and nm not in BACKBONE:
            nh += 1
        elif nm[0] in "NO":
            np_ += 1
    tot = nh + np_
    return {"pocket_hydrophobic": nh, "pocket_polar": np_,
            "pocket_ratio": float(nh / tot) if tot else -1.0}


# ── 4. 용매 접근성 ────────────────────────────────────────────────────
def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1)


def solvent_access(atoms, coords, metal_xyz, n_rays=128, reach=10.0, step=0.5):
    """물 자리에서 광선을 쏴 벌크 용매까지 빠져나가는 방향의 비율."""
    wp = _water_pos(coords, metal_xyz)
    if wp is None:
        return {"escape_frac": -1.0}
    wat, _ = wp
    # 배위원자도 실제로 길을 막으므로 여기선 빼지 않는다
    _, P = _protein(atoms, coords, drop_self=False)
    if len(P) == 0:
        return {"escape_frac": -1.0}

    P = P[np.linalg.norm(P - wat, axis=1) < reach + RAY_CUT + 0.5]   # 미리 가지치기
    if len(P) == 0:
        return {"escape_frac": 1.0}

    dirs = fibonacci_sphere(n_rays)
    ts = np.arange(step, reach + 1e-9, step)
    pts = (wat + dirs[:, None, :] * ts[None, :, None]).reshape(-1, 3)
    d = np.linalg.norm(pts[:, None, :] - P[None, :, :], axis=2)
    blocked = (d < RAY_CUT).any(axis=1).reshape(len(dirs), len(ts))
    return {"escape_frac": float((~blocked.any(axis=1)).mean())}


# ── 5. CO2 기질 자리 ──────────────────────────────────────────────────
# substrate_pocket() 은 '주변이 소수성인가'만 센다. 소수성 벽이 멀쩡해도
# 그 사이 공간이 메워져 있으면 CO2 는 못 들어간다. 아래는 공간 자체를 잰다.
#
# 파라미터 출처: PDB 3D92 (야생형 hCA II + CO2, 1.1 Å, 반응 전 복합체).
#   O(친핵체)–C(CO2)        2.79 Å
#   ∠Zn–O(친핵체)···C(CO2)  103.1°
R_OC = 2.79             # 친핵체 산소 → CO2 탄소 거리
ANG_OC = 103.1          # Zn–친핵체–탄소 각도
ANG_TOL = 15.0          # 각도 창 절반폭
R_CO2_C = VDW["C"]      # CO2 탄소도 그냥 탄소다

# 구면 표본수. 2000 은 부족하다 — 각도 창이 전체 구면의 25% 라 후보점이
# 500개밖에 안 남고, 2CBA 에서 n=500~4000 사이 값이 0.092~0.101 로 흔들린다.
# 20000 부터 0.0951/0.0943/0.0944/0.0948 (n=2e4/4e4/8e4/1.6e5) 로 ±0.001 안에
# 수렴하고 구조 하나당 0.5초다. 수렴값 0.095 는 3점 교정의 2CBA 값과 같다.
N_CO2_PTS = 20000


def co2_open_frac(atoms, coords, metal_xyz, n_pts=N_CO2_PTS, tol=ANG_TOL):
    """CO2 탄소가 앉을 수 있는 자리의 비율(0~1). 못 재면 -1.

    친핵체는 실측 OH- 가 아니라 가상 4번째 꼭짓점을 쓴다. 설계 모델에는
    물이 없어서 선택지가 없고, 그래서 기준선(결정구조·native 예측)도 같은
    가상 꼭짓점으로 재야 비교가 성립한다. 3D92 에서 가상 꼭짓점과 실측
    OH- 의 어긋남은 0.37 Å.

    물·헤테로 원자는 센 대상에서 빠진다(_protein 이 ATOM 만 고른다).
    AF3 출력에 물이 없으므로 결정구조 쪽 조건을 거기에 맞춘 것이다.

    배위원자 자신은 빼지 않는다(drop_self=False) — His 고리는 실제로
    기질 자리의 벽 한 면이다. solvent_access() 와 같은 이유.
    """
    out = {"co2_open_frac": -1.0, "co2_n_cand": 0}
    wp = _water_pos(coords, metal_xyz)
    if wp is None:
        return out
    nuc, w = wp              # w = Zn→친핵체 단위벡터

    # 친핵체에서 본 후보 방향 중, Zn 쪽(-w)과 103.1°±tol 를 이루는 것만.
    # cos 는 각도에 대해 감소함수라 lo/hi 가 뒤집힌다.
    dirs = fibonacci_sphere(n_pts)
    cosang = dirs @ (-w)
    lo = np.cos(np.radians(ANG_OC + tol))
    hi = np.cos(np.radians(ANG_OC - tol))
    cand = nuc + R_OC * dirs[(cosang >= lo) & (cosang <= hi)]
    out["co2_n_cand"] = int(len(cand))
    if len(cand) == 0:
        return out

    prot, P = _protein(atoms, coords, drop_self=False)
    if len(P) == 0:
        out["co2_open_frac"] = 1.0
        return out

    cut = np.array([R_CO2_C + VDW.get(a["elem"], VDW["C"]) for a in prot])
    d = np.linalg.norm(cand[:, None, :] - P[None, :, :], axis=2)
    out["co2_open_frac"] = float((~(d < cut[None, :]).any(axis=1)).mean())
    return out


# ── 통합  ─────────────────────────────────
def function_metrics(atoms, coords, metal_xyz, site):
    nb = classify_neighbors(atoms, coords, metal_xyz)
    out = {}
    out.update(water_site_open(nb))
    out.update(second_shell(nb))
    out.update(probe_distance(nb, atoms, site))
    out.update(substrate_pocket(atoms, coords, metal_xyz))
    out.update(solvent_access(atoms, coords, metal_xyz))
    out.update(co2_open_frac(atoms, coords, metal_xyz))
    # 분류 자체가 불가능했는지를 한 컬럼으로 남긴다. water_clash == -1 과
    # 중복이지만, shell_hbond_n 은 '폴백'과 '진짜 2차 배위권 없음'이 둘 다 0 이라
    # 값만으로 구분할 수 없다. 분석 전에 이 컬럼부터 세고 시작할 것.
    out["nb_status"] = "ok" if nb is not None else "undef"
    return out


if __name__ == "__main__":
    import sys, glob
    sys.path.insert(0, ".")
    from metal_site_score import parse_pdb, resolve_site, find_metal

    # score_one 과 같은 모양(dict)으로 둔다. 예전엔 여기만 coordinators 리스트라
    # 같은 이름 site 가 두 군데서 다른 물건이었다 — function_metrics 가 site 를
    # 받게 되면서 그 불일치가 터진다.
    site = {"coordinators": [{"chain": "A", "resi": r, "resn": "HIS", "atom": "auto"}
                             for r in (94, 96, 119)],
            "probe": {"chain": "A", "resi": 199, "atom": "OG1"}}   # native 2CBA 기준
    cols = ["water_clash", "water_min_dist", "water_blocker",
            "shell_hbond_n", "shell_hbond_dist", "shell_hbond_angle",
            "shell_hbond_resid", "probe_dist", "probe_resid",
            "pocket_ratio", "escape_frac", "co2_open_frac", "nb_status"]
    files = [f for p in sys.argv[1:] for f in sorted(glob.glob(p))]
    if not files:
        sys.exit("입력 PDB 없음.  사용:  python site_function.py <renum.pdb> ...")
    print(f"{'pdb':24s} " + " ".join(c[:11].rjust(11) for c in cols))
    for f in files:
        fit = resolve_site(parse_pdb(f), site["coordinators"],
                           metal_xyz=find_metal(parse_pdb(f)))
        if fit is None:
            print(f"{f.split('/')[-1]:24s} 배위원자 없음")
            continue
        # 폴백이던 coords.mean(0) 은 제거 — 중심점은 금속 위치가 아니다.
        # 그게 fit_virtual_metal 평면갇힘 버그의 원형이었다.
        r = function_metrics(parse_pdb(f), fit["coords"], fit["metal"], site)
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append((f"{v:.3f}" if isinstance(v, float) else str(v)).rjust(11))
        print(f"{f.split('/')[-1]:24s} " + " ".join(cells))