#!/usr/bin/env python3
"""
site_function.py — (E) 레이어: 배위 자리가 '촉매를 할 수 있는가'를 채점.

metal_site_score.py 의 (A)~(D)는 Zn을 얼마나 잘 '잡는가'만 본다.
이 모듈은 그 자리가 carbonic anhydrase처럼 '일을 할 수 있는가'를 본다.

  1. 열린 4번째 배위 자리 (촉매 물이 앉을 곳)  → water_site_open()
  2. 2차 배위권 (Thr199 등가물)                → second_shell()
  3. 소수성 기질 포켓 (CO2 도킹)               → substrate_pocket()
  4. 벌크 용매 통로 (양성자 배출)              → solvent_access()

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
CLASH_POLAR = 2.4    # 물 O ↔ N/O : 수소결합은 2.6~3.2 이므로 그 아래만 진짜 충돌
CLASH_APOLAR = 3.0   # 물 O ↔ C/S
HBOND_CUT = 3.4      # 2차 배위권 수소결합 상한
RAY_CUT = 2.8        # 광선이 원자에 이만큼 가까워지면 막힘
SELF_TOL = 0.1       # 배위원자 자기 자신 판정 허용오차
HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "MET", "PRO"}
DONOR_ATOMS = {"OG", "OG1", "OH", "ND1", "NE2", "OD1", "OD2", "OE1", "OE2", "N"}
BACKBONE = {"N", "CA", "C", "O"}


def _label(a):
    """잔기 라벨. renumber_site.py 의 +1000 오프셋을 되돌려 표시."""
    r = a["resi"]
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


def _protein(atoms, coords, drop_self=True):
    """단백질 원자만. drop_self면 배위원자 자신은 뺀다."""
    prot = [a for a in atoms if a["rec"] == "ATOM"]
    if not prot:
        return [], np.empty((0, 3))
    P = np.array([a["xyz"] for a in prot])
    if drop_self:
        keep = np.linalg.norm(P[:, None, :] - coords[None, :, :], axis=2).min(1) > SELF_TOL
        prot = [a for a, k in zip(prot, keep) if k]
        P = P[keep]
    return prot, P


# ── 1. 촉매 물 자리 ───────────────────────────────────────────────────
def water_site_open(atoms, coords, metal_xyz):
    wp = _water_pos(coords, metal_xyz)
    if wp is None:
        return {"water_clash": -1, "water_min_dist": 0.0, "water_blocker": "undef"}
    wat, _ = wp
    prot, P = _protein(atoms, coords)
    if len(prot) == 0:
        return {"water_clash": -1, "water_min_dist": 0.0, "water_blocker": "undef"}
    cuts = np.array([CLASH_POLAR if (a.get("elem") or a["name"][0]) in ("N", "O")
                     else CLASH_APOLAR for a in prot])
    d = np.linalg.norm(P - wat, axis=1)
    viol = d < cuts
    i = int(np.argmin(d - cuts))          # 가장 심한 위반
    return {
        "water_clash": int(viol.sum()),
        "water_min_dist": float(d.min()),
        "water_blocker": f"{_label(prot[i])}/{prot[i]['name']}" if viol.any() else "",
    }


# ── 2. 2차 배위권 ─────────────────────────────────────────────────────
def second_shell(atoms, coords, metal_xyz):
    """
    촉매 물에 수소결합할 수 있는 곁사슬 O/N (CA II 의 Thr199 역할).
    각도: Zn-Owat···X. 109±30° 정도가 '조준이 맞는' 범위.
    """
    empty = {"shell_hbond_n": 0, "shell_hbond_dist": 99.0,
             "shell_hbond_angle": -1.0, "shell_hbond_resid": ""}
    wp = _water_pos(coords, metal_xyz)
    if wp is None:
        return empty
    wat, _ = wp
    prot, P = _protein(atoms, coords)
    if len(prot) == 0:
        return empty

    d = np.linalg.norm(P - wat, axis=1)
    # 진짜 충돌 거리(<2.4Å)에 있는 건 수소결합 파트너가 아니다
    hits = [(float(d[i]), prot[i], P[i]) for i in range(len(prot))
            if prot[i]["name"] in DONOR_ATOMS and CLASH_POLAR <= d[i] <= HBOND_CUT]
    if not hits:
        return empty
    hits.sort(key=lambda h: h[0])

    d0, a0, x0 = hits[0]
    u, v = metal_xyz - wat, x0 - wat
    cos = u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
    ang = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return {"shell_hbond_n": len(hits), "shell_hbond_dist": d0,
            "shell_hbond_angle": ang, "shell_hbond_resid": _label(a0)}


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


# ── 통합 ──────────────────────────────────────────────────────────────
def function_metrics(atoms, coords, metal_xyz):
    out = {}
    out.update(water_site_open(atoms, coords, metal_xyz))
    out.update(second_shell(atoms, coords, metal_xyz))
    out.update(substrate_pocket(atoms, coords, metal_xyz))
    out.update(solvent_access(atoms, coords, metal_xyz))
    return out


if __name__ == "__main__":
    import sys, glob
    sys.path.insert(0, ".")
    from metal_site_score import parse_pdb, resolve_site, find_metal

    site = [{"chain": "A", "resi": r, "resn": "HIS", "atom": "auto"}
            for r in (94, 96, 119)]
    cols = ["water_clash", "water_min_dist", "water_blocker",
            "shell_hbond_n", "shell_hbond_dist", "shell_hbond_resid",
            "pocket_ratio", "escape_frac"]
    files = [f for p in sys.argv[1:] for f in sorted(glob.glob(p))]
    print(f"{'pdb':24s} " + " ".join(c[:11].rjust(11) for c in cols))
    for f in files:
        fit = resolve_site(parse_pdb(f), site, metal_xyz=find_metal(parse_pdb(f)))
        if fit is None:
            print(f"{f.split('/')[-1]:24s} 배위원자 없음")
            continue
        # 폴백이던 coords.mean(0) 은 제거 — 중심점은 금속 위치가 아니다.
        # 그게 fit_virtual_metal 평면갇힘 버그의 원형이었다.
        r = function_metrics(parse_pdb(f), fit["coords"], fit["metal"])
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append((f"{v:.3f}" if isinstance(v, float) else str(v)).rjust(11))
        print(f"{f.split('/')[-1]:24s} " + " ".join(cells))