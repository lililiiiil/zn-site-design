#!/usr/bin/env python3
"""
native_site_geometry.py — 2CBA(carbonic anhydrase II) 활성부위 기하 전수 측정.

Round 1 채점기는 배위 His 3개만 봤다. 이 스크립트는 CA가 실제로 촉매를
하는 데 필요한 전체 구성 — 촉매 물, Thr199, Glu106, 소수성 포켓, His64
양성자 셔틀 — 의 거리·각도를 native에서 직접 잰다.

여기서 나온 숫자가 Round 2 모티프 정의와 채점 임계값의 근거가 된다.
"내가 기억하는 값"이 아니라 "구조에서 잰 값"을 쓸 것.

사용:
  python native_site_geometry.py 2cba.pdb
  python native_site_geometry.py 2cba.pdb --json native_site_geometry.json

의존성: numpy, metal_site_score.py (같은 디렉터리)
"""
import argparse, json, sys
from itertools import combinations
import numpy as np

sys.path.insert(0, ".")
from metal_site_score import parse_pdb

CHAIN = "A"
COORDINATORS = [94, 96, 119]          # 배위 His
THR_GATE = (199, "OG1")               # 수산화물 방향 고정
GLU_GATE = (106, ("OE1", "OE2"))      # Thr199의 문지기
SHUTTLE = 64                          # 양성자 셔틀 His
RING_NEIGHBORS = {"NE2": ("CD2", "CE1"), "ND1": ("CG", "CE1")}
HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "MET", "PRO"}


def build_index(atoms):
    return {(a["chain"], a["resi"], a["name"]): a["xyz"] for a in atoms}


def norm(v):
    return float(np.linalg.norm(v))


def angle(a, b, c):
    """a-b-c 각도(도). b가 꼭짓점."""
    u, v = a - b, c - b
    cos = u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def lone_pair(idx, resi, nname):
    """His 배위 N의 고립전자쌍 방향(단위벡터). 금속이 이쪽에 있어야 정상."""
    nb = RING_NEIGHBORS.get(nname)
    if nb is None:
        return None
    try:
        x = idx[(CHAIN, resi, nname)]
        p1, p2 = idx[(CHAIN, resi, nb[0])], idx[(CHAIN, resi, nb[1])]
    except KeyError:
        return None
    u1 = (p1 - x) / np.linalg.norm(p1 - x)
    u2 = (p2 - x) / np.linalg.norm(p2 - x)
    lp = -(u1 + u2)
    n = np.linalg.norm(lp)
    return None if n < 1e-6 else lp / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdb")
    ap.add_argument("--json", default=None, help="측정값 JSON 저장 경로")
    args = ap.parse_args()

    atoms = parse_pdb(args.pdb)
    idx = build_index(atoms)
    out = {}

    # ── Zn 위치 ───────────────────────────────────────────────────────
    zns = [a for a in atoms if a["rec"] == "HETATM" and a["resn"].upper() == "ZN"]
    if not zns:
        sys.exit("ZN HETATM 없음. holo 구조인지 확인.")
    ZN = zns[0]["xyz"]
    print(f"Zn: {np.round(ZN, 3)}\n")

    # ── 1차 배위권 ────────────────────────────────────────────────────
    print("── 1차 배위권 (His3) " + "─" * 40)
    coord_atoms, coord_info = [], []
    for resi in COORDINATORS:
        cands = [(nm, idx[(CHAIN, resi, nm)]) for nm in ("NE2", "ND1")
                 if (CHAIN, resi, nm) in idx]
        if not cands:
            sys.exit(f"His{resi} 배위 후보 원자 없음")
        nm, x = min(cands, key=lambda c: norm(c[1] - ZN))
        d = norm(x - ZN)
        lp = lone_pair(idx, resi, nm)
        lp_dev = (float(np.degrees(np.arccos(np.clip(
            ((ZN - x) / np.linalg.norm(ZN - x)) @ lp, -1, 1))))
            if lp is not None else None)
        coord_atoms.append(x)
        coord_info.append({"resi": resi, "atom": nm, "zn_dist": d,
                           "lone_pair_dev": lp_dev})
        print(f"  His{resi:<4d} {nm:4s}  Zn–N = {d:5.2f} Å   "
              f"고립전자쌍 이탈 = {lp_dev:5.1f}°")
    coord_atoms = np.array(coord_atoms)
    out["coordinators"] = coord_info

    print("\n  N–Zn–N 각도:")
    angs = []
    for i, j in combinations(range(3), 2):
        a = angle(coord_atoms[i], ZN, coord_atoms[j])
        angs.append(a)
        print(f"    His{COORDINATORS[i]}–Zn–His{COORDINATORS[j]}  {a:6.2f}°")
    print(f"    평균 {np.mean(angs):.2f}°  (이상적 사면체 109.47°)")
    out["n_zn_n_angles"] = angs

    # 배위 N 3개가 만드는 평면에서 Zn이 얼마나 벗어나 있나
    # → fit_virtual_metal 평면갇힘 버그가 얼마나 큰 오차인지의 정답값
    n = np.cross(coord_atoms[1] - coord_atoms[0], coord_atoms[2] - coord_atoms[0])
    n /= np.linalg.norm(n)
    h = abs(float((ZN - coord_atoms.mean(0)) @ n))
    print(f"\n  Zn의 N3 평면 이탈 높이 = {h:.3f} Å")
    print(f"  → 가상금속 적합이 평면에 갇히면 최소 이만큼 틀린다")
    out["zn_out_of_plane"] = h

    # ── 촉매 물 ───────────────────────────────────────────────────────
    print("\n── 촉매 물 (4번째 배위) " + "─" * 36)
    waters = [a for a in atoms if a["rec"] == "HETATM"
              and a["resn"].upper() in ("HOH", "WAT") and a["name"] == "O"]
    if not waters:
        print("  물 분자 없음 — 이하 건너뜀")
        W = None
    else:
        w = min(waters, key=lambda a: norm(a["xyz"] - ZN))
        W = w["xyz"]
        dW = norm(W - ZN)
        print(f"  가장 가까운 물: HOH{w['resi']}  Zn–O = {dW:.2f} Å")
        if dW > 2.6:
            print("  ⚠ 2.6 Å 초과 — 배위된 물이 아닐 수 있음")
        out["zn_water_dist"] = dW
        for i, resi in enumerate(COORDINATORS):
            print(f"    His{resi}–Zn–Owat = {angle(coord_atoms[i], ZN, W):6.2f}°")

    # ── 2차 배위권: Thr199 ────────────────────────────────────────────
    if W is not None:
        print("\n── 2차 배위권 (Thr199 문지기) " + "─" * 30)
        key = (CHAIN, THR_GATE[0], THR_GATE[1])
        if key in idx:
            OG1 = idx[key]
            d = norm(OG1 - W)
            a = angle(ZN, W, OG1)
            print(f"  Thr199 OG1 ··· Owat  = {d:5.2f} Å")
            print(f"  Zn–Owat···OG1 각도    = {a:6.2f}°")
            print(f"  → site_function.py 의 shell_hbond_dist / shell_hbond_angle 목표값")
            out["thr_water_dist"], out["thr_water_angle"] = d, a

            # Thr199 → Glu106 (3차 배위권)
            for nm in GLU_GATE[1]:
                k2 = (CHAIN, GLU_GATE[0], nm)
                if k2 in idx:
                    print(f"  Thr199 OG1 ··· Glu106 {nm} = {norm(idx[k2] - OG1):5.2f} Å")
        else:
            print("  Thr199 OG1 없음")

    # ── 소수성 포켓 ───────────────────────────────────────────────────
    if W is not None:
        print("\n── CO2 포켓 (물 자리 4~8 Å 껍질) " + "─" * 26)
        prot = [a for a in atoms if a["rec"] == "ATOM"]
        P = np.array([a["xyz"] for a in prot])
        d = np.linalg.norm(P - W, axis=1)
        shell = (d >= 4.0) & (d <= 8.0)
        resis = {}
        nh = npol = 0
        for a, s in zip(prot, shell):
            if not s:
                continue
            rn, nm = a["resn"].upper(), a["name"]
            resis.setdefault((a["resi"], rn), 0)
            resis[(a["resi"], rn)] += 1
            if rn in HYDROPHOBIC and nm.startswith("C") and nm not in ("N", "CA", "C", "O"):
                nh += 1
            elif nm[0] in "NO":
                npol += 1
        ratio = nh / (nh + npol) if (nh + npol) else -1
        print(f"  소수성 {nh}  극성 {npol}  비율 {ratio:.3f}")
        print(f"  → pocket_ratio 목표값")
        top = sorted(resis.items(), key=lambda kv: -kv[1])[:10]
        print("  주요 잔기: " + ", ".join(f"{rn}{ri}" for (ri, rn), _ in top))
        out["pocket_ratio"] = ratio

    # ── 양성자 셔틀 ───────────────────────────────────────────────────
    print("\n── 양성자 셔틀 (His64) " + "─" * 37)
    found = False
    for nm in ("NE2", "ND1"):
        k = (CHAIN, SHUTTLE, nm)
        if k in idx:
            found = True
            print(f"  His64 {nm}  Zn 거리 = {norm(idx[k] - ZN):5.2f} Å"
                  + (f"   물 거리 = {norm(idx[k] - W):5.2f} Å" if W is not None else ""))
    if not found:
        print("  His64 없음 (구조에 따라 in/out 자세 미모델링일 수 있음)")
    else:
        print("  → Round 2 스캐폴드는 이 거리에 His를 놓을 공간이 있어야 함")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nJSON 저장: {args.json}")


if __name__ == "__main__":
    main()
