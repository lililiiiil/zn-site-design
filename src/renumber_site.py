#!/usr/bin/env python3
"""
renumber_site.py — 설계/AF2 출력 PDB의 배위 His 3개를 찾아 94/96/119로 리넘버.

왜 필요? RFdiffusion→MPNN→AF2를 거치면 잔기 번호가 1..N으로 새로 매겨져서,
native(94/96/119) 기준으로 짠 site.json이 설계 파일엔 안 먹힘.
이 스크립트가 "아연을 잡는 His 3개"를 기하로 자동 탐지해서 그 셋만
94/96/119(체인 A)로 다시 붙여줌. 그러면 하나의 site.json으로
native 레퍼런스(2CBA)와 설계 파일들을 똑같이 채점할 수 있음.

원리: 모든 His의 배위 후보 질소(NE2/ND1) 중에서, 가상 금속 하나에 가장
잘 모이는 His 3개 조합을 고름(= metal_site_score.py의 fit_virtual_metal과
같은 논리). 그 3개가 곧 배위 자리.

충돌 방지: 나머지 잔기는 전부 +1000 오프셋을 줘서, 94/96/119 번호가
다른 잔기와 겹치지 않게 함. (채점기는 배위 잔기만 보므로 나머지 번호는 무의미)

사용:
  python renumber_site.py design.pdb                 # design_renum.pdb 생성
  python renumber_site.py design.pdb -o out.pdb
  python renumber_site.py "outputs/*.pdb"            # glob, 각각 *_renum.pdb

의존성: numpy
"""
import argparse, glob, os, sys
from itertools import combinations, product
import numpy as np

IDEAL_N = 2.05          # His-N ~ Zn 이상 배위 거리(Å)
WARN_RESIDUAL = 0.5     # 이보다 크면 "진짜 배위 자리가 아닐 수 있음" 경고


def parse_atoms(path):
    atoms = []
    with open(path) as fh:
        for ln in fh:
            if ln[:6].strip() not in ("ATOM", "HETATM"):
                continue
            if ln[16].strip() not in ("", "A"):   # altloc 첫 등장만
                continue
            try:
                atoms.append({
                    "line": ln.rstrip("\n"),
                    "name": ln[12:16].strip(),
                    "resn": ln[17:20].strip(),
                    "chain": ln[21],
                    "resi": int(ln[22:26]),
                    "xyz": np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]),
                })
            except ValueError:
                continue
    return atoms


def fit_metal(ligands, ideal, iters=200, tol=1e-6):
    """ligand 좌표들에 이상거리를 맞추는 가상 금속 위치 + RMS 잔차."""
    m = ligands.mean(0)
    for _ in range(iters):
        v = m - ligands
        d = np.linalg.norm(v, axis=1, keepdims=True); d[d < 1e-8] = 1e-8
        m_new = (ligands + ideal[:, None] * v / d).mean(0)
        if np.linalg.norm(m_new - m) < tol:
            m = m_new; break
        m = m_new
    resid = np.linalg.norm(m - ligands, axis=1) - ideal
    return m, float(np.sqrt((resid ** 2).mean()))


def find_his_site(atoms):
    """배위 His 3개 (chain,resi) 튜플 리스트, 전체 His 개수, 적합 잔차 반환."""
    his = {}
    for a in atoms:
        if a["resn"] == "HIS" and a["name"] in ("NE2", "ND1"):
            his.setdefault((a["chain"], a["resi"]), {})[a["name"]] = a["xyz"]
    keys = list(his.keys())
    if len(keys) < 3:
        return None, len(keys), None
    best = None
    for trio in combinations(keys, 3):
        for pick in product(*[list(his[k].items()) for k in trio]):
            coords = np.array([p[1] for p in pick])
            _, resid = fit_metal(coords, np.full(3, IDEAL_N))
            if best is None or resid < best[0]:
                best = (resid, trio)
    resid, trio = best
    return list(trio), len(keys), resid


def rewrite(path, out):
    atoms = parse_atoms(path)
    trio, n_his, resid = find_his_site(atoms)
    if trio is None:
        print(f"  ✗ {os.path.basename(path)}: His {n_his}개 — 배위 자리(3개) 없음. 건너뜀.")
        return False

    trio_sorted = sorted(trio, key=lambda k: k[1])      # 원래 번호 오름차순
    remap = {trio_sorted[0]: 94, trio_sorted[1]: 96, trio_sorted[2]: 119}

    outlines = []
    with open(path) as fh:
        for ln in fh:
            if ln[:6].strip() not in ("ATOM", "HETATM"):
                outlines.append(ln.rstrip("\n")); continue
            try:
                chain, resi = ln[21], int(ln[22:26])
            except ValueError:
                outlines.append(ln.rstrip("\n")); continue
            if (chain, resi) in remap:
                new_chain, new_resi = "A", remap[(chain, resi)]
            else:
                new_chain, new_resi = chain, resi + 1000   # 충돌 방지 오프셋
            outlines.append(ln[:21] + new_chain + f"{new_resi:>4d}" + ln[26:].rstrip("\n"))

    with open(out, "w") as fh:
        fh.write("\n".join(outlines) + "\n")

    flag = "  ⚠ 잔차 큼 — 진짜 배위 자리가 아닐 수 있음" if resid > WARN_RESIDUAL else ""
    print(f"  ✓ {os.path.basename(path)} → {os.path.basename(out)} | "
          f"His {n_his}개 중 배위 3개 = "
          f"{', '.join(f'{c}{r}' for c, r in trio_sorted)} → 94/96/119 "
          f"| 가상금속 잔차 {resid:.3f}Å{flag}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdbs", nargs="+", help="설계/AF2 출력 PDB (glob 가능)")
    ap.add_argument("-o", "--out", default=None,
                    help="출력 경로 (단일 파일일 때만). 미지정 시 *_renum.pdb")
    args = ap.parse_args()

    files = [f for pat in args.pdbs for f in sorted(glob.glob(pat))] or args.pdbs
    if args.out and len(files) > 1:
        sys.exit("여러 파일엔 -o 못 씀. glob으로 돌리면 각자 *_renum.pdb 생성.")

    ok = 0
    for f in files:
        out = args.out or f.rsplit(".", 1)[0] + "_renum.pdb"
        ok += rewrite(f, out)
    print(f"\n리넘버 완료: {ok} / {len(files)}")


if __name__ == "__main__":
    main()
