#!/usr/bin/env python3
"""
metal_site_score.py — 금속 배위 자리 기하 채점기 (de novo 설계 PDB용)

설계 결과 PDB들을 받아서, 금속 배위 자리(Cys SG / His NE2·ND1 / Asp OD / Glu OE 등)의
기하가 얼마나 보존됐는지 점수화하고 정렬·필터한다.

레이어:
  (A) coord_rmsd_vs_ref   : 배위원자만 Kabsch 정렬 후 RMSD  (레퍼런스 필요)  ← 주 정렬 키
  (B) pair_dist_rmsd_vs_ref: 배위원자 쌍거리 편차          (레퍼런스 필요, 정렬 불필요)
  (C) metal_fit_residual  : 가상금속(또는 실제 금속) 대비 이상적 배위거리 잔차 (레퍼런스 프리)
  (D) angle_rmsd          : ligand-metal-ligand 각도 편차 (레퍼런스가 있으면 vs ref, 없으면 vs ideal)

의존성: numpy
"""
import argparse, glob, json, sys, math
from itertools import combinations, product, permutations
import numpy as np
import site_function as sf
import os
import gemmi

# ── 기본 배위원자 정의 ────────────────────────────────────────────────
# 잔기명 → 후보 배위원자 이름. His는 둘 중 자동선택.
DEFAULT_COORD_ATOMS = {
    "CYS": ["SG"],
    "HIS": ["NE2", "ND1"],   # 자동선택 대상
    "ASP": ["OD1", "OD2"],   # 자동선택 대상
    "GLU": ["OE1", "OE2"],
    "MET": ["SD"],
    "SER": ["OG"],
    "THR": ["OG1"],
    "TYR": ["OH"],
}
# 원소별 이상적 Zn–ligand 배위 거리(Å). ligand 원자 원소로 lookup.
IDEAL_DIST = {"N": 2.05, "S": 2.32, "O": 2.05}
METAL_NAMES = {"ZN", "FE", "CU", "NI", "MN", "CO", "MG", "CA"}  # HETATM 금속 후보

# ── 초경량 PDB 파서 ───────────────────────────────────────────────────
def parse_pdb(path):
    """ATOM/HETATM 레코드 → dict 리스트. altloc은 첫 등장(또는 'A')만."""
    atoms = []  
    with open(path) as fh:
        for ln in fh:
            rec = ln[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            altloc = ln[16].strip()
            if altloc not in ("", "A"):
                continue
            atoms.append({
                "rec": rec,
                "name": ln[12:16].strip(),
                "resn": ln[17:20].strip(),
                "chain": ln[21].strip(),
                "resi": int(ln[22:26]),
                "xyz": np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]),
                "elem": (ln[76:78].strip() or ln[12:16].strip()[0]).upper(),
            })
    return atoms


def parse_cif(path):
    """mmCIF(OF3/AF3 출력) → parse_pdb 와 동일한 dict 리스트.

    parse_pdb 와 거동을 맞추기 위한 세 가지 — 지우지 말 것:
      - altloc: gemmi 는 대안이 없을 때 빈 문자열이 아니라 널 문자('\\x00')를 준다.
        ("", "A") 로 거르면 모든 원자가 탈락해 조용히 빈 리스트가 나온다.
      - elem: gemmi 는 'Zn', PDB 경로는 'ZN' → upper() 로 정규화해야 두 경로가 같아진다.
      - st[0]: 첫 모델만. 여러 모델을 돌면 같은 원자가 중복되어 모든 거리가 깨진다.
    """
    st = gemmi.read_structure(path)
    result = []
    for cra in st[0].all():            # chain·residue·atom을 한 번에
        if cra.atom.altloc not in ("\x00", "A"):
            continue
        result.append({
            "rec":   "HETATM" if cra.residue.het_flag == "H" else "ATOM",
            "name":  cra.atom.name,
            "resn":  cra.residue.name,
            "chain": cra.chain.name,
            "resi":  cra.residue.seqid.num,
            "xyz":   np.array([cra.atom.pos.x, cra.atom.pos.y, cra.atom.pos.z]),
            "elem":  cra.atom.element.name.upper(),
            "plddt": cra.atom.b_iso,
        })
    return result


def read_atoms(path):
    """확장자로 파서를 고른다. 규칙을 여기 한 군데에만 둔다 —
    호출부(score_one, build_ref)에 if 를 복사하면 한쪽만 고치게 된다."""
    ext = os.path.splitext(path)[1].lower()   # .CIF 처럼 대문자로 오는 경우 대비
    return parse_cif(path) if ext in (".cif", ".mmcif") else parse_pdb(path)


def element_of(atom_name):
    """원자 이름에서 원소 추정 (배위 거리 lookup용)."""
    n = atom_name.strip()
    if n[0] in "SNOC" and not n[0].isdigit():
        return n[0]
    return n.lstrip("0123456789")[0]

# ── 기하 유틸 ─────────────────────────────────────────────────────────
def kabsch_rmsd(P, Q):
    """P를 Q에 최적 중첩(회전+이동)한 뒤 RMSD. P,Q: (N,3), 같은 순서 대응."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    P_al = (R @ Pc.T).T
    return float(np.sqrt(((P_al - Qc) ** 2).sum(1).mean()))

RING_NEIGHBORS = {"NE2": ("CD2", "CE1"), "ND1": ("CG", "CE1")}

def lone_pair_dir(idx, chain, resi, nname):
    """sp2 질소의 고립전자쌍 단위벡터. 금속은 반드시 이 방향에 있어야 한다.
    좌표계에 독립적이라 설계/native 어디서나 똑같이 쓸 수 있다."""
    nb = RING_NEIGHBORS.get(nname)
    if nb is None:
        return None
    try:
        x = idx[(chain, resi, nname)]
        p1, p2 = idx[(chain, resi, nb[0])], idx[(chain, resi, nb[1])]
    except KeyError:
        return None
    u1 = (p1 - x) / np.linalg.norm(p1 - x)
    u2 = (p2 - x) / np.linalg.norm(p2 - x)
    lp = -(u1 + u2)
    n = np.linalg.norm(lp)
    return None if n < 1e-6 else lp / n

def lp_deviation(m, coords, lps):
    """금속이 각 배위원자의 고립전자쌍 축에서 벗어난 평균 각도(도).
    가짜 His 삼중조를 걸러내는 주 판별 지표. 임계값은 native 2CBA 에서 잴 것."""
    a = [np.degrees(np.arccos(np.clip(
            float(((m - x) / np.linalg.norm(m - x)) @ lp), -1, 1)))
         for x, lp in zip(coords, lps) if lp is not None]
    return float(np.mean(a)) if a else 999.0

def _plane_normal(c):
    if len(c) != 3:
        return None
    n = np.cross(c[1] - c[0], c[2] - c[0])
    nn = np.linalg.norm(n)
    return None if nn < 1e-8 else n / nn

def fit_virtual_metal(ligands, ideal, iters=200, tol=1e-6, init=None):
    """
    ligand 좌표들에서 이상적 배위거리를 만족하는 금속 위치 추정.
    Weiszfeld류 고정점 반복. 반환: (metal_xyz, rms_residual).

    반복 자체는 옳다 — 고정점이 최소제곱 정류점과 정확히 일치한다.
    문제는 초기값이었다. 배위원자 3개가 만드는 평면은 이 사상의 불변다양체이고
    중심점이 그 안에 있어서, 금속이 영원히 평면을 벗어나지 못했다.
    → init 인자 추가. 직접 부르지 말고 fit_virtual_metal_3d 를 쓸 것.
    """
    m = ligands.mean(0) if init is None else np.array(init, float)
    for _ in range(iters):
        v = m - ligands
        dist = np.linalg.norm(v, axis=1, keepdims=True)
        dist[dist < 1e-8] = 1e-8
        m_new = (ligands + ideal[:, None] * v / dist).mean(0)
        if np.linalg.norm(m_new - m) < tol:
            m = m_new; break
        m = m_new
    resid = np.linalg.norm(m - ligands, axis=1) - ideal
    return m, float(np.sqrt((resid ** 2).mean()))

def fit_virtual_metal_3d(ligands, ideal, hint=None, offset=1.0):
    """
    평면 밖에서 출발시킨다. 평면 위/아래 두 해는 거울대칭이라 잔차가 항상
    같으므로 잔차로는 고를 수 없다 → hint(고립전자쌍 합)로 분지를 정한다.
    """
    n = _plane_normal(ligands)
    if n is None:
        return fit_virtual_metal(ligands, ideal)
    if hint is not None and abs(float(hint @ n)) > 1e-6:
        n = n * np.sign(float(hint @ n))
    return fit_virtual_metal(ligands, ideal, init=ligands.mean(0) + offset * n)

def all_angles(ligands, metal):
    """모든 ligand-metal-ligand 각도(도), 배위원자 쌍 순서대로."""
    out = []
    for i, j in combinations(range(len(ligands)), 2):
        a = ligands[i] - metal
        b = ligands[j] - metal
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        out.append(math.degrees(math.acos(np.clip(cos, -1, 1))))
    return np.array(out)

def pairwise_dists(coords):
    return np.array([np.linalg.norm(coords[i] - coords[j])
                     for i, j in combinations(range(len(coords)), 2)])

# ── 배위원자 선택 + 금속 적합 (함께 결정) ────────────────────────────
def resolve_site(atoms, coordinators, metal_xyz=None, ideal_dist=None):
    """
    배위원자 선택과 금속 위치를 '함께' 결정한다.
    분리하면 닭-달걀이 된다: 원자를 고르려면 금속이 필요하고,
    금속을 적합하려면 원자가 필요하다. 그래서 조합을 전부 시도한다.

      금속 있음 : 금속에 가장 가까운 원자 (기존과 동일 — 이건 옳았다)
      금속 없음 : NE2/ND1 조합 전부 시도 → 고립전자쌍 정렬이 최선인 조합
                  (기존은 무조건 목록 첫 항목 = 항상 NE2 였다. His119 처럼
                   ND1 로 배위하는 경우를 통째로 놓쳤다.)

    기존의 ref_coords 기반 선택은 제거했다. 설계와 native 는 좌표계가
    달라서 절대좌표 거리를 비교하는 것 자체가 성립하지 않는다.

    반환: dict{names, coords, elems, metal, source, residual, lp_dev} 또는 None
    """
    idx = {(a["chain"], a["resi"], a["name"]): a["xyz"]
           for a in atoms if a["rec"] == "ATOM"}
    per_res = []
    for c in coordinators:
        want = c.get("atom", "auto")
        cands = [want] if want != "auto" else DEFAULT_COORD_ATOMS.get(c["resn"].upper(), [])
        found = [(nm, idx[(c["chain"], c["resi"], nm)])
                 for nm in cands if (c["chain"], c["resi"], nm) in idx]
        if not found:
            return None
        per_res.append(found)

    def pack(pick):
        names = [p[0] for p in pick]
        coords = np.array([p[1] for p in pick])
        elems = [element_of(nm) for nm in names]   # 선택된 원자 기준 (기존은 첫 후보 기준 = 버그)
        ideal = np.array([(ideal_dist or {}).get(e, IDEAL_DIST.get(e, 2.1)) for e in elems])
        lps = [lone_pair_dir(idx, c["chain"], c["resi"], nm)
               for c, nm in zip(coordinators, names)]
        return names, coords, elems, ideal, lps

    if metal_xyz is not None:
        pick = [min(f, key=lambda t: np.linalg.norm(t[1] - metal_xyz)) for f in per_res]
        names, coords, elems, ideal, lps = pack(pick)
        resid = np.linalg.norm(metal_xyz - coords, axis=1) - ideal
        return {"names": names, "coords": coords, "elems": elems,
                "metal": metal_xyz, "source": "explicit",
                "residual": float(np.sqrt((resid ** 2).mean())),
                "lp_dev": lp_deviation(metal_xyz, coords, lps)}

    best = None
    for pick in product(*per_res):
        names, coords, elems, ideal, lps = pack(pick)
        valid = [lp for lp in lps if lp is not None]
        hint = np.sum(valid, axis=0) if valid else None
        m, res = fit_virtual_metal_3d(coords, ideal, hint=hint)
        dev = lp_deviation(m, coords, lps)
        if best is None or dev < best["lp_dev"]:
            best = {"names": names, "coords": coords, "elems": elems,
                    "metal": m, "source": "virtual", "residual": res, "lp_dev": dev}
    return best


def best_permutation(coords, resns, ref):
    """
    설계 배위원자 ↔ 레퍼런스 배위원자 대응 확정. 화학적으로 같은 잔기끼리만 교환.
    주의: 이건 '최선의 대응' RMSD(대칭 보정 RMSD)다. README 에 명시할 것.
    """
    n = len(coords)
    cands = [p for p in permutations(range(n))
             if all(resns[p[i]] == resns[i] for i in range(n))]
    return list(min(cands, key=lambda p: kabsch_rmsd(coords[list(p)], ref["coords"])))


def find_metal(atoms):
    for a in atoms:
        if a["rec"] == "HETATM" and a["resn"].upper() in METAL_NAMES:
            return a["xyz"]
    return None

# ── 한 설계 채점 ──────────────────────────────────────────────────────
def score_one(path, site, ref=None, force_virtual=False):
    atoms = read_atoms(path)
    r = {"pdb":os.path.basename(path)}

    fit = resolve_site(atoms, site["coordinators"],
                       metal_xyz=None if force_virtual else find_metal(atoms),
                       ideal_dist=site.get("ideal_dist"))
    r["ok"] = fit is not None
    if not r["ok"]:
        return r

    coords, m = fit["coords"], fit["metal"]
    r["coord_atoms"]        = "/".join(fit["names"])   # 어느 N을 썼는지 기록 (재현성)
    r["metal_source"]       = fit["source"]
    r["metal_fit_residual"] = fit["residual"]          # (C)
    r["lone_pair_dev"]      = fit["lp_dev"]            # 진짜 배위 자리인가

    # (E) 기능 레이어 — m 이 평면 밖 올바른 위치라야 물 자리가 제대로 잡힌다
    r.update(sf.function_metrics(atoms, coords, m))

    # (A)(B)(D) — ref 가 있으면 대응 순열을 먼저 확정하고 '그 순서로' 전부 계산.
    # angle_rmsd 를 이 분기 안으로 옮긴 게 핵심. 기존엔 순열 확정 전에 계산됐다.
    if ref is not None:
        resns = [c["resn"].upper() for c in site["coordinators"]]
        C = coords[best_permutation(coords, resns, ref)]
        r["coord_rmsd_vs_ref"] = kabsch_rmsd(C, ref["coords"])
        r["pair_dist_rmsd_vs_ref"] = float(np.sqrt(
            ((pairwise_dists(C) - ref["pair_dists"]) ** 2).mean()))
        r["angle_rmsd"] = float(np.sqrt(((all_angles(C, m) - ref["angles"]) ** 2).mean()))
    else:
        tgt = site.get("ideal_angle", 109.47)
        r["angle_rmsd"] = float(np.sqrt(((all_angles(coords, m) - tgt) ** 2).mean()))
    return r


def build_ref(path, site):
    atoms = read_atoms(path)
    fit = resolve_site(atoms, site["coordinators"], metal_xyz=find_metal(atoms),
                       ideal_dist=site.get("ideal_dist"))
    if fit is None:
        sys.exit(f"[ref] 배위원자 추출 실패: {path}")
    return {"coords": fit["coords"],
            "pair_dists": pairwise_dists(fit["coords"]),
            "angles": all_angles(fit["coords"], fit["metal"]),
            "names": fit["names"]}


# ── 정렬·필터·출력 ────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdbs", nargs="+", help="설계 PDB들 (glob 가능)")
    ap.add_argument("--site", required=True, help="배위 자리 정의 JSON")
    ap.add_argument("--ref", help="레퍼런스(native) PDB — 있으면 RMSD 채점 활성")
    ap.add_argument("--sort-key", default="auto",
                    help="정렬 키 (기본: ref 있으면 coord_rmsd_vs_ref, 없으면 metal_fit_residual)")
    ap.add_argument("--max-rmsd", type=float, default=1.0, help="통과선: 배위원자 RMSD(Å)")
    ap.add_argument("--max-residual", type=float, default=0.15, help="통과선: 금속 잔차(Å)")
    ap.add_argument("--max-angle", type=float, default=12.0,
                    help="통과선: 각도 RMSD(도). apo/ref-free 모드에서 특히 중요")
    ap.add_argument("--max-lp-dev", type=float, default=15.0,
                    help="통과선: 고립전자쌍 이탈 평균 각도(도). "
                         "※미보정 임시값 — native 2CBA 에서 재서 정할 것")
    ap.add_argument("--virtual", action="store_true",
                    help="HETATM 금속을 무시하고 가상 금속으로 채점 (AF2 모드 예행)")
    ap.add_argument("--out", default=None, help="CSV 저장 경로")
    args = ap.parse_args()

    site = json.load(open(args.site))
    ref = build_ref(args.ref, site) if args.ref else None
    files = [f for pat in args.pdbs for f in sorted(glob.glob(pat))] or args.pdbs

    rows = [score_one(f, site, ref, force_virtual=args.virtual) for f in files]
    good = [r for r in rows if r.get("ok")]

    if not good:
        print("평가 가능한 설계 없음."); return

    key = args.sort_key
    if key == "auto":
        # residual 은 배위 N 들이 4.1Å 안에만 있으면 0 으로 붙어서 판별력이 없다.
        # ref-free 모드의 주 지표는 angle_rmsd.
        key = "coord_rmsd_vs_ref" if ref else "angle_rmsd"
    good.sort(key=lambda r: r.get(key, float("inf")))

    def passes(r):
        p = (r.get("lone_pair_dev", 999) <= args.max_lp_dev
             and r.get("angle_rmsd", 999) <= args.max_angle
             # water_clash 는 측정불가일 때 -1 이다. '== 0' 이라야 그것도 함께
             # 떨어진다. '<= 0' 으로 바꾸면 분류 실패한 모델이 통과해버린다.
             and r.get("water_clash", 9) == 0)
        
        if r.get("metal_source") != "virtual":
            p = p and r["metal_fit_residual"] <= args.max_residual

        if "coord_rmsd_vs_ref" in r:
            p = p and r["coord_rmsd_vs_ref"] <= args.max_rmsd
        return p
    
    for r in good:
        r["pass"] = passes(r)

    # 콘솔 표
    cols = ["pdb", "coord_atoms", "coord_rmsd_vs_ref", "pair_dist_rmsd_vs_ref",
            "metal_fit_residual", "angle_rmsd", "lone_pair_dev", "water_clash",
            "escape_frac", "shell_hbond_n", "metal_source", "pass","nb_status"]
    cols = [c for c in cols if any(c in r for r in good)]
    w = {c: max(len(c), *(len(f"{r.get(c,''):.3f}") if isinstance(r.get(c), float)
              else len(str(r.get(c, ""))) for r in good)) for c in cols}
    print("  ".join(c.ljust(w[c]) for c in cols))
    for r in good:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append((f"{v:.3f}" if isinstance(v, float) else str(v)).ljust(w[c]))
        print("  ".join(cells))
    bad = [r for r in rows if not r.get("ok")]
    if bad:
        print(f"\n[누락] 배위원자 못 찾은 설계 {len(bad)}개: "
              + ", ".join(r["pdb"] for r in bad))
    print(f"\n통과 {sum(r['pass'] for r in good)} / 평가 {len(good)}  (정렬 키: {key})")

    if args.out:
        import csv
        allcols = ["pdb", "ok", "coord_atoms", "coord_rmsd_vs_ref", "pair_dist_rmsd_vs_ref",
                   "metal_fit_residual", "angle_rmsd", "lone_pair_dev",
                   "water_clash", "water_min_dist", "water_blocker",
                   "shell_hbond_n", "shell_hbond_dist", "shell_hbond_angle",
                   "shell_hbond_resid", "pocket_hydrophobic", "pocket_polar",
                   "pocket_ratio", "escape_frac", "metal_source", "nb_status", "pass"]
        with open(args.out, "w", newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=allcols, extrasaction="ignore")
            wtr.writeheader()
            for r in rows:
                wtr.writerow(r)
        print(f"CSV 저장: {args.out}")

if __name__ == "__main__":
    main()