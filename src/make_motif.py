#!/usr/bin/env python3
"""
make_motif.py — 2CBA에서 모티프 추출
  잔기 목록은 명령줄 인자로 받음 (모티프 A / B 공용)
  잔기 번호는 2CBA 원본 유지 (리넘버 안 함)
  ZN과 HOH263은 항상 포함
"""
import sys
import numpy as np

WATER_RESI = 263


def parse_lines(path):
    """ATOM/HETATM → dict. 원본 줄('line')과 삽입코드('icode')를 함께 보관."""
    atoms = []
    with open(path) as fh:
        for ln in fh:
            rec = ln[:6].strip()
            if rec not in ("ATOM", "HETATM"):   # ANISOU/TER/MODEL 여기서 탈락
                continue
            if ln[16].strip() not in ("", "A"):  # altloc 첫 등장만
                continue
            atoms.append({
                "line":  ln.rstrip("\n").rstrip("\r"),
                "rec":   rec,
                "name":  ln[12:16].strip(),
                "resn":  ln[17:20].strip().upper(),
                "chain": ln[21].strip(),
                "resi":  int(ln[22:26]),
                "icode": ln[26].strip(),
                "xyz":   np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]),
                "elem":  (ln[76:78].strip() or ln[12:16].strip()[0]).upper(),
            })
    return atoms


def count_models(path):
    with open(path) as fh:
        return sum(1 for ln in fh if ln[:5] == "MODEL")


def want_from_string(want_str):
    """'94:HIS,96:HIS' → {94: 'HIS', 96: 'HIS'}"""
    want = {}
    for a in want_str.split(","):
        parts = a.split(":")
        if len(parts) != 2:
            raise ValueError(f"형식 오류: {a!r}")
        num, name = parts
        int_num = int(num)
        name_strip = name.strip().upper()
        if int_num in want:
            raise ValueError(f"잔기 번호 중복: {int_num}")
        want[int_num] = name_strip
    return want


def inspect(path, atoms, chain, want):
    """생성 전 확인. 문제가 있으면 False."""
    ok = True

    n_model = count_models(path)
    print(f"MODEL 줄 개수: {n_model}")
    if n_model > 1:
        print("  ✗ 멀티모델 — 파서가 모델을 이어붙임")
        ok = False

    icodes = {a["icode"] for a in atoms}
    print(f"삽입코드 집합: {icodes}")
    hit = {(a["resi"], a["icode"]) for a in atoms
           if a["resi"] in want and a["icode"]}
    if hit:
        print(f"  ✗ 지정 잔기 번호에 삽입코드가 붙어 있음: {sorted(hit)}")
        ok = False

    chains = {a["chain"] for a in atoms if a["rec"] == "ATOM"}
    print(f"단백질 체인: {sorted(chains)}")
    if chain not in chains:
        print(f"  ✗ 지정한 체인 {chain!r}이(가) 파일에 없음")
        ok = False

    print(f"\n-- 지정 잔기 {len(want)}개 --")
    for resi, resn_want in sorted(want.items()):
        found = [a for a in atoms if a["rec"] == "ATOM" and a["resi"] == resi]
        if not found:
            print(f"  ✗ {resi}: 없음")
            ok = False
            continue
        for ch in sorted({a["chain"] for a in found}):
            sub = [a for a in found if a["chain"] == ch]
            resn = sub[0]["resn"]
            mark = "✓" if resn == resn_want else "✗"
            if resn != resn_want:
                ok = False
            print(f"  {mark} {ch}/{resi} {resn} (기대 {resn_want})  원자 {len(sub)}개")

    zns = [a for a in atoms if a["rec"] == "HETATM" and a["resn"] == "ZN"]
    print(f"\nZN 원자: {len(zns)}개  " +
          ", ".join(f"{a['chain']}/{a['resi']} elem={a['elem']}" for a in zns))
    if len(zns) != 1:
        print("  ✗ ZN이 1개가 아님")
        ok = False

    wats = [a for a in atoms if a["rec"] == "HETATM"
            and a["resn"] in ("HOH", "WAT") and a["resi"] == WATER_RESI]
    print(f"HOH{WATER_RESI}: {len(wats)}개  " +
          ", ".join(f"{a['chain']}/{a['name']}" for a in wats))
    if not wats:
        print("  ✗ 지정한 물 번호가 없음")
        ok = False
    elif zns:
        o = [a for a in wats if a["name"] == "O"]
        if o:
            d = float(np.linalg.norm(o[0]["xyz"] - zns[0]["xyz"]))
            print(f"  Zn–Owat = {d:.2f} Å")
            if d > 2.6:
                print("  ✗ 2.6 Å 초과 — 배위된 촉매 물이 아닐 수 있음")
                ok = False

    return ok


def select(atoms, chain, want):
    """모티프에 남길 원자. 조건 3개의 OR, altloc은 parse에서 이미 걸림."""
    out = []
    for a in atoms:
        keep = (
            (a["rec"] == "ATOM"
             and a["chain"] == chain
             and a["resi"] in want
             and a["resn"] == want[a["resi"]])
            or (a["rec"] == "HETATM" and a["resn"] == "ZN")
            or (a["rec"] == "HETATM" and a["resn"] in ("HOH", "WAT")
                and a["resi"] == WATER_RESI)
        )
        if keep:
            out.append(a)
    return out


def main():
    if len(sys.argv) < 5:
        print(f"사용법: python {sys.argv[0]} <src.pdb> <dst.pdb> <chain> <want>")
        print('  want 예시: "94:HIS,96:HIS,119:HIS,199:THR,106:GLU,92:GLN,117:GLU"')
        print("  PowerShell에서는 want를 반드시 따옴표로 감쌀 것")
        return 1

    src = sys.argv[1]
    dst = sys.argv[2]
    chain = sys.argv[3]
    want_str = sys.argv[4]

    want = want_from_string(want_str)
    print(f"chain 인자 = {chain!r}")
    print(f"want 인자  = {want!r}")

    atoms = parse_lines(src)
    print(f"{src}: ATOM/HETATM {len(atoms)}줄\n")

    if not inspect(src, atoms, chain, want):
        print("\n확인 실패 — 파일 생성 안 함")
        return 1

    sel = select(atoms, chain, want)
    print(f"\n선택된 원자: {len(sel)}개")
    for resi in sorted(want):
        n = sum(1 for a in sel if a["resi"] == resi and a["rec"] == "ATOM")
        print(f"  {resi}: {n}")

    with open(dst, "w") as fh:
        for a in sel:
            fh.write(a["line"] + "\n")
        fh.write("END\n")
    print(f"\n→ {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())