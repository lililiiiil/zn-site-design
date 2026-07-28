#!/usr/bin/env python3
"""
test_site_function.py — site_function.py 의 물 자리 판정 계약을 고정한다.

왜 합성 기하부터 하나: 2CBA 는 최종 관문이지만 실패했을 때 원인이 안 보인다.
Zn 을 원점에 놓고 이상적 사면체를 세우면 가상 물이 정확히 (0,0,2) 에 오므로,
원자 하나를 원하는 거리·각도에 정확히 놓고 단독으로 판정을 확인할 수 있다.

사용:
  python test_site_function.py                  # 합성 기하 테스트만
  python test_site_function.py 2cba_renum.pdb   # + native 검수까지

의존성: numpy, site_function.py (같은 디렉터리)
"""
import sys
import numpy as np
import site_function as sf

PASS, FAIL = [], []


# ── 합성 자리 만들기 ──────────────────────────────────────────────────
# 배위 N 3개를 이상적 사면체 세 꼭짓점에, Zn 을 원점에.
# 네 번째 꼭짓점이 +z 이므로 가상 물은 정확히 (0, 0, PROBE) 에 온다.
METAL = np.zeros(3)
_A = np.radians(70.5288)                       # 사면체: -z 축에서 이만큼
_DIRS = np.array([[np.sin(_A) * np.cos(np.radians(p)),
                   np.sin(_A) * np.sin(np.radians(p)),
                   -np.cos(_A)] for p in (0, 120, 240)])
COORDS = 2.05 * _DIRS                          # His N 3개
WAT = METAL + sf.PROBE * np.array([0.0, 0.0, 1.0])


def atom(name, resn, resi, xyz, elem=None, rec="ATOM"):
    return {"rec": rec, "name": name, "resn": resn, "chain": "A",
            "resi": resi, "xyz": np.asarray(xyz, float),
            "elem": (elem or name[0]).upper()}


def place(d, alpha_deg, phi_deg=0.0):
    """물 자리에서 거리 d, Zn–Owat 축에서 alpha 만큼 벌어진 지점.

    alpha=0   → Zn 쪽 (물과 금속 사이)
    alpha=109 → Thr199 가 앉는 자리
    alpha=180 → Zn 반대쪽. 친핵 공격 방향, 즉 CO2 통로 한복판.
    """
    a, p = np.radians(alpha_deg), np.radians(phi_deg)
    return WAT + d * np.array([np.sin(a) * np.cos(p),
                               np.sin(a) * np.sin(p), -np.cos(a)])


def base_atoms():
    """배위 His N 3개. _protein 이 SELF_TOL 로 걸러내야 하는 것들."""
    return [atom("NE2", "HIS", r, c, "N")
            for r, c in zip((94, 96, 119), COORDS)]


def classify(extra):
    return sf.classify_neighbors(base_atoms() + extra, COORDS, METAL)


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append((name, detail))


# ── 테스트 ────────────────────────────────────────────────────────────
def test_steric_cut_values():
    want = {"C": 3.22, "N": 3.07, "O": 3.04, "S": 3.32}
    for e, v in want.items():
        got = sf.steric_cut(e)
        check(f"steric_cut({e})", abs(got - v) < 0.005, f"{got:.3f} != {v}")
    check("steric_cut(미지 원소)=탄소값",
          abs(sf.steric_cut("X") - 3.22) < 0.005)


def test_carbon_far_is_clean():
    nb = classify([atom("CB", "LEU", 200, place(3.5, 100), "C")])
    check("C 3.5Å → 충돌 아님", len(nb["clash"]) == 0, f"{len(nb['clash'])}개")


def test_carbon_near_is_clash():
    nb = classify([atom("CB", "LEU", 200, place(2.9, 100), "C")])
    check("C 2.9Å → 충돌", len(nb["clash"]) == 1 and len(nb["hbond"]) == 0)


def test_polar_hbond_geometry_exempt():
    """Thr199 시나리오. 2.7Å 은 O 의 vdW 합(3.04) 아래지만 충돌이 아니다."""
    nb = classify([atom("OG1", "THR", 199, place(2.7, 110), "O")])
    check("O 2.7Å/110° → 수소결합, 충돌 아님",
          len(nb["hbond"]) == 1 and len(nb["clash"]) == 0,
          f"hbond={len(nb['hbond'])} clash={len(nb['clash'])}")


def test_polar_wrong_angle_is_clash():
    """같은 거리라도 180° 근처면 기질 통로를 막는 것이지 수소결합이 아니다."""
    nb = classify([atom("OD1", "ASN", 201, place(2.7, 175), "O")])
    check("O 2.7Å/175° → 충돌, 수소결합 아님",
          len(nb["clash"]) == 1 and len(nb["hbond"]) == 0)


def test_polar_too_close_is_clash():
    nb = classify([atom("OG1", "THR", 199, place(1.9, 110), "O")])
    check("O 1.9Å/110° → 충돌 (HB_LO 아래)",
          len(nb["clash"]) == 1 and len(nb["hbond"]) == 0)


def test_nitrogen_exemption_actually_fires():
    """2.9Å 은 N 의 vdW 합(3.07) 아래 — 면제가 없으면 충돌로 잡힌다."""
    nb = classify([atom("ND1", "HIS", 202, place(2.9, 100), "N")])
    check("N 2.9Å/100° → 수소결합 (면제 동작 확인)",
          len(nb["hbond"]) == 1 and len(nb["clash"]) == 0)


def test_mutual_exclusion():
    extra = [atom("OG1", "THR", 199, place(2.7, 110), "O"),
             atom("CB", "LEU", 200, place(2.9, 100, 90), "C"),
             atom("OD1", "ASN", 201, place(2.7, 175), "O"),
             atom("NZ", "LYS", 203, place(3.2, 95, 200), "N")]
    nb = classify(extra)
    ids_c = {id(a) for _, _, a in nb["clash"]}
    ids_h = {id(a) for _, _, a in nb["hbond"]}
    check("clash 와 hbond 가 겹치지 않음", not (ids_c & ids_h))
    check("모든 원자가 최대 한 번만 분류됨",
          len(ids_c) + len(ids_h) <= len(extra))


def test_coordinating_atoms_excluded():
    """배위 N 은 물에서 3.31Å 에 있다. _protein 이 안 빼면 여기 섞여 든다."""
    nb = classify([])
    labels = ([a["resi"] for _, _, a in nb["clash"]] +
              [a["resi"] for _, _, a in nb["hbond"]])
    check("배위원자 자신은 분류에서 제외", labels == [], f"{labels}")


def test_metric_keys_unchanged():
    """키 이름이 바뀌면 metal_site_score.py 의 passes() 와 CSV 가 조용히 깨진다."""
    nb = classify([atom("OG1", "THR", 199, place(2.7, 110), "O")])
    w, s = sf.water_site_open(nb), sf.second_shell(nb)
    for k in ("water_clash", "water_min_dist", "water_blocker"):
        check(f"키 존재: {k}", k in w)
    for k in ("shell_hbond_n", "shell_hbond_dist",
              "shell_hbond_angle", "shell_hbond_resid"):
        check(f"키 존재: {k}", k in s)
    check("Thr199 가 파트너로 보고됨", "199" in str(s.get("shell_hbond_resid")),
          str(s.get("shell_hbond_resid")))


def test_blocker_is_deepest_not_nearest():
    """TODO — 네가 채울 것.

    큰 원자가 살짝 닿은 것보다 작은 원자가 깊이 박힌 게 더 심각하다.
    S 를 3.1Å(침범 0.22)에, O 를 2.7Å 이 아닌 175° 방향 2.9Å(침범 0.14)에 놓으면
    '가장 가까운 것'과 '가장 깊이 박힌 것'이 달라진다.
    water_blocker 가 어느 쪽을 보고해야 하는지 정하고, 그걸 검증해라.
    """
    pass


def test_undefined_water_fallback():
    """TODO — 네가 채울 것.

    배위원자 3개를 일직선이나 한 점에 몰아넣으면 fourth_vertex 가 None 을 낸다.
    그때 classify_neighbors 가 None 을 반환하고,
    water_clash == -1, water_blocker == "undef", shell_hbond_n == 0
    이 나오는지 확인해라. 이 폴백은 원래 코드에 있던 계약이다.
    """
    pass


# ── native 검수 ───────────────────────────────────────────────────────
def native_check(path):
    from metal_site_score import parse_pdb, resolve_site, find_metal
    site = [{"chain": "A", "resi": r, "resn": "HIS", "atom": "auto"}
            for r in (94, 96, 119)]
    atoms = parse_pdb(path)
    fit = resolve_site(atoms, site, metal_xyz=find_metal(atoms))
    if fit is None:
        print(f"\n[native] 배위원자 추출 실패: {path}")
        print("  → renumber_site.py 를 먼저 돌렸는지 확인")
        return
    r = sf.function_metrics(atoms, fit["coords"], fit["metal"])

    print(f"\n── native 검수: {path} " + "─" * 30)
    print(f"  metal_source       {fit['source']}")
    print(f"  water_clash        {r['water_clash']}        기대: 0")
    print(f"  water_blocker      {r['water_blocker'] or '(없음)'}")
    print(f"  water_min_dist     {r['water_min_dist']:.2f} Å")
    print(f"  shell_hbond_n      {r['shell_hbond_n']}        기대: >= 1")
    print(f"  shell_hbond_resid  {r['shell_hbond_resid']}   기대: THR199")
    print(f"  shell_hbond_dist   {r['shell_hbond_dist']:.2f} Å   참고: 2.6~3.0")
    print(f"  shell_hbond_angle  {r['shell_hbond_angle']:.1f}°   참고: 100~120")
    print(f"  escape_frac        {r['escape_frac']:.3f}")
    print(f"  pocket_ratio       {r['pocket_ratio']:.3f}")

    check("native: Thr199 가 2차 배위권으로 검출됨",
          "199" in str(r["shell_hbond_resid"]),
          f"실제 = {r['shell_hbond_resid']!r}")

    print("\n  ※ 거리·각도는 합격/불합격이 아니라 '실측값'이다.")
    print("    HB_LO/HB_HI/HB_ANG_* 와 --max-lp-dev 를 이 숫자로 다시 정할 것.")
    print("    ※ 가상 물은 4번째 꼭짓점 방향 2.0Å 지점이라 결정 구조의")
    print("      수산화물과 0.2~0.4Å 어긋난다. 2.6 대신 2.9 가 나와도 정상일 수 있다.")


# ── 실행 ──────────────────────────────────────────────────────────────
def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        try:
            t()
        except NotImplementedError:
            FAIL.append((t.__name__, "아직 구현 안 됨"))
        except Exception as e:
            FAIL.append((t.__name__, f"{type(e).__name__}: {e}"))

    if len(sys.argv) > 1:
        try:
            native_check(sys.argv[1])
        except Exception as e:
            FAIL.append(("native_check", f"{type(e).__name__}: {e}"))

    print()
    for name, detail in FAIL:
        print(f"  ✗ {name}" + (f"  — {detail}" if detail else ""))
    for name, _ in PASS:
        print(f"  ✓ {name}")
    print(f"\n{len(PASS)} 통과 / {len(FAIL)} 실패")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
