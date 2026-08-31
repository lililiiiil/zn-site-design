#!/usr/bin/env python3
"""
compare_scores.py — 채점 CSV 두 개를 같은 pdb 행끼리 대조한다.

용도 두 가지 (같은 스크립트, 읽는 줄만 다르다):

  1) 앵커 검증   : 옛 CSV vs 새로 돌린 explicit CSV
                   → [숫자] 최대차가 전부 0 이어야 통과
  2) 모드 대조   : explicit CSV vs virtual CSV
                   → [coord_atoms] 불일치 개수 + [순위] 겹침을 본다

사용:
  python compare_scores.py A.csv B.csv
  python compare_scores.py A.csv B.csv --top 5 --pool 10
  python compare_scores.py A.csv B.csv --exclude id09     (붕괴 모델 분리)

의존성 없음 (표준 라이브러리만).
"""
import argparse, csv, os, re, sys

# 순위 비교에 쓸 컬럼과 방향. water_min_dist 는 클수록 물 자리가 열려 있다.
DEFAULT_RANK_COL = "water_min_dist"

# 숫자로 대조할 컬럼 (있는 것만 자동으로 씀)
NUM_COLS = [
    "metal_fit_residual", "angle_rmsd", "lone_pair_dev",
    "water_clash", "water_min_dist",
    "shell_hbond_n", "shell_hbond_dist", "shell_hbond_angle",
    "pocket_hydrophobic", "pocket_polar", "pocket_ratio", "escape_frac",
    "coord_rmsd_vs_ref", "pair_dist_rmsd_vs_ref",
]
# 문자로 대조할 컬럼
STR_COLS = ["coord_atoms", "metal_source", "water_blocker",
            "shell_hbond_resid", "nb_status", "pass"]


def norm_key(v):
    """pdb 컬럼 정규화. 옛 CSV엔 경로가 통째로 들어있을 수 있다
    (윈도우 백슬래시 포함). 양쪽 다 파일명만 남긴다."""
    v = (v or "").strip().replace("\\", "/")
    return os.path.basename(v)


def load(path):
    """pdb(정규화) -> 행 dict. 중복 키는 경고하고 첫 것을 쓴다."""
    rows, dup = {}, []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            k = norm_key(r.get("pdb"))
            if not k:
                continue
            if k in rows:
                dup.append(k)
                continue
            rows[k] = r
    return rows, dup


def as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_a", help="기준 CSV (옛것 / explicit)")
    ap.add_argument("csv_b", help="비교 CSV (새것 / virtual)")
    ap.add_argument("--top", type=int, default=5, help="A 기준 상위 몇 개를 볼지")
    ap.add_argument("--pool", type=int, default=10, help="B 기준 상위 몇 개 안에 남는지")
    ap.add_argument("--rank-col", default=DEFAULT_RANK_COL, help="순위 비교 컬럼")
    ap.add_argument("--rank-asc", action="store_true",
                    help="작을수록 좋은 컬럼일 때 (기본은 클수록 좋음)")
    ap.add_argument("--exclude", default=None,
                    help="이 문자열이 pdb 이름에 있으면 본 집계에서 빼고 따로 센다 (예: id09)")
    ap.add_argument("--group", action="store_true",
                    help="설계 단위(_model_N 제거)로 묶어 순위를 한 번 더 본다")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="앵커 검증에서 '같다'로 볼 최대 절대차")
    args = ap.parse_args()

    A, dupA = load(args.csv_a)
    B, dupB = load(args.csv_b)

    # ── 1. 행 수와 교집합 ────────────────────────────────────────────
    common_all = sorted(set(A) & set(B))
    print(f"[행] A {len(A)}개  /  B {len(B)}개  /  교집합 {len(common_all)}개")
    if dupA or dupB:
        print(f"  ! 중복 pdb 키 — A {len(dupA)}개, B {len(dupB)}개 (첫 행만 사용)")
    if not common_all:
        print("\n  ! 교집합 0개. pdb 값 형태가 서로 다르다. 앞 3개씩 찍어본다:")
        print("    A:", list(A)[:3])
        print("    B:", list(B)[:3])
        sys.exit(1)
    only_a = sorted(set(A) - set(B))
    only_b = sorted(set(B) - set(A))
    if only_a:
        print(f"  A에만 {len(only_a)}개: {', '.join(only_a[:5])}{' …' if len(only_a) > 5 else ''}")
    if only_b:
        print(f"  B에만 {len(only_b)}개: {', '.join(only_b[:5])}{' …' if len(only_b) > 5 else ''}")

    # 제외군 분리
    if args.exclude:
        common = [k for k in common_all if args.exclude not in k]
        excl = [k for k in common_all if args.exclude in k]
        print(f"  제외군('{args.exclude}') {len(excl)}개는 따로 집계 → 본 집계 {len(common)}개")
    else:
        common, excl = common_all, []

    def block(keys, title):
        if not keys:
            return
        print(f"\n===== {title} ({len(keys)}개) =====")

        # ── 2. 문자 컬럼 일치 ────────────────────────────────────────
        for c in STR_COLS:
            if not any(c in A[k] and c in B[k] for k in keys):
                continue
            bad = [k for k in keys
                   if (A[k].get(c) or "").strip() != (B[k].get(c) or "").strip()]
            # metal_source 가 explicit -> virtual 로만 갈리는 건 스위치가 먹었다는
            # 뜻이지 불일치가 아니다. 목록을 찍지 않는다.
            if c == "metal_source" and bad and all(
                    (A[k].get(c) or "").strip() == "explicit"
                    and (B[k].get(c) or "").strip() == "virtual" for k in bad):
                print(f"[{c}] explicit -> virtual {len(bad)}개 (스위치 작동 — 예상된 차이)")
                continue
            mark = "일치" if not bad else f"불일치 {len(bad)}개"
            print(f"[{c}] {len(keys) - len(bad)}/{len(keys)} 일치  → {mark}")
            for k in bad[:10]:
                print(f"    {k}:  A={A[k].get(c)!r}   B={B[k].get(c)!r}")
            if len(bad) > 10:
                print(f"    … 외 {len(bad) - 10}개")

        # ── 3. 숫자 컬럼 최대 절대차 ─────────────────────────────────
        print("\n[숫자] 컬럼별 최대 절대차  (앵커 검증이면 전부 0 이어야 함)")
        print(f"  {'컬럼':<24}{'최대차':>10}   {'그 행':<28}{'A':>10}{'B':>10}")
        for c in NUM_COLS:
            worst = None
            for k in keys:
                a, b = as_float(A[k].get(c)), as_float(B[k].get(c))
                if a is None or b is None:
                    continue
                d = abs(a - b)
                if worst is None or d > worst[0]:
                    worst = (d, k, a, b)
            if worst is None:
                continue
            d, k, a, b = worst
            flag = "" if d <= args.tol else "  <-- 다름"
            print(f"  {c:<24}{d:>10.4f}   {k[:28]:<28}{a:>10.3f}{b:>10.3f}{flag}")

        # ── 4. 순위 겹침 ─────────────────────────────────────────────
        rc = args.rank_col
        rank_keys = [k for k in keys
                     if as_float(A[k].get(rc)) is not None
                     and as_float(B[k].get(rc)) is not None]
        if len(rank_keys) >= 2:
            rev = not args.rank_asc
            sa = sorted(rank_keys, key=lambda k: as_float(A[k][rc]), reverse=rev)
            sb = sorted(rank_keys, key=lambda k: as_float(B[k][rc]), reverse=rev)
            top_a, pool_b = sa[:args.top], set(sb[:args.pool])
            kept = [k for k in top_a if k in pool_b]
            arrow = "작을수록 좋음" if args.rank_asc else "클수록 좋음"
            print(f"\n[순위] {rc} ({arrow}) — A 상위 {args.top} 중 "
                  f"B 상위 {args.pool} 안에 남은 것: {len(kept)}/{len(top_a)}")
            for k in top_a:
                ra = sa.index(k) + 1
                rb = sb.index(k) + 1
                print(f"    {'O' if k in pool_b else 'X'}  {k[:34]:<34}"
                      f"A {ra:>3}위 ({as_float(A[k][rc]):.3f})  →  "
                      f"B {rb:>3}위 ({as_float(B[k][rc]):.3f})")

    block(common, "본 집계")
    block(excl, f"제외군 '{args.exclude}'")

    # ── 5. 설계 단위 순위 ────────────────────────────────────────────
    # 모델 단위 순위는 한 설계의 5모델이 상위권을 독식하면 '설계를 고르는
    # 능력'을 시험하지 못한다. 실제 선별은 설계 단위로 하므로 여기서 한 번 더 본다.
    if args.group:
        rc = args.rank_col
        rev = not args.rank_asc
        pick = max if rev else min

        def by_design(src, keys):
            g = {}
            for k in keys:
                v = as_float(src[k].get(rc))
                if v is None:
                    continue
                d = re.sub(r"_model_\d+(\.pdb)?$", "", k)
                g.setdefault(d, []).append(v)
            return {d: pick(vs) for d, vs in g.items()}

        keys = common  # 제외군은 빼고 본다
        ga, gb = by_design(A, keys), by_design(B, keys)
        designs = sorted(set(ga) & set(gb))
        if len(designs) >= 2:
            arrow = "작을수록 좋음" if args.rank_asc else "클수록 좋음"
            sa = sorted(designs, key=lambda d: ga[d], reverse=rev)
            sb = sorted(designs, key=lambda d: gb[d], reverse=rev)
            top = min(args.top, len(designs))
            pool = min(args.pool, len(designs))
            kept = [d for d in sa[:top] if d in set(sb[:pool])]
            print(f"\n===== 설계 단위 순위 ({len(designs)}개 설계) =====")
            print(f"[집계] 설계마다 5모델 중 {'최대' if rev else '최소'} {rc} ({arrow})")
            print(f"[순위] A 상위 {top} 중 B 상위 {pool} 안에 남은 것: {len(kept)}/{top}")
            print(f"  {'':<3}{'설계':<30}{'A값':>8}{'A위':>5}   {'B값':>8}{'B위':>5}")
            for d in sa:
                ra, rb = sa.index(d) + 1, sb.index(d) + 1
                mark = ("O" if d in set(sb[:pool]) else "X") if ra <= top else " "
                print(f"  {mark:<3}{d[:30]:<30}{ga[d]:>8.3f}{ra:>5}   {gb[d]:>8.3f}{rb:>5}")


if __name__ == "__main__":
    main()