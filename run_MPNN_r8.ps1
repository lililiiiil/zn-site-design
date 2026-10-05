# LigandMPNN - Round 8 (8 backbones x 5 sequences = 40)
#
# 입력은 8라 RFD3 산출물 중 "물자리 열림 + 곁사슬충돌 0 + 끊김 0" 인 8개.
#   설정 rfd3/rfd3_r8_lean_co2oh_thr.json  (ligand ZN,CO2,OH + Thr199 로타머 고정)
#   채점 results/r8/scores_r8_lean_co2oh_thr_2026-10-05.csv
#
# r5 와 달라진 것: 백본 PDB 에 CO2 와 OH 가 들어 있다 (체인 B, 리간드 원자 5개).
# LigandMPNN 이 ZN 뿐 아니라 기질까지 보면서 서열을 설계한다. r5 는 ZN 1개만
# 봤다 ("The number of ligand atoms parsed is equal to: 1").
#
# Controlled: checkpoint, omit_AA "C", seed 20260911, temperature 0.2, batch_size 5
#             — r5 와 같게 둬서 라운드 간 비교가 성립하게 한다
# Variable  : pdb_path, fixed_residues (백본마다 모티프 8잔기, 설계 번호)
#
# NOTE 1: fixed_residues 는 각 모델의 diffused_index_map 에서 뽑았다.
#         RFD3 가 모티프의 사슬 내 위치를 정하므로 백본마다 번호가 다르다.
# NOTE 2: pdb_path MUST use forward slashes.
#         LigandMPNN builds the output name with  path.split("/")[-1]  --
#         with backslashes the whole absolute path becomes the filename and
#         open() fails with [Errno 22] Invalid argument.

$ROOT  = "C:\naver_news\ZN\zn-site-design"
$ROOTU = "C:/naver_news/ZN/zn-site-design"     # forward-slash form

Set-Location $ROOT
.\mpnn_env\Scripts\Activate.ps1
Set-Location "$ROOT\LigandMPNN"

$jobs = @(
  @{ n="b0_m1"; fix="A20 A62 A68 A108 A109 A153 A168 A170" },
  @{ n="b0_m2"; fix="A41 A68 A71 A160 A161 A164 A191 A195" },
  @{ n="b0_m3"; fix="A7 A12 A94 A122 A148 A175 A177 A178" },
  @{ n="b0_m4"; fix="A13 A14 A17 A18 A117 A179 A195 A198" },
  @{ n="b0_m6"; fix="A52 A73 A79 A88 A151 A152 A153 A198" },
  @{ n="b0_m7"; fix="A19 A39 A52 A137 A139 A156 A174 A193" },
  @{ n="b1_m0"; fix="A5 A53 A57 A69 A73 A91 A136 A139" },
  @{ n="b1_m2"; fix="A25 A80 A83 A84 A95 A120 A121 A140" }
)

$fail = 0
foreach ($j in $jobs) {
  $pdb = "$ROOTU/design/round_8/r8_$($j.n).pdb"
  $out = "$ROOTU/LigandMPNN/outputs/round8_$($j.n)"
  if (-not (Test-Path $pdb)) {
    Write-Host "MISSING: $pdb" -ForegroundColor Red
    $fail++
    continue
  }
  Write-Host "=== $($j.n) ===" -ForegroundColor Cyan
  python run.py --model_type ligand_mpnn `
    --checkpoint_ligand_mpnn "$ROOTU/LigandMPNN/model_params/ligandmpnn_v_32_020_25.pt" `
    --pdb_path $pdb `
    --out_folder $out `
    --fixed_residues $j.fix `
    --omit_AA "C" `
    --seed 20260911 --temperature 0.2 --batch_size 5
  if ($LASTEXITCODE -ne 0) { Write-Host "  FAILED: $($j.n)" -ForegroundColor Red; $fail++ }
}

Write-Host ""
Write-Host "done.  failed $fail / $($jobs.Count)" -ForegroundColor $(if ($fail -eq 0) {"Green"} else {"Yellow"})
