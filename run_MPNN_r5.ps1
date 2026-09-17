# LigandMPNN - Round 5 (11 backbones x 5 sequences = 55)
#
# Controlled: checkpoint, omit_AA "C", seed 20260911, temperature 0.2, batch_size 5
# Variable  : pdb_path, fixed_residues (per backbone)
#
# NOTE 1: backbone folder is  round_5  (underscore), not round5
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
  @{ n="b0_m1"; fix="A17 A43 A114 A128 A156 A185 A189" },
  @{ n="b0_m3"; fix="A33 A36 A50 A91 A96 A137 A162" },
  @{ n="b0_m5"; fix="A19 A35 A60 A63 A84 A113 A159" },
  @{ n="b0_m6"; fix="A8 A10 A12 A55 A81 A111 A112" },
  @{ n="b0_m7"; fix="A10 A53 A113 A115 A116 A134 A202" },
  @{ n="b1_m0"; fix="A13 A79 A93 A95 A97 A118 A120" },
  @{ n="b1_m1"; fix="A38 A84 A109 A116 A119 A144 A157" },
  @{ n="b1_m3"; fix="A17 A28 A51 A75 A85 A120 A122" },
  @{ n="b1_m5"; fix="A23 A24 A33 A91 A93 A126 A146" },
  @{ n="b1_m6"; fix="A59 A62 A87 A89 A90 A114 A137" },
  @{ n="b1_m7"; fix="A28 A49 A72 A79 A92 A95 A99" }
)

$fail = 0
foreach ($j in $jobs) {
  $pdb = "$ROOTU/design/round_5/r5_$($j.n).pdb"
  $out = "$ROOTU/LigandMPNN/outputs/round5_$($j.n)"
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
Write-Host "done.  failed $fail / 11" -ForegroundColor $(if ($fail -eq 0) {"Green"} else {"Yellow"})