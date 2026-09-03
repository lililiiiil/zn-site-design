#  LigandMPNN 서열 생성
# 2라운드 v2 명령과 동일. 두 인자만 교체:
#   --omit_AA_per_residue : ../design/round2/omit_his.json -> ../omit_r3.json
#   --out_folder          : ./outputs/round2_v2 -> ./outputs/round3_v1
# 나머지(checkpoint, pdb_path, fixed_residues, omit_AA, seed, temperature, batch_size)는 통제 변수.
# 주: 문서의 pdb_path ../design/round2/sample_0.pdb 는 파일 위치가 ../design/sample_0.pdb 로
#     바뀌어 있음. 동일 파일(754 원자, ZN B346 @ 1.609 1.312 0.485)이며 경로만 갱신.

Set-Location C:\naver_news\ZN\zn-site-design
.\mpnn_env\Scripts\Activate.ps1
Set-Location .\LigandMPNN

python run.py --model_type ligand_mpnn `
--checkpoint_ligand_mpnn ./model_params/ligandmpnn_v_32_020_25.pt `
--pdb_path ../design/sample_0.pdb `
--out_folder ./outputs/round4_t03 `
--fixed_residues "A31 A33 A35 A52 A77 A81 A112" `
--omit_AA "C" `
--omit_AA_per_residue ../omit_r3.json `
--seed 20260903 --temperature 0.3 --batch_size 30
