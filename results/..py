import pandas as pd
v1 = pd.read_csv('2026-07-23정리/zn-site-design/results/scores_v1.csv')
v2 = pd.read_csv('2026-07-23정리/zn-site-design/results/scores_v2.csv')

print(v2[['shell_hbond_n','escape_frac','metal_fit_residual','angle_rmsd','lone_pair_dev']].describe())
print(v2.nb_status.value_counts())      # 폴백 몇 개인지 먼저