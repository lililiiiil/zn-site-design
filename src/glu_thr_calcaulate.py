from make_motif import parse_lines as pl
import numpy as np
start = pl(path = 'motif_B.pdb')

compoun = [a for a in start if (a["resi"] == 199 and a["name"] == "OG1") or (a["resi"] == 106 and a["name"] in ("OE1","OE2"))]
print(len(compoun))

glu_oxygens = [a for a in compoun if a["name"] in ("OE1", "OE2")]
thr_og1 = [a for a in compoun if a["name"] == "OG1"]


for a in glu_oxygens :
    cc = np.linalg.norm(a["xyz"]- thr_og1[0]["xyz"])
    print(a["name"],cc)
    