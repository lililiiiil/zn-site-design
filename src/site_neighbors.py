from metal_site_score import parse_pdb as par
from metal_site_score import find_metal as fme
import numpy as np

stop = {31,33,35,52,77,81,112}
def parsing(path) :
    pa = par(path)
    fm = fme(pa)

    if fm is None: 
        raise ValueError(f"금속 원자 없음: {path}")
    
    calclated = [(a["resi"], np.linalg.norm(a["xyz"] - fm) ) for a in pa if a["name"] == "CA" and 
                 a["resi"] not in stop and 
                  a["rec"] == "ATOM"]


    return  sorted(calclated, key= lambda t:t[1])
