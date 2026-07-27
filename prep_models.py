import re, sys, zipfile
from pathlib import Path
from collections import Counter
import gemmi

zip_path, outdir = sys.argv[1], sys.argv[2]
DIR_RE = re.compile(r"^d5n3_(.+)_seed(\d+)$")
FILE_RE = re.compile(r"_model_(\d+)\.cif$")

out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
tmp = out / "_cif"; tmp.mkdir(exist_ok=True)

n = 0
with zipfile.ZipFile(zip_path) as z:
    cifs = sorted(e for e in z.namelist() if e.endswith(".cif"))
    print("zip 안 cif: %d개" % len(cifs))
    for e in cifs:
        p = e.split("/")
        md = DIR_RE.match(p[0]) if len(p) > 1 else None
        mf = FILE_RE.search(p[-1])
        if not md or not mf:
            print("  건너뜀:", e); continue
        name = "%s_s%s_m%s" % (md.group(1), md.group(2), mf.group(1))
        raw = tmp / (name + ".cif")
        raw.write_bytes(z.read(e))
        st = gemmi.read_structure(str(raw)); st.setup_entities()
        st.write_pdb(str(out / (name + ".pdb")))
        raw.unlink(); n += 1

try: tmp.rmdir()
except OSError: pass

print("\n변환 %d개 -> %s\n" % (n, out))
for k, v in sorted(Counter(f.name.rsplit("_s", 1)[0] for f in out.glob("*.pdb")).items()):
    print("  %-12s %d" % (k, v))
