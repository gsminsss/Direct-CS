"""Generalized copy of OPT4B: constants and inventory checks follow the variant."""
import sys
src=open(sys.argv[1]).read()
def rep(o,n,c=1):
    global src; assert src.count(o)==c,(src.count(o),o[:60]); src=src.replace(o,n)
rep('''        "CAT_POST":105,"CORE_POST":161,"DOWN_DIRECT_D5_D13":4264,''','''        "CAT_POST":105,"CORE_POST":161,"DOWN_DIRECT_D5_D13":2*DOWNSTREAM_LOCATIONS_PER_PATCH,''')
rep('''    if total!=expected or expected!=22804951904:''','''    if total!=expected:''')
rep('''    cert.EXPECTED_DETECTORS=431''','''    cert.EXPECTED_DETECTORS=OP_DETECTORS+TRUTH_PER_PROBE''')
rep('''    if rh!=263 or ra<265: raise RuntimeError(("matrix drift",rh,ra))''','''    if rh!=OP_DETECTORS or ra<OP_DETECTORS+2: raise RuntimeError(("matrix drift",rh,ra))''')
open(sys.argv[2],"w").write(src); print("ok")
