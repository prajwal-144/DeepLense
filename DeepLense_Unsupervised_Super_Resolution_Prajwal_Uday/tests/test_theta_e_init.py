import sys, os, warnings
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_ROOT, "src"), os.path.join(_ROOT, "scripts")]
warnings.filterwarnings("ignore")
import numpy as np
from data_a import ModelADataset
from theta_e_init import initial_guess

def main():
    # Model_A is expected three levels up from this file. MODELA_ROOT overrides.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    root = os.environ.get("MODELA_ROOT", repo_root)
    ds = ModelADataset(root, split="val", classes=["axion"], limit=120)
    e_t, e_b, comp = [], [], []
    for i in range(len(ds)):
        g = initial_guess(ds.image(i), 0.10593)
        t = ds.truth(i)
        e_t.append(abs(g["theta_E"] - t["theta_E"]))
        e_b.append(abs(np.hypot(g["sx"], g["sy"]) - t["beta"]))
        comp.append(g["ring_completeness"])
    e_t, e_b = np.array(e_t), np.array(e_b)
    print(f"n = {len(e_t)}")
    print(f"  |theta_E_est - theta_E_true|  median {np.median(e_t):.4f}\" "
          f"= {np.median(e_t)/0.10593:.2f} px   p90 {np.percentile(e_t,90)/0.10593:.2f} px")
    print(f"  |beta_est - beta_true|        median {np.median(e_b):.4f}\" "
          f"= {np.median(e_b)/0.10593:.2f} px")
    print(f"  ring completeness             median {np.median(comp):.2f}")
    ok = np.median(e_t) / 0.10593 < 1.5      # the old bank's bin width
    print("\n" + ("PASS  (below the 1.5 px bank bin width -> routing on the estimate "
                  "is indistinguishable from routing on truth)" if ok else "*** FAIL ***"))
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())