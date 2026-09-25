import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = ["test_lens_models.py", "test_sources.py", "test_jacobian.py",
         "test_raytrace.py", "test_theta_e_init.py"]
def main():
    fails = []
    for t in TESTS:
        print("\n" + "=" * 80); print(t); print("=" * 80)
        r = subprocess.run([sys.executable, os.path.join(HERE, t)])
        if r.returncode != 0:
            fails.append(t)
    print("\n" + "=" * 80)
    print("ALL PASSED" if not fails else "FAILED: " + ", ".join(fails))
    print("test_backend_parity.py needs torch and is not run here.")
    return 1 if fails else 0
if __name__ == "__main__":
    raise SystemExit(main())