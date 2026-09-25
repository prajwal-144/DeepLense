import sys, os, warnings
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_ROOT, "src"), os.path.join(_ROOT, "scripts")]
warnings.filterwarnings("ignore")
import numpy as np
from data_a import ModelADataset
from sources import SersicSource
from raytrace import render, image_plane_grid, gaussian_psf, area_downsample

# Model_A is expected three levels up from this file, i.e. beside the repo.
# Set MODELA_ROOT if it lives somewhere else.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.environ.get("MODELA_ROOT", _REPO_ROOT)
RES, N = 0.10593, 127
PSF_FWHM = 0.18

def corr(a, b, w=None):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    w = np.ones_like(a) if w is None else np.asarray(w, float).ravel()
    W = w.sum(); a = a - (w*a).sum()/W; b = b - (w*b).sum()/W
    return float((w*a*b).sum()/np.sqrt((w*a*a).sum()*(w*b*b).sum()+1e-30))

def truth_params(t, circular=True):
    lens = dict(theta_E=t["theta_E"], gamma=t["host_slope"],
                e1=0.0 if circular else t["host_e1"],
                e2=0.0 if circular else t["host_e2"],
                g1=t["gamma1_ext"], g2=t["gamma2_ext"], cx=0.0, cy=0.0)
    src = dict(amp=1.0, R_sersic=t["source_R_sersic"], n_sersic=t["source_n_sersic"],
               se1=t["source_e1"], se2=t["source_e2"], sx=t["source_x"], sy=t["source_y"])
    return lens, src

def main():
    ds = ModelADataset(ROOT, split="val", classes=["axion"], limit=40)
    print(ds.summary())
    grid = image_plane_grid(N, RES, 3)
    psf = gaussian_psf(PSF_FWHM, RES/3)
    cs, cs_amp, cs_rec, host_e = [], [], [], []
    for i in range(len(ds)):
        t = ds.truth(i)
        lens, sp = truth_params(t, circular=True)
        pred = render(SersicSource(sp), lens, N, RES, psf=psf, supersample=3, grid=grid)
        ref = ds.image_nss(i)
        w = (ref > 0.1*ref.max()).astype(float) + 0.02
        cs.append(corr(pred, ref, w))
        # amplitude-matched residual
        s = (pred*ref).sum()/max((pred*pred).sum(), 1e-30)
        cs_amp.append(float(np.sqrt(((s*pred-ref)**2).sum()/ (ref**2).sum())))
        # regression probe: the recorded ellipticity, NOT asserted on
        lens_r, _ = truth_params(t, circular=False)
        pred_r = render(SersicSource(sp), lens_r, N, RES, psf=psf, supersample=3,
                        grid=grid)
        cs_rec.append(corr(pred_r, ref, w))
        host_e.append(float(np.hypot(t["host_e1"], t["host_e2"])))
    cs = np.array(cs); cs_amp = np.array(cs_amp)
    cs_rec = np.array(cs_rec); host_e = np.array(host_e)
    print(f"\nrender(TRUE params, CIRCULAR macro lens) vs image_nss, n={len(cs)}")
    print(f"arc-weighted correlation : median {np.median(cs):.4f}   p10 {np.percentile(cs,10):.4f}   min {cs.min():.4f}")
    print(f"amplitude-matched nrmse  : median {np.median(cs_amp):.4f}")
    print(" (a second-order term is still unmodelled; this is not the")
    print(" detector noise floor)")

    # arm B: is the recorded ellipticity still inert?
    print("\nregression probe: same images rendered with the recorded host_e1/e2:")
    print(f"   arc-weighted correlation : median {np.median(cs_rec):.4f}   p10 {np.percentile(cs_rec,10):.4f}   min {cs_rec.min():.4f}")
    for lo, hi in ((0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 1.0)):
        m = (host_e >= lo) & (host_e < hi)
        if m.sum():
            print(f"     |host_e| in [{lo:.1f},{hi:.1f}): n={m.sum():2d}   "
                  f"circular {np.median(cs[m]):.4f}   recorded {np.median(cs_rec[m]):.4f}")
    if np.median(cs_rec) > np.median(cs):
        print("   *** the recorded ellipticity now beats the circular lens, so")
        print("       Model_A may have been regenerated with host_e1/e2 applied")
        print("       to the mass model. The circular assumption here and in")
        print("       metrics.py would then need revisiting. ***")
    else:
        print("   host_e1/e2 remain inert; the circular lens still wins")

    # supersampling convergence, measured
    print("\nsupersampling error vs an S=9 reference (same params, noiseless):")
    t = ds.truth(0); lens, sp = truth_params(t)
    ref9 = render(SersicSource(sp), lens, N, RES, psf=gaussian_psf(PSF_FWHM, RES/9),
                  supersample=9)
    for S in (1, 2, 3, 5):
        p = render(SersicSource(sp), lens, N, RES, psf=gaussian_psf(PSF_FWHM, RES/S),
                   supersample=S)
        rel = np.abs(p - ref9).max()/ref9.max()
        dflux = abs(p.sum() - ref9.sum())/ref9.sum()
        print(f"   S={S}   max |dI|/Imax = {rel:.2e}    total flux error = {dflux:.2e}")

    # area_downsample conserves the mean exactly
    a = np.random.default_rng(0).random((1, 1, 12, 12))
    d = area_downsample(a, 3)
    print(f"\narea_downsample mean preserved: |d.mean - a.mean| = {abs(d.mean()-a.mean()):.2e}")

    ok = np.median(cs) > 0.9 and abs(d.mean()-a.mean()) < 1e-12
    print("\n" + ("PASS" if ok else "*** FAIL ***"))
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())