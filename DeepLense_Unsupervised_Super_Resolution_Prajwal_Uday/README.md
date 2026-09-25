# Unsupervised Super-Resolution of Gravitational Lensing Images

GSoC 2026, ML4SCI DeepLense. Mid-term submission.

**Prajwal Uday** · [Mid-term blog post](https://medium.com/@uprajwal20/unsupervised-super-resolution-of-gravitational-lensing-images-gsoc-2026-mid-term-ml4sci-584a46d8d946)

## What this does

Recovers the background galaxy behind a strong gravitational lens from a single
low-resolution image, with no high-resolution target and no labels.

Euclid and LSST are expected to find of order 100,000 galaxy-galaxy lenses, and
only a small fraction will get follow-up imaging. Supervised super-resolution
needs matched low- and high-resolution pairs of the same object, which will not
exist for most of that population. The approach here uses the physics of lensing
as the supervision instead: a candidate lens and source are ray-traced, blurred
by the instrument, binned onto detector pixels, and compared with the observed
image. The chi-squared against those pixels is the only training signal.

## The problem this fixes

An earlier grid-based pipeline on this dataset recovered sources 10 to 13 times
too large, and that did not improve across a 100x sweep of the regularisation
strength. The cause was not the network. It was the forward model: the lens was
a fixed circular profile with one Einstein radius shared across every image, so
it had no way to match the elliptical geometry the data actually contain.

Replacing it with a lens fitted per image brings the recovered source size to
within about 6 per cent of truth.

## Method

Fourteen numbers are fitted to each image, with nothing read from the truth
tables:

| component | parameters |
|---|---|
| lens: elliptical power law + external shear | Einstein radius, radial slope, 2 ellipticity, 2 shear |
| source: elliptical Sersic | amplitude, half-light radius, index, 2 ellipticity, 2 position |
| sky | background level |

The forward model traces rays back from each image pixel to the source plane,
samples the source there, convolves with the instrument response, and
area-averages onto detector pixels. The deflection field is validated against
`lenstronomy` to 3e-15.

Two further points matter for the result:

- **The PSF is measured, not assumed.** Every previous run on this data used a
  0.10 arcsec Gaussian. Deconvolving the stored source against the analytic
  profile over 40 systems gives 0.18 to 0.20 arcsec with wings far broader than
  a Gaussian.
- **The starting point comes from the image itself.** The Einstein radius and
  source offset are measured from the ring geometry, to about 0.45 pixels. This
  is what keeps the method unsupervised.

## Results

On 2,000 held-out validation images across three dark matter classes.

Parameter recovery, Spearman rank correlation against truth:

| parameter | rho |
|---|---|
| Einstein radius | +0.973 |
| source half-light radius | +0.947 |
| source Sersic index | +0.956 |
| source offset | +0.925 |
| lens ellipticity | +0.813 |
| external shear | +0.645 |
| radial slope | +0.590 |

Source reconstruction against the stored truth:

| metric | this work | earlier grid pipeline |
|---|---|---|
| correlation | 0.992 | -- |
| size ratio | 1.061 | 10.6 to 12.8 |
| nmse | 0.017 | -- |

Quantities that never entered the fit, as a check on the lens itself: total
magnification 5.83 against a true 5.89, and the tangential critical curve
recovered to 0.56 detector pixels.

Is the sub-pixel detail real? A Gaussian clump of 0.76 detector pixels was
injected into the source, re-rendered through the full instrument model at
matched noise, and reconstructed with and without it. Over 2,400 injections
across 60 images, a median of 18 per cent of the injected contrast comes back.

## Layout

```
src/                       physics and models, imported by everything else
  backend.py               numpy / torch shim, so the physics is one source file
  lens_models.py           EPL + external shear deflection, convergence, magnification
  sources.py               elliptical Sersic, and a free pixel grid
  raytrace.py              the observation operator: source -> lens -> PSF -> pixels
  theta_e_init.py          ring geometry, for the unsupervised starting point
  data_a.py                dataset reader, raw units, per-image noise
  metrics.py               source-plane scoring
  backproject.py           source-plane back-projection and sampler
  sisr_net.py              the super-resolution network

scripts/                   everything that is meant to be run
  calibrate_psf.py         measures the instrument response
  fit_per_image.py         the per-image fit (Levenberg-Marquardt, 14 parameters)
  evaluate.py              scores a fit against truth
  truth_chi2.py            the chi-squared a perfect fit would reach
  superresolve.py          renders the fitted source on finer grids
  train_b4.py   eval_b4.py      back-projection + SISR on the fitted lens
  train_b5_mu.py eval_b5_mu.py  magnification-gated variant
  train_pathb.py eval_pathb.py  amortised network, one pass for the parameters
  refine_pathb.py          uses that network as a warm start for the fit
  magnification_extract.py magnification field and critical curves
  mu_resolution.py         does magnification predict where SR works?
  lens_ladder.py           deflection-family ablation
  make_figures*.py         all figures, built from files already on disk

tests/                     physics gates, run these first
results/                   fitted parameters, metrics, figures
```

Scripts add `src/` to `sys.path` themselves, so there is nothing to install and
no package to build.

## Running it

Everything runs from the repository root. `--root` points at the folder that
contains `Model_A/`.

```bash
pip install -r requirements.txt

python tests/run_all.py                                     # physics gates
python scripts/calibrate_psf.py --root . --n 40             # measure the PSF
python scripts/fit_per_image.py --root . --n 2000 --classes axion cdm wdm \
    --target image --psf-mode empirical --supersample 1 \
    --out results/fits_img.json
python scripts/evaluate.py --fits results/fits_img.json --root .
python scripts/superresolve.py --fits results/fits_img.json --root . --factors 1 2 4 8
python scripts/make_figures.py --root .
```

`tests/run_all.py` should end in five PASS blocks. If any of them fails,
everything downstream is meaningless. Two of the five read Model_A and look for
it beside the repository; set `MODELA_ROOT` if it lives somewhere else.

The dataset is the DeepLense Model_A simulation: 127x127 Roman-like images at
0.10593 arcsec per pixel, three dark matter classes, with the noiseless source
and the simulation parameters stored alongside for scoring. Truth is used only
after a fit is written to disk.

Checkpoints (`*.pt`) and example dumps (`*.npz`) are not versioned; the training
scripts write them into `results/` when you run them.

## Known limitations

- Simulated data only. Nothing here has met real galaxy morphology or a real PSF.
- One instrument, and one of the two available bands.
- The sources in this dataset really are Sersic profiles, so the 7-parameter fit
  has exactly the right prior. It is a ceiling, not a competitor. The question
  the free-form rows answer is how close a model with no parametric assumption
  can get, since that is what real galaxies will need.
- Point estimates only, no posteriors.