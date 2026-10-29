"""
segmentation_agreement.py
ISSUE 1: do different observers (and the network) segment structures alike?

Input layout (one folder per observer, same case names in each):
    <root>/Anna/102.1.xcf, <root>/Ari/102.1.xcf, ...
Folders that contain no .xcf files (e.g. Imatges) are ignored.

Area structures compared: cavum, cerebellum, cisterna magna, nuchal fold
(layers 'cavum', 'cerebel'/'cerbel', 'cisterna magna', 'plec nucal').
Line layers (midline, silvio) are skipped: Dice is not meaningful for lines.

1) Pairwise Dice, every observer against every other, per case and
   structure (network included if --network is given).
2) STAPLE consensus per case and structure from the doctors only (never the
   network). Optional seniority weighting through MAP-STAPLE priors.
3) Agreement with the consensus: Dice of each observer and of the network
   against it, plus STAPLE's sensitivity/specificity per observer. For a
   doctor, the consensus is rebuilt without that doctor (leave-one-out), so
   doctors are not compared with a consensus they helped build.
4) Fleiss' kappa among the doctors, per case and structure.

    python segmentation_agreement.py "Estudi compartiu expertVSnoexpert" \
        --out agreement_results [--network Predictions/] \
        [--levels Anna=3,Ari=2,Clara=2,Cris=1,Judith=3 --prior-strength 0.05]

--network: folder of <case>.png label maps (1 cavum, 2 cerebellum,
3 cisterna magna, 4 nuchal fold), same size as the XCF canvas.

Outputs in --out:
    pairwise_dice.csv           case, structure, observer pair, Dice
    pairwise_summary.csv        per structure and pair: mean, SD, median, min, max
    network_vs_doctors.csv      per case: range of doctor-doctor Dice and where
                                the network's Dice to the doctors falls
    consensus_agreement.csv     Dice / sensitivity / specificity vs consensus
    fleiss_kappa.csv            per case and structure
    pairwise_dice_<structure>.png   heatmap of mean pairwise Dice
    consensus_dice_<structure>.png  boxplot of Dice vs consensus per observer
"""
import os
import argparse
import itertools

import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes, binary_dilation

STRUCTURES = ['cavum', 'cerebellum', 'cisterna_magna', 'nuchal_fold']
LAYER_TO_STRUCTURE = {'cavum': 'cavum', 'cerebel': 'cerebellum', 'cerbel': 'cerebellum',
                      'cisterna magna': 'cisterna_magna', 'plec nucal': 'nuchal_fold'}
LINE_LAYERS = {'midline', 'silvio'}
LABELS = {'cavum': 1, 'cerebellum': 2, 'cisterna_magna': 3, 'nuchal_fold': 4}
NETWORK = 'network'


# ----------------------------------------------------------------- reading
def layer_mask(layer, canvas_hw):
    """Binary mask of one GIMP layer placed on the full canvas (layers can be
    smaller than the canvas and offset), holes filled as in foetal_yago.py."""
    a = np.asarray(layer.image)
    m = a[..., -1] > 0 if (a.ndim == 3 and a.shape[2] in (2, 4)) else \
        (a.max(axis=-1) if a.ndim == 3 else a) > 0
    H, W = canvas_hw
    out = np.zeros((H, W), bool)
    x0, y0 = int(getattr(layer, 'xOffset', 0) or 0), int(getattr(layer, 'yOffset', 0) or 0)
    ys, xs = slice(max(y0, 0), min(y0 + m.shape[0], H)), slice(max(x0, 0), min(x0 + m.shape[1], W))
    out[ys, xs] = m[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
    return binary_fill_holes(out)


def masks_from_layers(layers, canvas_hw, warn=None):
    """dict structure -> mask from a list of GIMP layers."""
    masks = {}
    for layer in layers:
        name = str(layer.name).strip().lower()
        if '.dcm' in name or name in LINE_LAYERS:
            continue
        s = LAYER_TO_STRUCTURE.get(name)
        if s is None:
            if warn is not None:
                warn.add(layer.name)
            continue
        if s in masks:
            raise ValueError('two layers for {} ({})'.format(s, layer.name))
        masks[s] = layer_mask(layer, canvas_hw)
    return masks


def read_xcf(path, warn=None):
    from gimpformats.gimpXcfDocument import GimpDocument
    doc = GimpDocument(path)
    layers = [l for l in doc.raw_layers if not l.isGroup]
    W, H = getattr(doc, 'width', None), getattr(doc, 'height', None)
    if not W or not H:                         # fall back on the image layer
        img = [l for l in layers if '.dcm' in str(l.name).lower()] or layers[-1:]
        H, W = np.asarray(img[0].image).shape[:2]
    return masks_from_layers(layers, (int(H), int(W)), warn)


def read_network(folder, case, shape):
    import cv2
    path = os.path.join(folder, case + '.png')
    if not os.path.exists(path):
        return None
    lab = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if lab.shape[:2] != shape:
        raise ValueError('{}: network {} != annotation {}'.format(case, lab.shape[:2], shape))
    return {s: lab == v for s, v in LABELS.items()}


# ----------------------------------------------------------------- metrics
def dice(a, b):
    s = a.sum() + b.sum()
    return np.nan if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


def roi(masks, margin=10):
    """Region where any observer annotated, plus a margin: STAPLE and kappa
    are computed here so the huge empty background does not dominate."""
    u = np.any(masks, axis=0)
    return binary_dilation(u, iterations=margin) if u.any() else u


def staple(D, prior_mean=None, prior_strength=0.0, n_iter=200, tol=1e-7):
    """Binary STAPLE (Warfield et al. 2004); with prior_strength > 0 it becomes
    MAP-STAPLE (Commowick et al. 2012): Beta priors on each rater's
    sensitivity p and specificity q, centred on prior_mean[j], worth
    prior_strength * n_voxels pseudo-observations.
    D: (n_raters, n_voxels) bool. Returns W (prob. of foreground), p, q."""
    D = D.astype(float)
    R, N = D.shape
    pm = np.full(R, 0.9) if prior_mean is None else np.asarray(prior_mean, float)
    p, q = pm.copy(), pm.copy()                     # initialisation
    a = 1.0 + prior_strength * N * pm               # Beta(a, b), mode near pm
    b = 1.0 + prior_strength * N * (1.0 - pm)
    f = np.clip(D.mean(), 1e-6, 1 - 1e-6)           # prior probability of foreground
    W = D.mean(axis=0)
    for _ in range(n_iter):
        lp1 = np.log(f) + (D * np.log(p[:, None]) + (1 - D) * np.log(1 - p[:, None])).sum(0)
        lp0 = np.log(1 - f) + ((1 - D) * np.log(q[:, None]) + D * np.log(1 - q[:, None])).sum(0)
        W_new = 1.0 / (1.0 + np.exp(np.clip(lp0 - lp1, -50, 50)))
        sw, s1w = W_new.sum(), (1 - W_new).sum()
        p = np.clip(((D * W_new).sum(1) + a - 1) / (sw + a + b - 2), 1e-6, 1 - 1e-6)
        q = np.clip((((1 - D) * (1 - W_new)).sum(1) + a - 1) / (s1w + a + b - 2), 1e-6, 1 - 1e-6)
        done = np.abs(W_new - W).max() < tol
        W = W_new
        if done:
            break
    return W, p, q


def fleiss_kappa(D):
    """Fleiss' kappa for binary labels. D: (n_raters, n_items) bool."""
    n, N = D.shape
    if n < 2 or N == 0:
        return np.nan
    k1 = D.sum(0).astype(float)
    counts = np.stack([n - k1, k1], 1)
    P_i = (np.sum(counts ** 2, 1) - n) / (n * (n - 1))
    p_j = counts.sum(0) / (N * n)
    P_e = np.sum(p_j ** 2)
    return np.nan if P_e >= 1 else (P_i.mean() - P_e) / (1 - P_e)


# ----------------------------------------------------------------- main
def parse_levels(text):
    if not text:
        return {}
    out = {}
    for item in text.split(','):
        k, v = item.split('=')
        out[k.strip()] = float(v)
    return out


def level_to_prior(levels, observers, lo=0.85, hi=0.95):
    """Map expertise levels linearly to prior sensitivity/specificity in
    [lo, hi] (lowest level -> lo, highest -> hi). Unlisted observers get the
    middle value."""
    if not levels:
        return None
    vals = np.array(list(levels.values()))
    vmin, vmax = vals.min(), vals.max()
    mid = (lo + hi) / 2
    return np.array([lo + (hi - lo) * (levels[o] - vmin) / (vmax - vmin) if o in levels
                     and vmax > vmin else mid for o in observers])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('root')
    p.add_argument('--out', default='agreement_results')
    p.add_argument('--network', help='folder of <case>.png label maps')
    p.add_argument('--levels', help='e.g. Anna=3,Ari=2 (higher = more senior)')
    p.add_argument('--prior-strength', type=float, default=0.0,
                   help='MAP-STAPLE prior weight, as a fraction of ROI pixels (0 = plain STAPLE)')
    p.add_argument('--no-figures', action='store_true')
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    observers = sorted(d for d in os.listdir(args.root)
                       if os.path.isdir(os.path.join(args.root, d))
                       and any(f.endswith('.xcf') for f in os.listdir(os.path.join(args.root, d))))
    cases = sorted(set.union(*[set(f[:-4] for f in os.listdir(os.path.join(args.root, o))
                                   if f.endswith('.xcf')) for o in observers]))
    levels = parse_levels(args.levels)
    print('Observers:', ', '.join(observers), '| cases:', len(cases))

    warn = set()
    pair_rows, cons_rows, kappa_rows, net_rows = [], [], [], []
    for case in cases:
        ann = {}
        for o in observers:
            path = os.path.join(args.root, o, case + '.xcf')
            if os.path.exists(path):
                try:
                    ann[o] = read_xcf(path, warn)
                except Exception as ex:
                    print('ERROR reading', path, ':', ex)
        if not ann:
            continue
        shape = next(iter(next(iter(ann.values())).values())).shape if any(ann.values()) else None
        if args.network and shape is not None:
            net = read_network(args.network, case, shape)
            if net is not None:
                ann[NETWORK] = net

        for s in STRUCTURES:
            have = [o for o in ann if s in ann[o]]
            docs = [o for o in have if o != NETWORK]
            # 1) pairwise Dice
            for a, b in itertools.combinations(have, 2):
                pair_rows.append(dict(case=case, structure=s, obs1=a, obs2=b,
                                      dice=dice(ann[a][s], ann[b][s])))
            # network within the doctors' range?
            if NETWORK in have and len(docs) >= 2:
                dd = [dice(ann[a][s], ann[b][s]) for a, b in itertools.combinations(docs, 2)]
                nd = [dice(ann[NETWORK][s], ann[d][s]) for d in docs]
                net_rows.append(dict(case=case, structure=s,
                                     doctors_min=np.nanmin(dd), doctors_median=np.nanmedian(dd),
                                     doctors_max=np.nanmax(dd), network_mean=np.nanmean(nd),
                                     network_min=np.nanmin(nd), network_max=np.nanmax(nd),
                                     network_within=np.nanmin(dd) <= np.nanmean(nd)))
            if len(docs) < 2:
                continue
            # 4) Fleiss' kappa among doctors
            stack = np.stack([ann[o][s] for o in docs])
            R = roi(stack)
            kappa_rows.append(dict(case=case, structure=s, n_raters=len(docs),
                                   fleiss_kappa=fleiss_kappa(stack[:, R]), roi_pixels=int(R.sum())))
            # 2-3) STAPLE consensus and agreement with it
            def consensus(raters):
                st = np.stack([ann[o][s] for o in raters])
                Rr = roi(np.stack([ann[o][s] for o in have]))
                pm = level_to_prior(levels, raters)
                W, pp, qq = staple(st[:, Rr], pm, args.prior_strength)
                full = np.zeros(st.shape[1:], bool)
                full[Rr] = W >= 0.5
                return full, dict(zip(raters, pp)), dict(zip(raters, qq))
            cons_all, sens, spec = consensus(docs)
            for o in have:
                if o == NETWORK:
                    ref, se, sp, loo = cons_all, np.nan, np.nan, False
                elif len(docs) >= 3:
                    ref, _, _ = consensus([d for d in docs if d != o])
                    se, sp, loo = sens[o], spec[o], True
                else:
                    ref, se, sp, loo = cons_all, sens[o], spec[o], False
                cons_rows.append(dict(case=case, structure=s, observer=o,
                                      dice_vs_consensus=dice(ann[o][s], ref),
                                      staple_sensitivity=se, staple_specificity=sp,
                                      leave_one_out=loo))
        print('done', case)

    if warn:
        print('Ignored unknown layer names:', sorted(warn))
    pw = pd.DataFrame(pair_rows)
    pw.to_csv(os.path.join(args.out, 'pairwise_dice.csv'), index=False)
    summ = (pw.groupby(['structure', 'obs1', 'obs2'])['dice']
            .agg(['count', 'mean', 'std', 'median', 'min', 'max']).reset_index())
    summ.to_csv(os.path.join(args.out, 'pairwise_summary.csv'), index=False)
    pd.DataFrame(cons_rows).to_csv(os.path.join(args.out, 'consensus_agreement.csv'), index=False)
    pd.DataFrame(kappa_rows).to_csv(os.path.join(args.out, 'fleiss_kappa.csv'), index=False)
    if net_rows:
        pd.DataFrame(net_rows).to_csv(os.path.join(args.out, 'network_vs_doctors.csv'), index=False)

    # short printed summary
    cons = pd.DataFrame(cons_rows)
    for s in STRUCTURES:
        print('\n==', s)
        sub = pw[pw.structure == s]
        if len(sub):
            print('  pairwise Dice (all pairs): mean {:.3f}  SD {:.3f}  min {:.3f}'.format(
                sub.dice.mean(), sub.dice.std(), sub.dice.min()))
        c = cons[cons.structure == s] if len(cons) else cons
        if len(c):
            print('  Dice vs consensus, mean per observer:')
            print(c.groupby('observer')['dice_vs_consensus'].agg(['mean', 'std', 'min'])
                  .round(3).to_string().replace('\n', '\n    '))
        k = pd.DataFrame(kappa_rows)
        if len(k):
            k = k[k.structure == s]
            print('  Fleiss kappa: mean {:.3f}  min {:.3f}'.format(k.fleiss_kappa.mean(), k.fleiss_kappa.min()))

    if not args.no_figures:
        make_figures(pw, cons, args.out)
    print('\nResults in', args.out)


def make_figures(pw, cons, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for s in STRUCTURES:
        sub = pw[pw.structure == s]
        if len(sub) == 0:
            continue
        obs = sorted(set(sub.obs1) | set(sub.obs2), key=lambda o: (o == NETWORK, o))
        M = np.full((len(obs), len(obs)), np.nan)
        S = np.full_like(M, np.nan)
        for (a, b), g in sub.groupby(['obs1', 'obs2'])['dice']:
            i, j = obs.index(a), obs.index(b)
            M[i, j] = M[j, i] = g.mean()
            S[i, j] = S[j, i] = g.std()
        fig, ax = plt.subplots(figsize=(1.2 * len(obs) + 2, 1.1 * len(obs) + 1.5))
        im = ax.imshow(M, vmin=np.nanmin(M) if np.isfinite(M).any() else 0, vmax=1, cmap='viridis')
        for i in range(len(obs)):
            for j in range(len(obs)):
                if i != j and np.isfinite(M[i, j]):
                    ax.text(j, i, '{:.2f}\n±{:.2f}'.format(M[i, j], S[i, j]), ha='center',
                            va='center', fontsize=8, color='w' if M[i, j] < np.nanmean(M) else 'k')
        ax.set_xticks(range(len(obs)), obs, rotation=45, ha='right')
        ax.set_yticks(range(len(obs)), obs)
        ax.set_title('Mean pairwise Dice (± SD over cases): ' + s)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'pairwise_dice_{}.png'.format(s)), dpi=130)
        plt.close(fig)

        c = cons[cons.structure == s] if len(cons) else cons
        if len(c):
            obs_c = sorted(c.observer.unique(), key=lambda o: (o == NETWORK, o))
            fig, ax = plt.subplots(figsize=(1.1 * len(obs_c) + 2, 4))
            ax.boxplot([c[c.observer == o].dice_vs_consensus.dropna() for o in obs_c],
                       labels=obs_c, showmeans=True)
            ax.set_ylabel('Dice vs consensus')
            ax.set_title('Agreement with STAPLE consensus: ' + s)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            fig.tight_layout()
            fig.savefig(os.path.join(out, 'consensus_dice_{}.png'.format(s)), dpi=130)
            plt.close(fig)


if __name__ == '__main__':
    main()
