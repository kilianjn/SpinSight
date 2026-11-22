"""Build the 3D brain phantom of SpinSight from the MNI ICBM152 2009c template.

Input (TemplateFlow, tpl-MNI152NLin2009cAsym, 1 mm):
    *_res-01_T1w.nii.gz, *_res-01_T2w.nii.gz,
    *_res-01_label-{GM,WM,CSF}_probseg.nii.gz
    (https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/...)

Output: assets/volumes/brain3d.png (mosaic of axial slices, RGBA = two tissue
labels + their fractions per voxel) and assets/volumes/brain3d.toml (metadata).

Brain tissues come from the template's probability maps; the head outside the
brain (skull/air, muscle/skin, fat, fluid such as the eyes) is classified softly
from the T1w/T2w intensities.

Usage: python make_brain3d.py <templateflow dir> <output dir>
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

src, out = Path(sys.argv[1]), Path(sys.argv[2])
pre = 'tpl-MNI152NLin2009cAsym_res-01_'
load = lambda n: nib.load(src / f'{pre}{n}.nii.gz')
img = load('T1w')
t1 = np.asarray(img.dataobj, dtype=np.float32)
t2 = np.asarray(load('T2w').dataobj, dtype=np.float32)
gm, wm, csf = (np.asarray(load(f'label-{t}_probseg').dataobj, dtype=np.float32) for t in ('GM', 'WM', 'CSF'))
# the probability maps do not always sum to one inside the brain: normalise them there, with a
# smooth transition towards the brain surface (where partial volume with other tissue is real)
total = gm + wm + csf
blend = np.clip((ndi.gaussian_filter(total, 1.0) - 0.4) / 0.4, 0, 1)
factor = 1 + blend * (1 / np.maximum(total, 1e-3) - 1)
gm, wm, csf = gm * factor, wm * factor, csf * factor
brain = np.clip(gm + wm + csf, 0, 1)

# ---- head mask: Otsu threshold on the smoothed T1w, holes filled in all directions
sm = ndi.gaussian_filter(t1, 1.0)
vals = sm[sm > 0]
hist, edges = np.histogram(np.log1p(vals), bins=256)
centers = (edges[:-1] + edges[1:]) / 2
w0 = np.cumsum(hist); w1 = w0[-1] - w0
m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1)
thr = np.expm1(centers[np.argmax(w0 * w1 * (m0 - m1) ** 2)])
head = (sm > thr) | (brain > 0.5)
head = ndi.binary_closing(head, iterations=4)
for axis in range(3):
    head = np.stack([ndi.binary_fill_holes(s) for s in np.moveaxis(head, axis, 0)], axis=axis)
lab, n = ndi.label(head)
head = lab == (1 + np.argmax(ndi.sum(head, lab, range(1, n + 1))))
head_frac = np.clip(ndi.gaussian_filter(head.astype(np.float32), 0.6), 0, 1)
head_frac = np.maximum(head_frac, brain)

# ---- soft classification of the non-brain head
t1n = t1 / np.percentile(t1[head], 99)
t2n = t2 / np.percentile(t2[head], 99)
classes = {  # (T1w, T2w) centre of each class
    'Cortical bone': (0.08, 0.08),
    'Muscle': (0.45, 0.28),
    'Adipose tissue': (0.85, 0.55),
    'CSF': (0.25, 0.95),
}
sigma = 0.14
memb = np.stack([np.exp(-((t1n - a) ** 2 + (t2n - b) ** 2) / (2 * sigma ** 2)) for a, b in classes.values()])
memb /= np.maximum(memb.sum(0), 1e-6)
rest = np.clip(head_frac - brain, 0, 1)

tissues = ['Gray matter', 'White matter', 'CSF', 'Adipose tissue', 'Muscle', 'Cortical bone']
frac = np.zeros((len(tissues),) + t1.shape, np.float32)
frac[0], frac[1], frac[2] = gm, wm, csf
for k, name in enumerate(classes):
    frac[tissues.index(name)] += rest * memb[k]

# ---- crop to the head
idx = np.where(head_frac > 0.02)
lo = [max(int(i.min()) - 1, 0) for i in idx]
hi = [min(int(i.max()) + 2, s) for i, s in zip(idx, t1.shape)]
frac = frac[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
nt, nx, ny, nz = frac.shape
origin = (img.affine @ np.array([lo[0], lo[1], lo[2], 1.0]))[:3]

# ---- two strongest tissues per voxel: labels in R/G, fractions in B/A (0-255)
order = np.argsort(-frac, axis=0)
t_a, t_b = order[0], order[1]
f_a = np.take_along_axis(frac, t_a[None], 0)[0]
f_b = np.take_along_axis(frac, t_b[None], 0)[0]
# keep the total amount of tissue: the two kept fractions take over the dropped remainder
total = frac.sum(0)
kept_sum = np.maximum(f_a + f_b, 1e-6)
f_a, f_b = f_a * total / kept_sum, f_b * total / kept_sum
empty_a = f_a < 1 / 255
rgba = np.stack([np.where(empty_a, 255, t_a), np.where(f_b < 1 / 255, 255, t_b),
                 np.round(np.clip(f_a, 0, 1) * 255), np.round(np.clip(f_b, 0, 1) * 255)], -1).astype(np.uint8)

# mosaic of axial slices: slice k (inferior→superior) at tile (k // cols, k % cols);
# within a tile, row = y index (posterior→anterior), column = x index (right→left in voxel order)
cols = int(np.ceil(np.sqrt(nz)))
rows = int(np.ceil(nz / cols))
mosaic = np.zeros((rows * ny, cols * nx, 4), np.uint8)
mosaic[..., 0:2] = 255
for k in range(nz):
    r, c = divmod(k, cols)
    mosaic[r * ny:(r + 1) * ny, c * nx:(c + 1) * nx] = rgba[:, :, k].transpose(1, 0, 2)
out.mkdir(parents=True, exist_ok=True)
Image.fromarray(mosaic, 'RGBA').save(out / 'brain3d.png', optimize=True)
with open(out / 'brain3d.toml', 'w') as f:
    f.write('# SpinSight 3D phantom: MNI ICBM152 2009c (see LICENSE-MNI.txt), built by tools/make_brain3d.py\n')
    f.write(f'dims = [{nx}, {ny}, {nz}]\nvoxel_size = [1.0, 1.0, 1.0]\n')
    f.write(f'origin = [{origin[0]:.1f}, {origin[1]:.1f}, {origin[2]:.1f}]  # RAS mm of voxel (0, 0, 0)\n')
    f.write(f'tile_columns = {cols}\n')
    f.write('tissues = [' + ', '.join(f'"{t}"' for t in tissues) + ']\n')
print('dims', (nx, ny, nz), 'origin', origin, 'png bytes', (out / 'brain3d.png').stat().st_size)
# error of keeping two tissues
kept = np.zeros_like(frac)
np.put_along_axis(kept, t_a[None], f_a[None], 0)
np.put_along_axis(kept, t_b[None], f_b[None], 0)
print('max dropped fraction', float((frac - kept).max()), 'mean dropped', float((frac - kept)[:, head[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]].sum(0).mean()))
