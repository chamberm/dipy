import numpy as np


d = 'D:\H_schijf\Data\MASSIVE\Processed\stuff'
do = 'D:\H_schijf\Data\MASSIVE\Processed\stuff\Otsu_mask'
f = 'DWIs_A_MD_C_native_rigid_2KS_ordered_shells_pos'
b0_t = 1

from os.path import join
fdwi = join(d, ''.join((f, '.nii')))
fbval = join(d, ''.join((f, '.bval')))
fbvec = join(d, ''.join((f, '.bvec')))

import nibabel as nib
img = nib.load(fdwi)
data = img.get_data()
from dipy.io import read_bvals_bvecs
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
bvecs = bvecs[:, [1, 0, 2]]
bvecs[:, 2] = -bvecs[:, 2]
from dipy.core.gradients import gradient_table
gtab = gradient_table(bvals, bvecs, b0_threshold=b0_t)


from dipy.data import get_sphere

sphere = get_sphere('symmetric724')

from dipy.segment.mask import median_otsu
maskdata, mask = median_otsu(data, 2, 1)


import dipy.reconst.dti as dti

tenmodel = dti.TensorModel(gtab)

tenfit = tenmodel.fit(maskdata)

from dipy.viz import fvtk
ren = fvtk.ren()

evals = tenfit.evals[:, 48:49, :]
evecs = tenfit.evecs[:, 48:49, :]

from dipy.reconst.dti import fractional_anisotropy, color_fa, lower_triangular

FA = fractional_anisotropy(tenfit.evals)

FA = np.clip(FA, 0, 1)
RGB = color_fa(FA, tenfit.evecs)

cfa = RGB[:, 48:49, :]
cfa /= cfa.max()

fvtk.add(ren, fvtk.tensor(evals, evecs, cfa, sphere))

fvtk.show(ren)