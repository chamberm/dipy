#! /usr/bin/env python

import sys
import nibabel as nib
import numpy as np

def nans(shape, dtype=float):
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a


def unvec_CT(x, mask):
    y = nans(np.shape(mask))
    y[mask] = x
    return y


def unvec(x, mask):
    dims = np.array([np.shape(mask)[0], np.shape(mask)[1], np.shape(mask)[2]])
    y = nans([dims[0], dims[1], dims[2], np.shape(x)[0]])
    for k in range(np.shape(x)[0]):
        dummy = nans(dims)
        dummy[mask] = x[k, :]
        y[:, :, :, k] = dummy
    return y

DWI='dwi_nlm.nii'
img = nib.load(DWI)
data = img.get_data()

wm='wm_2x2x2.nii'
wm_img = nib.load(wm)
wm_mask = wm_img.get_data().astype(np.bool)

gm='gm_2x2x2.nii'
gm_img = nib.load(gm)
gm_mask = gm_img.get_data().astype(np.bool)

csf='csf_2x2x2.nii'
csf_img = nib.load(csf)
csf_mask = csf_img.get_data().astype(np.bool)

fa='fa_2x2x2.nii'
fa_img = nib.load(fa)
fa_mask = fa_img.get_data().astype(np.bool)

wb='b0_2x2x2_brain_mask.nii.gz'
wb_img = nib.load(wb)
whole_mask = wb_img.get_data().astype(np.bool)

from dipy.core.gradients import gradient_table_from_bvals_bvecs
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.peaks import peaks_from_model, reshape_peaks_for_visualization

bval = 'encoding.bval'
bvec = 'encoding.bvec'
b_vals, b_vecs = read_bvals_bvecs(bval, bvec)
gtab = gradient_table_from_bvals_bvecs(b_vals, b_vecs, b0_threshold=10)

from dipy.reconst.csdeconv import recursive_response

# """dti for fa"""

# import dipy.reconst.dti as dti
# tenmodel = dti.TensorModel(gtab)
# tenfit = tenmodel.fit(data, mask=whole_mask)

# from dipy.reconst.dti import fractional_anisotropy
# FA = fractional_anisotropy(tenfit.evals)
# MD = dti.mean_diffusivity(tenfit.evals)
# fake_wm_mask = (np.logical_or(FA >= 0.4, (np.logical_and(FA >= 0.15, MD >= 0.0011))))

# FA[np.isnan(FA)] = 0
# FA = np.clip(FA, 0, 1)
# fa_img = nib.Nifti1Image(FA.astype(np.float32), wm_img.get_affine())
# nib.save(fa_img, 'fa_2x2x2.nii')


response, msk = recursive_response(gtab, data, mask=wm_mask, sh_order=8,
                              peak_thr=0.01, init_fa=0.08,
                              init_trace=0.0021, iter=8, convergence=0.001,
                              parallel=True)

"""HACK"""
n_iter = np.shape(msk)[0]

masks = np.zeros([n_iter, np.shape(data[wm_mask])[0]])
for i in range(0, n_iter):
  if i == 0 :
      n_vox = np.shape(msk[0])[0]
  else:
      n_vox = np.sum(msk[i-1])
  m = np.ones(n_vox)
  for j in range(i)[::-1]:
      m = unvec_CT(m, msk[j])
  m = ~np.isnan(m)
  masks[i, m] = 1

m = unvec(masks, wm_mask)

m = np.sum(m, 3)

nib.save(nib.Nifti1Image(m.astype(np.float32), wm_img.get_affine()), 'single_fiber_mask.nii')
r_sh = response.dwi_response
np.savetxt('response.txt', r_sh)

"""END HACK """


"""
Now, that we have the response function, we are ready to start the deconvolution
process. Let's import the CSD model and fit the datasets.
"""

from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
csd_model = ConstrainedSphericalDeconvModel(gtab, response)

"""
For illustration purposes we will fit only a small portion of the data.
"""
#data_small = data[40:70, 40:70, 44:45]
#csd_fit = csd_model.fit(data_small)

"""
Show the CSD-based ODFs also known as FODFs (fiber ODFs).
"""

"""based on scilpy"""
from dipy.data import get_sphere
sphere = get_sphere('symmetric724')

peaks_csd = peaks_from_model(model=csd_model,
                                 data=data,
                                 sphere=sphere,
                                 relative_peak_threshold=.5,
                                 min_separation_angle=25,
                                 mask=whole_mask,
                                 return_sh=True,
                                 normalize_peaks=True,
                                 parallel=True)

csd_sh = nib.Nifti1Image(peaks_csd.shm_coeff.astype(np.float32), wm_img.get_affine())
nib.save(csd_sh, 'fodf_2x2x2.nii')

nib.save(nib.Nifti1Image(reshape_peaks_for_visualization(peaks_csd),
                                 wm_img.get_affine()), 'maxima_2x2x2.nii')





