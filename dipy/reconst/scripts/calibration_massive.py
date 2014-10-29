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


def prepare_data_for_multi_shell(gtab, data):
    ind500 = (gtab.bvals < b0_t) | ((gtab.bvals < 600) & (gtab.bvals > 400))
    ind1000 = (gtab.bvals < b0_t) | ((gtab.bvals < 1100) & (gtab.bvals > 900))
    ind2000 = (gtab.bvals < b0_t) | ((gtab.bvals < 2100) & (gtab.bvals > 1900))
    ind3000 = (gtab.bvals < b0_t) | ((gtab.bvals < 3100) & (gtab.bvals > 2900))
    ind4000 = (gtab.bvals < b0_t) | ((gtab.bvals < 4100) & (gtab.bvals > 3900))

    S500 = data[..., ind500]
    S1000 = data[..., ind1000]
    S2000 = data[..., ind2000]
    S3000 = data[..., ind3000]
    S4000 = data[..., ind4000]

    bvals = gtab.bvals
    bvecs = gtab.bvecs

    gtab500 = gradient_table(bvals[ind500], bvecs[ind500, :], b0_threshold=b0_t)
    gtab1000 = gradient_table(bvals[ind1000], bvecs[ind1000, :], b0_threshold=b0_t)
    gtab2000 = gradient_table(bvals[ind2000], bvecs[ind2000, :], b0_threshold=b0_t)
    gtab3000 = gradient_table(bvals[ind3000], bvecs[ind3000, :], b0_threshold=b0_t)
    gtab4000 = gradient_table(bvals[ind4000], bvecs[ind4000, :], b0_threshold=b0_t)

    return (gtab500, S500), (gtab1000, S1000), (gtab2000, S2000), (gtab3000, S3000), (gtab4000, S4000)


#from dipy.data import fetch_stanford_hardi, read_stanford_hardi
#fetch_stanford_hardi()
#img, gtab = read_stanford_hardi()
#data = img.get_data()
#data = data[20:50, 55:85, 38:40]

d = 'D:\H_schijf\Data\MASSIVE\Processed\stuff'
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
from dipy.core.gradients import gradient_table
gtab = gradient_table(bvals, bvecs, b0_threshold=b0_t)


gd0, gd1, gd2, gd3, gd4 = prepare_data_for_multi_shell(gtab, data)
gtab = gd3[0]
data = gd3[1]


from dipy.data import get_sphere

sphere = get_sphere('symmetric724')

from dipy.reconst.csdeconv import recursive_response

#import dipy.reconst.dti as dti
#
#tenmodel = dti.TensorModel(gtab)
#
#tenfit = tenmodel.fit(data, mask=data[..., 0] > 200)
#
#from dipy.reconst.dti import fractional_anisotropy
#
#FA = fractional_anisotropy(tenfit.evals)
#
#MD = dti.mean_diffusivity(tenfit.evals)
#
#wm_mask = (np.logical_or(FA >= 0.4, (np.logical_and(FA >= 0.15, MD >= 0.0011))))
#
#response, msk = recursive_response(gtab, data, mask=wm_mask, sh_order=8,
#                              peak_thr=0.05, init_fa=0.08,
#                              init_trace=0.0021, iter=10, convergence=0.01,
#                              parallel=False)

wm_mask = np.ones(data.shape[0:3], dtype=bool)

response, msk = recursive_response(gtab, data, mask=None, sh_order=8,
                              peak_thr=0.05, init_fa=0.08,
                              init_trace=0.0021, iter=15, convergence=0.01,
                              parallel=False)


from dipy.reconst.shm import sh_to_sf

from dipy.viz import fvtk

response_signal = sh_to_sf(response, sphere, sh_order=8, basis_type=None)

response_actor = fvtk.sphere_funcs(response_signal, sphere)

ren = fvtk.ren()

fvtk.add(ren, response_actor)

fvtk.show(ren)


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

nib.save(nib.Nifti1Image(m, img.get_affine()), join(d, ''.join((f, '_single_fiber_mask_t005_SS3000_nomask.nii'))))
import scipy.io
scipy.io.savemat(join(d, ''.join((f, '_single_fiber_response_t005_SS3000_nomask.mat'))), mdict={'r_sh': response})
np.save(join(d, ''.join((f, '_single_fiber_response_t005_SS3000_nomask'))), response)