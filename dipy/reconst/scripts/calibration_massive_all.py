import numpy as np
from multiprocessing import freeze_support


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
from dipy.core.gradients import gradient_table
gtab = gradient_table(bvals, bvecs, b0_threshold=b0_t)



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
#wm_mask = (FA >= 0.1)


#wm_mask = np.ones(data.shape[0:3], dtype=bool)


from dipy.segment.mask import median_otsu
b0_mask, wm_mask = median_otsu(data, 2, 1)


#if __name__=='__main__':
#    freeze_support()

#response, msk = recursive_response(gtab, data, mask=wm_mask, sh_order=8,
#                              peak_thr=0.01, init_fa=0.08,
#                              init_trace=0.0021, iter=15, convergence=0.01,
#                              parallel=False)
#
#
#
#n_iter = np.shape(msk)[0]
#
#masks = np.zeros([n_iter, np.shape(data[wm_mask])[0]])
#for i in range(0, n_iter):
#    if i == 0 :
#        n_vox = np.shape(msk[0])[0]
#    else:
#        n_vox = np.sum(msk[i-1])
#    m = np.ones(n_vox)
#    for j in range(i)[::-1]:
#        m = unvec_CT(m, msk[j])
#    m = ~np.isnan(m)
#    masks[i, m] = 1
#
#m = unvec(masks, wm_mask)
#
#m = np.sum(m, 3)
#
#nib.save(nib.Nifti1Image(m, img.get_affine()), join(do, ''.join((f, '_single_fiber_mask_t001_nomask.nii'))))
#import scipy.io
#scipy.io.savemat(join(do, ''.join((f, '_single_fiber_response_t001_nomask.mat'))), mdict={'r_sh': response})
#np.save(join(do, ''.join((f, '_single_fiber_response_t001_nomask'))), response)


gd0, gd1, gd2, gd3, gd4 = prepare_data_for_multi_shell(gtab, data)


for shell in range(0, 5):
    if shell == 0:
        gtab = gd0[0]
        data = gd0[1]
        she = '500'
    elif shell == 1:
        gtab = gd1[0]
        data = gd1[1]
        she = '1000'
    elif shell == 2:
        gtab = gd2[0]
        data = gd2[1]
        she = '2000'
    elif shell == 3:
        gtab = gd3[0]
        data = gd3[1]
        she = '3000'
    elif shell == 4:
        gtab = gd4[0]
        data = gd4[1]
        she = '4000'

    response, msk = recursive_response(gtab, data, mask=wm_mask, sh_order=8,
                                  peak_thr=0.01, init_fa=0.05,
                                  init_trace=0.0021, iter=15, convergence=0.01,
                                  parallel=False)

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

    nib.save(nib.Nifti1Image(m, img.get_affine()), join(do, ''.join((f, '_single_fiber_mask_t005_SS', she, '_nomask.nii'))))
    scipy.io.savemat(join(do, ''.join((f, '_single_fiber_response_t005_SS', she, '_nomask.mat'))), mdict={'r_sh': response})
    np.save(join(do, ''.join((f, '_single_fiber_response_t005_SS', she, '_nomask'))), response)