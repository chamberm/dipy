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
bvecs = bvecs[:, [1, 0, 2]]
bvecs[:, 2] = -bvecs[:, 2]
from dipy.core.gradients import gradient_table
gtab = gradient_table(bvals, bvecs, b0_threshold=b0_t)


from dipy.data import get_sphere

sphere = get_sphere('symmetric724')

from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel

from dipy.segment.mask import median_otsu
b0_mask, wm_mask = median_otsu(data, 2, 1)

del b0_mask


#if __name__=='__main__':
#    freeze_support()

#response = np.load(join(do, ''.join((f, '_single_fiber_response_t001_nomask.npy'))))
#
#csd_model = ConstrainedSphericalDeconvModel(gtab, response)
#
#csd_fit = csd_model.fit(data[wm_mask])
#
#csd_sh_coeff = unvec(np.transpose(csd_fit.shm_coeff), wm_mask)
#
#nib.save(nib.Nifti1Image(csd_sh_coeff, img.get_affine()), join(do, ''.join((f, '_odf_sh_t001_nomask.nii'))))


gd0, gd1, gd2, gd3, gd4 = prepare_data_for_multi_shell(gtab, data)


for shell in [4]:#range(0, 5):
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

    response = np.load(join(do, ''.join((f, '_single_fiber_response_t005_SS', she, '_nomask.npy'))))

    from dipy.reconst.csdeconv import auto_response

    response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)

    csd_model = ConstrainedSphericalDeconvModel(gtab, response)

    csd_fit = csd_model.fit(data[wm_mask])

    from dipy.viz import fvtk
    ren = fvtk.ren()

    csd_odf = unvec(np.transpose(csd_fit.odf(sphere)), wm_mask)


    fodf_spheres = fvtk.sphere_funcs(csd_odf[:, 48:49], sphere, scale=1.3, norm=False)

    fvtk.add(ren, fodf_spheres)


    fvtk.show(ren)


#    csd_sh_coeff = unvec(np.transpose(csd_fit.shm_coeff), wm_mask)
#
#    nib.save(nib.Nifti1Image(csd_sh_coeff, img.get_affine()), join(do, ''.join((f, '_odf_sh_t005_SS', she, '_nomask.nii'))))