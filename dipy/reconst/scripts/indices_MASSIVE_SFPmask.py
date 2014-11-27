import nibabel as nib
import numpy as np
from os.path import join
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
sphere = get_sphere('symmetric724')
from dipy.reconst.peaks import peaks_from_model,  reshape_peaks_for_visualization

# goede uit data halen voor elk model of aparte datasets maken
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


#from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
#csd_model = ConstrainedSphericalDeconvModel(gtab, response)
#from dipy.reconst.peaks import peaks_from_model
#csd_peaks = peaks_from_model(model=csd_model,
#                             data=data,
#                             sphere=sphere,
#                             relative_peak_threshold=.5,
#                             min_separation_angle=25, mask=SFPmask)
#sh = csd_peaks.shm_coeff
#dirs = csd_peaks.peak_dirs
#vals = csd_peaks.peak_values # save as AFD/HMOA
#ind = csd_peaks.peak_indices

d = 'D:\H_schijf\Data\MASSIVE\Processed\stuff'
do = 'D:\H_schijf\Data\MASSIVE\Processed\stuff'
f = 'DWIs_A_MD_C_native_rigid_2KS_ordered_shells_pos'
b0_t = 1
fdwi = join(d, ''.join((f, '.nii')))
fbval = join(d, ''.join((f, '.bval')))
fbvec = join(d, ''.join((f, '.bvec')))
img = nib.load(fdwi)
data = img.get_data()
affine = img.get_affine()
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
bvecs = bvecs[:, [1, 0, 2]]
bvecs[:, 2] = -bvecs[:, 2]
gtab = gradient_table(bvals, bvecs, b0_threshold=b0_t)
#data = data[SFPmask]
from dipy.reconst.shore import ShoreModel
asm = ShoreModel(gtab)
asmfit = asm.fit(data)
#rtop_pdf = unvec_CT(asmfit.rtop_pdf(), SFPmask) # save as RTOP
#msd = unvec_CT(asmfit.msd(), SFPmask) # save as MSD
#data = unvec(data, SFPmask)
rtop_pdf = asmfit.rtop_pdf() # save as RTOP
nib.save(nib.Nifti1Image(rtop_pdf.astype(np.float32), affine), join(d, ''.join((f, '_SHORE_rtop_pdf.nii'))))
rtop_signal = asmfit.rtop_signal()
nib.save(nib.Nifti1Image(rtop_signal.astype(np.float32), affine), join(d, ''.join((f, '_SHORE_rtop_signal.nii'))))
msd = asmfit.msd() # save as MSD
nib.save(nib.Nifti1Image(msd.astype(np.float32), affine), join(d, ''.join((f, '_SHORE_rtop_msd.nii'))))
shore_peaks = peaks_from_model(model=asm,
                             data=data,
                             sphere=sphere,
                             relative_peak_threshold=.5,
                             min_separation_angle=25)#, mask=SFPmask)
del data
sh = shore_peaks.shm_coeff
dirs = shore_peaks.peak_dirs
vals = shore_peaks.peak_values
ind = shore_peaks.peak_indices
nib.save(nib.Nifti1Image(sh.astype('float32'), affine), join(d, ''.join((f, '_SHORE_odf_sh.nii'))))
nib.save(nib.Nifti1Image(dirs.astype('float32'), affine), join(d, ''.join((f, '_SHORE_dirs.nii'))))
nib.save(nib.Nifti1Image(vals.astype('float32'), affine), join(d, ''.join((f, '_SHORE_vals.nii'))))
nib.save(nib.Nifti1Image(ind.astype('float32'), affine),join(d, ''.join((f, '_SHORE_ind.nii'))))
del sh, dirs, vals, ind


d = 'D:\H_schijf\Data\MASSIVE\Processed\stuff'
do = 'D:\H_schijf\Data\MASSIVE\Processed\stuff'
f = 'DWIs_A_MD_C_native_rigid_2KS_ordered_grid1_pos'
b0_t = 1
fdwi = join(d, ''.join((f, '.nii')))
fbval = join(d, ''.join((f, '.bval')))
fbvec = join(d, ''.join((f, '.bvec')))
img = nib.load(fdwi)
data = img.get_data()
affine = img.get_affine()
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
bvecs = bvecs[:, [1, 0, 2]]
bvecs[:, 2] = -bvecs[:, 2]
gtab = gradient_table(bvals, bvecs, b0_threshold=b0_t)
from dipy.reconst.dsi import DiffusionSpectrumModel
dsmodel = DiffusionSpectrumModel(gtab, qgrid_size=35, filter_width=18.5)
data = data / (data[..., 0, None]).astype(np.float)
#data = data[SFPmask]
#rtop_pdf = unvec_CT(dsmodel.fit(data).rtop_pdf(normalized=False), SFPmask)
#rtop_pdf_norm = unvec_CT(dsmodel.fit(data).rtop_pdf(), SFPmask)
#msd_norm = unvec_CT(dsmodel.fit(data).msd_discrete(), SFPmask)
#msd = unvec_CT(dsmodel.fit(data).msd_discrete(normalized=False), SFPmask)
#data = unvec(data, SFPmask)
rtop_signal = dsmodel.fit(data).rtop_signal()
rtop_pdf = dsmodel.fit(data).rtop_pdf(normalized=False)
rtop_pdf_norm = dsmodel.fit(data).rtop_pdf()
msd_norm = dsmodel.fit(data).msd_discrete()
msd = dsmodel.fit(data).msd_discrete(normalized=False)
dsi_peaks = peaks_from_model(model=dsmodel,
                             data=data,
                             sphere=sphere,
                             relative_peak_threshold=.5,
                             min_separation_angle=25)#, mask=SFPmask)
del data
sh = dsi_peaks.shm_coeff
dirs = dsi_peaks.peak_dirs
vals = dsi_peaks.peak_values
ind = dsi_peaks.peak_indices
nib.save(nib.Nifti1Image(sh.astype('float32'), affine), join(d, ''.join((f, '_DSI1_odf_sh.nii'))))
nib.save(nib.Nifti1Image(dirs.astype('float32'), affine), join(d, ''.join((f, '_DSI1_dirs.nii'))))
nib.save(nib.Nifti1Image(vals.astype('float32'), affine), join(d, ''.join((f, '_DSI1_vals.nii'))))
nib.save(nib.Nifti1Image(ind.astype('float32'), affine),join(d, ''.join((f, '_DSI1_ind.nii'))))
del sh, dirs, vals, ind
