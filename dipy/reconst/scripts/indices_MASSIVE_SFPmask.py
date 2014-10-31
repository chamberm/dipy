import nibabel as nib
import numpy as np

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


from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
csd_model = ConstrainedSphericalDeconvModel(gtab, response)
from dipy.reconst.peaks import peaks_from_model
csd_peaks = peaks_from_model(model=csd_model,
                             data=data,
                             sphere=sphere,
                             relative_peak_threshold=.5,
                             min_separation_angle=25, mask=SFPmask)
sh = csd_peaks.shm_coeff
dirs = csd_peaks.peak_dirs
vals = csd_peaks.peak_values # save as AFD/HMOA
ind = csd_peaks.peak_indices


data = data[SFPmask]
from dipy.reconst.shore import ShoreModel
asm = ShoreModel(gtab)
asmfit = asm.fit(data)
rtop_pdf = unvec_CT(asmfit.rtop_pdf(), SFPmask) # save as RTOP
msd = unvec_CT(asmfit.msd(), SFPmask) # save as MSD
data = unvec(data, SFPmask)
shore_peaks = peaks_from_model(model=asm,
                             data=data,
                             sphere=sphere,
                             relative_peak_threshold=.5,
                             min_separation_angle=25, mask=SFPmask)
sh = shore_peaks.shm_coeff
dirs = shore_peaks.peak_dirs
vals = shore_peaks.peak_values
ind = shore_peaks.peak_indices


from dipy.reconst.dsi import DiffusionSpectrumModel
dsmodel = DiffusionSpectrumModel(gtab, qgrid_size=35, filter_width=18.5)
data = data / (data[..., 0, None]).astype(np.float)
data = data[SFPmask]
rtop_pdf = unvec_CT(dsmodel.fit(data).rtop_pdf(normalized=False), SFPmask)
rtop_pdf_norm = unvec_CT(dsmodel.fit(data).rtop_pdf(), SFPmask)
msd_norm = unvec_CT(dsmodel.fit(data).msd_discrete(), SFPmask)
msd = unvec_CT(dsmodel.fit(data).msd_discrete(normalized=False), SFPmask)
data = unvec(data, SFPmask)
dsi_peaks = peaks_from_model(model=dsmodel,
                             data=data,
                             sphere=sphere,
                             relative_peak_threshold=.5,
                             min_separation_angle=25, mask=SFPmask)
sh = dsi_peaks.shm_coeff
dirs = dsi_peaks.peak_dirs
vals = dsi_peaks.peak_values
ind = dsi_peaks.peak_indices


import dipy.reconst.dti as dti