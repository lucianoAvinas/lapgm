import gzip
import requests
import numpy as np

from pathlib import Path


# Inspired by Casper O. da Costa-Luis's brainweb download routines.
# ref: https://github.com/casperdcl/brainweb/blob/master/brainweb/utils.py

MODALITY_TYPES = ('T1', 'T2', 'PD')
THICKNESS_TYPES = (1, 3, 5, 7, 9)
NOISE_TYPES = (0, 1, 3, 5, 7, 9)
RF_TYPES = (0, 20, 40)

ARR_SHAPE = (181, 217, 181)


def write_compressed(url: str, file_path: str):
    # writing a compressed response: https://stackoverflow.com/a/13137873
    req = requests.get(url, stream=True)
    if req:
        with open(file_path, 'wb') as fp:
            for chunk in req:
                fp.write(chunk)
    else:
        raise RuntimeError(f'Bad url request: {url}')


def gz_to_numpy(gz_fname: str, shape: tuple[int], dtype: type[np.dtype]):
    with gzip.open(gz_fname, 'rb') as fp:
        arr = np.frombuffer(fp.read(), dtype=dtype)
    return arr.reshape(shape)


def download_and_read_phantom(data_path: str = 'data'):
    file_path = data_path / Path('phantom.rawb.gz')

    if not file_path.exists():
        url = 'http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?do_download_alias=' \
              'phantom_1.0mm_normal_crisp&format_value=raw_byte&zip_value=gnuzip'
        write_compressed(url, file_path)

    arr = gz_to_numpy(file_path, ARR_SHAPE, np.uint8)

    return arr


def download_and_read_normal(data_path: str = 'data', slice_thickness: int = 1, noise: int = 3, 
                             rf_intensity: int = 20):
    # check validity of inputs
    assert slice_thickness in THICKNESS_TYPES
    assert noise in NOISE_TYPES
    assert rf_intensity in RF_TYPES

    data_arrays = dict()
    url_base = 'http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?do_download_alias='

    for modality in ['T1', 'T2', 'PD']:
        file_name = f'{modality.lower()}_{slice_thickness}mm_pn{noise}_rf{rf_intensity}.raws.gz'
        file_path = data_path / Path(file_name)

        # if no file, download from https://brainweb.bic.mni.mcgill.ca/selection_normal.html
        if not file_path.exists():
            url = f'{url_base}{modality}+ICBM+normal+{slice_thickness}mm+pn{noise}+' \
                  f'rf{rf_intensity}&format_value=raw_short&zip_value=gnuzip'
            write_compressed(url, file_path)

        # read from gz file
        arr = gz_to_numpy(file_path, ARR_SHAPE, np.int16)

        # cast to float and [0,1]-normalize
        arr = arr.astype(float) / arr.max()

        data_arrays[modality] = arr

    return data_arrays
