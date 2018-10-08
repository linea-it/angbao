""" loadreddat.py --- load and reduce input datasets
"""
from __future__ import (print_function)
from numpy import (load)
from pandas import (read_csv)
from numpy import (array, loadtxt, concatenate, mean, cov)
from numpy import where
from numpy.linalg import (inv, block_diag)
from scipy.interpolate import (InterpolatedUnivariateSpline)
from os.path import (isfile)


def loadreddat(measfn,
               kind='A0B1', data='y1',
               **kwargs):
    """Reduce datasets for aB0A3 modeling
    Arguments
    ---------
    measfn: str [CSV format]
        Input dataset filename.
    kind: str
        Desired kind of fit.
    data: str
        Select column of dataset to befined as the data.
        If 'mean' -> the mean of 'mock<N>' names is considered.
    covfn: (opt) str [NPY format]
        Input covariance filename.
    xlim: (opt) tuple
        Scale cuts (xmin, xmax).
    Return
    ------
    ll: list
        Data scales.
    dd: list
        Data.
    icc: list
        Inverse covariance matrices.
    model: PyDict
        Pieces for model APS, apswg and apsnw splined.
    """
    # Load datasets
    print("Loading dataset", ifn)
    cldf = read_csv(measfn, index_col=0, header=[0, 1])

    # Scales
    if 'xlim' in kwargs:    # cuts
        cldf = cldf.query("%lf<ell<%lf" % xlim)
    ll = [cldf[iz].index.values for iz in cldf.columns.levels[0]]

    # Mocks
    mock = cldf.drop(labels='y1', level='data', axis=1)
    mock = [mock[iz] for iz in mock.columns.levels[0]]
    nb = cov.shape[0]       # No. bins
    ns = mock[0].shape[1]   # No. mock samples

    # Data
    if data == 'mean':
        dd = [m.mean(axis=1).values for m in mock]
    else:
        dd = cldf.xs(data, level='data', axis=1)
        dd = [dd[iz].values for iz in data.columns]

    # Covariance
    if 'covfn' in kwargs:
        cov = load(kwargs['covfn'])
        if 'xlim' in kwargs:    # scale cuts
            lmask = [(l > kwargs['xlim'][0]) * (l < kwargs['xlim'][1])
                     for l in ll]
            dshell = covar.shape[0] / len(ll)
            mtmp = [where(lmask == 1)[0] + ii * dshell
                    for ii in range(len(ll))]
            mtmp = concatenate(mtmp)
            cov = cov[mtmp, :]
            cov = cov[:, mtmp]
    else:
        if kwargs['blockcov']:
            cov = block_diag(*[m.T.cov().values for m in mock])
        else:
            cov = concat(mock).T.cov().values

    # Precision
    icc = inv(cov)

    # De-bias inverse covariance matrix
    if 'covfn' not in kwargs:
        D = (nb + 1.) / (ns - 1.)
        print("Debiasing Prec.Mat. from mocks",
              "(nb, ns)=(%d, %d). 1-D=%le"
              % (nb, ns, 1. - D))
        icc *= (1. - D)

    # Model
    ofn = "%s/aps_templates_mc_sm-nw_rsd.dat" % (pkd)
    laps = loadtxt(ofn, usecols=[0], unpack=True)
    apswg = loadtxt(ofn, usecols=[1, 2, 3, 4], unpack=True).tolist()
    apswg = [array(aps) for aps in apswg]
    apsnw = loadtxt(ofn, usecols=[5, 6, 7, 8], unpack=True).tolist()
    apsnw = [array(aps) for aps in apsnw]
    model = {'apswg': [InterpolatedUnivariateSpline(laps, wg)
                       for wg in apswg],
             'apsnw': [InterpolatedUnivariateSpline(laps, nw)
                       for nw in apsnw]}

    return(ll, dd, icc, model)


if __name__ == '__main__':
    dl, lmin, lmax = 20, 60, 290
    kind = 'aA2B0'
    data = 'y1'
    nooutl = False
    Sigma = 5.6
    hc = True
    template = 'wg'
    B0prior = True
    fullcov = False

    ll, dd, icc, model = reduce_datasets_all(dl, lmin, lmax,
                                             kind=kind, data=data,
                                             nooutl=nooutl,
                                             Sigma=Sigma, hc=hc,
                                             fullcov=fullcov)
