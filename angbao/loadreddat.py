""" loadreddat.py --- load and reduce input datasets
    - Construct/computes datavector and precision matrix.
    - Interpolate templates.
"""
from __future__ import (print_function)
from numpy import (load)
from pandas import (read_csv, concat)
from numpy import (concatenate, cov, where)
from numpy.linalg import (inv)
from scipy.linalg import (block_diag)
from scipy.interpolate import (InterpolatedUnivariateSpline)


def loadreddat(measfn, tempfn, data='y1', **kwargs):
    """Reduce datasets for aB0A3 modeling
    Arguments
    ---------
    measfn: str [CSV format]
        Input measurements dataset filename.
    tempfn: str [CSV format]
        Input model dataset filename.
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
    print("Loading datasets:", end=' ')
    print(measfn, ",", sep='', end=' ')
    cldf = read_csv(measfn, index_col=0, header=[0, 1])
    print(tempfn)
    tpdf = read_csv(tempfn, index_col=0, header=[0, 1])

    # Scales
    if 'xlim' in kwargs:    # cuts
        print("Applying scale-cuts:", "%lf<ell<%lf" % kwargs['xlim'])
        cldf = cldf.query("%lf<ell<%lf" % kwargs['xlim'])
    ll = [cldf[iz].index.values for iz in cldf.columns.levels[0]]

    # Mocks
    mock = cldf.drop(labels='y1', level='data', axis=1)
    mock = [mock[iz] for iz in mock.columns.levels[0]]
    nb = sum([len(l) for l in ll])  # No. bins
    ns = mock[0].shape[1]           # No. mock samples

    # Data
    if data == 'mean':
        print("Selecting datavector: mean of mocks")
        dd = [m.mean(axis=1).values for m in mock]
    else:
        print("Selecting datavector:", data)
        dd = cldf.xs(data, level='data', axis=1)
        dd = [dd[iz].values for iz in dd.columns]
    dd = concatenate(dd)    # The full datavector! 

    # Covariance
    if 'covfn' in kwargs:
        print("Loading covariance dataset:", kwargs['covfn'])
        cov = load(kwargs['covfn'])
        if 'xlim' in kwargs:    # scale cuts
            print("Applying scale-cuts to input cov")
            lmask = [(l > kwargs['xlim'][0]) * (l < kwargs['xlim'][1])
                     for l in ll]
            dshell = covar.shape[0] / len(ll)
            mtmp = [where(lmask == 1)[0] + ii * dshell
                    for ii in range(len(ll))]
            mtmp = concatenate(mtmp)
            cov = cov[mtmp, :]
            cov = cov[:, mtmp]
    else:
        if 'blockcov' in kwargs and kwargs['blockcov']:
            print("Selecting block-diagonal cov from mocks")
            cov = block_diag(*[m.T.cov().values for m in mock])
        else:
            print("Selecting full cov from mocks")
            cov = concat(mock).T.cov().values

    # Precision
    print("Computing prec. matrix")
    icc = inv(cov)

    # De-bias inverse covariance matrix
    if 'covfn' not in kwargs:
        D = (nb + 1.) / (ns - 1.)
        print("Debiasing Prec.Mat. from mocks",
              "(nb, ns)=(%d, %d). 1-D=%le"
              % (nb, ns, 1. - D))
        icc *= (1. - D)

    # Model
    print("Interpolating templates")
    laps = tpdf.index.values
    # Here we deconstruct pandas.DataFrame into list of numpy.array
    apswg = tpdf.xs('wg', level='type', axis=1).values
    apswg = [apswg[:, i] for i in range(apswg.shape[1])]
    apsnw = tpdf.xs('nw', level='type', axis=1).values
    apsnw = [apsnw[:, i] for i in range(apsnw.shape[1])]
    model = {'apswg': [InterpolatedUnivariateSpline(laps, wg)
                       for wg in apswg],
             'apsnw': [InterpolatedUnivariateSpline(laps, nw)
                       for nw in apsnw]}

    return(ll, dd, icc, model)


if __name__ == '__main__':
    print("Testing")
    ll, dd, icc, model = loadreddat(
        '../dat/apsmeasurements_nside1024_dl15lmin0lmax450.csv.gz',
        '../dat/apstemplates_mc_rsd.csv.gz')
