""" loadreddat.py --- load and reduce input datasets
"""
from __future__ import (print_function)
from numpy import (array, loadtxt, concatenate, mean, cov)
from numpy.linalg import (inv, block_diag)
from scipy.interpolate import (InterpolatedUnivariateSpline)
from os.path import (isfile)


# Datasets
mockd = "../../dat/mockmeasure"  # mock measurements
y1d = "../../dat/datameasure"   # y1 measurements
pkd = "../../dat/pktemplates"   # Pk templates
seld = "../../dat/dndz"         # selections
modd = "../../dat"              # model
mockmaskd = "../../dat/mockstats"  # mock mask
# bias = [1.83, 1.77, 1.79, 2.15] # Bias from mocks.
bias = [1., 1., 1., 1.]  # trivial thing


def reduce_datasets_all(dl, xmin, xmax,
                        kind='aA0B1', data='y1', nooutl=False, hc=False,
                        **kwargs):
    """Reduce datasets for aB0A3 modeling
    Arguments
    ---------
    dl: float
        Binning width.
    xmin, xmax: float
        Scale (\ell) limits for fit.
    data: str
        Select data 'y1' -> Y1 measurements, 'mock' -> Mean of mocks.
    nooutl: bool
        If True remove outliers from a mask file.
    imock: int (Optional)
        Select one mock as data.
    Sigma: float (Optional)
        Damping scale, used for kind!='aB0A3'.
    hc: Bool (Optional)
        If True, HC's measurements for the mocks are used.
    Return
    ------
    ll: list
        Data scales.
    dd: list
        Data.
    icc: list
        Inverse covariance matrices.
    model: PyDict
        Pieces for model APS. If kind=='aB0A3' contains zsel, pklin
        and pknw, otherwise apswg and apsnw splined.
    """
    # Datasets names
    if hc:
        if dl is None:
            mockfn = ["%s/mocksmeasure_zbin%d_fidbinning_nside1024.dat"
                      % (mockd, iz + 1) for iz in range(4)]
            y1fn = ["%s/datameasure_zbin%d_fidbinning_nside1024.dat"
                    % (y1d, iz + 1) for iz in range(4)]
        else:
            lmin, lmax = (0, 1000)
            if dl == 15:
                lmax = 450
            mockfn = ["%s/mocksmeasure_zbin%d_dl%d_lmin%dlmax%d_nside1024.dat"
                      % (mockd, iz + 1, dl, lmin, lmax)
                      for iz in range(4)]
            y1fn = ["%s/datameasure_zbin%d_dl%d_lmin%dlmax%d_nside1024.dat"
                    % (y1d, iz + 1, dl, lmin, lmax)
                    for iz in range(4)]
    else:
        lmin, lmax = (40, 720)
        mockfn = ["%s/mocksmeasure_zbin%d_dl%d_lmin%dlmax%d.dat"
                  % (mockd, iz + 1, dl, lmin, lmax) for iz in range(4)]
        y1fn = ["%s/datameasure_zbin%d_dl%d_lmin%dlmax%d.dat"
                % (y1d, iz + 1, dl, lmin, lmax) for iz in range(4)]

    pkfn = "%s/pk_templates_mc.dat" % (pkd)
    # pkfn = "%s/pk_templates_planckhc.dat" % (pkd)
    # selfn = ["%s/micenz_%d.txt" % (seld, iz + 1) for iz in range(4)]
    selfn = [seld + "/dNdzZ30meanSVCMOF-DNF0.60.7.dat", seld +
             "/dNdzZ30meanSVCMOF-DNF0.70.8.dat", seld +
             "/dNdzZ30meanSVCMOF-DNF0.80.9.dat", seld +
             "/dNdzZ30meanSVCMOF-DNF0.91.0.dat"]

    # Load datasets
    ell = loadtxt(y1fn[0], usecols=[0], unpack=True)
    mock = [loadtxt(fn, unpack=True) for fn in mockfn]
    y1 = [loadtxt(fn, usecols=[1], unpack=True) for fn in y1fn]
    sel = [loadtxt(fn, unpack=True) for fn in selfn]
    pk = loadtxt(pkfn, unpack=True)
    if nooutl:                  # Mock mask (Outliers)
        from numpy import isfinite
        if 'lmax' in kwargs:
            if kwargs['B0prior']:
                ifnstats = "%s/Y1clBAOmock_%s_dl%dlmin%dlmax%d_B0prior.csv" %\
                           (mockmaskd, kind, dl, xmin, kwargs['lmax'])
            else:
                ifnstats = "%s/Y1clBAOmock_%s_dl%dlmin%dlmax%d.csv" %\
                           (mockmaskd, kind, dl, xmin, kwargs['lmax'])
        else:
            if kwargs['B0prior']:
                ifnstats = "%s/Y1clBAOmock_%s_dl%dlmin%dlmax%d_B0prior.csv" %\
                    (mockmaskd, kind[1:], dl, xmin, xmax)
            else:
                ifnstats = "%s/Y1clBAOmock_%s_dl%dlmin%dlmax%d.csv" %\
                    (mockmaskd, kind[1:], dl, xmin, xmax)
        print(ifnstats)
        mockmask = isfinite(loadtxt(ifnstats, usecols=[1], delimiter=','))
        print(" Removing %d outlier mocks." % (mockmask.shape[0] -
                                               sum(mockmask)))
        mock = [m[:, mockmask] for m in mock]

    # Select scales
    print(" ** Measurements: %1.2f < ell < %1.2f, %d bins"
          % (ell[0], ell[-1], ell.shape[0]))
    lmask = (ell > xmin) * (ell < xmax)
    ll = [ell[lmask] for iz in range(4)]
    mock = [m[lmask, :] for m in mock]
    y1 = [d[lmask] for d in y1]
    print(" ** Used: %1.2f < ell < %1.2f, %d bins"
          % (ll[0][0], ll[0][-1], ll[0].shape[0]))

    # Data
    if data == 'mock':
        if 'imock' in kwargs:
            if kwargs['imock'] < 0 or kwargs['imock'] >= mock[0].shape[1]:
                raise Exception("Invalid index of mock:", kwargs['imock'])
            else:
                aps = [m[:, kwargs['imock']] for m in mock]
        else:
            aps = [mean(m, axis=1) for m in mock]
    elif data == 'y1':
        aps = y1
    else:
        raise Exception("Invalid data type:", data)
    dd = concatenate(aps)

    # Covariance
    mock = concatenate(mock)
    if 'njk' in kwargs and kwargs['njk'] is not None:
        print(" *** Using covariance from %d JK." % (kwargs['njk']))
        ifncov = [
            "%s/jkcov_zbin%d_dl%d_lmin%dlmax%d_%djkregions_nside1024.dat" %
            (y1d, iz + 1, dl, lmin, lmax, kwargs['njk']) for iz in range(4)]
        covar = [loadtxt(fn) for fn in ifncov]
        covar = [c[lmask, :] for c in covar]
        covar = [c[:, lmask] for c in covar]
        covar = block_diag(*covar)
    else:
        covar = cov(mock)

    if not kwargs['fullcov']:
        print(" ** Using block-diagonal covariance matrix.")
        cc = [None] * 4
        dim = covar.shape[0] / 4
        for iz in range(4):
            cc[iz] = covar[iz * dim:(iz + 1) * dim, iz * dim:(iz + 1) * dim]
        covar = block_diag(*cc)

    # Covariance
    covar = loadtxt("../../dat/covtheory_dl15_lmin0lmax450_rsdnl.dat")
    dshell = covar.shape[0] / 4
    from numpy import where
    mtmp = [where(lmask == 1)[0] + ii * dshell for ii in range(4)]
    mtmp = concatenate(mtmp)
    covar = covar[mtmp, :]
    covar = covar[:, mtmp]

    # Precision
    icc = inv(covar)

    # De-bias inverse covariance matrix
    D = (dd.shape[0] + 1.) / (mock.shape[1] - 1.)
    print(" ** Debiasing inv.cov. (nb, ns)=(%d, %d). 1-D=%le"
          % (dd.shape[0], mock.shape[1], 1. - D))
    icc *= (1. - D)

    # Model
    if kind == 'aA3B0_rr':
        ofn = "%s/aps_templates_mc_lin-nw_rsd.dat" % (pkd)
        if isfile(ofn):
            print(" *** File was already generated.\n"
                  "     -> Reading %s" % (ofn))
            laps = loadtxt(ofn, usecols=[0], unpack=True)
            apswg = loadtxt(ofn, usecols=[1, 2, 3, 4], unpack=True).tolist()
            apswg = [array(aps) for aps in apswg]
            apsnw = loadtxt(ofn, usecols=[5, 6, 7, 8], unpack=True).tolist()
            apsnw = [array(aps) for aps in apsnw]
        else:
            print("ERROR: You should provide the data")
            exit(1)
        model = {'apswg': [InterpolatedUnivariateSpline(laps, wg)
                           for wg in apswg],
                 'apsnw': [InterpolatedUnivariateSpline(laps, nw)
                           for nw in apsnw]}
    elif kind == 'aB0A3':
        model = {'sel': sel, 'pklin': [pk[0], pk[1]],
                 'pknw': [pk[0], pk[2]]}
    else:
        ofn = "%s/aps_templates_mc_sm-nw_rsd.dat" % (pkd)
        if isfile(ofn):
            print(" *** File was already generated.\n"
                  "     -> Reading %s" % (ofn))
            laps = loadtxt(ofn, usecols=[0], unpack=True)
            apswg = loadtxt(ofn, usecols=[1, 2, 3, 4], unpack=True).tolist()
            apswg = [array(aps) for aps in apswg]
            apsnw = loadtxt(ofn, usecols=[5, 6, 7, 8], unpack=True).tolist()
            apsnw = [array(aps) for aps in apsnw]
        else:
            print("ERROR: you should provide the data")
            exit(1)
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
