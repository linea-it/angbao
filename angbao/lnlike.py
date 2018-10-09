""" lnlike.py --- Likelihood PDF routines
"""
from numpy import (inf, isfinite, dot, concatenate)


def gaussian_lnprior(x, mu, sigma):
    return(-0.5 * (x - mu)**2 / sigma**2)


def lnprior_aA0B1(theta, **kwargs):
    a,\
        A00, B00, B10,\
        A01, B01, B11,\
        A02, B02, B12,\
        A03, B03, B13 = theta

    if True:  # B00 >= 0.0 and B01 >= 0.0 and B02 >= 0.0 and B03 >= 0.0:
        return(0.0)
    else:
        return(-inf)


def apsmod_aA0B1(theta, x, model, template='wg'):
    a,\
        A00, B00, B10,\
        A01, B01, B11,\
        A02, B02, B12,\
        A03, B03, B13 = theta

    if template == 'wg':
        yt_spline = model['apswg']
    elif template == 'nw':
        yt_spline = model['apsnw']
    else:
        raise Exception("Invalid template:", template)

    ym = [A00 + (B00 + B10 * x[0]) * yt_spline[0](x[0] / a),
          A01 + (B01 + B11 * x[1]) * yt_spline[1](x[1] / a),
          A02 + (B02 + B12 * x[2]) * yt_spline[2](x[2] / a),
          A03 + (B03 + B13 * x[3]) * yt_spline[3](x[3] / a)]

    return(ym)


def lnprior_aA1B1(theta, **kwargs):
    a,\
        A00, A10, B00, B10,\
        A01, A11, B01, B11,\
        A02, A12, B02, B12,\
        A03, A13, B03, B13 = theta

    if True:  # B00 >= 0.0 and B01 >= 0.0 and B02 >= 0.0 and B03 >= 0.0:
        return(0.0)
    else:
        return(-inf)


def apsmod_aA1B1(theta, x, model, template='wg'):
    a,\
        A00, A10, B00, B10,\
        A01, A11, B01, B11,\
        A02, A12, B02, B12,\
        A03, A13, B03, B13 = theta

    if template == 'wg':
        yt_spline = model['apswg']
    elif template == 'nw':
        yt_spline = model['apsnw']
    else:
        raise Exception("Invalid template:", template)

    ym = [A00 + A10 * x[0] + (B00 + B10 * x[0]) * yt_spline[0](x[0] / a),
          A01 + A11 * x[1] + (B01 + B11 * x[1]) * yt_spline[1](x[1] / a),
          A02 + A12 * x[2] + (B02 + B12 * x[2]) * yt_spline[2](x[2] / a),
          A03 + A13 * x[3] + (B03 + B13 * x[3]) * yt_spline[3](x[3] / a)]

    return(ym)


def lnprior_aA1B0(theta, **kwargs):
    a,\
        A00, A10, B00,\
        A01, A11, B01,\
        A02, A12, B02,\
        A03, A13, B03 = theta

    if B00 >= 0.0 and B01 >= 0.0 and B02 >= 0.0 and B03 >= 0.0:
        return(0.0)
    else:
        return(-inf)


def apsmod_aA1B0(theta, x, model, template='wg'):
    a,\
        A00, A10, B00,\
        A01, A11, B01,\
        A02, A12, B02,\
        A03, A13, B03 = theta

    if template == 'wg':
        yt_spline = model['apswg']
    elif template == 'nw':
        yt_spline = model['apsnw']
    else:
        raise Exception("Invalid template:", template)

    ym = [(A00 + A10 * x[0]) + B00 * yt_spline[0](x[0] / a),
          (A01 + A11 * x[0]) + B01 * yt_spline[1](x[1] / a),
          (A02 + A12 * x[0]) + B02 * yt_spline[2](x[2] / a),
          (A03 + A13 * x[0]) + B03 * yt_spline[3](x[3] / a)]

    return(ym)


def lnprior_aA0B0(theta, **kwargs):
    a,\
        A00, B00,\
        A01, B01,\
        A02, B02,\
        A03, B03 = theta

    if B00 >= 0.0 and B01 >= 0.0 and B02 >= 0.0 and B03 >= 0.0:
        return(0.0)
    else:
        return(-inf)


def apsmod_aA0B0(theta, x, model, template='wg'):
    a,\
        A00, B00,\
        A01, B01,\
        A02, B02,\
        A03, B03 = theta

    if template == 'wg':
        yt_spline = model['apswg']
    elif template == 'nw':
        yt_spline = model['apsnw']
    else:
        raise Exception("Invalid template:", template)

    ym = [(A00) + B00 * yt_spline[0](x[0] / a),
          (A01) + B01 * yt_spline[1](x[1] / a),
          (A02) + B02 * yt_spline[2](x[2] / a),
          (A03) + B03 * yt_spline[3](x[3] / a)]

    return(ym)


def lnprior_aA2B0(theta, **kwargs):
    a,\
        A00, A10, A20, B00,\
        A01, A11, A21, B01,\
        A02, A12, A22, B02,\
        A03, A13, A23, B03 = theta

    if B00 >= 0.0 and B01 >= 0.0 and B02 >= 0.0 and B03 >= 0.0:
        return(0.0)
    else:
        return(-inf)


def apsmod_aA2B0(theta, x, model, template='wg'):
    a,\
        A00, A10, A20, B00,\
        A01, A11, A21, B01,\
        A02, A12, A22, B02,\
        A03, A13, A23, B03 = theta

    if template == 'wg':
        yt_spline = model['apswg']
    elif template == 'nw':
        yt_spline = model['apsnw']
    else:
        raise Exception("Invalid template:", template)

    ym = [(A00 + A10 * x[0] + A20 * x[0]**2) + B00 * yt_spline[0](x[0] / a),
          (A01 + A11 * x[0] + A21 * x[0]**2) + B01 * yt_spline[1](x[1] / a),
          (A02 + A12 * x[0] + A22 * x[0]**2) + B02 * yt_spline[2](x[2] / a),
          (A03 + A13 * x[0] + A23 * x[0]**2) + B03 * yt_spline[3](x[3] / a)]

    return(ym)


def lnprior_aA2B0i(theta, **kwargs):
    a,\
        A00, A10, A20, B00,\
        A01, A11, A21, B01,\
        A02, A12, A22, B02,\
        A03, A13, A23, B03 = theta

    if B00 > 0.0 and B01 > 0.0 and B02 > 0.0 and B03 > 0.0:
        return(0.0)
    else:
        return(-inf)


def apsmod_aA2B0i(theta, x, model, template='wg'):
    a,\
        A00, A10, A20, B00,\
        A01, A11, A21, B01,\
        A02, A12, A22, B02,\
        A03, A13, A23, B03 = theta

    if template == 'wg':
        yt_spline = model['apswg']
    elif template == 'nw':
        yt_spline = model['apsnw']
    else:
        raise Exception("Invalid template:", template)

    ym = [(A00 * x[0] + A10 + A20 / x[0]**2) + B00 * yt_spline[0](x[0] / a),
          (A01 * x[0] + A11 + A21 / x[0]**2) + B01 * yt_spline[1](x[1] / a),
          (A02 * x[0] + A12 + A22 / x[0]**2) + B02 * yt_spline[2](x[2] / a),
          (A03 * x[0] + A13 + A23 / x[0]**2) + B03 * yt_spline[3](x[3] / a)]

    return(ym)


def lnprior_aA2B0ii(theta, **kwargs):
    a,\
        A00, A10, A20, B00,\
        A01, A11, A21, B01,\
        A02, A12, A22, B02,\
        A03, A13, A23, B03 = theta

    if B00 >= 0.0 and B01 >= 0.0 and B02 >= 0.0 and B03 >= 0.0:
        return(0.0)
    else:
        return(-inf)


def apsmod_aA2B0ii(theta, x, model, template='wg'):
    a,\
        A00, A10, A20, B00,\
        A01, A11, A21, B01,\
        A02, A12, A22, B02,\
        A03, A13, A23, B03 = theta

    if template == 'wg':
        yt_spline = model['apswg']
    elif template == 'nw':
        yt_spline = model['apsnw']
    else:
        raise Exception("Invalid template:", template)

    ym = [(A00 * x[0] + A10 + A20 / x[0]) + B00 * yt_spline[0](x[0] / a),
          (A01 * x[0] + A11 + A21 / x[0]) + B01 * yt_spline[1](x[1] / a),
          (A02 * x[0] + A12 + A22 / x[0]) + B02 * yt_spline[2](x[2] / a),
          (A03 * x[0] + A13 + A23 / x[0]) + B03 * yt_spline[3](x[3] / a)]

    return(ym)


def lnprior_aA3B0(theta, **kwargs):
    a,\
        A00, A10, A20, A30, B00,\
        A01, A11, A21, A31, B01,\
        A02, A12, A22, A32, B02,\
        A03, A13, A23, A33, B03 = theta

    if B00 >= 0.0 and B01 >= 0.0 and B02 >= 0.0 and B03 >= 0.0:
        return(0.0)
    else:
        return(-inf)


def apsmod_aA3B0(theta, x, model, template='wg'):
    a,\
        A00, A10, A20, A30, B00,\
        A01, A11, A21, A31, B01,\
        A02, A12, A22, A32, B02,\
        A03, A13, A23, A33, B03 = theta

    if template == 'wg':
        yt_spline = model['apswg']
    elif template == 'nw':
        yt_spline = model['apsnw']
    else:
        raise Exception("Invalid template:", template)

    ym = [(A00 + A10 * x[0] + A20 * x[0]**2 + A30 * x[0]**3)
          + B00 * yt_spline[0](x[0] / a),
          (A01 + A11 * x[0] + A21 * x[0]**2 + A31 *
           x[0]**3) + B01 * yt_spline[1](x[1] / a),
          (A02 + A12 * x[0] + A22 * x[0]**2 + A32 *
           x[0]**3) + B02 * yt_spline[2](x[2] / a),
          (A03 + A13 * x[0] + A23 * x[0]**2 + A33 * x[0]**3)
          + B03 * yt_spline[3](x[3] / a)]

    return(ym)


def lnprior_aA3B0i(theta, **kwargs):
    a,\
        A00, A10, A20, A30, B00,\
        A01, A11, A21, A31, B01,\
        A02, A12, A22, A32, B02,\
        A03, A13, A23, A33, B03 = theta

    if B00 >= 0.0 and B01 >= 0.0 and B02 >= 0.0 and B03 >= 0.0:
        return(0.0)
    else:
        return(-inf)


def apsmod_aA3B0i(theta, x, model, template='wg'):
    a,\
        A00, A10, A20, A30, B00,\
        A01, A11, A21, A31, B01,\
        A02, A12, A22, A32, B02,\
        A03, A13, A23, A33, B03 = theta

    if template == 'wg':
        yt_spline = model['apswg']
    elif template == 'nw':
        yt_spline = model['apsnw']
    else:
        raise Exception("Invalid template:", template)

    ym = [(A00 / x[0] + A10 + A20 * x[0] + A30 * x[0]**2)
          + B00 * yt_spline[0](x[0] / a),
          (A01 / x[0] + A11 + A21 * x[0] + A31 *
           x[0]**2) + B01 * yt_spline[1](x[1] / a),
          (A02 / x[0] + A12 + A22 * x[0] + A32 *
           x[0]**2) + B02 * yt_spline[2](x[2] / a),
          (A03 / x[0] + A13 + A23 * x[0] + A33 * x[0]**2)
          + B03 * yt_spline[3](x[3] / a)]

    return(ym)


def lnprior_aA3B0ii(theta, **kwargs):
    a,\
        A00, A10, A20, A30, B00,\
        A01, A11, A21, A31, B01,\
        A02, A12, A22, A32, B02,\
        A03, A13, A23, A33, B03 = theta

    if B00 >= 0.0 and B01 >= 0.0 and B02 >= 0.0 and B03 >= 0.0:
        return(0.0)
    else:
        return(-inf)


def apsmod_aA3B0ii(theta, x, model, template='wg'):
    a,\
        A00, A10, A20, A30, B00,\
        A01, A11, A21, A31, B01,\
        A02, A12, A22, A32, B02,\
        A03, A13, A23, A33, B03 = theta

    if template == 'wg':
        yt_spline = model['apswg']
    elif template == 'nw':
        yt_spline = model['apsnw']
    else:
        raise Exception("Invalid template:", template)

    ym = [(A00 / x[0]**2 + A10 / x[0] + A20 + A30 * x[0])
          + B00 * yt_spline[0](x[0] / a),
          (A01 / x[0]**2 + A11 / x[0] + A21 + A31 * x[0]) +
          B01 * yt_spline[1](x[1] / a),
          (A02 / x[0]**2 + A12 / x[0] + A22 + A32 * x[0]) +
          B02 * yt_spline[2](x[2] / a),
          (A03 / x[0]**2 + A13 / x[0] + A23 + A33 * x[0])
          + B03 * yt_spline[3](x[3] / a)]

    return(ym)


def lnprior_aA4B0(theta, **kwargs):
    a,\
        A00, A10, A20, A30, A40, B00,\
        A01, A11, A21, A31, A41, B01,\
        A02, A12, A22, A32, A42, B02,\
        A03, A13, A23, A33, A43, B03 = theta

    if B00 >= 0.0 and B01 >= 0.0 and B02 >= 0.0 and B03 >= 0.0:
        return(0.0)
    else:
        return(-inf)


def apsmod_aA4B0(theta, x, model, template='wg'):
    a,\
        A00, A10, A20, A30, A40, B00,\
        A01, A11, A21, A31, A41, B01,\
        A02, A12, A22, A32, A42, B02,\
        A03, A13, A23, A33, A43, B03 = theta

    if template == 'wg':
        yt_spline = model['apswg']
    elif template == 'nw':
        yt_spline = model['apsnw']
    else:
        raise Exception("Invalid template:", template)

    ym = [(A00 + A10 * x[0] + A20 * x[0]**2 + A30 * x[0]**3 + A40 * x[0]**4)
          + B00 * yt_spline[0](x[0] / a),
          (A01 + A11 * x[0] + A21 * x[0]**2 + A31 * x[0] **
           3 + A41 * x[0]**4) + B01 * yt_spline[1](x[1] / a),
          (A02 + A12 * x[0] + A22 * x[0]**2 + A32 * x[0] **
           3 + A42 * x[0]**4) + B02 * yt_spline[2](x[2] / a),
          (A03 + A13 * x[0] + A23 * x[0]**2 + A33 * x[0]**3 + A43 * x[0]**4)
          + B03 * yt_spline[3](x[3] / a)]

    return(ym)


def lnprior_aA5B0(theta, **kwargs):
    a,\
        A00, A10, A20, A30, A40, A50, B00,\
        A01, A11, A21, A31, A41, A51, B01,\
        A02, A12, A22, A32, A42, A52, B02,\
        A03, A13, A23, A33, A43, A53, B03 = theta

    if B00 >= 0.0 and B01 >= 0.0 and B02 >= 0.0 and B03 >= 0.0:
        return(0.0)
    else:
        return(-inf)


def apsmod_aA5B0(theta, x, model, template='wg'):
    a,\
        A00, A10, A20, A30, A40, A50, B00,\
        A01, A11, A21, A31, A41, A51, B01,\
        A02, A12, A22, A32, A42, A52, B02,\
        A03, A13, A23, A33, A43, A53, B03 = theta

    if template == 'wg':
        yt_spline = model['apswg']
    elif template == 'nw':
        yt_spline = model['apsnw']
    else:
        raise Exception("Invalid template:", template)

    ym = [(A00 + A10 * x[0] + A20 * x[0]**2 + A30 * x[0]**3 + A40 * x[0]**4
           + A50 * x[0]**5) + B00 * yt_spline[0](x[0] / a),
          (A01 + A11 * x[0] + A21 * x[0]**2 + A31 * x[0]**3 + A41 *
           x[0]**4 + A51 * x[0]**5) + B01 * yt_spline[1](x[1] / a),
          (A02 + A12 * x[0] + A22 * x[0]**2 + A32 * x[0]**3 + A42 *
           x[0]**4 + A52 * x[0]**5) + B02 * yt_spline[2](x[2] / a),
          (A03 + A13 * x[0] + A23 * x[0]**2 + A33 * x[0]**3 + A43 * x[0]**4
              + A53 * x[0]**5) + B03 * yt_spline[3](x[3] / a)]

    return(ym)


def apsmod(theta, xd, model, template='wg', kind='aA0B1'):
    if kind == 'aA0B1':
        ym = apsmod_aA0B1(theta, xd, model, template=template)
    elif kind == 'aA1B0':
        ym = apsmod_aA1B0(theta, xd, model, template=template)
    elif kind == 'aA3B0':
        ym = apsmod_aA3B0(theta, xd, model, template=template)
    elif kind == 'aA0B0':
        ym = apsmod_aA0B0(theta, xd, model, template=template)
    elif kind == 'aA2B0':
        ym = apsmod_aA2B0(theta, xd, model, template=template)
    elif kind == 'aA2B0i':
        ym = apsmod_aA2B0i(theta, xd, model, template=template)
    elif kind == 'aA2B0ii':
        ym = apsmod_aA2B0ii(theta, xd, model, template=template)
    elif kind == 'aA3B0i':
        ym = apsmod_aA3B0i(theta, xd, model, template=template)
    elif kind == 'aA3B0ii':
        ym = apsmod_aA3B0ii(theta, xd, model, template=template)
    elif kind == 'aA4B0':
        ym = apsmod_aA4B0(theta, xd, model, template=template)
    elif kind == 'aA5B0':
        ym = apsmod_aA5B0(theta, xd, model, template=template)
    else:
        raise Exception("Invalid kind of fit:", kind)

    ym = concatenate(ym)

    return(ym)


def lnlike(theta, xd, yd, icov, model, template='wg', kind='aA0B1',
           verbose=False):
    """ Log of likelihood PDF
    Arguments
    ---------
    theta: list
           Parameters.
    xd: ndarray
        Scales of data.
    yd: ndarray
        Data.
    icov: ndarray
          Data covariance.
    model: PyDict
           Model dictionary.
    template: str
              Desired template for model, 'wg'->Wiggle, 'nw'->No-Wiggle.
    kind: str
          Kind of fit desired.
    Return
    ------
    lnlike: float
            Log of likelihood at theta
    """
    if kind == 'aA0B1':
        ym = apsmod_aA0B1(theta, xd, model, template=template)
    elif kind == 'aA1B0':
        ym = apsmod_aA1B0(theta, xd, model, template=template)
    elif kind == 'aA3B0':
        ym = apsmod_aA3B0(theta, xd, model, template=template)
    elif kind == 'aA0B0':
        ym = apsmod_aA0B0(theta, xd, model, template=template)
    elif kind == 'aA2B0':
        ym = apsmod_aA2B0(theta, xd, model, template=template)
    elif kind == 'aA2B0i':
        ym = apsmod_aA2B0i(theta, xd, model, template=template)
    elif kind == 'aA2B0ii':
        ym = apsmod_aA2B0ii(theta, xd, model, template=template)
    elif kind == 'aA3B0i':
        ym = apsmod_aA3B0i(theta, xd, model, template=template)
    elif kind == 'aA3B0ii':
        ym = apsmod_aA3B0ii(theta, xd, model, template=template)
    elif kind == 'aA4B0':
        ym = apsmod_aA4B0(theta, xd, model, template=template)
    elif kind == 'aA5B1':
        ym = apsmod_aA5B0(theta, xd, model, template=template)
    elif kind == 'aA1B1':
        ym = apsmod_aA1B1(theta, xd, model, template=template)
    else:
        raise Exception("Invalid kind of fit:", kind)

    ym = concatenate(ym)

    diff = ym - yd
    chi2 = dot(dot(icov, diff), diff)

    return(-0.5 * chi2)


def lnprob(theta, xd, yd, icov, model, template='wg', kind='aA0B1',
           verbose=False, **kwargs):
    """ Log of Posterior PDF, likelihood PDF times prior PDF
    Arguments
    ---------
    theta: list
           Parameters.
    xd: ndarray
        Scales of data.
    yd: ndarray
        Data.
    icov: ndarray
          Data covariance.
    model: PyDict
           Model dictionary.
    template: str
              Desired template for model, 'wg'->Wiggle, 'nw'->No-Wiggle.
    kind: str
          Kind of fit desired.
    Return
    ------
    lnprob: float
            Log of posterior PDF at theta.
    """
    if kind == 'aA0B1':
        lp = lnprior_aA0B1(theta, **kwargs)
    elif kind == 'aA1B0':
        lp = lnprior_aA1B0(theta, **kwargs)
    elif kind == 'aA3B0':
        lp = lnprior_aA3B0(theta, **kwargs)
    elif kind == 'aA0B0':
        lp = lnprior_aA0B0(theta, **kwargs)
    elif kind == 'aA2B0':
        lp = lnprior_aA2B0(theta, **kwargs)
    elif kind == 'aA2B0i':
        lp = lnprior_aA2B0i(theta, **kwargs)
    elif kind == 'aA2B0ii':
        lp = lnprior_aA2B0ii(theta, **kwargs)
    elif kind == 'aA3B0i':
        lp = lnprior_aA3B0i(theta, **kwargs)
    elif kind == 'aA3B0ii':
        lp = lnprior_aA3B0ii(theta, **kwargs)
    elif kind == 'aA4B0':
        lp = lnprior_aA4B0(theta, **kwargs)
    elif kind == 'aA5B0':
        lp = lnprior_aA5B0(theta, **kwargs)
    elif kind == 'aA1B1':
        lp = lnprior_aA1B1(theta, **kwargs)
    else:
        raise Exception("Invalid kind of fit:", kind)

    if not isfinite(lp):
        return -inf

    ll = lnlike(theta, xd, yd, icov, model, template=template,
                kind=kind, verbose=verbose)

    return(lp + ll)
