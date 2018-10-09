""" mle.py --- Maximum Likelihood Estimator routines
    -
"""
from __future__ import (print_function)
from numpy import (array, dot, concatenate, argmin)
from numpy.linalg import (inv)
from scipy.linalg import (block_diag)
from scipy.optimize import (minimize)
from lnlike import (lnprob)


def design_matrix(a, x, model, template='wg', kind='aA1B0', **kwargs):
    """Construtct design matrix
    """
    # parameter order: Ai, B0
    from numpy import ones_like
    if kind == 'aA0B0':
        A = [array([ones_like(x[iz])])
             for iz in range(4)]
    elif kind == 'aA1B0':
        A = [array([ones_like(x[iz]), x[iz]]) for iz in range(4)]
    elif kind == 'aA2B0i':
        A = [array([x[iz], ones_like(x[iz]), 1.0 / x[iz]**2])
             for iz in range(4)]
    elif kind == 'aA2B0ii':
        A = [array([x[iz], ones_like(x[iz]), 1.0 / x[iz]])
             for iz in range(4)]
    elif kind == 'aA3B0i':
        A = [array([1.0 / x[iz], ones_like(x[iz]), x[iz], x[iz]**2])
             for iz in range(4)]
    elif kind == 'aA3B0ii':
        A = [array([1.0 / x[iz]**2, 1.0 / x[iz], ones_like(x[iz]), x[iz]])
             for iz in range(4)]
    elif kind == 'aA2B0':
        A = [array([ones_like(x[iz]), x[iz], x[iz]**2])
             for iz in range(4)]
    elif kind == 'aA3B0':
        A = [array([ones_like(x[iz]), x[iz], x[iz]**2, x[iz]**3])
             for iz in range(4)]
    elif kind == 'aA4B0':
        A = [array([ones_like(x[iz]), x[iz], x[iz]**2, x[iz]**3, x[iz]**4])
             for iz in range(4)]
    elif kind == 'aA5B0':
        A = [array([ones_like(x[iz]), x[iz], x[iz]**2, x[iz]**3, x[iz]**4,
                    x[iz]**5])
             for iz in range(4)]
    else:
        raise Exception("Invalid kind of fit:", kind)
    A = block_diag(*A)

    return(A)


def funcB0(B0, a, A, xd, yd, icov, model, template='wg', kind='aA2B0',
           verbose=False, **kwargs):
    ydloc = yd_noB0(a, B0, yd, xd, model, template=template,
                    kind=kind, **kwargs)
    D1 = dot(A, dot(icov, A.T))
    D2 = dot(A, dot(icov, ydloc))
    # print(abs(dot(inv(D1), D1) - eye(D1.shape[0], D1.shape[1])).max())
    lam = dot(inv(D1), D2).tolist()

    if kind == 'aA2B0' or kind == 'aA2B0i' or kind == 'aA2B0ii':
        th = [a] + lam[0:3] + [B0[0]] + lam[3:6] + [B0[1]] +\
            lam[6:9] + [B0[2]] + lam[9:12] + [B0[3]]
    elif kind == 'aA1B0':
        th = [a] + lam[0:2] + [B0[0]] + lam[2:4] + [B0[1]] +\
            lam[4:6] + [B0[2]] + lam[6:8] + [B0[3]]
    elif kind == 'aA0B0':
        th = [a] + lam[0:1] + [B0[0]] + lam[1:2] + [B0[1]] +\
            lam[2:3] + [B0[2]] + lam[3:4] + [B0[3]]
    elif kind in ['aA3B0', 'aA3B0i', 'aA3B0ii']:
        th = [a] + lam[0:4] + [B0[0]] + lam[4:8] + [B0[1]] +\
            lam[8:12] + [B0[2]] + lam[12:16] + [B0[3]]
    elif kind in ['aA4B0']:
        th = [a] + lam[0:5] + [B0[0]] + lam[5:10] + [B0[1]] +\
            lam[10:15] + [B0[2]] + lam[15:20] + [B0[3]]
    elif kind in ['aA5B0']:
        th = [a] + lam[0:6] + [B0[0]] + lam[6:12] + [B0[1]] +\
            lam[12:18] + [B0[2]] + lam[18:24] + [B0[3]]
    else:
        raise Exception("Invalid kind of fit:", kind)

    c2 = -2. * lnprob(th, xd, yd, icov, model,
                      template=template, kind=kind,
                      verbose=False,
                      **kwargs)
    return(c2)


def hybridminimization(a, xd, yd, icov, model, template='wg',
                       kind='aA2B0', verbose=False,
                       **kwargs):
    """
    """
    A = design_matrix(a, xd, model, template=template, kind=kind,
                      **kwargs)

    # Minimize B0
    def func(B0): return funcB0(B0, a, A, xd, yd, icov, model,
                                template=template, kind=kind,
                                verbose=False, **kwargs)
    B0 = minimize(func, [1, 1, 1, 1], method='Powell')  # Carefull here!
    B0 = B0['x']

    # B0 is assoc with Ai's
    ydloc = yd_noB0(a, B0, yd, xd, model, template=template,
                    kind=kind, **kwargs)
    D1 = dot(A, dot(icov, A.T))
    D2 = dot(A, dot(icov, ydloc))
    lam = dot(inv(D1), D2).tolist()

    if kind == 'aA0B0':
        th = [a] + lam[0:1] + [B0[0]] + lam[1:2] + \
            [B0[1]] + lam[2:3] + [B0[2]] + lam[3:4] + [B0[3]]
    elif kind == 'aA1B0':
        th = [a] + lam[0:2] + [B0[0]] + lam[2:4] + \
            [B0[1]] + lam[4:6] + [B0[2]] + lam[6:8] + [B0[3]]
    elif kind == 'aA2B0' or kind == 'aA2B0i' or kind == 'aA2B0ii':
        th = [a] + lam[0:3] + [B0[0]] + lam[3:6] + [B0[1]] +\
            lam[6:9] + [B0[2]] + lam[9:12] + [B0[3]]
    elif kind in ['aA3B0', 'aA3B0i', 'aA3B0ii']:
        th = [a] + lam[0:4] + [B0[0]] + lam[4:8] + [B0[1]] +\
            lam[8:12] + [B0[2]] + lam[12:16] + [B0[3]]
    elif kind in ['aA4B0']:
        th = [a] + lam[0:5] + [B0[0]] + lam[5:10] + [B0[1]] +\
            lam[10:15] + [B0[2]] + lam[15:20] + [B0[3]]
    elif kind in ['aA5B0']:
        th = [a] + lam[0:6] + [B0[0]] + lam[6:12] + [B0[1]] +\
            lam[12:18] + [B0[2]] + lam[18:24] + [B0[3]]
    else:
        raise Exception("Invalid kind of fit:", kind)

    return(th)


def yd_noB0(a, B0, yd, x, model, template='wg', kind='aA1B0', **kwargs):
    """Construtct design matrix for aA1B0 fit
    """
    if template == 'wg':
        yt_spline = model['apswg']
    elif template == 'nw':
        yt_spline = model['apsnw']
    else:
        raise Exception("Invalid template:", template)

    # parameter order: Ai, B0
    correction = [B0[0] * yt_spline[0](x[0] / a),
                  B0[1] * yt_spline[1](x[1] / a),
                  B0[2] * yt_spline[2](x[2] / a),
                  B0[3] * yt_spline[3](x[3] / a)]

    correction = concatenate(correction)
    yd = yd - correction

    return(yd)


def profiled_likelihood(alpha, xd, yd, icov, model, template='wg',
                        kind='aA0B1', **kwargs):
    theta = []
    chi2 = []
    for a in alpha:
        th = hybridminimization(a, xd, yd, icov, model, template=template,
                                kind=kind, verbose=False,
                                **kwargs)
        c2 = -2. * lnprob(th, xd, yd, icov, model, template=template,
                          kind=kind, verbose=False,
                          **kwargs)
        theta.append(th)
        chi2.append(c2)

    imin = argmin(chi2)
    print("Minimum found at:", end=' ')
    print(chi2[imin], "/", len(xd) * len(xd[0]) - len(th), theta[imin])

    out = concatenate((array(theta),
                       array(chi2).reshape(len(alpha), 1)),
                      axis=1)
    return(out)
