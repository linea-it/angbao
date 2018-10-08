""" mle.py --- Maximum Likelihood Estimator routines
"""
from __future__ import (print_function)
from numpy import (array, exp, dot, concatenate, argmin, zeros, diag)
from numpy.linalg import (inv, block_diag)
from lnlike import (lnprob)
from minimization import (find_tempamp)
from scipy.optimize import (minimize)
from sys import (argv, stderr)


def design_matrix_A3B0_(a, x, model, template='wg',
                        r_zmean=[1697.151505201687, 1904.8679873075332,
                                 2104.755175184404, 2299.5412779720123],
                        **kwargs):
    """Construct design matrix for aA3B0 fit
    """
    nw_spline = model['apsnw']
    if template == 'wg':
        yt_spline = model['apswg']
    elif template == 'nw':
        yt_spline = model['apsnw']
    else:
        raise Exception("Invalid template:", template)

    Sigma = kwargs['Sigma']
    Olin = [yt_spline[iz](x[iz] / a) / nw_spline[iz](x[iz] / a)
            for iz in range(4)]
    Odamp = [1.0 + (Olin[iz] - 1.0) *
             exp(- Sigma**2 * ((x[iz] + 0.5) / r_zmean[iz])**2)
             for iz in range(4)]

    # parameter order: Ai, B0
    A = [array([x[iz] * Odamp[iz], Odamp[iz], Odamp[iz] / x[iz]])
         for iz in range(4)]
    A = block_diag(*A)

    return(A)


def funcB0(B0, a, A, xd, yd, icov, model, template='wg', kind='aA2B0',
           verbose=False, B0prior=False, **kwargs):
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
                      verbose=False, B0prior=B0prior,
                      **kwargs)
    return(c2)


def hybridminimization(a, xd, yd, icov, model, template='wg',
                       kind='aA2B0', verbose=False, B0prior=False,
                       **kwargs):

    A = design_matrix_(a, xd, model, template=template, kind=kind,
                       **kwargs)

    # Minimize B0
    def func(B0): return funcB0(B0, a, A, xd, yd, icov, model,
                                template=template, kind=kind,
                                verbose=False, B0prior=False, **kwargs)
    B0 = minimize(func, [1, 1, 1, 1], method='Powell')  # ,
    # options={'disp': False, 'ftol': 0.0})
    B0 = B0['x']

    # B0 is assoc with Ai's
    ydloc = yd_noB0(a, B0, yd, xd, model, template=template,
                    kind=kind, **kwargs)
    D1 = dot(A, dot(icov, A.T))
    D2 = dot(A, dot(icov, ydloc))
    # print(abs(dot(inv(D1), D1) - eye(D1.shape[0], D1.shape[1])).max())
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


def design_matrix_(a, x, model, template='wg', kind='aA1B0', **kwargs):
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


def profiled_likelihood_(alpha, xd, yd, icov, model, template='wg',
                         kind='aA0B1', B0prior=True, **kwargs):

    # Number of parameters per shell
    # TODO: Can be done better...
    if kind == 'aA0B0':
        npar = 1
    elif kind == 'aA1B0':
        npar = 2
    elif kind == 'aA2B0' or kind == 'aA2B0i':
        npar = 3
    elif kind in ['aA3B0', 'aA3B0i', 'aA3B0ii']:
        npar = 4
    elif kind in ['aA4B0']:
        npar = 4
    elif kind in ['aA5B0']:
        npar = 5
    else:
        raise Exception("Invalid kind of fit:", kind)

    if B0prior:
        T = [zeros(npar) for iz in range(4)]
        L0 = [zeros(npar) for iz in range(4)]
        for iz in range(4):
            L0[iz][-1] = kwargs['mu'][iz]
            T[iz][-1] = 1. / kwargs['sigma'][iz]**2
        T = [diag(t) for t in T]
        T = block_diag(*T)
        L0 = concatenate(L0)
        print(" ** Applying prior on B parameters")
        print("    mu =", kwargs['mu'], "sigma =", kwargs['sigma'])

    theta = []
    chi2 = []
    for a in alpha:
        th = hybridminimization(a, xd, yd, icov, model, template=template,
                                kind=kind, verbose=False, B0prior=False,
                                **kwargs)
        c2 = -2. * lnprob(th, xd, yd, icov, model, template=template,
                          kind=kind, verbose=False,
                          B0prior=B0prior, **kwargs)
        theta.append(th)
        chi2.append(c2)

    imin = argmin(chi2)
    print(chi2[imin], "/", len(xd) * len(xd[0]) - len(th), theta[imin])

    out = concatenate((array(theta),
                       array(chi2).reshape(len(alpha), 1)),
                      axis=1)
    return(out)


if __name__ == '__main__':
    if (len(argv) != 7):
        print("Usage:", argv[0], "<A> <B> <dl> <lmin> <lmax> <data>")
        exit(2)

    print(argv)
    # Parameters
    dl, lmin, lmax = int(argv[3]), float(argv[4]), float(argv[5])
#    kind = "aA%dB%di" % (int(argv[1]), int(argv[2]))
    kind = "aA%dB%d" % (int(argv[1]), int(argv[2]))
    data = argv[6]
    nooutl = False
    Sigma = 5.6
    hc = True
    template = 'wg'
    B0prior = False
    fullcov = True
    njk = None

    # Case name
    odir = "/tmp"
    stat = 'mle'
    if dl is None:
        pref = "%s/%s_%s_%s_%s_fidbinning"\
               % (odir, stat, kind, template, data)
    else:
        pref = "%s/%s_%s_%s_%s_dl%d_lmin%dlmax%d"\
               % (odir, stat, kind, template, data, dl, lmin, lmax)
    if B0prior:
        pref = "%s_B0prior" % (pref)
    if fullcov:
        pref = "%s_fullcov" % (pref)

    from reddat import reduce_datasets_all
    ll, dd, icc, model = reduce_datasets_all(dl, lmin, lmax,
                                             kind=kind, data=data,
                                             nooutl=nooutl,
                                             Sigma=Sigma, hc=hc,
                                             fullcov=fullcov, B0prior=B0prior)

    # Put a prior on B0:
    kwargs = {}
    if B0prior:
        args = reduce_datasets_all(dl, lmin, ll[0][3], kind=kind,
                                   data=data, nooutl=nooutl,
                                   Sigma=Sigma, hc=hc,
                                   fullcov=fullcov, lmax=lmax,
                                   B0prior=B0prior)
        kwargs['mu'] = find_tempamp(*args, template=template)
        kwargs['sigma'] = [0.4 * mu for mu in kwargs['mu']]

    from numpy import linspace
    da, amin, amax = 0.001, 0.8, 1.2
    alpha = linspace(amin, amax, int((amax - amin) / da) + 1)
    out = profiled_likelihood_(alpha, ll, dd, icc, model,
                               template=template, kind=kind,
                               B0prior=B0prior, Sigma=Sigma, **kwargs)

    # Save MLE results.
    from numpy import savetxt
    dof = (len(ll) * len(ll[0]) - (out.shape[1] - 1))
    out_red = concatenate((out,
                           out[:, -1].reshape(out.shape[0], 1) / dof),
                          axis=1)
    savetxt("%s.dat.gz" % (pref), out_red, fmt='%le',
            header="alpha [nuiscence]x4 chi2 chi2_red")

    # Print for tables...
    from scipy.interpolate import InterpolatedUnivariateSpline
    from scipy.optimize import brentq
    chi2 = out[:, -1]
    imin = argmin(chi2)
    _chi2 = InterpolatedUnivariateSpline(alpha, chi2 -
                                         chi2[imin] - 1.0)
    left = brentq(_chi2, alpha[0], alpha[imin])
    right = brentq(_chi2, alpha[imin], alpha[-1])
    alpha_err = 0.5 * (right - left)

    print("$%.3f \pm %.3f\, %.3f/%d(%.4f)$" %
          (alpha[imin], alpha_err, chi2[imin], dof, chi2[imin] / dof))

    if data == 'mock':
        print("%d,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%d,%.4f\n" %
              (ll[0][0] - int(dl / 2),
               ll[0][-1] + int(dl / 2),
               dl,
               len(ll[0]),
               int(argv[1]), int(argv[2]),
               alpha[imin], alpha_err, chi2[imin] * 1800., dof,
               chi2[imin] / dof * 1800.),
              file=stderr)
    else:
        print("%d,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%d,%.4f\n" %
              (ll[0][0] - int(dl / 2),
               ll[0][-1] + int(dl / 2),
               dl,
               len(ll[0]),
               int(argv[1]), int(argv[2]),
               alpha[imin], alpha_err, chi2[imin], dof, chi2[imin] / dof),
              file=stderr)
