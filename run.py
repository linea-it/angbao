""" test.py --- Simple test suite
"""
from angbao import (loadreddat, profiled_likelihood)
from numpy import (linspace, concatenate)
from pandas import (DataFrame)
from sys import (argv)


if __name__ == '__main__':
    if (len(argv) != 7):
        print("Usage:", argv[0], "<measfn> <tempfn> <A> <B> <data> <outfn>")
        exit(2)

    measfn = argv[1]
    tempfn = argv[2]
    kind = "aA%dB%d" % (int(argv[3]), int(argv[4]))
    data = argv[5]
    ofn = argv[6]
    template = 'wg'

    print("Loading data")
    ll, dd, icc, model = loadreddat(measfn, tempfn)

    print("Profiling likelihood PDF")
    da, amin, amax = 0.001, 0.8, 1.2
    alpha = linspace(amin, amax, int((amax - amin) / da) + 1)
    out = profiled_likelihood(alpha, ll, dd, icc, model, template=template,
                              kind=kind)

    print("Saving data")
    dof = (len(ll) * len(ll[0]) - (out.shape[1] - 1))
    out_red = concatenate((out,
                           out[:, -1].reshape(out.shape[0], 1) / dof),
                          axis=1)
    names = ['alpha']\
            + ['nuis%d' % (i) for i in range(out_red.shape[1] - 3)]\
            + ['chi2', 'chi2_red']
    DataFrame(data=out_red, columns=names).set_index('alpha').to_csv(ofn)
