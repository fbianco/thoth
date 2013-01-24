#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Savitzky-Golay smoother

    Copyright © 2005-8 Vladimir Likic, Uwe Schmitt
    New BSD License (python website)
    or GPL v.2 (bioinformatics)

    http://code.google.com/p/pyms/
"""

from math import *
from numpy import zeros, dot, concatenate, convolve, linalg, size

def resub(D, rhs):
    """ solves D D^T = rhs by resubstituion.
        D is lower triangle-matrix from cholesky-decomposition """

    M = D.shape[0]
    x1= zeros((M,),float)
    x2= zeros((M,),float)

    # resub step 1
    for l in range(M):
        sum = rhs[l]
        for n in range(l):
            sum -= D[l,n]*x1[n]
        x1[l] = sum/D[l,l]

    # resub step 2
    for l in range(M-1,-1,-1):
        sum = x1[l]
        for n in range(l+1,M):
            sum -= D[n,l]*x2[n]
        x2[l] = sum/D[l,l]

    return x2


def calcCoeff(numPoints, polyDegree, diffOrder=0):
    """ calculates filter coefficients for symmetric savitzky-golay filter.
        see: http://www.nrbook.com/a/bookcpdf/c14-8.pdf

        numPoints   means that 2*numPoints+1 values contribute to the
                     smoother.

        polyDegree   is degree of fitting polynomial

        diffOrder   is degree of implicit differentiation.
                     0 means that filter results in smoothing of function
                     1 means that filter results in smoothing the first
                                                 derivative of function.
                     and so on ...

    """

    # setup normal matrix
    A = zeros((2*numPoints+1, polyDegree+1), float)
    for i in range(2*numPoints+1):
        for j in range(polyDegree+1):
            A[i,j] = pow(i-numPoints, j)

    # calculate diff_order-th row of inv(A^T A)
    ATA = dot(A.transpose(), A)
    rhs = zeros((polyDegree+1,), float)
    rhs[diffOrder] = 1
    D = linalg.cholesky(ATA)
    wvec = resub(D, rhs)

    # calculate filter-coefficients
    coeff = zeros((2*numPoints+1,), float)
    for n in range(-numPoints, numPoints+1):
        x = 0.0
        for m in range(polyDegree+1):
            x += wvec[m]*pow(n, m)
        coeff[n+numPoints] = x
    return coeff

def smooth(signal, coeff):
    """ applies coefficients calculated by calc_coeff()
        to signal """

    N = size(coeff-1)/2

    # Padded signal with first/last values, to avoid border effect for non zeroes data
    signal = concatenate((zeros(N)+signal[0],signal,zeros(N)+signal[-1]))
    res = convolve(signal, coeff)
    return res[2*N:-2*N]

def savitzky(signal, numPoints, polyDegree, diffOrder=0):
    coeff = calcCoeff(numPoints, polyDegree, diffOrder)
    return smooth(signal, coeff)