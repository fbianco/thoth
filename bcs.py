#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    \package BCS


    \file bcs.py
    \author François Bianco, University of Geneva - francois.bianco@unige.ch
    \date 2011.08.05
    \version 0.01


    \mainpage BCS fitting

    \section Copyright

    Copyright (C) 2011 François Bianco, University of Geneva - francois.bianco@unige.ch

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

from __future__ import division
from pylab import *
from scipy import optimize

kb = 0.13807/1.602 # meV/K

#def discrete_convolution(a, b):
    ## (a*b)[n] = sum a[m]b[n-m]
    #assert len(a) == len(b)
    #length = len(a)
    #result = []
    #for n in arange(length):
        #intermediate_result = 0
        #for m in arange(n+1):
            #intermediate_result += a[m]*b[n-m]
        #result.append(intermediate_result)
    #return array(result)

def fft_convolution(a, b):
    """Compute the convolution of a and b using fft:
    (a*b) = fft(ifft(b)*ifft(a))"""
    length = len(a)
    middle = int(length/2)
    a1 = a[middle:]
    a2 = a[:middle]
    # padding of a, makes it periodic
    padded_a = np.concatenate((a1,a2))
    ifft_a = ifft(a)
    result = real(fft(ifft_a*ifft(b)))/real(ifft_a[0])
    result1 = result[middle:]
    result2 = result[:middle]
    # need to unpad the results
    return np.concatenate((result1,result2))

def BCS_fit(G_delta0, x, y) :
    """ Usage G_delta0,x in meV. max of y normalized to 1, gap = 0.

    Optimize only the temperature T parameter of the BCS function to fit the
    data (x,y) with a superconducting gap of G_delta0.

    Return the fitted temperature in K.
    """

    errfunc = lambda T, G_delta0, x, y: BCS(G_delta0, T, x) - y

    T = 4.11 # Initial guess for the temperature
    T_fit,success = optimize.leastsq(errfunc, T, args=(G_delta0,x, y))

    return T_fit

def BCS(G_delta0, G_temp, x, Shift_x=0, G_sigma=0, G_gamma=0) :
    """ Usage BCS(gap,temperature,energy) in meV and K
    G_delta0 superconducting gap in meV
    G_temp temperature in K
    Shift_x for non symmetrical x axis
    G_sigma experimental broadening
    G_gamma lifetime
    """

    YBCS = abs(real((x - 1j * G_gamma) / \
           sqrt((x - 1j * G_gamma)**2 - G_delta0**2)))

    # Thermal broadening
    if G_temp > 0:
        g = BCS_thermal_broadening(G_temp, x, Shift_x)
        YBCS = fft_convolution(g, YBCS)

    # Experimental broadening
    if G_sigma > 0 :
        g = BCS_exeprimental_broadening(G_sigma, x)
        YBCS = fft_convolve(g, YBCS)

    return YBCS

def BCS_thermal_broadening(G_temp, x, Shift_x=0):
    """Add thermal broadening to to the spectra"""
    return -1. / (4 * kb * G_temp * (cosh((x - Shift_x) / \
                  (2 * kb * G_temp)))**2)

def BCS_exeprimental_broadening(G_sigma, x):
    """Add experimental broadening to to the spectra"""
    return exp((-(x - Shift_x / G_sigma)**2) / 2) / \
                    (sqrt(2 * pi) * G_sigma)

def BCS_plot(G_delta0=2, G_temp=0, x=linspace(-100,100,10000)) :
    """Plot a BCS curve"""
    plot(x,BCS(G_delta0, G_temp, x))
    xlabel('Energy (meV)')
    ylabel('Density of state (a.u.)')
    legend(('BCS gap at %g K' % G_temp,))
    grid('on')

if __name__ == "__main__":
    pass