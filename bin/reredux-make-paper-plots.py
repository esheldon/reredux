#!/usr/bin/env python
from __future__ import print_function

import os
import biggles
from biggles import plot_hist, Table
import numpy
from argparse import ArgumentParser
import reredux
import fitsio
import esutil as eu
import ngmix

#parser = ArgumentParser()

run='mcal-v14s01'
nrows=1000000

pixel_scale=0.263
nsim_color='steelblue'

def read_cosmos():
    f='$COSMOS_DIR/COSMOS_23.5_training_sample/real_galaxy_catalog_23.5_fits.fits'

    t=fitsio.read(f,lower=True)

    w, = numpy.where((t['fit_status'][:,4] != 0)
                     & (t['fit_status'][:,4] != 5)
                     & (t['sersicfit'][:,3] > 0.001) )

    t=t[w]
    return t

def write_cosmos_matched_r50(r50_des_pixels):
    fname='$LENSDIR/great3-cosmos-matched/great3-cosmos-r50-des-pixels.fits'
    print("writing:",fname)
    fitsio.write(fname, r50_des_pixels, clobber=True)

def read_reredux():
    t=reredux.files.read_collated(run, columns=['flags','gauss_s2n_w','cosmos_id'],
                                  rows=numpy.arange(nrows))
    return t

def read_data():

    rdata = read_reredux()
    cdata=read_cosmos()

    w, = numpy.where(rdata['flags']==0)

    uid = numpy.unique(rdata['cosmos_id'][w])
    mr, mc = eu.numpy_util.match(uid, cdata['ident'])

    # in HST pixels
    r50_raw = cdata['sersicfit'][mc,1]
    # in DES pixels
    r50_des_pixels = r50_raw*0.03/0.265
    q=cdata['sersicfit'][mc,3]

    r50_pixels =  numpy.sqrt(q)*r50_des_pixels

    s2n=rdata['gauss_s2n_w'][w[mr]]

    write_cosmos_matched_r50(r50_pixels)

    r50 = r50_pixels*pixel_scale
    return s2n,r50

def get_bdk_data():
    import nsim
    # the nsim bdk sim for comparison

    run='run-bdj03mcal02'

    run_conf=nsim.files.read_config(run)
    sim_conf=nsim.files.read_config(run_conf['sim'])
    sim_conf['seed']=3

    sim=nsim.simgs.SimBDJointDiffshape(sim_conf)

    nsample=1000000
    rdata = sim.joint_pdf.sample(nsample)
    r50 = rdata[:,0]

    dir=nsim.files.get_output_dir(run)

    fname='s2n-expected.txt'
    fname=os.path.join(dir, fname)
    s2n = numpy.fromfile(fname, sep='\n')

    return s2n, r50

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def make_s2n_plot(s2n, s2n_bdk):

    log_s2n=numpy.log10( s2n.clip(min=0.0001,max=10000.0) )
    log_s2n_bdk=numpy.log10( s2n_bdk.clip(min=0.0001,max=10000.0) )

    s2n_min=0.1
    s2n_max=2.5
    nbin=200

    logs2n_hdict = eu.stat.histogram(
        log_s2n,
        nbin=nbin,
        min=s2n_min,
        max=s2n_max,
        more=True,
    )
    logs2n_bdk_hdict = eu.stat.histogram(
        log_s2n_bdk,
        nbin=nbin,
        min=s2n_min,
        max=s2n_max,
        more=True,
    )

    h = logs2n_hdict['hist']
    h = h/(1.0*h.sum())
    winsize=11
    h = savitzky_golay(h, winsize, 3)
    s2n_curve = biggles.Curve(
        10.0**logs2n_hdict['center'],
        h,
        color='black',
    )
    s2n_curve.label='Real Galaxy Sim'



    hbdk = logs2n_bdk_hdict['hist']
    hbdk = hbdk/(1.0*hbdk.sum())
    s2n_bdk_curve = biggles.Curve(
        10.0**logs2n_bdk_hdict['center'],
        hbdk,
        color=nsim_color,
        type='dashed',
    )
    s2n_bdk_curve.label='BDK Sim'


    plt=biggles.FramedPlot()
    plt.xlabel=r'$log_{10}(S/N)$'
    plt.aspect_ratio=1.0
    plt.y1.draw_ticklabels=False
    plt.xlog=True
    plt.xrange=[0.9, 110.0]
    plt.yrange=[0,1.3*h.max()]

    key=biggles.PlotKey(0.9, 0.9, [s2n_curve,s2n_bdk_curve],
                        halign='right')
    plt.add(s2n_curve, s2n_bdk_curve, key)

    return plt

def make_r50_plot(r50, r50_bdk):

    r50min=0.051
    r50max=1.5
    print("r50:",r50.mean(), r50.std())
    nbin=100

    hdict = eu.stat.histogram(
        r50,
        nbin=nbin,
        min=r50min,
        max=r50max,
        more=True,
    )
    bdk_hdict = eu.stat.histogram(
        r50_bdk,
        nbin=nbin,
        min=r50min,
        max=r50max,
        more=True,
    )

    h = hdict['hist']
    h = h/(1.0*h.sum())
    winsize=7
    h = savitzky_golay(h, winsize, 3)
    curve = biggles.Curve(
        hdict['center'],
        h,
        color='black',
    )


    hbdk = bdk_hdict['hist']
    hbdk = hbdk/(1.0*hbdk.sum())
    bdk_curve = biggles.Curve(
        bdk_hdict['center'],
        hbdk,
        color=nsim_color,
        type='dashed',
    )

    plt=biggles.FramedPlot()
    plt.xlabel=r'$r_{50} [arcsec]$'
    plt.aspect_ratio=1.0
    plt.y1.draw_ticklabels=False
    plt.xrange=[0.051,1.049]
    plt.yrange=[0,1.1*hbdk.max()]

    plt.add(curve, bdk_curve)

    return plt

def main():
    biggles.configure('default','fontsize_min',2.0)
    s2n,r50=read_data()

    s2n_bdk, r50_bdk = get_bdk_data()

    tab=Table(1,2)

    tab[0,0] = make_s2n_plot(s2n, s2n_bdk)
    tab[0,1] = make_r50_plot(r50, r50_bdk)

    epsfile='/u/ki/esheldon/public_html/tmp/tmp.eps'
    print(epsfile)
    tab.write_eps(epsfile)

    pngfile=epsfile.replace('.eps','.png')
    print(pngfile)
    tab.write_img(800,800,pngfile)


main()
