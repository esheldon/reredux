#!/usr/bin/env python
from __future__ import print_function

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

    return s2n,r50_pixels

def make_s2n_plot(s2n):
    log_s2n=numpy.log10( s2n.clip(min=0.001,max=10000.0) )
    plt=plot_hist(log_s2n,
                  nbin=100,
                  visible=False,
                  min=0.5,
                  max=2.5,
                  xlabel=r'$log_{10}(S/N)$',
                  aspect_ratio=1,
                  norm=1)
    plt.y1.draw_ticklabels=False
    return plt

def make_r50_plot(r50):

    print("r50:",r50.mean(), r50.std())
    nbin=50
    plt=plot_hist(r50,
                  nbin=nbin,
                  visible=False,
                  norm=1,
                  min=0,
                  max=6,
                  aspect_ratio=1,
                  xlabel=r'$r_{50} [pixels]$')


    # the r50 from the parametric sims
    p=ngmix.priors.LogNormal(1.54, 1.0)
    r=p.sample(r50.size*10)

    plot_hist(r,
              nbin=nbin,
              visible=False,
              norm=1,
              min=0,
              max=6,
              plt=plt,
              color='red',
              smooth=True,
              type='dashed')
    plt.y1.draw_ticklabels=False


    return plt

def main():
    biggles.configure('default','fontsize_min',2.0)
    s2n,r50=read_data()

    tab=Table(1,2)

    tab[0,0] = make_s2n_plot(s2n)
    tab[0,1] = make_r50_plot(r50)

    epsfile='/u/ki/esheldon/public_html/tmp/tmp.eps'
    print(epsfile)
    tab.write_eps(epsfile)

    pngfile=epsfile.replace('.eps','.png')
    print(pngfile)
    tab.write_img(800,800,pngfile)


main()
