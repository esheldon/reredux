from __future__ import print_function

import os
from pprint import pprint
import numpy
from numpy import arange, sqrt, array, zeros, ones, where

import fitsio
import ngmix
import esutil as eu
from esutil.numpy_util import between

from nsim.averaging_new import Summer

from . import files

SHAPENOISE=0.22
S2N_SOFT=20.0
DEFAULT_TEST_SIZE=100000

class Namer(object):
    """
    create strings with a specified front prefix
    """
    def __init__(self, front=None):
        self.front=front
    def __call__(self, name):
        if self.front is None or self.front=='':
            return name
        else:
            return '%s_%s' % (self.front, name)

class SummerReredux(Summer):
    def __init__(self, args):

        # assume same underlying sim for all runs, so same shears
        runs=args.runs
        conf=files.read_config(runs[0])
        sconf = files.read_sim_config(conf)
        shears_orig = sconf['shearmaker']['shears']

        # new convention
        shears=[]
        for shear in shears_orig:
            shear[0] *= -1

            shears += [shear]

        super(SummerReredux,self).__init__(conf, shears, args)

    def _preselect(self, data):
        """
        sub-classes might make a pre-selection, e.g. of some flags
        """
        w,=numpy.where(data['flags']==0)
        print("        kept %d/%d from flags" % (w.size, data.size))

        if w.size != data.size:
            data=data[w]

        return data

    def get_run_output(self, run):
        """
        collated file
        """
        fname=files.get_collated_file(run)
        return fname

class SummerDeconv(SummerReredux):
    def __init__(self, args):
        super(SummerDeconv,self).__init__(args)

        self.namer=Namer(front=None)
        self.gpsf_name='psfrec_g'

    def _get_g(self, data, w, type):
        n=self.namer
        if type=='noshear':
            name=n('e')
        else:
            name=n('e_%s' % type)

        if name not in data.dtype.names:
            g = None
        else:
            g = data[name][w]

        return g

    def _get_s2n_name(self, data):
        return self.namer('s2n_w')

'''
class Averager(dict):
    def __init__(self, run, **kw):

        """
        this show is for the detrend fits
        """
        self['run'] = run
        self['weights'] = kw.get('weights',None)
        self['show'] = kw.get('show',False)

        self.shear_true_in_struct=False

        if self['weights'] is not None:
            print("Using weights")

        conf=files.read_config(run)
        sconf = files.read_sim_config(conf)

        self.update(conf)
        self['sconf'] = sconf

        if 'model_pars' in self:
            model = self['model_pars'].keys()[0]
        else:
            model=None

        self.namer=Namer(model)

        if 'shearmaker' in self['sconf']:
            self.shears = self['sconf']['shearmaker']['shears']
        else:
            self.shears=None


    def do_averages(self, data):
        means= self._get_averages(data)
        self._write_means(means)

    def do_fits(self):
        self.means = self._read_means()

        self.fits = fit_m_c(self.means)
        self.fits_onem = fit_m_c(self.means, onem=True)

        self._write_fits(self.fits)

    def doplot(self,args):
        import biggles

        means=self.means
        fits=self.fits

        Q=calc_q(fits)

        if args.yrange is not None:
            yrange=[float(r) for r in args.yrange.split(',')]
        else:
            yrange=[-0.005,0.005]

        xrng=args.xrange
        if xrng is not None:
            xrng=[float(r) for r in args.xrange.split(',')]

        tab=biggles.Table(1,2)
        tab.aspect_ratio=0.5

        diff = means['shear'] - means['shear_true']
        differr = means['shear_err']

        plts=[]
        for i in [0,1]:

            x = means['shear_true'][:,i]
            plt =biggles.plot(
                x,
                diff[:,i],
                yerr=differr[:,i],
                xlabel='shear%d true' % (i+1,),
                ylabel='shear%d diff' % (i+1,),
                yrange=yrange,
                xrange=xrng,
                visible=False,
            )
            yfit=fits['m'][0,i]*x + fits['c'][0,i]

            c=biggles.Curve(x, yfit, color='red')
            z=biggles.Curve(x, x*0, color='black')
            plt.add(c,z)

            mstr='m%d: %.2g +/- %.2g' % (i+1,fits['m'][0,i],fits['merr'][0,i])
            cstr='c%d: %.2g +/- %.2g' % (i+1,fits['c'][0,i],fits['cerr'][0,i])
            mlab=biggles.PlotLabel(0.1,0.9,
                                   mstr,
                                   halign='left')
            clab=biggles.PlotLabel(0.1,0.85,
                                   cstr,
                                   halign='left')
            plt.add(mlab,clab)
            if i==0:
                Qstr='Q: %d' % (int(Q),)
                Qlab=biggles.PlotLabel(0.1,0.8,
                                       Qstr,
                                       halign='left')
                plt.add(Qlab)


            tab[0,i] = plt

        fname=files.get_fit_file(self['run'],
                                 extra='fit-m-c',
                                 ext='eps')
        eu.ostools.makedirs_fromfile(fname)
        print("writing:",fname)
        tab.write_eps(fname)

        if args.show:
            tab.show(width=1000, height=1000)

    def _get_arrays(self, data):

        print("getting arrays")
        n=self.namer

        gfield = n('mcal_g')
        gpsf_field = n('mcal_gpsf')

        g = data[gfield]
        gpsf = data[gpsf_field]

        return g, gpsf

    def _get_R(self, data):

        print("getting R")
        n=self.namer

        Rfield = n('mcal_R')
        Rpsf_field = n('mcal_Rpsf')

        R = data[Rfield].mean(axis=0)
        Rpsf = data[Rpsf_field].mean(axis=0)

        return R, Rpsf

    def _get_averages_noweight(self, data, sel=None, show_progress=False):

        h,rev = eu.stat.histogram(data['shear_index'], rev=True)
        nind = h.size

        g, gpsf = self._get_arrays(data)
        Rmean, Rpsf_mean = self._get_R(data)
        Rinv = numpy.linalg.inv(Rmean)

        if sel is None:
            sel=ones(2)

        means=get_mean_struct(nind)
        if 'shear_true' in data.dtype.names:
            shear_true_in_struct=True
        else:
            shear_true_in_struct=False
        for i in xrange(nind):
            if show_progress:
                print("shear index:",i)

            if rev[i] != rev[i+1]:
                w=rev[ rev[i]:rev[i+1] ]

                if self.shear_true_in_struct:
                    shear_true = data['shear_true'][w].mean(axis=0)
                else:
                    shear_true = self.shears[i]

                psf_corr  = Rpsf_mean*gpsf[w].mean(axis=0)
                gmean     = g[w].mean(axis=0) - psf_corr

                gmean *= sel

                shear     = numpy.dot(Rinv, gmean)
                shear_err = g[w].std(axis=0)/numpy.sqrt(w.size)
                shear_err = numpy.dot(Rinv, shear_err)

                means['count'][i] = w.size
                means['shear'][i] = shear
                means['shear_err'][i] = shear_err
                means['shear_true'][i] = shear_true

        w,=numpy.where(means['count'] > 0)
        if w.size != means.size:
            print("    keeping %d/%d positive count" % (w.size,means.size))
            means=means[w]
        return means


        return means

    def _get_averages_weighted(self, data, sel=None, show_progress=False):
        from numpy import newaxis

        h,rev = eu.stat.histogram(data['shear_index'], rev=True)
        nind = h.size

        weights = self._get_weights(data)

        g, gpsf = self._get_arrays(data)
        Rmean, Rpsf_mean = self._get_R_weighted(data,weights)
        Rinv = numpy.linalg.inv(Rmean)

        if sel is None:
            sel=ones(2)

        means=get_mean_struct(nind)

        for i in xrange(nind):
            if show_progress:
                print("shear index:",i)

            if rev[i] != rev[i+1]:
                w=rev[ rev[i]:rev[i+1] ]

                wts = weights[w]
                wsum=wts.sum()

                wna=wts[:,newaxis]
                wna2=wna**2

                if self.shear_true_in_struct:
                    shear_true = (data['shear_true'][w]*wna).mean(axis=0)/wsum
                else:
                    shear_true = self.shears[i]

                means['shear_true'][i] = shear_true

                psf_corr = Rpsf_mean*(gpsf[w]*wna).sum(axis=0)/wsum
                gmean = (g[w]*wna).sum(axis=0)/wsum - psf_corr

                gmean *= sel

                gerr2 = ( wna2 * (g[w]-gmean)**2 ).sum(axis=0)
                gerr = numpy.sqrt(gerr2)/wsum

                shear     = numpy.dot(Rinv, gmean)
                shear_err = numpy.dot(Rinv, gerr)

                means['count'][i] = w.size
                means['shear'][i] = shear
                means['shear_err'][i] = shear_err

        w,=numpy.where(means['count'] > 0)
        if w.size != means.size:
            print("    keeping %d/%d positive count" % (w.size,means.size))
            means=means[w]
        return means



    def _set_shear_true_in_struct(self):
        fname=files.get_collated_file(self['run'])
        with fitsio.FITS(fname) as fobj:
            colnames=fobj[1].get_colnames()
            if 'shear_true' in colnames:
                self.shear_true_in_struct=True
            else:
                self.shear_true_in_struct=False


    def _get_columns(self):
        n=self.namer
        columns=[
            n('s2n_r'),
            n('mcal_s2n_r'),
            n('mcal_g'),
            n('mcal_gpsf'),
            n('mcal_R'),
            n('mcal_Rpsf'),
            'shear_index',
            'flags'
        ]

        if self['weights'] == 's2n':
            columns += [n('mcal_s2n_r')]
        elif self['weights'] == 'noise':
            columns += [n('mcal_g_cov')]

        self._set_shear_true_in_struct()

        if self.shear_true_in_struct:
            columns.append('shear_true')

        return columns



    def _get_noise_weights(self, g_cov):
        covsum = g_cov[:,0,0] + g_cov[:,1,1]

        weights = 1.0/(2*SHAPENOISE**2 + covsum)
        return weights

    def _get_weights(self, data):
        n=self.namer
        if self['weights'] == 's2n':
            col = n('mcal_s2n_r')
            s2n = data[col]
            return self._get_s2n_weights(s2n)
        elif self['weights']=='noise':
            col = n('mcal_g_cov')
            cov = data[col]
            return self._get_noise_weights(cov)
        else:
            raise RuntimError("bad weights: '%s'" % self['weights'])

    def _get_s2n_weights(self, s2n):
        wts = 1.0/(1.0 + (S2N_SOFT/s2n)**2 )
        return wts

    def _get_averages(self, data, sel=None, show_progress=False):
        if self['weights'] is not None:
            return self._get_averages_weighted(data, sel=sel, show_progress=show_progress)
        else:
            return self._get_averages_noweight(data, sel=sel, show_progress=show_progress)

    def read_data(self, **kw):
        """
        read data from original for or cache

        parameters
        ----------
        cache: bool
            Read from the cache, creating it if needed.  Objects
            with flags are removed during cache creation
        ntest: int
            Number to read for test
        select: string
            select to apply
        select_cosmos: bool
            Make select based on cosmos catalog
        """
        cache=kw.get("cache",False)
        if cache:
            data = self._read_cached_data(**kw)
        else:
            data = self._read_uncached_data(**kw)

        return data

    def _determine_test(self, **kw):
        ntest=kw.get('ntest',DEFAULT_TEST_SIZE)
        return ntest

    def _read_cached_data(self, **kw):
        """
        selection occurs after cacheing
        """
        cache_file=get_cache_file(self['run'])
        if not os.path.exists(cache_file):
            self._cache_in_chunks()

        print("reading cache:",cache_file)

        ntest = self._determine_test(**kw)
        if ntest is not None:
            rows=arange(ntest)
        else:
            rows=None

        data=fitsio.read(cache_file, rows=rows)

        data=self.select(data, **kw)
        return data


    def _read_uncached_data(self, **kw):
        """
        we select after reading
        """
        columns=self._get_columns()

        if 'extra_cols' in kw:
            ecols=kw['extra_cols']
            if ecols is not None:
                ecols=ecols.split(',')
                columns += ecols

        print("reading columns:",columns)

        ntest = self._determine_test(**kw)
        if ntest is not None:
            rows=arange(ntest)
        else:
            rows=None

        data=files.read_collated(self['run'], columns=columns, rows=rows)

        # always make the flag select
        print("    cutting flags")
        w,=where(data['flags']==0)
        print("        keeping %d/%d from flags" % (w.size,data.size))
        if w.size != data.size:
            data=data[w]

        data=self.select(data, **kw)

        return data

    def select(self, data, **kw):
        """
        select and get the selected data and selection effect
        """

        w=self._do_select(data, **kw)
        if w is not None:
            data=data[w]

        return data

    def _cache_in_chunks(self):

        cache_file=get_cache_file(self['run'])
        print("cacheing to:",cache_file)

        columns=self._get_columns()
        print("cacheing columns:")
        pprint(columns)

        chunksize=1000000

        first=True
        with fitsio.FITS(cache_file,'rw',clobber=True) as output:
            fname=files.get_collated_file(self['run'])

            print("cacheing from:",fname)

            with fitsio.FITS(fname) as fits:

                hdu=fits[1]

                nrows=hdu.get_nrows()
                nchunks = nrows//chunksize

                if (nrows % chunksize) > 0:
                    nchunks += 1

                beg=0
                for i in xrange(nchunks):
                    print("chunk %d/%d" % (i+1,nchunks))

                    end=beg+chunksize

                    data = hdu[columns][beg:end]

                    print("    cutting flags")
                    w,=where(data['flags']==0)
                    print("        keeping %d/%d from flags" % (w.size,data.size))
                    data=data[w]

                    #w=self._do_select(data)
                    #if w is not None:
                    #    data=data[w]

                    if first:
                        output.write(data)
                        first=False
                    else:
                        output[-1].append(data)

                    beg = beg + chunksize

    def _do_select(self, data, **kw):

        select_cosmos=kw.get('select_cosmos',False)
        select=kw.get('select',None)
        if not select_cosmos and select is None:
            return None

        if select_cosmos and not hasattr(self,'_good_cosmos_ids'):
            self._good_cosmos_ids = get_good_cosmos()


        if self['weights'] is not None:
            raise RuntimeError("don't ask for weights and select")

        # some cut on cosmos goodness
        if select_cosmos:
            print("    cutting to good cosmos")
            mc,mdata=eu.numpy_util.match_multi(
                self._good_cosmos_ids,
                data['cosmos_id']
            )

            print("        keeping %d/%d good cosmos" % (mdata.size,data.size))

            logic = numpy.zeros(data.size, dtype=bool)
            logic[mdata] = True
        else:
            logic = ones(data.size, dtype=bool)


        # optional additional select given on the command line
        if select is not None:
            print("    cutting: '%s'" % select)
            tlogic = eval(select)
            w,=where(tlogic)
            print("        keeping %d/%d from select" % (w.size,data.size))

            logic = logic & tlogic

        w,=where( logic )
        print("    keeping %d/%d" % (w.size,data.size))

        return w


    def _get_means_file(self):
        return files.get_fit_file(self['run'], extra='shear-means')

    def _get_fit_file(self):
        return files.get_fit_file(self['run'], extra='fit-m-c')

    def _read_means(self):
        fname=self._get_means_file()
        print("reading:",fname)
        return fitsio.read(fname)

    def _write_means(self, means):
        fname=self._get_means_file()
        eu.ostools.makedirs_fromfile(fname)

        print("writing:",fname)
        fitsio.write(fname, means, clobber=True)

    def _write_fits(self, fits):
        fname=self._get_fit_file()

        eu.ostools.makedirs_fromfile(fname)
        print("writing fit data to file:",fname)
        fitsio.write(fname, fits, clobber=True)

class AveragerNoCorr(Averager):

    def _get_means_file(self):
        fname=files.get_fit_file(self['run'], extra='shear-means')
        return fname.replace('.fits','-nocorr.fits')

    def _get_fit_file(self):
        fname=files.get_fit_file(self['run'], extra='fit-m-c')
        return fname.replace('.fits','-nocorr.fits')


    def _get_arrays(self, data):

        print("getting arrays")
        n=self.namer

        gfield = n('mcal_g')
        g = data[gfield]

        return g


    def _get_averages_noweight(self, data, sel=None, show_progress=False):

        if sel is None:
            sel=ones(2)

        shears = self.shears

        h,rev = eu.stat.histogram(data['shear_index'], rev=True)
        nind = h.size

        g = self._get_arrays(data)

        means=get_mean_struct(nind)

        for i in xrange(nind):
            if show_progress:
                print("shear index:",i)

            w=rev[ rev[i]:rev[i+1] ]

            shear_true = shears[i]

            shear     = g[w].mean(axis=0)
            shear_err = g[w].std(axis=0)/numpy.sqrt(w.size)

            means['shear'][i] = shear
            means['shear_err'][i] = shear_err
            means['shear_true'][i] = shear_true

        return means

    def _get_columns(self):
        n=self.namer
        columns=[
            n('mcal_g'),
            'shear_index',
            'flags'
        ]

        return columns
'''

def fit_m_c_boot(data, nboot=1000):
    fits=fit_m_c(data)
    fitsone=fit_m_c(data, onem=True)

    bfits = numpy.zeros(nboot, dtype=fits.dtype)
    bfitsone = numpy.zeros(nboot, dtype=fitsone.dtype)

    print("bootstrapping m-c fits")
    for i in xrange(nboot):
        rint = numpy.random.randint(0, data.size, data.size)
        tfits=fit_m_c(data[rint], doprint=False)

        rint = numpy.random.randint(0, data.size, data.size)
        tfitsone=fit_m_c(data[rint], doprint=False, onem=True)

        bfits[i] = tfits
        bfitsone[i] = tfitsone

    fits['merr']=bfits['m'].std(axis=0)
    fits['cerr']=bfits['m'].std(axis=0)
    for i in [0,1]:
        print_m_c(i+1,
                  fits['m'][0,i],
                  fits['merr'][0,i],
                  fits['c'][0,i],
                  fits['cerr'][0,i])


    fitsone['merr'] = bfitsone['m'].std()
    fitsone['c1err'] = bfitsone['c1'].std()
    fitsone['c2err'] = bfitsone['c2'].std()

    print('  m:  %.3e +/- %.3e' % (fitsone['m'][0],fitsone['merr'][0]))
    print('  c1: %.3e +/- %.3e' % (fitsone['c1'][0],fitsone['c1err'][0]))
    print('  c2: %.3e +/- %.3e' % (fitsone['c2'][0],fitsone['c2err'][0]))

    return fits, fitsone


def fit_m_c(data, doprint=True, onem=False, max_shear=None, nocorr_select=False):

    strue = data['shear_true']

    sdiff = data['shear'] - data['shear_true']

    serr  = data['shear_err']

    if max_shear is not None:
        stot_true = sqrt(strue[:,0]**2 + strue[:,1]**2)
        w,=where(stot_true < max_shear)
        if w.size == 0:
            raise ValueError("no shears less than %g" % max_shear)
        print("kept %d/%d with shear < %g" % (w.size,data.size,max_shear))
        strue=strue[w,:]
        sdiff=sdiff[w,:]
        serr=serr[w,:]


    m = numpy.zeros(2)
    merr = numpy.zeros(2)
    c = numpy.zeros(2)
    cerr = numpy.zeros(2)

    if onem:
        fits=numpy.zeros(1, dtype=[('m','f8'),
                                   ('merr','f8'),
                                   ('c1','f8'),
                                   ('c1err','f8'),
                                   ('c2','f8'),
                                   ('c2err','f8')])


        fitter=MCFitter(strue, sdiff, serr)
        fitter.dofit()
        res=fitter.get_result()

        pars=res['pars']
        perr=res['perr']
        fits['m'] = pars[0]
        fits['c1'] = pars[1]
        fits['c2'] = pars[2]
        fits['merr'] = perr[0]
        fits['c1err'] = perr[1]
        fits['c2err'] = perr[2]

        if doprint:
            print('  m:  %.3e +/- %.3e' % (pars[0],perr[0]))
            print('  c1: %.3e +/- %.3e' % (pars[1],perr[1]))
            print('  c2: %.3e +/- %.3e' % (pars[2],perr[2]))
        return fits

    fits=numpy.zeros(1, dtype=[('m','f8',2),
                               ('merr','f8',2),
                               ('c','f8',2),
                               ('cerr','f8',2),
                               ('r','f8',2)])

    for i in [0,1]:

        #c, c_err, m, m_err, covar = fitline(strue[:,i], sdiff[:,i])
        res = fitline(strue[:,i], sdiff[:,i])
        r = res['cov']/numpy.sqrt(res['slope_err']**2 * res['offset_err']**2)
        fits['m'][0,i] = res['slope']
        fits['merr'][0,i] = res['slope_err']
        fits['c'][0,i] = res['offset']
        fits['cerr'][0,i] = res['offset_err']
        fits['r'][0,i] = r

        if doprint:
            print_m_c(i+1,
                      res['slope'],
                      res['slope_err'],
                      res['offset'],
                      res['offset_err'],
                      r=r)

    return fits


#def plot_line_fit(args, extra, x, y, res, xlabel, ylabel):
def plot_line_fit(run, extra, x, y, res, xlabel, ylabel, show=False, label_error=True):
    import biggles
    plt=biggles.FramedPlot()

    ymin=y.min()
    ymax=y.max()
    if ymin < 0:
        yr = [1.1*ymin, 0.0]
    else:
        yr = [0, 1.1*ymax]

    xr = [0.0, 1.1*x.max()]

    plt.xrange=xr
    plt.yrange=yr
    plt.xlabel=xlabel
    plt.ylabel=ylabel
    plt.aspect_ratio=1

    xfit = numpy.linspace(0, xr[1])
    yfit = res['offset'] + res['slope']*xfit

    pts = biggles.Points(x,y,type='filled circle')
    c = biggles.Curve(xfit, yfit, color='blue')

    if label_error:
        alab=r'$slope = %.3g \pm %.3g' % (res['slope'],res['slope_err'])
        blab=r'$offset = %.3g \pm %.3g' % (res['offset'],res['offset_err'])
    else:
        alab=r'$slope = %.3g' % (res['slope'],)
        blab=r'$offset = %.3g' % (res['offset'],)

    alabel=biggles.PlotLabel(0.9, 0.9, alab, halign='right')
    blabel=biggles.PlotLabel(0.9, 0.85, blab, halign='right')

    plt.add(c, pts, alabel, blabel)

    plotfile=files.get_plot_file(run, extra=extra)

    print("writing:",plotfile)
    eu.ostools.makedirs_fromfile(plotfile)
    plt.write_eps(plotfile)

    if show:
        plt.show()

    return plt



SIGMA2_MIN_CONSTANT_GROUND = 4.     # 2**2
NORMALIZATION_CONSTANT_SPACE = 1.232
NORMALIZATION_CONSTANT_GROUND = NORMALIZATION_CONSTANT_SPACE

CFID = 2.e-4
MFID = 2.e-3

def calc_q(fits):

    if 'c1' in fits.dtype.names:
        m1=fits['m'][0]
        m2=fits['m'][0]
        c1=fits['c1'][0]
        c2=fits['c2'][0]
    else:
        m1=fits['m'][0,0]
        m2=fits['m'][0,1]
        c1=fits['c'][0,0]
        c2=fits['c'][0,1]

    sigma2_min = SIGMA2_MIN_CONSTANT_GROUND 
    norm = NORMALIZATION_CONSTANT_GROUND 

    num = 2000.0 * norm
    denom = numpy.sqrt(  (c1 / CFID)**2 
                       + (c2 / CFID)**2
                       + (m1 / MFID)**2
                       + (m2 / MFID)**2
                       + sigma2_min)

    Q = num/denom
    return Q

# quick line fit pulled from great3-public code
def _calculateSvalues(xarr, yarr, sigma2=1.):
    """Calculates the intermediate S values required for basic linear regression.

    See, e.g., Numerical Recipes (Press et al 1992) Section 15.2.
    """
    if len(xarr) != len(yarr):
        raise ValueError("Input xarr and yarr differ in length!")
    if len(xarr) <= 1:
        raise ValueError("Input arrays must have 2 or more values elements.")

    S = len(xarr) / sigma2
    Sx = numpy.sum(xarr / sigma2)
    Sy = numpy.sum(yarr / sigma2)
    Sxx = numpy.sum(xarr * xarr / sigma2)
    Sxy = numpy.sum(xarr * yarr / sigma2)
    return (S, Sx, Sy, Sxx, Sxy)

def fitline(xarr, yarr):
    """Fit a line y = a + b * x to input x and y arrays by least squares.

    Returns the tuple (a, b, Var(a), Cov(a, b), Var(b)), after performing an internal estimate of
    measurement errors from the best-fitting model residuals.

    See Numerical Recipes (Press et al 1992; Section 15.2) for a clear description of the details
    of this simple regression.
    """
    # Get the S values (use default sigma2, best fit a and b still valid for stationary data)
    S, Sx, Sy, Sxx, Sxy = _calculateSvalues(xarr, yarr)
    # Get the best fit a and b
    Del = S * Sxx - Sx * Sx
    a = (Sxx * Sy - Sx * Sxy) / Del
    b = (S * Sxy - Sx * Sy) / Del
    # Use these to estimate the sigma^2 by residuals from the best-fitting model
    ymodel = a + b * xarr
    sigma2 = numpy.mean((yarr - ymodel)**2)
    # And use this to get model parameter error estimates
    var_a  = sigma2 * Sxx / Del
    cov_ab = - sigma2 * Sx / Del
    var_b  = sigma2 * S / Del

    a_err = numpy.sqrt(var_a)
    b_err = numpy.sqrt(var_b)
    return {'offset':a,
            'offset_err':a_err,
            'slope':b,
            'slope_err':b_err,
            'cov':cov_ab}

    #return a, a_err, b, b_err, cov_ab

class MCFitter(object):
    """
    fit single m
    """
    def __init__(self, shear_true, shear_diff, shear_err):
        self.shear_true=shear_true
        self.shear_diff=shear_diff
        self.shear_err=shear_err

        self.npars=3

        self._set_guess()

    def get_result(self):
        return self._result

    def dofit(self):
        from scipy.optimize import leastsq
        res=leastsq(self._errfunc, self.guess, full_output=1)

        pars, pcov0, infodict, errmsg, ier = res
        if ier == 0:
            # wrong args, this is a bug
            raise ValueError(errmsg)

        numiter = infodict['nfev']
        pcov=None
        perr=None

        if pcov0 is not None:
            pcov = self._scale_leastsq_cov(pars, pcov0)

            d=numpy.diag(pcov)
            w,=numpy.where(d < 0)

            if w.size == 0:
                # only do if non negative
                perr = numpy.sqrt(d)
        
        self._result={'pars':pars, 'pcov':pcov, 'perr':perr}


    def _set_guess(self):
        res1 = fitline(self.shear_true[:,0], self.shear_diff[:,0])
        res2 = fitline(self.shear_true[:,1], self.shear_diff[:,1])

        m = 0.5*(res1['slope'] + res2['slope'])

        self.guess = numpy.array([m, res1['offset'], res2['offset']])

    def eval_pars(self, pars):
        """
        [m, c1, c2]
        """
        return pars[0]*self.shear_true + pars[1:1+2]

    def _errfunc(self, pars):
        """
        [m, c1, c2]
        """

        model = self.eval_pars(pars)
        diff = (model-self.shear_diff)/self.shear_err
        return diff.ravel()

    def _scale_leastsq_cov(self, pars, pcov):
        """
        Scale the covariance matrix returned from leastsq; this will
        recover the covariance of the parameters in the right units.
        """

        # is total size right here?
        dof = (self.shear_diff.size-len(pars))
        s_sq = (self._errfunc(pars)**2).sum()/dof
        return pcov * s_sq 

    def __call__(self, x):
        """
        pars order same as for numpy.poly1d
        """
        return self.eval_pars(self._result['pars'])

    def __repr__(self):
        if hasattr(self,'_result'):
            pars=self._result['pars']
            perr=self._result['perr']

            if perr is not None:
                rep = """(d1,d2) = m*(s1,s2) + (c1,c2)
    m:  %g +/- %g
    c1: %g +/- %g
    c2: %g +/- %g""" % (pars[0],perr[0],pars[1],perr[1],pars[2],perr[2])

            else:
                rep = """(d1,d2) = m*(s1,s2) + (c1,c2)
    m:  %g +/- None
    c1: %g +/- None
    c2: %g +/- None""" % (pars[0],pars[1],pars[2])

        else:
            rep=""

        return rep



def get_cache_file(run):
    tdir=os.environ['TMPDIR']
    cache_file= files.get_collated_file(run)
    cache_file=os.path.basename(cache_file).replace('.fits','-cache.fits')

    cache_file=os.path.join(tdir, cache_file)
    return cache_file

def get_mean_struct(n):
    dt=[('count','i8'),
        ('shear','f8',2),
        ('shear_true','f8',2),
        ('shear_err','f8',2)]

    means = numpy.zeros(n, dtype=dt)
    return means

def print_m_c(sindex, m, merr, c, cerr, r=None):
    fmt = '  m{i}: %.3e +/- %.3e c{i}: %.3e +/- %.3e'.format(i=sindex)
    if r is not None:
        fmt += '  r{i}: %.3g'.format(i=sindex)
        print(fmt % (m, merr, c, cerr,r))
    else:
        print(fmt % (m, merr, c, cerr))


def get_good_cosmos(cat=None):
    """
    cuts from
    https://github.com/GalSim-developers/GalSim/blob/%23654/galsim/scene.py#L212
    """
    if cat is None:
        d=os.environ['COSMOS_DIR']
        d=os.path.join(d,'COSMOS_23.5_training_sample')
        fname=os.path.join(d,'real_galaxy_catalog_23.5_fits.fits')
        print("    reading cosmos cat:",fname)
        cat = fitsio.read(fname)

    mask=numpy.ones(cat.size,dtype=bool)

    min_hlr = 0.0
    max_hlr = 2.5
    min_flux = 8.0
    max_flux = 50.0

    cosmos_pix_scale = 0.03
    sparams = cat['sersicfit']

    sersicfit_status = cat['fit_status'][:,4]
    bulgefit_status = cat['fit_status'][:,0]
    mask &= ( (sersicfit_status > 0) &
              (sersicfit_status < 5) &
              (bulgefit_status > 0) &
              (bulgefit_status < 5) )

    # this cut is somehow different than below...
    hlr_pix = sparams[:,1]
    n = sparams[:,2]
    mask &= ( (n < 5) | (hlr_pix < 1./cosmos_pix_scale) )

    # different type of cut on hlr
    q = sparams[:,3]
    hlr = cosmos_pix_scale*hlr_pix*numpy.sqrt(q)
    mask &= ( (hlr > min_hlr) & (hlr < max_hlr) )

    # The prefactor for n=4 is 3.607.  For n=1, it is 1.901.
    # It's not linear in these values, but for the sake of efficiency and the 
    # ability to work on the whole array at once, just linearly interpolate.
    # Hopefully, this can be improved as part of issue #693.  Maybe by storing the
    # calculated directly flux in the catalog, rather than just the amplitude of the
    # surface brightness profile at the half-light-radius?
    #prefactor = ( (n-1.)*3.607 + (4.-n)*1.901 ) / (4.-1.)

    flux_hlr = sparams[:,0]
    prefactor = ((3.607-1.901)/3.) * n + (4.*1.901 - 1.*3.607)/3.
    flux = 2.0*numpy.pi*prefactor*(hlr**2)*flux_hlr/cosmos_pix_scale**2
    mask &= ( (flux > min_flux) & (flux < max_flux) )

    w,=numpy.where(mask)

    return cat['IDENT'][w]
