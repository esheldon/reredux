from __future__ import print_function

import os
from pprint import pprint
import numpy
from numpy import arange, sqrt, array, zeros

import fitsio
import esutil as eu
from esutil.numpy_util import between

from . import files

S2N_SOFT=20.0

MIN_LOG_T=-3.0
#MAX_LOG_T=0.0
MAX_LOG_T=-0.4
#MAX_LOG_T=-0.3

class Averager(dict):
    def __init__(self, run, fit_only=False, use_weights=False, use_cache=False,
                 do_test=False, ntest=100000, show=False, cuts=None):

        """
        this show is for the detrend fits
        """
        self['run'] = run
        self['use_weights'] = use_weights
        self['use_cache'] = use_cache
        self['fit_only'] = fit_only
        self['show'] = show

        self['cuts'] = cuts

        self['do_test'] = do_test
        self['ntest'] = ntest

        if self['use_weights']:
            print("Using weights")

        conf=files.read_config(run)
        sconf=files.read_egret_config(conf['reredux_config'])

        self.update(conf)
        self['sconf'] = sconf

        self['model'] = self['model_pars'].keys()[0]
        #self.shears = self['sconf']['shear']['shears']
        self.shears = self['sconf']['shearmaker']['shears']

        if not self['fit_only']:
            self._load_data()

    def go(self):
        print("fit only:",self['fit_only'])
        if self['fit_only']:
            self.means = self._read_means()
        else:
            self.means= self._get_averages()
            self._write_means()

        '''
        maxdiff=0.0015
        w,=numpy.where(
            (numpy.abs(self.means['shear'][:,0] - self.means['shear_true'][:,0]) < maxdiff)
            &
            (numpy.abs(self.means['shear'][:,1] - self.means['shear_true'][:,1]) < maxdiff)
        )
        means_use=self.means[w]
        print("kept means %d/%d" % (w.size,self.means.size))
        self.fits = fit_m_c(means_use)
        '''
        self.fits = fit_m_c(self.means)
        #self.Q=calc_q(self.fits)
        #print("  Q: %d" % self.Q)

        #self.fits_onem = fit_m_c(means_use, onem=True)
        self.fits_onem = fit_m_c(self.means, onem=True)
        #self.Q_onem=calc_q(self.fits_onem)

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

    def _get_averages_noweight(self, show_progress=False):

        data=self.data

        shears = self.shears

        h,rev = eu.stat.histogram(data['shear_index'], rev=True)
        nind = h.size

        g, gpsf, R, Rpsf = self._get_arrays()

        means=get_mean_struct(nind)

        for i in xrange(nind):
            if show_progress:
                print("shear index:",i)

            w=rev[ rev[i]:rev[i+1] ]

            shear_true = shears[i]

            Rmean = R[w].mean(axis=0)
            Rinv = numpy.linalg.inv(Rmean)

            psf_corr  = (gpsf[w]*Rpsf[w]).mean(axis=0)

            gmean     = g[w].mean(axis=0)
            shear     = numpy.dot(Rinv, gmean-psf_corr)
            shear_err = g[w].std(axis=0)/numpy.sqrt(w.size)
            shear_err = numpy.dot(Rinv, shear_err)

            means['shear'][i] = shear
            means['shear_err'][i] = shear_err
            means['shear_true'][i] = shear_true

        return means

    def _get_averages_weighted(self, show_progress=False):

        data=self.data

        shears = self.shears

        h,rev = eu.stat.histogram(data['shear_index'], rev=True)
        nind = h.size

        g, gpsf, R, Rpsf = self._get_arrays()

        means=get_mean_struct(nind)

        model=self['model_pars'].keys()[0]
        wfield = '%s_mcal_s2n_r' % model

        Rmean = zeros( (2,2) )
        psf_corr = zeros( 2 )
        gmean = zeros( 2 )
        gerr  = zeros( 2 )

        for i in xrange(nind):
            if show_progress:
                print("shear index:",i)

            w=rev[ rev[i]:rev[i+1] ]

            s2n = data[wfield][w]
            wts = self._get_s2n_weights(s2n)
            #wts = 1.0/(1.0 + (S2N_SOFT/s2n)**2 )

            wts2=wts**2
            wsum=wts.sum()

            shear_true = shears[i]

            for ii in xrange(2):
                gmean[ii] = (g[w,ii]*wts).sum()/wsum
                gerr2 = ( wts2 * (g[w,ii]-gmean[ii])**2 ).sum()
                gerr[ii] = numpy.sqrt(gerr2)
                psf_corr[ii]  = (gpsf[w,ii]*Rpsf[w,ii]*wts).sum()/wsum

                for jj in xrange(ii,2):
                    Rmean[ii,jj] = (R[w,ii,jj]*wts).sum()/wsum
                    if jj != ii:
                        Rmean[jj,ii] = Rmean[ii,jj]


            Rinv = numpy.linalg.inv(Rmean)

            shear     = numpy.dot(Rinv, gmean-psf_corr)
            shear_err = numpy.dot(Rinv, gerr)

            means['shear'][i] = shear
            means['shear_err'][i] = shear_err
            means['shear_true'][i] = shear_true

        return means

    def _get_arrays(self):

        data=self.data

        print("getting arrays")
        model=self['model']

        gfield = '%s_mcal_g' % model
        gpsf_field = '%s_mcal_gpsf' % model

        Rfield = '%s_mcal_R' % model
        Rpsf_field = '%s_mcal_Rpsf' % model

        g = data[gfield]
        gpsf = data[gpsf_field]
        R = data[Rfield]
        Rpsf = data[Rpsf_field]

        return g, gpsf, R, Rpsf

    def _get_columns(self):
        model=self['model']
        columns=[
            '%s_mcal_g' % model,
            '%s_mcal_gpsf' % model,
            '%s_mcal_R' % model,
            '%s_mcal_Rpsf' % model,
            'shear_index',
            'flags'
        ]

        if self['use_weights']:
            columns += ['%s_mcal_s2n_r' % model]

        return columns



    def _get_s2n_weights(s2n):
        wts = 1.0/(1.0 + (S2N_SOFT/s2n)**2 )
        return wts

    def _get_averages(self, show_progress=False):
        if self['use_weights']:
            return self._get_averages_weighted(show_progress=show_progress)
        else:
            return self._get_averages_noweight(show_progress=show_progress)

    def _load_data(self):
        if self['use_cache']:
            data = self._read_cached_data()
        else:
            data = self._read_data()
        self.data=data

    def _read_cached_data(self):
        cache_file=get_cache_file(self['run'])
        if not os.path.exists(cache_file):
            self._cache_in_chunks()

        print("reading cache:",cache_file)

        if self['do_test']:
            rows=arange(self['ntest'])
        else:
            rows=None

        data=fitsio.read(cache_file, rows=rows)

        return data


    def _read_data(self):

        model=self['model']

        columns=self._get_columns()

        print("reading columns:",columns)

        # if we get here and use_cache is specified, we are supposed to
        # read the whole thing and write the cache, *then* read a test
        # subset
        if self['do_test'] and not self['use_cache']:
            rows=arange(self['ntest'])
        else:
            rows=None

        data=files.read_collated(self['run'], columns=columns, rows=rows)

        print("cutting flags")
        logic = data['flags']==0

        if self['cuts'] is not None:
            print("cutting: '%s'" % self['cuts'])
            logic = logic & eval(self['cuts'])

        w,=numpy.where( logic )
        print("    keeping %d/%d" % (w.size,data.size))
        data=data[w]

        return data

    def _cache_in_chunks(self):

        model=self['model']

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
                    print("    chunk %d/%d" % (i+1,nchunks))

                    end=beg+chunksize

                    data = hdu[columns][beg:end]

                    logic = (data['flags']==0)
                    if self['cuts'] is not None:
                        #print("cutting: '%s'" % self['cuts'])
                        logic = logic & eval(self['cuts'])

                    w,=numpy.where(logic)

                    print("        %d/%d" % (w.size, data.size))
                    data=data[w]

                    if first:
                        output.write(data)
                        first=False
                    else:
                        output[-1].append(data)

                    beg = beg + chunksize


    def _read_means(self):
        return files.read_fit_file(self['run'], extra='shear-means')

    def _write_means(self):
        fname=files.get_fit_file(self['run'], extra='shear-means')
        eu.ostools.makedirs_fromfile(fname)

        print("writing:",fname)
        fitsio.write(fname, self.means, clobber=True)

    def _write_fits(self, fits):
        fname=files.get_fit_file(self['run'], extra='fit-m-c')

        eu.ostools.makedirs_fromfile(fname)
        print("writing fit data to file:",fname)
        fitsio.write(fname, fits, clobber=True)


class AveragerRmean(Averager):
    def _get_averages_noweight(self, show_progress=False):

        data=self.data

        shears = self.shears

        h,rev = eu.stat.histogram(data['shear_index'], rev=True)
        nind = h.size

        g, gpsf, R, Rpsf = self._get_arrays()

        Rmean = R.mean(axis=0)
        Rpsf_mean = Rpsf.mean(axis=0)
        Rinv = numpy.linalg.inv(Rmean)

        means=get_mean_struct(nind)

        for i in xrange(nind):
            if show_progress:
                print("shear index:",i)

            w=rev[ rev[i]:rev[i+1] ]

            shear_true = shears[i]

            psf_corr  = (gpsf[w]).mean(axis=0)*Rpsf_mean

            gmean     = g[w].mean(axis=0)
            shear     = numpy.dot(Rinv, gmean-psf_corr)
            shear_err = g[w].std(axis=0)/numpy.sqrt(w.size)
            shear_err = numpy.dot(Rinv, shear_err)

            means['shear'][i] = shear
            means['shear_err'][i] = shear_err
            means['shear_true'][i] = shear_true

        return means

    def _get_averages_weighted(self, show_progress=False):

        data=self.data

        shears = self.shears

        h,rev = eu.stat.histogram(data['shear_index'], rev=True)
        nind = h.size

        g, gpsf, R, Rpsf = self._get_arrays()

        means=get_mean_struct(nind)

        model=self['model_pars'].keys()[0]
        wfield = '%s_mcal_s2n_r' % model

        Rmean = zeros( (2,2) )
        psf_corr = zeros( 2 )
        gmean = zeros( 2 )
        gerr  = zeros( 2 )

        s2n = data[wfield]
        wts = self._get_s2n_weights(s2n)
        wsum = wts.sum()

        for ii in xrange(2):
            Rpsf_mean[ii] = (Rpsf[:,ii]*wts).sum()/wsum
            for jj in xrange(ii,2):
                Rmean[ii,jj] = (R[:,ii,jj]*wts).sum()/wsum
                if jj != ii:
                    Rmean[jj,ii] = Rmean[ii,jj]

        Rinv = numpy.linalg.inv(Rmean)

        for i in xrange(nind):
            if show_progress:
                print("shear index:",i)

            w=rev[ rev[i]:rev[i+1] ]

            s2n = data[wfield][w]
            wts = self._get_s2n_weights(s2n)

            wts2=wts**2
            wsum=wts.sum()

            shear_true = shears[i]

            for ii in xrange(2):
                gmean[ii] = (g[w,ii]*wts).sum()/wsum
                gerr2 = ( wts2 * (g[w,ii]-gmean[ii])**2 ).sum()
                gerr[ii] = numpy.sqrt(gerr2)
                psf_corr[ii]  = Rpsf_mean*(gpsf[w,ii]*wts).sum()/wsum

                for jj in xrange(ii,2):
                    Rmean[ii,jj] = (R[w,ii,jj]*wts).sum()/wsum
                    if jj != ii:
                        Rmean[jj,ii] = Rmean[ii,jj]

            shear     = numpy.dot(Rinv, gmean-psf_corr)
            shear_err = numpy.dot(Rinv, gerr)

            means['shear'][i] = shear
            means['shear_err'][i] = shear_err
            means['shear_true'][i] = shear_true

        return means


class AveragerDetrend(AveragerRmean):
    def _get_Rnoise(self, weights=None, show=False):
        import images
        from numpy import newaxis

        data=self.data

        noise0 = self['target_noise']
        print("noise0:",noise0)

        target_noises = array( self['detrend_noises'] )
        ndiff = target_noises - noise0
        xvals = 2*noise0*ndiff

        model=self['model']
        dtR_field = '%s_mcal_dt_Rnoise' % model
        dtR_psf_field = '%s_mcal_dt_Rnoise_psf' % model
        if weights is not None:
            wsum=weights.sum()
            wna1=weights[:,newaxis]
            wna2=weights[:,newaxis,newaxis]

            Rdt = (data[dtR_field]*wna2).sum(axis=0)/wsum

            if dtR_psf_field in data.dtype.names:
                Rdt_psf = (data['mcal_dt_Rnoise_psf']*wna1).sum(axis=0)/wsum
            else:
                print("No Rnoise psf found")
                Rdt_psf = None
        else:
            Rdt = data[dtR_field].mean(axis=0)
            if dtR_psf_field in data.dtype.names:
                Rdt_psf = data['mcal_dt_Rnoise_psf'].mean(axis=0)
            else:
                print("No Rnoise psf found")
                Rdt_psf = None

        plot_psf=True
        if Rdt_psf is None:
            plot_psf=False
            Rdt_psf = 0.0*Rdt[:,:,0]

        A = zeros( (2,2) )
        Apsf = zeros(2)

        p='%s (%.4g +/- %.4g) + (%.4g +/ %.4g) deltan'
        for i in xrange(2):
            res = fitline(xvals, Rdt_psf[:,i])
            Apsf[i] = res['slope']

            if plot_psf:
                plot_line_fit(
                    self['run'],
                    'Rnoise-detrend-Rpsf%d' % (i+1),
                    xvals, Rdt_psf[:,i],res,
                    r'$2 n \Delta n$',
                    r'$\Delta R^{PSF}_%d$' % (i+1),
                    show=show,
                )

            for j in xrange(2):
                res = fitline(xvals, Rdt[:,i,j])
                A[i,j] = res['slope']

                n='A[%d,%d]' % (i+1,j+1)
                s=res['slope']
                serr=res['slope_err']
                o=res['offset']
                oerr=res['offset_err']
                print(p % (n,o,oerr,s,serr))

                plt=plot_line_fit(
                    self['run'],
                    'Rnoise-detrend-R%d%d' % (i+1,j+1),
                    xvals,
                    Rdt[:,i,j],
                    res,
                    r'$2 n \Delta n$',
                    r'$\Delta R_{%d,%d}$' % (i+1,j+1),
                    show=show,
                )

        Rnoise = A*noise0**2
        Rnoise_psf = Apsf*noise0**2

        print("Rnoise")
        images.imprint(Rnoise, fmt='%.4g')
        print("Rnoise_psf")
        print('%.4g %.4g' % tuple(Rnoise_psf))

        return Rnoise, Rnoise_psf

    def _get_arrays(self):

        data=self.data

        print("getting arrays")
        model=self['model']

        gfield = '%s_mcal_g' % model
        gpsf_field = '%s_mcal_gpsf' % model

        Rfield = '%s_mcal_R' % model
        Rpsf_field = '%s_mcal_Rpsf' % model


        g = data[gfield]
        gpsf = data[gpsf_field]
        R = data[Rfield].copy()
        Rpsf = data[Rpsf_field].copy()

        Rnoise, Rnoise_psf = self._get_Rnoise()

        R -= Rnoise
        Rpsf -= Rnoise_psf

        return g, gpsf, R, Rpsf


    def _get_columns(self):

        model=self['model']
        columns=[
            '%s_mcal_g' % model,
            '%s_mcal_gpsf' % model,
            '%s_mcal_R' % model,
            '%s_mcal_Rpsf' % model,
            '%s_mcal_dt_Rnoise' % model,
            '%s_s2n_r' % model,
            '%s_log_T_r' % model,
            'shear_index',
            'flags'
        ]

        fname=files.get_collated_file(self['run'])
        with fitsio.FITS(fname) as fobj:
            colnames=fobj[1].get_colnames()
            Rpcol='%s_mcal_dt_Rnoise_psf' % model
            if Rpcol in colnames:
                columns.append(Rpcol)


        if self['use_weights']:
            columns += ['%s_mcal_s2n_r' % model]

        return columns


class AveragerSimn(Averager):
    def _get_arrays(self):

        data=self.data

        print("getting arrays")
        model=self['model']

        gfield = '%s_mcal_g' % model
        gpsf_field = '%s_mcal_gpsf' % model

        Rfield = '%s_mcal_R' % model
        Rpsf_field = '%s_mcal_Rpsf' % model

        Rnoise_field = '%s_mcal_Rnoise' % model
        Rpsf_noise_field = '%s_mcal_Rpsf_noise' % model

        g = data[gfield]
        gpsf = data[gpsf_field]
        R = data[Rfield].copy()
        Rpsf = data[Rpsf_field].copy()

        R -= data[Rnoise_field]
        Rpsf -= data[Rpsf_noise_field]

        return g, gpsf, R, Rpsf

    def _get_columns(self):
        model=self['model']
        columns=[
            '%s_mcal_g' % model,
            '%s_mcal_gpsf' % model,
            '%s_mcal_R' % model,
            '%s_mcal_Rpsf' % model,
            '%s_mcal_Rnoise' % model,
            '%s_mcal_Rpsf_noise' % model,
            'shear_index',
            'flags'
        ]

        if self['use_weights']:
            columns += ['%s_mcal_s2n_r' % model]

        return columns

class AveragerSimnRmean(AveragerSimn):
    def _get_arrays(self):

        print("subtracting Rnoise_mean")
        data=self.data

        print("getting arrays")
        model=self['model']

        gfield = '%s_mcal_g' % model
        gpsf_field = '%s_mcal_gpsf' % model

        Rfield = '%s_mcal_R' % model
        Rpsf_field = '%s_mcal_Rpsf' % model

        Rnoise_field = '%s_mcal_Rnoise' % model
        Rpsf_noise_field = '%s_mcal_Rpsf_noise' % model

        g = data[gfield]
        gpsf = data[gpsf_field]
        R = data[Rfield].copy()
        Rpsf = data[Rpsf_field].copy()

        R -= data[Rnoise_field].mean(axis=0)
        Rpsf -= data[Rpsf_noise_field].mean(axis=0)

        return g, gpsf, R, Rpsf


class AveragerRef(Averager):
    """
    we added a correlated noise term to cancel that in the
    measurement
    """
    def __init__(self, run, refrun, **kw):
        self['refrun']=refrun
        super(AveragerRef,self).__init__(run, **kw)

        if self['use_weights']:
            raise RuntimeError("weights don't work when using "
                               "a reference set")

    def _load_data(self):

        data= files.read_collated(self['refrun'])
        w,=numpy.where(data['flags']==0)
        print("    keeping %d/%d ref data" % (w.size,data.size))

        data=data[w]

        self.refdata=data

        super(AveragerRef,self)._load_data()

    def _get_arrays(self):

        g, gpsf, R, Rpsf = super(AveragerRef,self)._get_arrays()

        model=self['model']

        Rfield = '%s_mcal_R' % model
        Rpsf_field = '%s_mcal_Rpsf' % model
        refR = self.refdata[Rfield].mean(axis=0)
        refRpsf = self.refdata[Rpsf_field].mean(axis=0)

        for i in xrange(2):
            Rpsf[:,i] = refRpsf[i]
            for j in xrange(2):
                R[:,i,j] = refR[i,j]

        return g, gpsf, R, Rpsf


class AveragerRefFix(AveragerRef):
    """
    we added a correlated noise term to cancel that in the
    measurement
    """

    def _get_arrays(self):

        # note this is calling the parent of the parent
        g, gpsf, R, Rpsf = super(AveragerRef,self)._get_arrays()

        print("    calculating R fix")
        model=self['model']
        Rfield = '%s_mcal_R' % model
        Rpsf_field = '%s_mcal_Rpsf' % model
        refR = self.refdata[Rfield]
        refRpsf = self.refdata[Rpsf_field]

        Rcrud = R.mean(axis=0) - refR.mean(axis=0)
        Rpsf_crud = Rpsf.mean(axis=0) - refRpsf.mean(axis=0)

        R -= Rcrud
        Rpsf -= Rpsf_crud
        return g, gpsf, R, Rpsf



def fit_m_c(data, doprint=True, onem=False):
    import fitting
    import mcmc
    import esutil as eu

    strue = data['shear_true']
    sdiff = data['shear'] - data['shear_true']
    serr  = data['shear_err']

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

        fmt = '  m: %.3e +/- %.3e c1: %.3e +/- %.3e c2: %.3e +/- %.3e'
        print(fmt % (pars[0],perr[0],
                     pars[1],perr[1],
                     pars[2],perr[2]))
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
def plot_line_fit(run, extra, x, y, res, xlabel, ylabel, show=False):
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

    alab=r'$slope = %.3g \pm %.3g' % (res['slope'],res['slope_err'])
    blab=r'$offset = %.3g \pm %.3g' % (res['offset'],res['offset_err'])
    alabel=biggles.PlotLabel(0.9, 0.9, alab, halign='right')
    blabel=biggles.PlotLabel(0.9, 0.85, blab, halign='right')

    plt.add(c, pts, alabel, blabel)

    plotfile=files.get_plot_file(run, extra)

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
    cache_file=os.path.basename(cache_file)

    cache_file=os.path.join(tdir, cache_file)
    return cache_file

def get_mean_struct(n):
    dt=[('shear','f8',2),
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


