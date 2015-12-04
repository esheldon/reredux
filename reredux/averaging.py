from __future__ import print_function

import os
import numpy

import fitsio
import esutil as eu

from . import files

S2N_SOFT=10.0

class AveragerBase(dict):
    def __init__(self, run, fit_only=False, use_weights=False, use_cache=False):
        self['run'] = run
        self['use_weights'] = use_weights
        self['use_cache'] = use_cache
        self['fit_only'] = fit_only

        if self['use_weights']:
            print("Using weights")

        conf=files.read_config(run)
        sconf=files.read_egret_config(conf['reredux_config'])

        self.update(conf)
        self['sconf'] = sconf

        self['model'] = self['model_pars'].keys()[0]
        self.shears = self['sconf']['shear']['shears']

        self._load_data()

    def go(self):
        if self['fit_only']:
            self.means = self._read_means()
        else:
            self.means= self._get_averages()
            self._write_means()

        self.fits = fit_m_c(self.means)
        self.Q=calc_q(self.fits)
        print("  Q: %d" % self.Q)

        self.fits_onem = fit_m_c(self.means, onem=True)
        self.Q_onem=calc_q(self.fits_onem)

        self._write_fits(self.fits)

    def doplot(self,args):
        import biggles

        means=self.means
        fits=self.fits

        Q=calc_q(fits)

        if args.yrange is not None:
            yrange=[float(r) for r in args.yrange.split(',')]
        else:
            yrange=[-0.01,0.01]

        xrng=args.xrange
        if xrng is not None:
            xrng=[float(r) for r in args.xrange.split(',')]

        tab=biggles.Table(1,2)
        tab.aspect_ratio=0.5

        diff = means['shear'] - means['shear_true']

        plts=[]
        for i in [0,1]:

            x = means['shear_true'][:,i]
            plt =biggles.plot(
                x,
                diff[:,i],
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

    def _get_averages(self, show_progress=True):
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
        tfile=get_cache_file(self['run'])
        if os.path.exists(tfile):
            print("reading cache:",tfile)
            data=fitsio.read(tfile)#, rows=numpy.arange(1000000))
        else:
            data=self._read_data()
            print("writing cache:",tfile)
            fitsio.write(tfile, data, clobber=True)

        return data


    def _read_data(self):

        model=self['model']

        columns=self._get_columns()

        print("reading columns:",columns)
        data=files.read_collated(self['run'], columns=columns)

        w,=numpy.where(data['flags']==0)
        data=data[w]

        return data

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


class AveragerAddn(AveragerBase):
    """
    we added a correlated noise term to cancel that in the
    measurement
    """
    def _get_averages_noweight(self, show_progress=True):

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

    def _get_averages_weighted(self, show_progress=True):

        data=self.data

        shears = self.shears

        h,rev = eu.stat.histogram(data['shear_index'], rev=True)
        nind = h.size

        g, gpsf, R, Rpsf = self._get_arrays()

        means=get_mean_struct(nind)

        model=self['model_pars'].keys()[0]
        wfield = '%s_mcal_s2n_r' % model

        Rmean = numpy.zeros( (2,2) )
        psf_corr = numpy.zeros( 2 )
        gmean = numpy.zeros( 2 )
        gerr  = numpy.zeros( 2 )

        for i in xrange(nind):
            if show_progress:
                print("shear index:",i)

            w=rev[ rev[i]:rev[i+1] ]

            s2n = data[wfield][w]
            wts = 1.0/(1.0 + (S2N_SOFT/s2n)**2 )

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


class AveragerSimn(AveragerAddn):
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

        c, c_err, m, m_err, covar = fitline(strue[:,i], sdiff[:,i])
        r = covar/numpy.sqrt(m_err**2 * c_err**2)
        fits['m'][0,i] = m
        fits['merr'][0,i] = m_err
        fits['c'][0,i] = c
        fits['cerr'][0,i] = c_err
        fits['r'][0,i] = r

        if doprint:
            print_m_c(i+1, m,m_err,c,c_err, r=r)

    return fits

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
    return a, a_err, b, b_err, cov_ab

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
        c1, c1_err, m1, m1_err, covar = fitline(self.shear_true[:,0],
                                                self.shear_diff[:,0])
        c2, c2_err, m2, m2_err, covar = fitline(self.shear_true[:,1],
                                                self.shear_diff[:,1])

        m = 0.5*(m1 + m2)

        self.guess = numpy.array([m, c1, c2])

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
    tfile= files.get_collated_file(run)
    tfile=os.path.basename(tfile)

    tfile=os.path.join(tdir, tfile)
    return tfile

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


