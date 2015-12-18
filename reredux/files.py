from __future__ import print_function
import os


def get_reredux_basedir():
    """
    base directory for great3reredux images
    """
    return os.environ['REREDUX_DATA_DIR']

def get_data_dir(version):
    """
    directory holding images
    """
    basedir = get_reredux_basedir()

    #subdirs='work/data'
    #return os.path.join(basedir, version, subdirs)
    return os.path.join(basedir, version)

def get_meds_file(version, fnum):
    """
    the meds file
    """
    d = get_data_dir(version)

    fname = '%s_meds%06d.fits.fz' % (version,fnum)

    return os.path.join(d, fname)

def get_psf_file(version, fnum):
    """
    the psf file
    """
    d = get_data_dir(version)

    fname = '%s_psf%06d.fits' % (version,fnum)

    return os.path.join(d, fname)

'''
def get_data_dir(version, deep=False):
    """
    directory holding images
    """
    basedir = get_reredux_basedir()

    if deep:
        subdirs='work_deep/data'
    else:
        subdirs='work/data'

    return os.path.join(basedir, version, subdirs)


def get_meds_file(version, fnum, deep=False):
    """
    the meds file
    """
    d = get_data_dir(version, deep=deep)

    fname = 'meds%03d.fits.fz' % fnum

    return os.path.join(d, fname)

def get_psf_file(version, fnum, deep=False):
    """
    the psf file
    """
    d = get_data_dir(version, deep=deep)

    fname = 'psfs%03d.fits' % fnum

    return os.path.join(d, fname)
'''

#
# run base dir
#

def get_rundir(run):
    """
    my outputs
    """
    d=os.environ['REREDUX_OUTPUTS_DIR']
    return os.path.join(d, run)


#
# output files
#

def get_output_dir(run):
    """
    output files
    """

    basedir = get_rundir(run)
    return os.path.join(basedir, 'outputs')

def get_output_file(run, fnum, beg, end):
    """
    location of output file
    """

    dir=get_output_dir(run)
    fname = '%s-%06d-%06d-%06d.fits' % (run, fnum, beg, end)
    return os.path.join(dir, fname)

def get_log_file(run, fnum, beg, end):
    """
    location of output file
    """

    url = get_output_file(run, fnum, beg, end)
    return url.replace('.fits', '.log')

#
# collated files
#

def get_collated_dir(run):
    """
    output files
    """

    basedir = get_rundir(run)
    return os.path.join(basedir, 'collated')

def get_collated_file(run):
    """
    location of output file
    """

    dir=get_collated_dir(run)
    fname = '%s.fits' % run
    return os.path.join(dir, fname)

def read_collated(run, **kw):
    """
    read in the entire collated fits file
    """
    import fitsio

    fname=get_collated_file(run)
    print("reading:",fname)
    return fitsio.read(fname, **kw)

#
# files holding fit data
#

def get_fit_dir(run):
    """
    files holding the fits
    """

    basedir = get_rundir(run)
    return os.path.join(basedir, 'bias-fits')

def get_fit_file(run, extra=None, ext='fits'):
    """
    files holding fits for bias
    """

    dir=get_fit_dir(run)
    fname = '%s' % run
    if extra is not None:
        fname = '%s-%s' % (fname, extra)

    fname = '%s.%s' % (fname, ext)
    return os.path.join(dir, fname)

def read_fit_file(run, extra=None):
    """
    files holding fits for bias
    """
    import fitsio

    dir=get_fit_dir(run)
    fname = '%s' % run
    if extra is not None:
        fname = '%s-%s.fits' % (fname, extra)

    fname=os.path.join(dir, fname)
    print("reading:",fname)
    return fitsio.read(fname)


#
# priors
#

def get_fitprior_dir(run):
    """
    output files
    """

    basedir = get_rundir(run)
    return os.path.join(basedir, 'fitprior')

def get_fitprior_file(run, name, ext='fits'):
    """
    location of output file
    """

    dir=get_fitprior_dir(run)

    fname='%s-fitprior-%s.%s' % (run, name, ext)

    return os.path.join(dir, fname)


#
# shear averaged files
#

def get_averaged_dir(run):
    """
    output files
    """

    basedir = get_rundir(run)
    return os.path.join(basedir, 'averaged')

def get_averaged_file(run, corr_run):
    """
    location of output file
    """

    dir=get_averaged_dir(run)
    fname = '%s-%s-avg.fits' % (run, corr_run)
    return os.path.join(dir, fname)

def read_averaged(run, corr_run, **kw):
    """
    read in the entire collated fits file
    """
    import fitsio

    fname=get_averaged_file(run, corr_run)
    print("reading:",fname)
    return fitsio.read(fname, **kw)


#
# config files
# this works for both run and reredux version configs
#

def get_config_file(name):
    """
    the yaml config file

    name could represent a run or reredux version
    """
    d=os.environ['REREDUX_DIR']

    d = os.path.join(d, 'share/reredux_config')

    fname = '%s.yaml' % name
    return os.path.join(d, fname)

def read_config(name):
    """
    read a yaml config file

    name could represent a run or reredux version
    """
    import yaml

    fname=get_config_file(name)
    print("reading config:",fname)
    with open(fname) as fobj:
        data = yaml.load(fobj)
    
    return data

#
# egretsims config files
#

def get_egret_config_dir():
    """
    location of config files for egret sims
    """
    return os.environ['EGRET_CONFIG_DIR']

def get_egret_config_file(version):
    """
    location of config file for egret sim
    """
    d=get_egret_config_dir()
    name='%s.yaml' % version
    return os.path.join(d, name)

def read_egret_config(version):
    """
    location of config file for egret sim
    """
    import yaml
    fname=get_egret_config_file(version)

    print("reading config:",fname)
    with open(fname) as fobj:
        data = yaml.load(fobj)
    
    return data


#
# lsf batch system
#

def get_lsf_dir(run):
    """
    output files
    """

    basedir = get_rundir(run)
    return os.path.join(basedir, 'lsf')


def get_lsf_file(run, fnum, beg, end, missing=False):
    """
    location of output file
    """

    dir=get_lsf_dir(run)
    fname = '%s-%06d-%06d-%06d' % (run, fnum, beg, end)

    if missing:
        fname = '%s-missing' % fname

    fname = '%s.lsf' % fname
    return os.path.join(dir, fname)

#
# wq batch system
#

def get_wq_dir(run):
    """
    output files
    """

    basedir = get_rundir(run)
    return os.path.join(basedir, 'wq')


def get_wq_file(run, fnum, beg, end, missing=False):
    """
    location of output file
    """

    dir=get_wq_dir(run)
    fname = '%s-%06d-%06d-%06d' % (run, fnum, beg, end)

    if missing:
        fname = '%s-missing' % fname

    fname = '%s.yaml' % fname
    return os.path.join(dir, fname)


#
# slr batch system
#

def get_slr_dir(run):
    """
    output files
    """

    basedir = get_rundir(run)
    return os.path.join(basedir, 'slr')


def get_slr_file(run, fnum, beg, end, missing=False):
    """
    location of output file
    """

    dir=get_slr_dir(run)
    fname = '%s-%06d-%06d-%06d' % (run, fnum, beg, end)

    if missing:
        fname = '%s-missing' % fname

    fname = '%s.slr' % fname
    return os.path.join(dir, fname)


