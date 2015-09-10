from __future__ import print_function
import os


def get_reredux_basedir():
    """
    base directory for great3reredux images
    """
    return os.environ['REREDUX_DATA_DIR']

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


def get_output_basedir():
    """
    my outputs
    """
    return os.environ['REREDUX_OUTPUTS_DIR']


#
# output files
#

def get_output_dir(run):
    """
    output files
    """

    basedir = get_output_basedir()
    return os.path.join(basedir, 'outputs')

def get_output_file(run, fnum, beg, end):
    """
    location of output file
    """

    dir=get_output_dir(run)
    fname = '%s-%03d-%06d-%06d.fits' % (run, fnum, beg, end)
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

    basedir = get_output_basedir()
    return os.path.join(basedir, 'collated')

def get_collated_file(run):
    """
    location of output file
    """

    dir=get_collated_dir(run)
    fname = '%s.fits' % run
    return os.path.join(dir, fname)

#
# shear averaged files
#

def get_averaged_dir(run):
    """
    output files
    """

    basedir = get_output_basedir()
    return os.path.join(basedir, 'averaged')

def get_averaged_file(run, beg, end):
    """
    location of output file
    """

    dir=get_averaged_dir(run)
    fname = '%s-avg.fits' % run
    return os.path.join(dir, fname)

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
# lsf batch system
#

def get_lsf_dir(run):
    """
    output files
    """

    basedir = get_output_basedir()
    return os.path.join(basedir, 'lsf')


def get_lsf_file(run, fnum, beg, end, missing=False):
    """
    location of output file
    """

    dir=get_lsf_dir(run)
    fname = '%s-%03d-%06d-%06d' % (run, fnum, beg, end)

    if missing:
        fname = '%s-missing' % fname

    fname = '%s.lsf' % fname
    return os.path.join(dir, fname)


