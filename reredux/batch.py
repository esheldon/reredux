from __future__ import print_function
import os
import numpy
from . import files

def get_splits(ntot, nper):
    """
    get split ranges, where the range is inclusive, not
    exclusive like a python slice
    """
    beglist = numpy.arange(0,ntot,nper)
    endlist = numpy.arange(0,ntot,nper) + nper - 1

    if (ntot % nper) != 0:
        endlist[-1] = ntot-1
    return beglist, endlist

class BatchMaker(dict):
    def __init__(self, run, missing=False, **kw):
        """
        send extra things for template through kw
        """

        self.update(kw)

        self['run'] = run
        self['missing'] = missing

        self._load_configs()

        self['deep'] = self.reredux_conf['deep']

        self._set_walltime()
        self._setup_splits()
        self._make_dirs()

    def write_scripts(self):
        """
        Write all scripts
        """

        nfiles = self.reredux_conf['nfiles']
        for fnum in xrange(nfiles):
            for beg,end in zip(beglist, endlist):

                self.write_script(fnum, beg, end)

        print("scripts are in:",self.get_batch_dir())
    def write_script(self, fnum, beg, end):
        """
        write the script for a single split
        """
        fname = self.get_batch_file(fnum, beg, end, missing=self['missing'])

        self['job_name'] = self.get_job_name(fnum, beg, end)

        vers=self.reredux_conf['version']
        self['meds_file'] = files.get_meds_file(vers, fnum, deep=self['deep'])
        self['psf_file'] = files.get_psf_file(vers, fnum, deep=self['deep'])

        self['config_file'] = files.get_config_file(self['run'])
        self['output_file'] = files.get_output_file(self['run'], fnum, beg, end)
        self['log_file'] = files.get_log_file(self['run'], fnum, beg, end)
        self['beg'] = beg
        self['end'] = end

        if self['missing'] and os.path.exists(self['output_file']):
            if os.path.exists(fname):
                os.remove(fname)
            return

        print("writing:",fname)
        with open(fname,'w') as fobj:
            text = self.get_text()
            fobj.write(text)

    def get_job_name(self, fnum, beg, end):
        return '%s-%03d-%06d-%06d' % (self['run'], fnum, beg, end)

    def get_batch_file(self, run, fnum, beg, end, missing=False):
        raise NotImplementedError("implement in child class")


    def _load_configs(self):
        self.runconf = files.read_config(self['run'])
        self.reredux_conf = files.read_config(self.runconf['reredux_config'])

    def _set_walltime(self):
        time_per_seconds = self.runconf['time_per']
        time_seconds = time_per_seconds*self.runconf['nper']
        time_hours = time_seconds / 3600.0

        hours_round = int(round(time_hours))
        if hours_round < 1:
            hours_round=1
        if time_hours < 10:
            tstr = '0%d:00' % hours_round
        else:
            tstr = '%d:00' % hours_round

        self['walltime'] = tstr
        print("nper:",self.runconf['nper'],
              "time per:",self.runconf['time_per'],
              "hours:",time_hours,
              "walltime:",self['walltime'])


    def _setup_splits(self):
        """
        we expect the same splits for each file
        """
        ntot = self.reredux_conf['nperfile']
        npersplit = self.runconf['nper']

        self.beglist, self.endlist = get_splits(ntot, npersplit)

    def _make_dirs(self):
        dir = files.get_output_dir(self['run'])
        if not os.path.exists(dir):
            print("making dir:",dir)
            os.makedirs(dir)

        dir = self.get_batch_dir()
        if not os.path.exists(dir):
            print("making dir:",dir)
            os.makedirs(dir)



class LSFMaker(BatchMaker):
    def get_batch_dir(self):
        return files.get_lsf_dir(self['run'])

    def get_batch_file(self, fnum, beg, end, missing=False):
        return files.get_lsf_file(self['run'], fnum, beg, end, missing=missing)

    def get_text(self):
        return _lsf_template % self




_lsf_template="""#!/bin/bash
#BSUB -J %(job_name)s
#BSUB -n 1
#BSUB -oo ./%(job_name)s.oe
#BSUB -W %(walltime)s
#BSUB -R "linux64 && rhel60 && scratch > 2"

echo "working on host: $(hostname)"

config_file=%(config_file)s
psf_file=%(psf_file)s
meds_file=%(meds_file)s

output_file=%(output_file)s
log_file=%(log_file)s

beg=%(beg)s
end=%(end)s

# we need to make the scratch directory
tmpdir="/scratch/esheldon/${LSB_JOBID}"
mkdir -p $tmpdir
cd $tmpdir

tmplog=$(basename $log_file)

python -u $(which ngmixit)         \\
        --psf-file=${psf_file}     \\
        --fof-range $beg,$end      \\
        ${config_file}             \\
        ${output_file}             \\
        ${meds_file} &> ${tmplog}

status=$?

echo "moving log file ${tmplog} -> ${log_file}" >> ${tmplog}

# errors go to the jobs stderr
mv -fv "${tmplog}" "${log_file}" 1>&2
status2=$?

if [[ $status2 != "0" ]]; then
    # this error message will go to main error file
    echo "error ${status2} moving log to: ${log_file}" 1>&2

    status=$status2
fi

cd $HOME

echo "removing temporary directory"
rm -rfv ${tmpdir}

exit $status
"""

_batch_systems=['lsf']


