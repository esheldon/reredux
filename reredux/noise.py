from __future__ import print_function
import numpy
import meds
import pprint

from . import files

def calc_stats(version):
    n = Noise(version)
    n.go()

    stats=n.get_stats()

    pprint.pprint(stats)

    return stats

class Noise(object):
    """

    calculate the noise for shallow and deep fields

    currently the noise is different for each object,
    and varies within the stamp!

    """
    def __init__(self, version):
        self.version=version

        self.conf=files.read_config(self.version)

    def go(self):
        self.calculate_noise()
        self.calc_stats()

    def get_stats(self):
        return self.stats

    def calculate_noise(self):
        conf=self.conf

        nfiles = conf['nfiles']
        nperfile = conf['nperfile']
        nobj = nfiles*nperfile

        noises = numpy.zeros(nobj)

        itot=0
        for fnum in xrange(nfiles):
            fname = files.get_meds_file(self.conf['version'],
                                        fnum,
                                        deep=conf['deep'])
            print("meds file:",fname)

            with meds.MEDS(fname) as m:
                for i in xrange(m.size):

                    wt = m.get_cutout(i,0,type='weight')

                    noise = numpy.sqrt( (1.0/wt).mean() )

                    noises[itot] = noise

                    itot += 1

        self.noises = noises

    def calc_stats(self):
        import esutil as eu


        stats = eu.stat.get_stats(self.noises)
        stats['deep'] = self.conf['deep']

        self.stats=stats
        
