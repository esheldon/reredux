import os
import glob
from distutils.core import setup

scripts=['reredux-make-batch',
         'reredux-collate',
         'reredux-average',
         'reredux-vary-deepsize',
         'reredux-fit-prior',
         'reredux-fit-m-c']

scripts=[os.path.join('bin',s) for s in scripts]

conf_files=glob.glob('config/*.yaml')

data_files=[]
for f in conf_files:
    data_files.append( ('share/reredux_config',[f]) )


setup(name="reredux", 
      version="0.1.0",
      description="Run code on great3reredux simulations",
      license = "GPL",
      author="Erin Scott Sheldon",
      author_email="erin.sheldon@gmail.com",
      scripts=scripts,
      data_files=data_files,
      packages=['reredux'])
