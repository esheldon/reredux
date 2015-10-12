great3reredux v3
----------------
v3.yaml
v3deep.yaml
    - had bug (was sheared).  Matt is rerunning.
    I will want to redo priors

- max-v3d01
    - max like run to get priors
- mcal-v3d01
    - deep data
- mcal-v3s01
    - shallow data

    s1: 0.07996 +/- 0.00019
    s2: 0.00045 +/- 0.00017
    delta g/g = -0.0005 +/- 0.0024

# runs with gaussians
- max-g-v3d01
    - max like run to get priors
- mcal-g-v3d01
    - deep data
- mcal-g-v3s01
    - shallow data
    s1: 0.0800 +/- 0.0002   s2: 0.0004 +/- 0.0002
    delga t/g = -0.00059 +/- 0.00232

v04 in my directory
-------------------
can use priors from v3 for now

- max-v04d01
    - for prior
- mcal-v04d01
    running
- mcal-v04s01
    - shallow data

v05 larger sim, 20,000,000
--------------------------
- using deep run mcal-v04d01
    s1: 0.08007 +/- 0.00013   s2: -0.00006 +/- 0.00010
    delta g/g 0.0008 +/- 0.0017

    errors larger than I anticipated, may have to do with proper
    bootstrapping of the deep data. If I don't bootstrap the
    deep data as well, the frac error is 0.0007

v05deep larger deep sim, 1,000,000
- max-v05d01
    - for priors
- mcal-v05d01
    - priors from max-v5d01
    - 10,000 jobs each over an hour
- mcal-v05s02
    - priors from max-v5d01
    - responses from mcal-v05d01 

    s1: 0.07996 +/- 0.00007   s2: -0.00017 +/- 0.00007
    delta g/g = -0.0005 +/- 0.0008
    7.99257069e-02

    1000 bootstrap, but not sure that makes sense; it might
    since we are also bootstrapping the shallow.  Need to
    think about this more

    s1: 0.07996 +/- 0.00007   s2: -0.00017 +/- 0.00006
    frac1: -0.00048 +/- 0.00087

    - looking at variance just from the deep set, as a function of deep data
      set size (each out of 1,000,000 set, so the 100,000 one only has 10
      examples)

      10000 s1: 0.079966 +/- 0.000037   s2: -0.000163 +/- 0.000023 frac1: -0.00042 +/- 0.00046

    # variances are from deep data alone
    # 100 bootstraps
    10000 s1: 0.080017 +/- 0.000042   s2: -0.000130 +/- 0.000023 frac1: 0.00022 +/- 0.00053
    20000 s1: 0.079967 +/- 0.000030   s2: -0.000157 +/- 0.000019 frac1: -0.00041 +/- 0.00038
    30000 s1: 0.079956 +/- 0.000025   s2: -0.000166 +/- 0.000014 frac1: -0.00056 +/- 0.00031
    40000 s1: 0.079936 +/- 0.000022   s2: -0.000158 +/- 0.000013 frac1: -0.0008 +/- 0.00027
    50000 s1: 0.079974 +/- 0.000020   s2: -0.000157 +/- 0.000011 frac1: -0.00033 +/- 0.00025
    60000 s1: 0.079957 +/- 0.000016   s2: -0.000184 +/- 0.000011 frac1: -0.00054 +/- 0.0002
    70000 s1: 0.079982 +/- 0.000015   s2: -0.000169 +/- 0.000010 frac1: -0.00023 +/- 0.00019
    80000 s1: 0.079958 +/- 0.000015   s2: -0.000170 +/- 0.000009 frac1: -0.00053 +/- 0.00018
    90000 s1: 0.079966 +/- 0.000013   s2: -0.000167 +/- 0.000008 frac1: -0.00043 +/- 0.00016
    100000 s1: 0.079952 +/- 0.000012   s2: -0.000169 +/- 0.000008 frac1: -0.0006 +/- 0.00015

    # 1000 bootstraps
    10000 s1: 0.079958 +/- 0.000014   s2: -0.000160 +/- 0.000008 frac1: -0.00052 +/- 0.00017
    20000 s1: 0.079977 +/- 0.000009   s2: -0.000161 +/- 0.000006 frac1: -0.00029 +/- 0.00012
    30000 s1: 0.079965 +/- 0.000007   s2: -0.000168 +/- 0.000005 frac1: -0.00043 +/- 9.3e-05
    40000 s1: 0.079963 +/- 0.000007   s2: -0.000165 +/- 0.000004 frac1: -0.00046 +/- 8.8e-05
    50000 s1: 0.079958 +/- 0.000006   s2: -0.000165 +/- 0.000004 frac1: -0.00052 +/- 7.3e-05
    60000 s1: 0.079953 +/- 0.000005   s2: -0.000168 +/- 0.000003 frac1: -0.00058 +/- 6.5e-05
    70000 s1: 0.079965 +/- 0.000005   s2: -0.000162 +/- 0.000003 frac1: -0.00044 +/- 6.2e-05
    80000 s1: 0.079966 +/- 0.000005   s2: -0.000162 +/- 0.000003 frac1: -0.00042 +/- 5.8e-05
    90000 s1: 0.079964 +/- 0.000004   s2: -0.000165 +/- 0.000003 frac1: -0.00045 +/- 5.6e-05
    100000 s1: 0.079969 +/- 0.000004   s2: -0.000166 +/- 0.000003 frac1: -0.00039 +/- 5.2e-05


- mcal-v05bs01
    s1: 0.07988 +/- 0.00005   s2: -0.00004 +/- 0.00004
    frac1: -0.0016 +/- 0.00063
    7.98399143e-02

    wtf? Maybe the deep sim was different? I don't think so,
    at least egret itself is unchanged

    - should always use a new deep field as well, the above are correlated.
    Might make it worse but there it is.

    - using deep field mcal-v06d
        7.98399143e-02

    contribution from deep fields to error is about 
    0.000037 in shear
    0.00046 in fracdiff


- mcal-v06s01
    - shear -0.08 
    - deep mcal-v05d01
    s1: -0.07987 +/- 0.00008   s2: -0.00001 +/- 0.00007
    frac1: -0.0016 +/- -0.00094


- mcal-v07s01
    - shear 0.04
    s1: 0.03995 +/- 0.00006   s2: -0.00010 +/- 0.00006
    frac1: -0.0012 +/- 0.0014

- mcal-v08s01
    - shear -0.04
    - deep mcal-v05d01
    s1: -0.03996 +/- 0.00006   s2: 0.00010 +/- 0.00006
    frac1: -0.0011 +/- -0.0014

- full m-c
    - deep mcal-v05d01
    m: -1.061e-03 +/- 5.401e-04 c: 1.632e-05 +/- 3.187e-05

    assuming etac is 1.12
    Q = 
        1.12*2000.0/sqrt(1.0**2 + 2*( (-1.061e-03/2.0e-3)**2 +
                                      (1.632e-05/2.0e-4)**2 ) )
       = 1784


- mcal-v05s04, v06s04, v07s04, v08s04
    - fix using abs(g)
    - fix not interpolating twice

    - mcal-v05d02
        - deep run including above fixes

        s1
            m: -3.8e-04 +/- 4.8e-04
            c:  3.0e-05 +/- 3.1e-05
        s2
           mean: -4.232e-05 +/- 2.979e-05

    - using mcal-v05d-hinoise01 for reference set.  This has
      same noise as shallow fields but zero shear
      
        s1 
          m: -4.6e-04 +/- 5.3e-04
          c:  2.8e-05 +/- 3.3e-05
        s2
             -3.2e-05 +/- 2.9e-05



great3reredux v1
-----------------

- max-deep01
    - v1 sims
    - max like run on deep data, no extra noise added
    - flat g prior  

- max-deep02
    - v1 sims
    - added g prior, but no better, will use deep01

- mcal-01
    - v1 sims
    - flux, flat prior spanning what we see in deep01
    - T, gmixnd with 1.1 cov factor

- mcal-degrade01
    - v2 sims
    - flux, flat prior spanning what we see in deep01
    - T, gmixnd with 1.1 cov factor
        - was not refinding center in the noisy image
        - also bug, wrong target_noise


- mcal-degrade02
    - v2 sims
    - flux, flat prior spanning what we see in deep01
    - T, gmixnd with 1.1 cov factor
    - this version finding cen for the noisy images, 
      and proper mean noise from shallow of 0.0877

- mcal-02
    - v2 sims
    - flux, flat prior spanning what we see in deep01
    - T, gmixnd with 1.1 cov factor
        - using mcal-degrade02
        s1: 0.0806 +/- 0.0008   s2: 0.0011 +/- 0.0009
        fracerr: 0.00706 +/- 0.00984

- mcal-03
    - v2 sims
    - same as mcal-02 but just testing with new code
    - flux, flat prior spanning what we see in deep01
    - T, gmixnd with 1.1 cov factor
        - exactlyl the same, good

