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
    - for priors on exp
- max-v05d02
    - for priors on gauss



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

- mcal-v10s01
    - 300 different shears, exp

- mcal-v10s02
    - regauss

- mcal-v10s03
    - using v05d04 for deep
    - gaussian fit
    - looks consistent with zero within large noise

- mcal-v10s04
    - new Rnoise measurement
    - works within noise!

- mcal-v10s05
    - Rnoise doing model fit
    - using mean psf corr
      s1 m:  7.399e-03 +/- 8.174e-03 c: -3.221e-04 +/- 1.773e-04
      s2 m: -1.062e-02 +/- 7.706e-03 c:  3.800e-05 +/- 1.822e-04
    - using each psf corr
      s1 m:  7.805e-03 +/- 8.163e-03 c: -3.277e-04 +/- 1.771e-04
      s2 m: -1.068e-02 +/- 7.694e-03 c:  3.161e-05 +/- 1.819e-04

- mcal-v10s06,07
    - alt. method adding a bit of noise to the original image
        - ran 100 as first test

- mcal-v10s08
    - Rnoise doing model fit
    - extra noise added to get to s/n ~ 10
      s1 m: 2.430e-02 +/- 1.037e-02 c: -2.507e-04 +/- 2.246e-04
      s2 m: 5.865e-03 +/- 1.052e-02 c: -1.770e-04 +/- 2.481e-04

- mcal-v10s09
    - same but used "extra_sim_noise" feature and
    added different noise
      s1 m: 1.312e-02 +/- 1.105e-02 c: -2.757e-04 +/- 2.396e-04
      s2 m: -1.052e-02 +/- 9.761e-03 c: -2.445e-04 +/- 2.308e-04

- mcal-v10s10
    - used "extra_sim_noise" feature and added different noise


- mcal-v10s11
    - subtracting sheared correlated noise
    m1: 1.154e-02 +/- 9.207e-03 c1: -2.301e-04 +/- 2.000e-04  r1: 0.0404
    m2: -5.863e-03 +/- 8.401e-03 c2: -1.384e-04 +/- 1.986e-04  r2: -0.0438

- mcal-v10s12
    - subtracting sheared correlated noise
    - using 4 realizations

    - had messed up priors from gauss run
      m1: 1.127e-01 +/- 1.009e-02 c1: -3.041e-04 +/- 2.187e-04
      m2: 9.170e-02 +/- 9.300e-03 c2: -4.193e-05 +/- 2.197e-04
      Q: 33.8929

- mcal-v10s13
    - back to 1 realization
      m1: 1.237e-02 +/- 9.138e-03 c1: -2.205e-04 +/- 1.978e-04
      m2: 5.104e-03 +/- 8.688e-03 c2: -5.022e-05 +/- 2.051e-04
- mcal-v10s14
    - back to 1 realization
      m1: 7.372e-03 +/- 9.159e-03 c1: -3.316e-04 +/- 1.984e-04
      m2: -6.852e-03 +/- 8.777e-03 c2: -1.532e-04 +/- 2.071e-04
- mcal-v10s15
    - back to 1 realization
      m1: 2.995e-03 +/- 9.192e-03 c1: -4.682e-04 +/- 1.990e-04
      m2: -2.810e-03 +/- 8.404e-03 c2: 5.644e-05 +/- 1.985e-04





- mcal-v10s01d
    - redo fit to shallow but with a little extra noise added
    - average 160
    - see if distribution of R tightens.
        - It looks essentially the same

        - the "noise" in R is not because the fitter just ends up at some very
          noisy spot for the plus and minus shears. There is such noise as
          well, but it does not cause this tail to low R values.  I think this
          is consistent with my view that this is due to the properties of the
          +/- sheared images actually being different.  It may be that the
          only way to eliminate this is to add much more noise than was in the
          original image, which isn't very useful for s/n=10 galaxies.

        - another view is that you can only tighten this if the noise is not
          very dominant compared to the signal in the image in the first
          place.  Because otherwise the noise becomes a significant part
          of the structure of the reconvolved images.

          Why whitening or isotropizing didn't help is not clear.


- mcal-v11s01
    - 300 different shears.  Using v05d02 for ref run

    No bootstrapping

      s1 m: -1.326e-03 +/- 7.629e-04 c: 5.421e-05 +/- 2.672e-05
      s2 m: -5.576e-04 +/- 7.328e-04 c: -2.195e-05 +/- 2.643e-05
      m: -9.311e-04 +/- 5.284e-04

    With bootstrapping


      s1 m: -1.326e-03 +/- 9.404e-04 c: 5.421e-05 +/- 2.402e-05
      s2 m: -5.576e-04 +/- 9.009e-04 c: -2.195e-05 +/- 2.503e-05

      mean m from 1 and 2 gives (0.9 +/- 0.9) x 10^{-3} 
      but not sure what the error should be, probably fairly correlated

- going to do a run with twice the noise
- mcal-v05d03
    - deep run with target_noise: 0.1754
- mcal-v11s02
    - also target noies 0.1754

        s1 m: -2.284e-03 +/- 9.775e-04 c: 7.765e-05 +/- 3.420e-05
        s2 m: -2.644e-03 +/- 9.818e-04 c: 2.936e-06 +/- 3.536e-05

        mean m: (-2.5 +/- 0.9) x 10^{-3}

- mcal-v11s03
    - using v05d04 for deep
    - gaussian fit

        s1 m: -9.794e-04 +/- 7.677e-04 c: 5.442e-05 +/- 2.689e-05
        s2 m: -5.936e-04 +/- 7.408e-04 c: -2.601e-05 +/- 2.672e-05
        Q: 1172.15

- mcal-v11s04
    - sim noise Rnoise correction, no deep

        m1: -1.649e-03 +/- 7.699e-04 c1: 6.236e-05 +/- 2.700e-05  r1: -0.0627
        m2: -2.052e-04 +/- 7.359e-04 c2: 6.787e-06 +/- 2.658e-05  r2: -0.08
        Q: 1126.02
        mean m ~ (-0.9 +/- 0.7) x 10^{-3}

- mcal-v11s05
    - sim noise Rnoise correction, no deep
    - extra noise to get to min s/n ~ 10

    - no weights
      m1: -3.801e-03 +/- 9.794e-04 c1: 8.434e-05 +/- 3.434e-05  r1: -0.0627
      m2: -2.030e-03 +/- 9.444e-04 c2: 1.663e-05 +/- 3.410e-05  r2: -0.08
      m: -2.917e-03 +/- 6.847e-04

    - s2n_r weights
      m1: -5.620e-03 +/- 9.698e-04 c1: 8.771e-05 +/- 3.401e-05  r1: -0.0627
      m2: -4.330e-03 +/- 9.197e-04 c2: 5.391e-05 +/- 3.321e-05  r2: -0.08
      m: -4.978e-03 +/- 6.716e-04

    - using mean Rnoise
      m1: -3.853e-03 +/- 9.606e-04 c1: 8.555e-05 +/- 3.369e-05  r1: -0.0627
      m2: -2.200e-03 +/- 9.405e-04 c2: 1.728e-05 +/- 3.396e-05  r2: -0.08
      m: -3.026e-03 +/- 6.768e-04

    - using overall R mean
      m1: -6.958e-03 +/- 9.567e-04 c1: 7.040e-05 +/- 3.355e-05  r1: -0.0627
      m2: -5.369e-03 +/- 9.192e-04 c2: 1.421e-05 +/- 3.319e-05  r2: -0.08
      m: -6.148e-03 +/- 6.671e-04 c1: 6.879e-05 +/- 3.343e-05 c2:

- mcal-v11s06
    - gauss fit
    - subtracting correlated noise image
    - standard great3 noise level
      m1: 1.164e-03 +/- 8.397e-04 c1: 4.968e-05 +/- 2.936e-05
      m2: 3.218e-03 +/- 8.478e-04 c2: -2.483e-06 +/- 3.050e-05

    - error weights 0.25
      m1: 4.864e-03 +/- 8.292e-04 c1: 4.672e-05 +/- 2.899e-05
      m2: 6.894e-03 +/- 8.175e-04 c2: 3.976e-06 +/- 2.940e-05

    - s2n_r weights
        1/(1 + (20/s2n_r)^2 )
        m1: 2.046e-05 +/- 8.351e-04 c1: 5.793e-05 +/- 2.922e-05
        m2: 2.103e-03 +/- 8.198e-04 c2: -5.566e-07 +/- 2.951e-05
            m: 1.090e-03 +/- 5.860e-04

    - using mean corrections
      m1: -2.174e-03 +/- 8.266e-04 c1: 4.080e-05 +/- 2.899e-05  r1: -0.0627
      m2: -4.948e-04 +/- 8.371e-04 c2: 1.154e-06 +/- 3.023e-05  r2: -0.08
      m: -1.308e-03 +/- 5.918e-04

- mcal-v11s07
    - exp fit
    - subtracting correlated noise image
    - no weights
      m1: 1.205e-03 +/- 8.230e-04 c1: 4.033e-05 +/- 2.886e-05  r1: -0.0627
      m2: 3.885e-03 +/- 8.049e-04 c2: 6.882e-06 +/- 2.907e-05  r2: -0.08
      m: 2.575e-03 +/- 5.812e-04

    - weighted s2n_r, s2n_soft=20
      m1: -6.769e-05 +/- 8.273e-04 c1: 5.515e-05 +/- 2.901e-05  r1: -0.0627
      m2: 2.859e-03 +/- 7.609e-04 c2: 5.828e-06 +/- 2.748e-05  r2: -0.08
      m: 1.428e-03 +/- 5.674e-04

    - weighted s2n_r, s2n_soft=40
    - weighted s2n_r, s2n_soft=10

    - using mean corrections
      m1: -2.248e-03 +/- 8.246e-04 c1: 4.102e-05 +/- 2.892e-05  r1: -0.0627
      m2: 8.658e-05 +/- 7.981e-04 c2: -8.449e-08 +/- 2.882e-05  r2: -0.08
      m: -1.049e-03 +/- 5.780e-04

- v13
    - small new run 2M with fixes for psf center and kolmogorov
- v14
    - same as v13 but 100M
    - mcal-v14s02
        - extra noise to get to mode s/n ~12
        - add minus-sheared random noise
          m1: 5.545e-04 +/- 8.376e-04 c1: -4.193e-05 +/- 2.906e-05
          m2: 5.111e-04 +/- 8.124e-04 c2: 5.878e-05 +/- 2.873e-05
          m:  5.332e-04 +/- 5.849e-04
          c1: -4.207e-05 +/- 2.895e-05
          c2: 5.877e-05 +/- 2.895e-05

          s2n_r > 10
          m1: -4.823e-04 +/- 8.344e-04 c1: -4.132e-05 +/- 2.895e-05  r1: 0.0335
          m2: 2.283e-04 +/- 8.103e-04 c2: 4.260e-05 +/- 2.866e-05  r2: 0.0593
          m:  -1.193e-04 +/- 5.832e-04
          c1: -4.100e-05 +/- 2.887e-05
          c2: 4.182e-05 +/- 2.887e-05

          s2n_r > 15
            m1: -1.216e-03 +/- 9.220e-04 c1: -2.188e-05 +/- 3.198e-05  r1: 0.0335
            m2: 2.178e-03 +/- 8.646e-04 c2: 3.215e-05 +/- 3.058e-05  r2: 0.0593
            m:  5.193e-04 +/- 6.368e-04
            c1: -2.002e-05 +/- 3.154e-05
            c2: 2.858e-05 +/- 3.154e-05
          s2n_r > 20
            m1: -2.080e-03 +/- 9.962e-04 c1: 1.167e-05 +/- 3.456e-05  r1: 0.0335
            m2: 1.926e-03 +/- 1.002e-03 c2: 5.320e-05 +/- 3.544e-05  r2: 0.0593
            m:  -3.679e-05 +/- 7.126e-04
            c1: 1.415e-05 +/- 3.530e-05
            c2: 4.895e-05 +/- 3.530e-05



    - mcal-v14s01
        - extra noise to get to mode s/n ~12
        - detrend

    # no cuts
      m1: 2.063e-03 +/- 7.454e-04 c1: -2.424e-05 +/- 2.586e-05  r1: 0.0335
      m2: 1.174e-03 +/- 7.399e-04 c2: 9.781e-06 +/- 2.617e-05  r2: 0.0593

      m:   1.610e-03 +/- 5.267e-04
      c1: -2.479e-05 +/- 2.609e-05
      c2:  1.067e-05 +/- 2.608e-05

      A[1,1] (-1.04e-05 +/- 8.76e-07) + (-1.11 +/ 0.000454) deltan
      A[1,2] (1.97e-07 +/- 2.22e-06) + (-0.00057 +/ 0.00115) deltan
      A[2,1] (6.55e-06 +/- 9.26e-06) + (-0.00401 +/ 0.0048) deltan
      A[2,2] (-1.46e-05 +/- 2.02e-06) + (-1.11 +/ 0.00105) deltan
      Rnoise
      -0.0694 -3.56e-05
      -0.000251 -0.0693

       m1:  2.287e-03 +/- 7.394e-04 c1: -2.235e-05 +/- 2.565e-05  r1: 0.0335
       m2:  1.476e-03 +/- 7.296e-04 c2: 7.132e-06 +/- 2.580e-05  r2: 0.0593
       m:   1.874e-03 +/- 5.209e-04
       c1: -2.288e-05 +/- 2.580e-05
       c2:  7.922e-06 +/- 2.579e-05

    #
    # using responses
    #

    # s/n > 8 using responses
    99495052/99996971.=  0.995
    sel: [ 1.00007024  1.00015094]
    m1: 1.979e-03 +/- 7.390e-04 c1: -2.246e-05 +/- 2.564e-05  r1: 0.0335
    m2: 1.442e-03 +/- 7.323e-04 c2: 6.152e-06 +/- 2.590e-05  r2: 0.0593
    m: 1.706e-03 +/- 5.215e-04 c1: -2.283e-05 +/- 2.583e-05 c2: 6.655e-06 +/- 2.582e-05

    # s/n > 10 using responses
    94481942/99996971. = 0.94
    sel: [ 1.00051723  1.0011313 ]
    m1: -2.280e-05 +/- 7.543e-04 c1: -1.892e-05 +/- 2.617e-05  r1: 0.0335
    m2: -1.438e-04 +/- 7.493e-04 c2: 4.919e-06 +/- 2.650e-05  r2: 0.0593
    m: -8.473e-05 +/- 5.329e-04 c1: -1.905e-05 +/- 2.639e-05 c2: 4.991e-06 +/- 2.639e-05


    # s/n > 15 using responses
    67548722/99996971 = 0.68
    sel: [ 1.00256499  1.00501757]
    m1: -1.008e-03 +/- 8.287e-04 c1: -1.255e-05 +/- 2.875e-05  r1: 0.0335
    m2: -1.412e-03 +/- 8.359e-04 c2: 4.170e-06 +/- 2.956e-05  r2: 0.0593
    m: -1.212e-03 +/- 5.898e-04 c1: -1.293e-05 +/- 2.923e-05 c2: 4.460e-06 +/- 2.921e-05
    using mean s2l
    m1: 2.140e-04 +/- 8.297e-04 c1: -1.257e-05 +/- 2.879e-05  r1: 0.0335
    m2: -2.630e-03 +/- 8.349e-04 c2: 4.165e-06 +/- 2.953e-05  r2: 0.0593
    m: -1.233e-03 +/- 5.926e-04 c1: -1.441e-05 +/- 2.936e-05 c2: 6.967e-06 +/- 2.935e-05


    # s/n > 20 using responses
    48040349/99996971 = 0.48
    sel: [ 1.00370716  1.00688406]
    m1: -2.002e-03 +/- 9.373e-04 c1: -8.024e-06 +/- 3.251e-05  r1: 0.0335
    m2: -4.145e-04 +/- 9.720e-04 c2: 7.134e-06 +/- 3.437e-05  r2: 0.0593
    m: -1.192e-03 +/- 6.772e-04 c1: -7.283e-06 +/- 3.356e-05 c2: 5.616e-06 +/- 3.356e-05
    # using mean sel
    m1: -4.224e-04 +/- 9.387e-04 c1: -8.038e-06 +/- 3.257e-05  r1: 0.0335
    m2: -1.991e-03 +/- 9.704e-04 c2: 7.123e-06 +/- 3.432e-05  r2: 0.0593
    m: -1.220e-03 +/- 6.772e-04 c1: -9.188e-06 +/- 3.355e-05 c2: 8.858e-06 +/- 3.355e-05

    # s/n > 30 using responses
    24426946/99996971 = 0.24
    sel: [ 1.00370898  1.0080212 ]
    m1: -3.720e-03 +/- 1.219e-03 c1: 1.879e-05 +/- 4.229e-05  r1: 0.0335
    m2: 3.702e-04 +/- 1.309e-03 c2: 9.726e-06 +/- 4.629e-05  r2: 0.0593
    m: -1.632e-03 +/- 9.006e-04 c1: 2.122e-05 +/- 4.463e-05 c2: 5.898e-06 +/- 4.462e-05
    # using mean sel
    m1: -1.580e-03 +/- 1.222e-03 c1: 1.883e-05 +/- 4.238e-05  r1: 0.0335
    m2: -1.770e-03 +/- 1.306e-03 c2: 9.706e-06 +/- 4.619e-05  r2: 0.0593
    m: -1.673e-03 +/- 8.966e-04 c1: 1.871e-05 +/- 4.443e-05 c2: 1.026e-05 +/- 4.442e-05


            

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

