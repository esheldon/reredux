wombat
-------
v006
    - pretty standard, with exp
    - mcal-w006-01
    	- usual gauss/gauss fitting
			R: [ 0.55699758  0.55846702]
			Rpsf: [ 0.13382072  0.13298203]
			  m1: -7.402e-03 +/- 6.092e-03 c1: -2.817e-04 +/- 1.954e-04  r1: 0.0373
			  m2: -8.971e-03 +/- 6.278e-03 c2: -2.207e-04 +/- 1.923e-04  r2: -0.0506
			  m:  -8.150e-03 +/- 4.388e-03
			  c1: -2.826e-04 +/- 1.945e-04
			  c2: -2.220e-04 +/- 1.945e-04

		s/n > 10
        	kept: 648791/999389 = 0.649188
			R: [ 0.66737265  0.66915955]
			Rpsf: [ 0.14562377  0.14503495]

			Rsel: [ 0.00281923  0.00282607]
			Rpsf_sel: [ 0.00506003  0.00460277]
			without correction
			  m1: -1.493e-03 +/- 6.513e-03 c1: 4.268e-05 +/- 2.089e-04  r1: 0.0373
			  m2: -5.237e-03 +/- 6.493e-03 c2: -3.948e-05 +/- 1.988e-04  r2: -0.0506
			  m:  -3.277e-03 +/- 4.617e-03
			  c1: 4.054e-05 +/- 2.046e-04
			  c2: -4.252e-05 +/- 2.047e-04

			with correction
			  m1: -5.691e-03 +/- 6.486e-03 c1: -1.019e-05 +/- 2.081e-04  r1: 0.0373
			  m2: -9.426e-03 +/- 6.465e-03 c2: -8.726e-05 +/- 1.980e-04  r2: -0.0506
			  m:  -7.471e-03 +/- 4.597e-03
			  c1: -1.232e-05 +/- 2.038e-04
			  c2: -9.028e-05 +/- 2.038e-04


v006ln
    - low noise, no shape noise, exp
    - mcal-w006ln-01
        - deconv
		R: [ 0.25808106  0.26646678]
		Rpsf: [ 0.  0.]
		  m1: 4.806e-03 +/- 3.814e-03 c1: 5.980e-05 +/- 1.224e-04  r1: 0.0373
		  m2: -3.886e-02 +/- 2.973e-03 c2: -3.025e-05 +/- 9.105e-05  r2: -0.0506
		  m:  -1.600e-02 +/- 2.675e-03
		  c1: 3.490e-05 +/- 1.186e-04
		  c2: -6.564e-05 +/- 1.186e-04

    - mcal-w006ln-02
        - deconv not forcing same size for all k images
		R: [ 0.25803248  0.26641912]
		Rpsf: [ 0.  0.]
		  m1: 5.541e-03 +/- 4.146e-03 c1: 6.751e-05 +/- 1.330e-04  r1: 0.0373
		  m2: -3.879e-02 +/- 2.942e-03 c2: -3.103e-05 +/- 9.012e-05  r2: -0.0506
		  m:  -1.558e-02 +/- 2.801e-03
		  c1: 4.222e-05 +/- 1.242e-04
		  c2: -6.696e-05 +/- 1.242e-04
	- mcal-w006ln-03
		- using sigma_weight_factor = 1.2
        kept 934071/1000000 from flags
		R: [ 0.49061458  0.48975723]
		Rpsf: [ 0.  0.]
		  m1: -3.780e-02 +/- 2.732e-03 c1: 1.307e-05 +/- 8.765e-05  r1: 0.0373
		  m2: -5.039e-02 +/- 4.294e-03 c2: -1.254e-05 +/- 1.315e-04  r2: -0.0506
		  m:  -4.380e-02 +/- 2.549e-03
		  c1: 5.886e-06 +/- 1.130e-04
		  c2: -2.274e-05 +/- 1.130e-04
	- mcal-w006ln-04
		- using sigma_weight_factor = sqrt(5) = 2.236

		R: [ 0.21611933  0.22905206]
		Rpsf: [ 0.  0.]
		  m1: 3.529e-02 +/- 5.854e-03 c1: 1.005e-04 +/- 1.878e-04  r1: 0.0373
		  m2: -3.892e-02 +/- 2.579e-03 c2: -3.547e-05 +/- 7.897e-05  r2: -0.0506
		  m:  -7.470e-05 +/- 3.754e-03
		  c1: 5.813e-05 +/- 1.664e-04
		  c2: -9.563e-05 +/- 1.664e-04

	- mcal-w006ln-05
        - no noise correction
		- using sigma_weight_factor = sqrt(5) = 2.236
        R: [ 0.22778043  0.22776754]
        Rpsf: [ 0.  0.]
          m1: -2.432e-02 +/- 1.950e-03 c1: -4.393e-07 +/- 6.255e-05  r1: 0.0373
          m2: -3.479e-02 +/- 2.904e-03 c2: -1.626e-05 +/- 8.892e-05  r2: -0.0506
          m:  -2.931e-02 +/- 1.759e-03
          c1: -6.413e-06 +/- 7.798e-05
          c2: -2.474e-05 +/- 7.800e-05

	- mcal-w006ln-06
		- fix sigma weight
		R: [ 0.23107353  0.24214065]
		Rpsf: [ 0.  0.]
		  m1: 2.265e-02 +/- 5.096e-03 c1: 8.524e-05 +/- 1.635e-04  r1: 0.0373
		  m2: -3.900e-02 +/- 2.722e-03 c2: -3.003e-05 +/- 8.336e-05  r2: -0.0506
		  m:  -6.728e-03 +/- 3.319e-03
		  c1: 5.007e-05 +/- 1.471e-04
		  c2: -8.002e-05 +/- 1.472e-04

	- mcal-w006ln-07
		- used x_interpolant='lanczos15'
        - the two R values agree now, and the errors are not *so* different
          between m1 and m2. why don't thy agree?
		R: [ 0.26573648  0.26565307]
		Rpsf: [ 0.  0.]
		  m1: -2.671e-02 +/- 2.070e-03 c1: 2.714e-06 +/- 6.641e-05  r1: 0.0373
		  m2: -3.666e-02 +/- 3.070e-03 c2: -1.783e-05 +/- 9.403e-05  r2: -0.0506
		  m:  -3.145e-02 +/- 1.859e-03
		  c1: -2.961e-06 +/- 8.239e-05
		  c2: -2.589e-05 +/- 8.241e-05

clues
    - getting kmax too large errors, never had those before
        - don't see them for noisy data, but still odd
    - errors in m1,m2 don't agree, better with l15 but still difer
    - not working at high s/n!
	- hmm... galaxies seem awfully big

v007ln
--------

Want to see if non-trivial wcs is causing a problem wcs is now a simpler; well
it is still Tan but simple.  Could also try pixel_scale: 0.263

v006ln was highly non-trivial

	- mcal-w007ln-01
		- used x_interpolant='lanczos15'
		- wow, the errors are factor of 5 less. I had expected low errors since
		  there is no shape noise
		R: [ 0.26547729  0.26535603]
		Rpsf: [ 0.  0.]
		  m1: -3.140e-02 +/- 4.078e-04 c1: -1.539e-05 +/- 1.310e-05  r1: -0.0398
		  m2: -3.088e-02 +/- 5.268e-04 c2: -2.330e-05 +/- 1.751e-05  r2: -0.0557
		  m:  -3.113e-02 +/- 3.360e-04
		  c1: -1.573e-05 +/- 1.552e-05
		  c2: -2.284e-05 +/- 1.552e-05

v007ln
---------
- pure pixel scale
- psf pure e2, high ellip

- mcal-w008ln-01
    - used x_interpolant='lanczos15'
		R: [ 0.26559465  0.26552768]
		Rpsf: [ 0.  0.]
		  m1: -3.132e-02 +/- 4.407e-04 c1: 3.817e-05 +/- 1.513e-05  r1: -0.0248
		  m2: -3.075e-02 +/- 4.951e-04 c2: -2.716e-05 +/- 1.608e-05  r2: 0.132
		  m:  -3.105e-02 +/- 3.317e-04
		  c1: 3.794e-05 +/- 1.562e-05
		  c2: -2.848e-05 +/- 1.568e-05

v008ln
--------
- pure pixel scale 1.0 (not even using dudx etc.)
- psf pure e2, high ellip

- mcal-w008ln-02
	- fitting version, gauss/gauss
    -  more accurate, and less noisy
        R: [ 0.83006652  0.83085815]
        Rpsf: [ 0.17082828  0.17011149]
          m1: -4.613e-05 +/- 6.200e-05 c1: -1.481e-06 +/- 2.129e-06  r1: -0.0248
          m2: -2.432e-04 +/- 6.307e-05 c2: 3.173e-05 +/- 2.049e-06  r2: 0.132
          m:  -1.384e-04 +/- 4.464e-05
          c1: -1.403e-06 +/- 2.101e-06
          c2: 3.218e-05 +/- 2.110e-06

v009ln
---------
- similar to BFD sim

- mcal-w009ln-03
    - get dk from not-deconvolved, interpolated galaxy image
    - may want to do this differently
        - get dk from deconvolved?
        - fix dk?  0.1 seems to be a favorite for this sim
        - not sure stepK is what we want; docs say
        "Returns sampling in k space necessary to avoid folding of image in x
        space."
        so this is about optimizing real space not k space
    - might want to draw k then shear it?

- mcal-w009ln-04
    - fixing dk and sizes


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
    - deconv
        - adding noise fix helped with the additive, not the m though
        - trying trim image to see if that fixes it
            - whoa, worse again
        - 06 trim k image in circle
            - looks ok for c, but still bad m
        - 07 fixed dk=0.25
            - same
        - 08 temporarily turning off full wcs
            - better in g1, hmm....
            - but turns out I was still using sigma_weight in arcsec assuming
            old wcs
        - 09 temporarily turning off full wcs and turning off fix noise to see
          how much the correlated noise may be a problem
        - 10 same as 08 but fixed sigma weight
            - now looks like ones with wcs.  So the weight size matters.
        - 11 fixed sigma weight 1.5 in pixels
        - 12 fixed sigma weight 2.0 in pixels
            - unbiased... note no extra noise though
        - 13 same as 12 but no fix noise
            - looks worse
        - 14 for weight use factor of two bigger than psf
            R: [ 0.26099306  0.26574938]
            Rpsf: [ 0.  0.]
              m1: 4.784e-03 +/- 4.445e-03 c1: -7.762e-05 +/- 1.496e-04  r1: 0.0818
              m2: -2.384e-03 +/- 4.478e-03 c2: -1.711e-05 +/- 1.616e-04  r2: 0.0189
              m:  9.403e-04 +/- 3.167e-03
              c1: -8.821e-05 +/- 1.563e-04
              c2: -1.485e-05 +/- 1.560e-04

        - 15 extra noise
            - looks worse
            R: [ 0.25643193  0.2697692 ]
            Rpsf: [ 0.  0.]
              m1: 2.759e-02 +/- 6.513e-03 c1: 7.815e-05 +/- 2.193e-04  r1: 0.0818
              m2: -1.406e-02 +/- 5.939e-03 c2: 9.953e-05 +/- 2.143e-04  r2: 0.0189
              m:  5.257e-03 +/- 4.485e-03
              c1: 1.663e-05 +/- 2.213e-04
              c2: 1.127e-04 +/- 2.210e-04

        - 16 same as 14 but deweight
            - has higher sensitivity but about the same noise, which
              seems to be dominated by scatter in the shear fields?
            R: [ 0.51611103  0.52558817]
            Rpsf: [ 0.  0.]
              m1: 6.936e-03 +/- 4.346e-03 c1: -5.260e-05 +/- 1.463e-04
              m2: -4.418e-04 +/- 4.380e-03 c2: -1.049e-05 +/- 1.580e-04
              m:  2.980e-03 +/- 3.097e-03
              c1: -6.349e-05 +/- 1.528e-04
              c2: -8.164e-06 +/- 1.526e-04
        - 17 same as 16 but extra noise
            R: [ 0.43081698  0.4529229 ]
            Rpsf: [ 0.  0.]
              m1: 3.884e-02 +/- 6.234e-03 c1: -1.275e-04 +/- 2.099e-04  r1: 0.0818
              m2: -7.173e-03 +/- 5.708e-03 c2: 4.795e-06 +/- 2.060e-04  r2: 0.0189
              m:  1.417e-02 +/- 4.326e-03
              c1: -1.954e-04 +/- 2.135e-04
              c2: 1.932e-05 +/- 2.132e-04
        - 18 turn off k image trimming
            R: [ 0.43055414  0.45292981]
            Rpsf: [ 0.  0.]
              m1: 3.496e-02 +/- 6.091e-03 c1: -1.639e-04 +/- 2.051e-04  r1: 0.0818
              m2: -3.261e-03 +/- 5.524e-03 c2: 1.974e-05 +/- 1.993e-04  r2: 0.0189
              m:  1.447e-02 +/- 4.181e-03
              c1: -2.204e-04 +/- 2.063e-04
              c2: 3.180e-05 +/- 2.060e-04


        - other ideas
            - maybe stamps too small
            - maybe constant sigma weight is a bad idea
            - maybe wcs and k space is not working?

- v14 deconv
    - mcal-v14s04
        - no extra noise
        - about a percent in each component, opposite signs
    - mcal-v14s05
        - sigma weight = sqrt(5)*1.05*0.265
        - use full wcs
        - not as good, -4% in g1. because of weight or using full wcs?
    - mcal-v14s06
        - same as s05 but just sigma_weight_factor=2
        R: [ 0.26098296  0.26567978]
        Rpsf: [ 0.  0.]
          m1: 1.190e-02 +/- 6.244e-04 c1: 4.697e-05 +/- 2.166e-05  r1: -0.0335
          m2: -6.159e-03 +/- 6.239e-04 c2: 1.699e-05 +/- 2.207e-05  r2: 0.0593
          m:  2.707e-03 +/- 5.764e-04
          c1: 5.766e-05 +/- 2.853e-05
          c2: 3.559e-05 +/- 2.855e-05


TODO - extra noise
     - non-symmetric psfs from nsim?

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

        maximum likelihood
          m1: -4.367e-01 +/- 4.718e-04 c1: -2.401e-05 +/- 1.637e-05  r1: 0.0335
          m2: -4.350e-01 +/- 4.590e-04 c2: 3.345e-05 +/- 1.623e-05  r2: 0.0593
          m:  -4.359e-01 +/- 3.317e-04
          c1: -2.308e-05 +/- 1.640e-05
          c2: 3.170e-05 +/- 1.644e-05

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


- mcal-v14s03
    - new style metacal
    ngmixer/master:    6df6ffc070dba40adefbb2ca824193d6fca6f2d8
    ngmix/keep_prepix: 60ce3ba996ea220a5c781e078611262883faf7e5
    reredux/master:    3da81ca42438b9695faa2ff6da6ded83ededfbe9

    R: [ 0.56305554  0.56482219]
    Rpsf: [ 0.15606533  0.15631355]
    
    m1: 1.492e-03 +/- 8.353e-04 c1: 2.422e-05 +/- 2.898e-05  r1: -0.0335
    m2: -9.985e-04 +/- 7.882e-04 c2: 1.978e-05 +/- 2.788e-05  r2: 0.0593
    m:  2.245e-04 +/- 5.776e-04
    c1: 2.570e-05 +/- 2.859e-05
    c2: 2.234e-05 +/- 2.861e-05

    s/n > 7

        R: [ 0.57713748  0.57895808]
        Rpsf: [ 0.15454412  0.15482336]
        Rsel: [ 0.00040494  0.00038973]
        Rpsf_sel: [ 0.00289235  0.00288153]
        without correction
          m1: 2.571e-03 +/- 8.408e-04 c1: 1.309e-05 +/- 2.917e-05  r1: -0.0335
          m2: -2.415e-04 +/- 7.940e-04 c2: 1.772e-05 +/- 2.808e-05  r2: 0.0593
          m:  1.140e-03 +/- 5.822e-04
          c1: 1.475e-05 +/- 2.882e-05
          c2: 2.061e-05 +/- 2.883e-05

        with correction
          m1: 1.871e-03 +/- 8.402e-04 c1: 1.307e-05 +/- 2.915e-05  r1: -0.0335
          m2: -9.140e-04 +/- 7.935e-04 c2: 1.769e-05 +/- 2.806e-05  r2: 0.0593
          m:  4.533e-04 +/- 5.818e-04
          c1: 1.472e-05 +/- 2.879e-05
          c2: 2.056e-05 +/- 2.881e-05


    - mcal-v14s08
        - moments in k space

            - s/n > 15 (s/n calc seems to be off, this is probably more like 7)
                kept 0.965
                R: [ 0.2729822  0.2729822]
                Rpsf: [ 0.  0.]
                Rsel: [-0.00363915 -0.003668  ]
                Rpsf_sel: [ 0.  0.]
                without correction
                  m1: -2.414e-02 +/- 8.046e-04 c1: 6.655e-05 +/- 2.791e-05  r1: -0.0335
                  m2: -2.416e-02 +/- 8.499e-04 c2: -6.415e-06 +/- 3.006e-05  r2: 0.0593
                  m:  -2.415e-02 +/- 5.869e-04
                  c1: 6.656e-05 +/- 2.905e-05
                  c2: -6.391e-06 +/- 2.907e-05

                with correction
                  m1: -1.095e-02 +/- 8.154e-04 c1: 6.745e-05 +/- 2.829e-05  r1: -0.0335
                  m2: -1.087e-02 +/- 8.614e-04 c2: -6.503e-06 +/- 3.047e-05  r2: 0.0593
                  m:  -1.091e-02 +/- 5.949e-04
                  c1: 6.740e-05 +/- 2.944e-05
                  c2: -6.587e-06 +/- 2.946e-05


            - s/n > 20 (s/n calc seems to be off, this is probably more like 10)

                kept 0.76
                looks like a selection effect correction is still needed

                - Using R[1] for both
                R: [ 0.28773126  0.28773126]
                Rpsf: [ 0.  0.]

                Rsel: [-0.01063463 -0.01067082]
                Rpsf_sel: [ 0.  0.]
                without correction
                  m1: -3.649e-02 +/- 7.853e-04 c1: 7.682e-05 +/- 2.724e-05  r1: -0.0335
                  m2: -3.690e-02 +/- 8.245e-04 c2: -1.004e-04 +/- 2.916e-05  r2: 0.0593
                  m:  -3.670e-02 +/- 5.711e-04
                  c1: 7.706e-05 +/- 2.827e-05
                  c2: -1.000e-04 +/- 2.828e-05

                with correction
                  m1: 4.909e-04 +/- 8.154e-04 c1: 7.976e-05 +/- 2.829e-05  r1: -0.0335
                  m2: 1.942e-04 +/- 8.563e-04 c2: -1.043e-04 +/- 3.028e-05  r2: 0.0593
                  m:  3.398e-04 +/- 5.930e-04
                  c1: 7.994e-05 +/- 2.935e-05
                  c2: -1.040e-04 +/- 2.937e-05

                - using calculated R
                  as usual R[0] is off

                  R: [ 0.2766102   0.28773126]
                  Rsel: [-0.01063463 -0.01067082]

                  m1: 4.232e-02 +/- 8.495e-04 c1: 8.310e-05 +/- 2.947e-05  r1: -0.0335
                  m2: 1.942e-04 +/- 8.563e-04 c2: -1.043e-04 +/- 3.028e-05  r2: 0.0593
                  m:  2.088e-02 +/- 1.053e-03
                  c1: 1.081e-04 +/- 5.211e-05
                  c2: -6.092e-05 +/- 5.215e-05

            - s/n > 30
                kept 0.39

                R: [ 0.3104545  0.3104545]
                Rpsf: [ 0.  0.]

                Rsel: [-0.01244995 -0.01261258]
                Rpsf_sel: [ 0.  0.]
                without correction
                  m1: -3.706e-02 +/- 1.027e-03 c1: 2.621e-05 +/- 3.563e-05  r1: -0.0335
                  m2: -3.925e-02 +/- 1.016e-03 c2: -1.105e-04 +/- 3.594e-05  r2: 0.0593
                  m:  -3.818e-02 +/- 7.255e-04
                  c1: 2.751e-05 +/- 3.591e-05
                  c2: -1.083e-04 +/- 3.593e-05

                with correction
                  m1: 3.168e-03 +/- 1.070e-03 c1: 2.731e-05 +/- 3.711e-05  r1: -0.0335
                  m2: 1.430e-03 +/- 1.059e-03 c2: -1.152e-04 +/- 3.746e-05  r2: 0.0593
                  m:  2.283e-03 +/- 7.554e-04
                  c1: 2.834e-05 +/- 3.739e-05
                  c2: -1.134e-04 +/- 3.741e-05

            - no s/n cut
                R: [ 0.26981947  0.26981947]
                Rpsf: [ 0.  0.]
                  m1: -1.732e-02 +/- 7.998e-04 c1: 7.408e-05 +/- 2.775e-05  r1: -0.0335
                  m2: -1.735e-02 +/- 8.642e-04 c2: 4.594e-05 +/- 3.056e-05  r2: 0.0593
                  m:  -1.733e-02 +/- 5.906e-04
                  c1: 7.410e-05 +/- 2.923e-05
                  c2: 4.597e-05 +/- 2.925e-05



- v15small
    - larger psf
    - mcal-v15small01
        R: [ 0.35079861  0.39969921]
        Rpsf: [ 0.  0.]
          m1: 1.413e-01 +/- 5.268e-03 c1: 2.665e-04 +/- 1.887e-04  r1: -1.03e-17
          m2: -4.324e-03 +/- 4.505e-03 c2: 1.098e-04 +/- 1.610e-04  r2: -0
          m:  6.866e-02 +/- 4.578e-03
          c1: 2.665e-04 +/- 2.316e-04
          c2: 1.098e-04 +/- 2.316e-04
    - note if I use R[1] for both corrections it looks fine
        R: [ 0.39969921  0.39969921]
        Rpsf: [ 0.  0.]
          m1: 1.666e-03 +/- 4.623e-03 c1: 2.339e-04 +/- 1.656e-04  r1: -1.03e-17
          m2: -4.324e-03 +/- 4.505e-03 c2: 1.098e-04 +/- 1.610e-04  r2: -0
          m:  -1.322e-03 +/- 3.238e-03
          c1: 2.339e-04 +/- 1.639e-04
          c2: 1.098e-04 +/- 1.639e-04

      I wonder if this means the stamps are too small

- v16small
    - ARGH, turns out fixing size wasn't working , but in 
      egret
    - same as v15small but 48x48 stamps for all objects

    - mcal-v16small01
        - looks better!

        R: [ 0.39821887  0.39999097]
        Rpsf: [ 0.  0.]
          m1: 1.291e-03 +/- 4.414e-03 c1: -7.902e-07 +/- 1.581e-04  r1: -1.03e-17
          m2: 3.712e-03 +/- 4.742e-03 c2: -3.948e-04 +/- 1.695e-04  r2: -0
          m:  2.499e-03 +/- 3.248e-03
          c1: -7.933e-07 +/- 1.643e-04
          c2: -3.948e-04 +/- 1.643e-04

    - mcal-v16small02
        - extra noise
        - percent level bias
        R: [ 0.29853781  0.31816298]
        Rpsf: [ 0.  0.]
          m1: 4.660e-02 +/- 7.669e-03 c1: -1.453e-04 +/- 2.747e-04  r1: -1.03e-17
          m2: -1.235e-02 +/- 6.817e-03 c2: -3.659e-04 +/- 2.436e-04  r2: -0
          m:  1.720e-02 +/- 5.284e-03
          c1: -1.453e-04 +/- 2.674e-04
          c2: -3.659e-04 +/- 2.674e-04
        - using mean of R values
        R: [ 0.3083504  0.3083504]
        Rpsf: [ 0.  0.]
          m1: 1.330e-02 +/- 7.425e-03 c1: -1.407e-04 +/- 2.660e-04  r1: -1.03e-17
          m2: 1.908e-02 +/- 7.034e-03 c2: -3.775e-04 +/- 2.514e-04  r2: -0
          m:  1.618e-02 +/- 5.128e-03
          c1: -1.407e-04 +/- 2.595e-04
          c2: -3.775e-04 +/- 2.595e-04


    - mcal-v16small03
        - extra noise
        - not doing deweight

            R: [ 0.18960937  0.20352403]
            Rpsf: [ 0.  0.]
              m1: 3.719e-02 +/- 8.565e-03 c1: -1.311e-04 +/- 3.068e-04  r1: -1.03e-17
              m2: -2.645e-02 +/- 7.442e-03 c2: -2.564e-04 +/- 2.660e-04  r2: -0
              m:  5.445e-03 +/- 5.835e-03
              c1: -1.311e-04 +/- 2.953e-04
              c2: -2.564e-04 +/- 2.953e-04

        - using mean R of two components
            R: [ 0.1965667  0.1965667]
            Rpsf: [ 0.  0.]
              m1: 4.807e-04 +/- 8.262e-03 c1: -1.265e-04 +/- 2.960e-04  r1: -1.03e-17
              m2: 8.006e-03 +/- 7.705e-03 c2: -2.655e-04 +/- 2.754e-04  r2: -0
              m:  4.234e-03 +/- 5.665e-03
              c1: -1.265e-04 +/- 2.867e-04
              c2: -2.655e-04 +/- 2.867e-04

    - mcal-v16small04
        - extra noise but no fix noise
            R: [ 0.20550386  0.205277  ]
            Rpsf: [ 0.  0.]
              m1: -4.166e-02 +/- 7.145e-03 c1: 2.728e-04 +/- 2.560e-04  r1: -1.03e-17
              m2: -2.953e-02 +/- 8.275e-03 c2: -7.658e-04 +/- 2.957e-04  r2: -0
              m:  -3.561e-02 +/- 5.485e-03
              c1: 2.728e-04 +/- 2.775e-04
              c2: -7.658e-04 +/- 2.775e-04

    - mcal-v16small05
        - gaussian weight
        - no extra noise
        - not doing deweight

- v17small
    - still a remaining bug here... some 42,44,46 sizes still

    - mcal-v17small01
        - gaussian weight and now have some s/n to cut on when we
          use more noise

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

