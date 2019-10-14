# THIS CODE CHECKS THAT THE FIRST DERIVATIVE WRT TO BOTH DNDZ AND BZ WORKS WHERE BZ NOW HAS NT*NZ PARAMS
# SECOND DERIVATIVE NOT FULLY WORKING
# FOR NOW WORKS ONLY WITH NEAREST
import matplotlib
matplotlib.use('Agg')
import numpy as np
from astropy.io import fits
import astropy.wcs
from astropy.wcs import WCS
from astropy.table import Table
import os
import matplotlib
import copy
import pyccl as ccl
import sys
#export PYTHONPATH=../LSSLike/desclss/:$PYTHONPATH
import hod 
#%reload(hod)
import hod_funcs_evol_fit
#%reload(hod_funcs_evol_fit)
import matplotlib.pyplot as plt
from itertools import combinations
import scipy.linalg as la
from scipy.interpolate import interp1d
from scipy import asarray as ar,exp



# where to save plots
direc = '/users/boryanah/HSC/figs/'
# ask the user kindly to tell you those
tomo = sys.argv[1]
zsam = sys.argv[2]
steps = int(sys.argv[3])
functional = int(sys.argv[4])
# ___________________________________
#             PARAMETERS
# ___________________________________

# what interpolation scheme do we want
# OUR EQUATIONS SHOULD WORK FOR NEAREST (AND NOT YET LINEAR)
interp = 'nearest'#'log'#'nearest'#'spline' #'linear'

# constants
deg_to_rad = np.pi/180.
arcmin_to_rad = (1./60.*np.pi/180.)

# redshift parameters
z_ini_sample = 0.
z_end_sample = 2.
z_ini_bin = 0.
z_end_bin = 2.

# number of tomographic bins and zsamples in each
N_tomo = int(tomo) 
N_zsamples_theo = int(zsam) 

# How many samples do we want to take of the interpolation functions
fac = 100
N_zsamples = fac*N_zsamples_theo

# redshift parameters
z_s_edges = np.linspace(z_ini_sample,z_end_sample,N_zsamples+1)
z_s_cents = (z_s_edges[1:]+z_s_edges[:-1])*.5
z_s_edges_theo = np.linspace(z_ini_sample,z_end_sample,N_zsamples_theo+1)
z_s_cents_theo = (z_s_edges_theo[1:]+z_s_edges_theo[:-1])*.5
z_bin_edges = np.linspace(z_ini_bin,z_end_bin,N_tomo+1)
z_bin_cents = (z_bin_edges[1:]+z_bin_edges[:-1])*.5
sigma_e2 = np.ones(N_tomo)*(.4**2) # THIS VALUE ASSUMED

# Power spectrum parameters
N_ell = 10
ell_ini = 100.
ell_end = 2000.
ells = np.linspace(ell_ini, ell_end, N_ell) 
delta_ell = np.mean(np.diff(ells))

# area of sample
area_HSC = 100. # sq deg
area_COSMOS = 1.7 # sq deg
tot_area = 41253. # sq deg
f_sky = area_HSC/tot_area
area_HSC *= deg_to_rad**2 # in rad^2
area_COSMOS *= deg_to_rad**2 # in rad^2

# Power spectrum integration parameters 
k_ar = np.logspace(-4.3, 3, 1000)
z_ar = np.linspace(0., 3., 50)[::-1]
a_ar = 1./(1. + z_ar)

# THOSE ARE NOT USED IN THIS CODE SINCE WE USE SIMPLE BIAS
hod_params = {
    'sigm_0': 0.4,
    'sigm_1': 0.,
    'alpha_0': 1.0,
    'alpha_1': 0.,
    'fc_0': 1.,
    'fc_1': 0.,
    'lmmin_0': 3.71,#
    'lmmin_1': 9.99,#
    'm0_0': 1.28,#
    'm0_1': 10.34,#
    'm1_0': 7.08,#
    'm1_1': 9.34}#

# ___________________________________
#          END OF PARAMETERS
# ___________________________________


# ___________________________________
#          CATALOGUE LOADING
# ___________________________________

# Read catalog
cat=fits.open("data/cosmos_weights.fits")[1].data

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

# 1d gaussian
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def lorentz(x, A, mu, sig):
    return A/(1+(np.abs(x-mu)/(0.5*sig))**2.7)

def get_nz_from_photoz_bins(zp_code,zp_ini,zp_end,zt_edges,zt_nbins):
    # Select galaxies in photo-z bin
    sel=(cat[zp_code]<=zp_end) & (cat[zp_code]>zp_ini)

    # Effective number of galaxies
    ngal=len(cat) * np.sum(cat['weight'][sel])/np.sum(cat['weight'])

    # Make a normalized histogram
    nz,z_bins=np.histogram(cat['PHOTOZ'][sel],          # 30-band photo-zs
                           bins=zt_nbins,               # Number of bins
                           range=zt_edges,              # Range in z_true
                           weights=cat['weight'][sel],  # Color-space weights
                           density=True)

    zs=cat['PHOTOZ'][sel]                                                                          
    we=cat['weight'][sel]

    mean = (zs*we).sum()/we.sum()
    sigma = np.sqrt((zs**2*we).sum()/we.sum()-mean**2) 
    return nz, z_bins, ngal, mean, sigma

# initiating all
dndz_data = np.zeros((N_tomo,N_zsamples))
dndz_data_theo = np.zeros((N_tomo,N_zsamples_theo))
N_gal_bin = np.zeros(N_tomo)

# Get a redshift distribution for galaxies
# with z_bin_ini < z_ephor_ab < z_bin_end
# fill the dndz_data values with the interpolated samples
# dndz_data_theo values with the discrete samples
for i in range(N_tomo):
    z_bin_ini = z_bin_edges[i]
    z_bin_end = z_bin_edges[i+1]
    z_bin_mid = 0.5*(z_bin_ini+z_bin_end)        

   
    dndz_this, z_edges_theo, N_gal_this, mean, sigma = \
                    get_nz_from_photoz_bins(zp_code='pz_best_eab',# Photo-z code
                                            zp_ini=z_bin_ini, zp_end=z_bin_end, # Bin edges
                                            zt_edges=(z_ini_sample, z_end_sample),          # Sampling range
                                            zt_nbins=N_zsamples_theo)         # Number of samples

    
    # area under the curve (must be 1)
    sum_dndz = np.sum(dndz_this*(z_edges_theo[1]-z_edges_theo[0])) # equals 1

    # Important second line of normalization
    #dndz_this/=sum_dndz # DOESN'T MATTER AS SUM_DNDZ = 1
    dndz_this*=(z_edges_theo[1]-z_edges_theo[0]) # VERY VERY VERY IMPORTANT
    
    
    # this is what values will be interpolated
    # (here because I have other codes where I use lorentzian, avg, etc)
    dndz_theo_fn = dndz_this

    # interpolating from N_zsamples_theo points
    if interp == 'spline': # NOT USED IN THIS CODE
        f = interp1d(0.5*(z_edges_theo[:-1]+z_edges_theo[1:]),dndz_theo_fn,kind='cubic',bounds_error=0,fill_value=(dndz_theo_fn[0],dndz_theo_fn[-1]))
    elif interp == 'log': # NOT USED IN THIS CODE
        f = interp1d(np.append(0.5*(z_edges_theo[:-1]+z_edges_theo[1:]),np.array([z_s_cents[0],z_s_cents[-1]])),np.append(np.log10(dndz_theo_fn),np.log10(np.array([dndz_this[0],dndz_this[-1]]))),kind='cubic',fill_value='extrapolate')  
    elif interp == 'nearest':

        f = interp1d(np.append(0.5*(z_edges_theo[:-1]+z_edges_theo[1:]),np.array([z_s_cents[0],z_s_cents[-1]])),np.append(dndz_theo_fn,np.array([dndz_theo_fn[0],dndz_theo_fn[-1]])),kind='nearest',bounds_error=0,fill_value=0.)
        
    elif interp == 'linear':
        #f = interp1d(0.5*(z_edges_theo[:-1]+z_edges_theo[1:]),dndz_theo_fn,kind='linear',bounds_error=0,fill_value=0.)
        f = interp1d(np.append(0.5*(z_edges_theo[:-1]+z_edges_theo[1:]),np.array([z_s_cents[0],z_s_cents[-1]])),np.append(dndz_theo_fn,np.array([dndz_theo_fn[0],dndz_theo_fn[-1]])),kind='linear',bounds_error=0,fill_value=0.)

    # record discrete dndzs
    dndz_data_theo[i,:] = dndz_this
    
    if interp == 'log': dndz_data[i,:] = 10**f(z_s_cents)
    elif functional == 1:
        dndz_data_theo[i,:] = lorentz(z_s_cents_theo,np.max(dndz_this),mean,sigma)
        sum_dndz_theo = np.sum(dndz_data_theo[i,:])# equals 1
        dndz_data_theo[i,:] /= sum_dndz_theo
        # norm does not matter too much but just for direct checks with answer
        dndz_data[i,:] = lorentz(z_s_cents,np.max(dndz_this),mean,sigma)
    else: dndz_data[i,:] = f(z_s_cents)

    # Normalization not necessary
    #sum_dndz = np.sum(dndz_data[i,:]*(z_s_cents[1]-z_s_cents[0])) # equals 1
    #dndz_data[i,:] = dndz_data[i,:]/sum_dndz # DOESN'T MATTER SINCE PYCCL NORMALIZES

    # Record number of galaxies
    N_gal_bin[i] = N_gal_this
    
    # Plot interpolation and discrete samples
    plt.plot(z_s_cents_theo, dndz_this, label='theory')
    plt.plot(z_s_cents, dndz_data[i,:], label='z = %f'%(z_bin_mid))
    
plt.legend()
plt.xlabel("z", fontsize=14)
plt.ylabel("p(z)", fontsize=14)
plt.savefig('dndz.png')
plt.close()

# Sanity checks
print(N_gal_bin)

# ___________________________________
#          COMPUTING FULL CLS
# ___________________________________

# THIS CODE IS EQUIVALENT TO WHAT IS IN COMPUTE FISHER
# BUT WITHOUT THE COMMENTS. IT IS TO BE ALWAYS USED
# WITH SIMPLE BIAS = TRUE AND HALOFIT AS THIS IS WHAT
# THE CURLY C PART OF THE CODE EXPECTS. CURRENTLY
# I HAVE SET ALL BIAS PARAMS TO 1 TO MAKE SURE IT WORKS
# IF SOMETHING DOESN'T MAKE SENSE, SEE COMMENTS IN OTHER
# FILE AND YELL IF THERE ARE MISTAKES

def compute_Cls(par,hod_par=hod_params,z_cent=z_s_cents,N_gal_sample=N_gal_bin,k_arr=k_ar,z_arr=z_ar,a_arr=a_ar,ell=ells,compute_inv_cov=False,plot_for_sanity=False,powerspec='halofit',simple_bias=True):
    
    if powerspec == 'halofit':
        # Compute matter pk using halofit
        pk_mm_arr = np.array([ccl.nonlin_matter_power(cosmo_fid, k_arr, a) for a in a_arr])
    elif powerspec == 'halomodel':
        # Alternatively use halo model
        pk_mm_arr = np.array([ccl.halomodel.halomodel_matter_power(cosmo_fid, k_arr, a) for a in a_arr])
        
    if simple_bias == True:
        pk_gg_arr = pk_mm_arr.copy() # because we later include bias
        pk_gm_arr = pk_mm_arr.copy() # because we later include bias
    else:
        # Compute HOD
        hodpars = hod_funcs_evol_fit.HODParams(hod_par, islogm0=True, islogm1=True)
        
        hodprof = hod.HODProfile(cosmo_fid, hodpars.lmminf, hodpars.sigmf,\
                                 hodpars.fcf, hodpars.m0f, hodpars.m1f, hodpars.alphaf)
        # Compute galaxy pk using halofit
        pk_gg_arr = np.array([hodprof.pk(k_arr, a_arr[i]) for i in range(a_arr.shape[0])])
        # Compute galaxy-matter pk using halofit
        pk_gm_arr = np.array([hodprof.pk_gm(k_arr, a_arr[i]) for i in range(a_arr.shape[0])])
        
    # Create pk2D objects for interpolation
    pk_mm = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_arr), pk_arr=np.log(pk_mm_arr), is_logp=True)
    pk_gg = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_arr), pk_arr=np.log(pk_gg_arr), is_logp=True)
    pk_gm = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_arr), pk_arr=np.log(pk_gm_arr), is_logp=True)

    # load the bias parameters
    # k # indicates which of N_tomo sampling bins
    
    
    # load the dndz parameters
    dndz_z = np.zeros((N_tomo,N_zsamples))
    b_z = np.zeros((N_tomo,N_zsamples))
    for i_z in range(N_tomo):
        dndz_z[i_z,:] = np.array([par['dndz_%02d_%02d'%(i_z,k)]['val'] for k in range(N_zsamples)])
        b_z[i_z,:] = np.array([par['b_%02d_%02d'%(i_z,k)]['val'] for k in range(N_zsamples)])

    # per type gg gs ss
    tot_corr = N_ell*(N_tomo*(2*N_tomo+1))
    
    # make Cl_all of size N_ell*N_tomo*(2N_tomo+1)
    CL_ALL = np.zeros(tot_corr)
    temp = np.arange(2*N_tomo)
    temp = np.vstack((temp,temp)).T
    combs = np.array(list(combinations(range(2*N_tomo),2)))
    all_combos = np.vstack((temp,combs))
    
    for c, comb in enumerate(all_combos):
        i = comb[0]%N_tomo # first redshift bin
        j = comb[1]%N_tomo # second redshift bin
        t_i = comb[0]//N_tomo # tracer type 0 means g and 1 means s
        t_j = comb[1]//N_tomo # tracer type 0 means g and 1 means s
        
        # NOISE
        if (i == j):
            # number density of galaxies
            N_gal = N_gal_sample[i]            
            n_gal = N_gal/area_COSMOS # in rad^-2
            
            
            # Adding noise
            noise_gal = 1./n_gal
            noise_shape = sigma_e2[i]/n_gal
        else:
            noise_gal = 0.
            noise_shape = 0.
        
        # Now create corresponding Cls with pk2D objects matched to pk
        if t_i*2+t_j == 0: # this is gg 
            tracer_z1 = ccl.NumberCountsTracer(cosmo_fid, bias=(z_cent, b_z[i,:]), \
                                               dndz=(z_cent, dndz_z[i,:]),mag_bias=None, \
                                               has_rsd=False)
            tracer_z2 = ccl.NumberCountsTracer(cosmo_fid, bias=(z_cent, b_z[j,:]), \
                                               dndz=(z_cent, dndz_z[j,:]),mag_bias=None, \
                                               has_rsd=False)

            cl_gg = ccl.angular_cl(cosmo_fid, tracer_z1, tracer_z2, ell, p_of_k_a=pk_gg)
            cl_gg_no = cl_gg + noise_gal

            CL_ALL[(N_ell*c):(N_ell*c)+N_ell] = cl_gg_no
        elif t_i*2+t_j == 1: # this is gs
            tracer_z1 = ccl.NumberCountsTracer(cosmo_fid, bias=(z_cent, b_z[i,:]), \
                                               dndz=(z_cent, dndz_z[i,:]),mag_bias=None, \
                                               has_rsd=False)
            tracer_z2 = ccl.WeakLensingTracer(cosmo_fid, dndz=(z_cent, dndz_z[j,:]))
            cl_gs = ccl.angular_cl(cosmo_fid, tracer_z1, tracer_z2, ell, p_of_k_a=pk_gm)
            cl_gs_no = cl_gs
            CL_ALL[(N_ell*c):(N_ell*c)+N_ell] = cl_gs_no
            
        elif t_i*2+t_j == 3: # this is ss
            
            tracer_z1 = ccl.WeakLensingTracer(cosmo_fid, dndz=(z_cent, dndz_z[i,:]))
            tracer_z2 = ccl.WeakLensingTracer(cosmo_fid, dndz=(z_cent, dndz_z[j,:]))
            cl_ss = ccl.angular_cl(cosmo_fid, tracer_z1, tracer_z2, ell, p_of_k_a=pk_mm)
            cl_ss_no = cl_ss + noise_shape

            CL_ALL[(N_ell*c):(N_ell*c)+N_ell] = cl_ss_no

        if plot_for_sanity == True:
            if c == 0:
                # Plot for sanity
                plt.loglog(ells, cl_gg, label=r'$\mathrm{gg}$')
                plt.loglog(ells, cl_gg_no, label=r'$\mathrm{gg}+{\rm noise}$')
                plt.loglog(ells, cl_ss, label=r'$\mathrm{ss}$')
                plt.loglog(ells, cl_gs, label=r'$\mathrm{gs}$')
                plt.xlabel(r'$\ell$')
                plt.ylabel(r'$C_{\ell}$')
                plt.legend()
                plt.savefig('Cls.png')
                plt.close()

    if compute_inv_cov == False:
        print(len(CL_ALL))
        return CL_ALL

    print(len(CL_ALL))
    print(np.sum(CL_ALL>0))
    
    COV_ALL = np.zeros((len(CL_ALL),len(CL_ALL)))    
    # COMPUTE COVARIANCE MATRIX 
    for c_A, comb_A in enumerate(all_combos):
        for c_B, comb_B in enumerate(all_combos):
            i = comb_A[0]%N_tomo # first redshift bin
            j = comb_A[1]%N_tomo # second redshift bin
            m = comb_B[0]%N_tomo # first redshift bin
            n = comb_B[1]%N_tomo # second redshift bin                        
            
            c_im = np.argmax(np.product(np.sort(np.array([comb_A[0],comb_B[0]]))==all_combos,axis=1))
            c_jn = np.argmax(np.product(np.sort(np.array([comb_A[1],comb_B[1]]))==all_combos,axis=1))
            c_in = np.argmax(np.product(np.sort(np.array([comb_A[0],comb_B[1]]))==all_combos,axis=1))
            c_jm = np.argmax(np.product(np.sort(np.array([comb_A[1],comb_B[0]]))==all_combos,axis=1))

            # PAIRS A,B ARE (ti,tj),(tm,tn) at same ell
            #cov(ij,mn) = im,jn + in,jm                    
            C_im = CL_ALL[(c_im*N_ell):(c_im*N_ell)+N_ell]
            C_jn = CL_ALL[(c_jn*N_ell):(c_jn*N_ell)+N_ell]
            C_in = CL_ALL[(c_in*N_ell):(c_in*N_ell)+N_ell]
            C_jm = CL_ALL[(c_jm*N_ell):(c_jm*N_ell)+N_ell]

            
           # Knox formula
            Cov_ijmn = (C_im*C_jn+C_in*C_jm)/((2*ell+1.)*delta_ell*f_sky)
            
            if plot_for_sanity == True:
                if (c_A == c_B):
                    print(i,j,'=',m,n)
                    print(c_A)
                    Cl_err = np.sqrt(Cov_ijmn)

                    t_i = comb_A[0]//N_tomo
                    t_j = comb_A[1]//N_tomo
                    print(N_tomo*i+j+1)
                    
                    plt.subplot(N_tomo, N_tomo, N_tomo*i+j+1)
                    plt.title("z=%f x z=%f"%(z_bin_cents[i],z_bin_cents[j]))
                    c_ij = np.argmax(np.product(np.sort(np.array([comb_A[0],comb_A[1]]))==all_combos,axis=1))
                    C_ij = CL_ALL[(c_ij*N_ell):(c_ij*N_ell)+N_ell]
                    # maybe add legend
                    (_,caps,eb)=plt.errorbar(ell,C_ij,yerr=Cl_err,lw=2.,ls='-',capsize=5,label=str(t_i*2+t_j)) 
                    plt.legend()
                    plt.xscale('log')
                    plt.yscale('log')



            COV_ALL[(N_ell*c_A):(N_ell*c_A)+\
                    N_ell,(N_ell*c_B):(N_ell*c_B)+N_ell] = np.diag(Cov_ijmn)
            COV_ALL[(N_ell*c_B):(N_ell*c_B)+\
                    N_ell,(N_ell*c_A):(N_ell*c_A)+N_ell] = np.diag(Cov_ijmn)


            

    evals,evecs = la.eig(COV_ALL)
    
    if (is_pos_def(COV_ALL) != True): print("Covariance is not positive definite!"); exit(0)
    ICOV_ALL = la.inv(COV_ALL)
    return ICOV_ALL 

# ___________________________________
# COMPUTING FOR PARTICULAR COSMOLOGY
# ___________________________________

# Cosmological parameters
# Those 6 will be sampled
params={}
params['Omb']= {'val':0.0493,'dval':0.005,'label':'$\\Omega_b$','isfree':False}
params['Omc']= {'val':0.264,'dval':0.005,'label':'$\\Omega_c$','isfree':False}
params['Omk']   = {'val':0.0  ,'dval':0.02 ,'label':'$\\Omega_k$','isfree':False}
params['s8']  = {'val':0.8111   ,'dval':0.02 ,'label':'$\\sigma_8$'      ,'isfree':False}
params['h']  = {'val':0.6736   ,'dval':0.02 ,'label':'$\\h'      ,'isfree':False}
params['n_s']  = {'val':0.9649   ,'dval':0.02 ,'label':'$\\n_s$'      ,'isfree':False}

Omb = params['Omb']['val']
Omk = params['Omk']['val']
s8 = params['s8']['val']
h = params['h']['val']
n_s = params['n_s']['val']
Omc = params['Omc']['val']

    
# SETTING THE COSMOLOGY
FID_COSMO_PARAMS = {'Omega_b': Omb,
                    'Omega_k': Omk,
                    'sigma8': s8,
                    'h': h,
                    'n_s': n_s,
                    'Omega_c': Omc,
                    'transfer_function': 'boltzmann_class',
                    'matter_power_spectrum': 'halofit',
                    'mass_function': 'tinker10'
                    }
    
cosmo_fid = ccl.Cosmology(**FID_COSMO_PARAMS)

# load biases
a_s_cents_theo = 1./(1.+z_s_cents_theo)
b_zsamples_theo = 0.95/ccl.growth_factor(cosmo_fid,a_s_cents_theo)

# create the matrix
bz_data_theo = np.repeat(b_zsamples_theo.reshape(1,N_zsamples_theo),N_tomo,axis=0)
bz_data = np.zeros((N_tomo,N_zsamples))

for i in range(N_tomo):
    bz_data_theo[i,:] += .2*np.random.randn(N_zsamples_theo)
    # interpolating to nearest to pass to code
    if interp == 'nearest':
        f = interp1d(np.append(z_s_cents_theo,np.array([z_s_cents[0],z_s_cents[-1]])),np.append(bz_data_theo[i,:],np.array([bz_data_theo[i,0],bz_data_theo[i,-1]])),kind='nearest',bounds_error=0,fill_value=0.)
    else:
        print("STICK TO NEAREST FOR NOW"); exit(0)

    b_zsamples = f(z_s_cents)
    bz_data[i,:] = b_zsamples
    plt.plot(z_s_cents,b_zsamples)
    plt.plot(z_s_cents_theo,bz_data_theo[i,:])
plt.savefig('biases.png')
plt.close()
    
# matrices used for the fast calculation
b_mat = np.repeat(b_zsamples_theo.reshape(N_zsamples_theo,1),N_zsamples_theo,axis=1)
b2_mat = b_mat.T*b_mat # yes this is what i need
nob_mat = np.ones((N_zsamples_theo,N_zsamples_theo))

# when num differentiating
dndz_dval = 0.02
bz_dval = 0.02

for i in np.arange(N_tomo):
    for k in np.arange(N_zsamples):
        params['b_%02d_%02d'%(i,k)]  = {'val':bz_data[i,k]  ,'dval':0.02 ,'label':'$b_%02d_%02d$'%(i,k) ,'isfree':False}
        params['dndz_%02d_%02d'%(i,k)]  = {'val':dndz_data[i,k] ,'dval':0.02 ,'label':'dndz_%02d_%02d'%(i,k),'isfree':False}

Cl_true = compute_Cls(params)
# add noise
#R = np.random.normal(0.,1.,N_ell*N_tomo*(2*N_tomo+1))  
#X = la.cholesky(Cov).T                                                                      
#Nl = np.dot(X,R)                                                                            
# adding noise                                                                                   
#Cl_true += Nl

# _______________________________________________
#               REGULARIZATION PRIOR
# _______________________________________________

dndz_dval = 0.02
sigma0 = 0.000001
sigma1 = 0.2
sigma2 = 0.2
# This is regularization of first derivative
D1 = np.zeros((N_zsamples_theo*N_tomo,N_zsamples_theo*N_tomo))
# This ensures sum is 1
D0 = np.zeros((N_zsamples_theo*N_tomo,N_zsamples_theo*N_tomo))

lam0 = np.zeros(N_zsamples_theo*N_tomo)
lam = np.zeros(N_zsamples_theo*N_tomo)
for j in range(N_zsamples_theo*N_tomo):
    i_tomo = j//N_zsamples_theo
    i_zsam = j%N_zsamples_theo
    lam *= 0
    lam0 *= 0
    lam0[i_tomo*N_zsamples_theo:(i_tomo+1)*N_zsamples_theo]=1.

    if (i_zsam+1<N_zsamples_theo):
        ip1 = j+1
        lam[ip1] = -1./dndz_dval 
    if (i_zsam-1>=0):
        im1 = j-1
        lam[im1] = 1./dndz_dval
    tmp0 = lam0.reshape(N_zsamples_theo*N_tomo,1)
    tmp = lam.reshape(N_zsamples_theo*N_tomo,1)    

    D0 += np.dot(tmp0,tmp0.T)
    D1 += np.dot(tmp,tmp.T)

#print(D0)
D0 /= N_tomo**2#sigma0**2.
D1 /= sigma1**2.


D = D0#+D1


# ___________________________________
#          COMPUTING CURLY C
# ___________________________________
# linear shape
def D_i(x,x_i,Delta_x):
    D = np.zeros(len(x))
    x_sel = x[np.logical_and(x_i < x, x <= x_i+Delta_x)]
    D[np.logical_and(x_i < x, x <= x_i+Delta_x)] = 1.-((x_sel-x_i)/Delta_x)
    x_sel = x[np.logical_and(x_i >= x, x > x_i-Delta_x)]
    D[np.logical_and(x_i >= x, x > x_i-Delta_x)] = 1.-((x_i-x_sel)/Delta_x)

    sum_D = np.sum(D*(x[1]-x[0]))    
    #D /= sum_D # normalization doesn't matter
    return D

# nearest shape
def D_i_near(x,x_i,Delta_x):
    D = np.zeros(len(x))
    D[np.logical_and(x > x_i-Delta_x/2., x <= x_i+Delta_x/2.)] = 1.

    sum_D = np.sum(D*(x[1]-x[0]))
    #D /= sum_D # normalization doesn't matter
    return D 

# Computing curly C matrix entries for every N_t_s*(2 N_t_s+1)*N_ell = 1(2*1+1)*10 = 30 entries
# NOTE THAT N_t_s, i.e. N_tomo_single IS ALWAYS 1. THIS IS BECAUSE CURLY C IS INDEP OF TOMO BINS
def compute_mCs(par,i_zs,j_zs,z_cent=z_s_cents,k_arr=k_ar,z_arr=z_ar,a_arr=a_ar,ell=ells):    
    # Compute matter pk using halofit
    pk_mm_arr = np.array([ccl.nonlin_matter_power(cosmo_fid, k_arr, a) for a in a_arr])
    pk_gg_arr = pk_mm_arr.copy() # because we later include bias
    pk_gm_arr = pk_mm_arr.copy() # because we later include bias

    # Create pk2D objects for interpolation
    pk_mm = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_arr), pk_arr=np.log(pk_mm_arr), is_logp=True)
    pk_gg = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_arr), pk_arr=np.log(pk_gg_arr), is_logp=True)
    pk_gm = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_arr), pk_arr=np.log(pk_gm_arr), is_logp=True)

    # bias parameters should be set to 1
    b_z = np.ones(N_zsamples)
    
    # load the dndz parameters for curly C matrix element i_zs,j_zs
    dndz_z_single_i = np.zeros((N_tomo_single,N_zsamples))
    dndz_z_single_j = np.zeros((N_tomo_single,N_zsamples))
    # D_i_many is defined below
    for i_z in range(N_tomo_single):
        if interp == 'nearest':
            dndz_z_single_i[i_z,:] = D_i_many_near[i_zs,:]
            dndz_z_single_j[i_z,:] = D_i_many_near[j_zs,:]
        elif interp == 'linear':
            dndz_z_single_i[i_z,:] = D_i_many[i_zs,:]
            dndz_z_single_j[i_z,:] = D_i_many[j_zs,:]

    # per type gg gs ss
    tot_corr = N_ell*(N_tomo_single*(2*N_tomo_single+1))
    
    # make Cl_all of size N_ell*N_tomo_single*(2N_tomo_single+1)=30
    CL_ALL = np.zeros(tot_corr)
    temp = np.arange(2*N_tomo_single)
    temp = np.vstack((temp,temp)).T
    combs = np.array(list(combinations(range(2*N_tomo_single),2)))
    all_combos = np.vstack((temp,combs))

    for c, comb in enumerate(all_combos):
        # this is artifact from the other code -- here N_tomo_single is always 1
        i = comb[0]%N_tomo_single # first redshift bin (ALWAYS 0 HERE)
        j = comb[1]%N_tomo_single # second redshift bin (ALWAYS 0 HERE)
        t_i = comb[0]//N_tomo_single # tracer type 0 means g and 1 means s
        t_j = comb[1]//N_tomo_single # tracer type 0 means g and 1 means s

        
        # Now create corresponding Cls with pk2D objects matched to pk
        if t_i*2+t_j == 0: # this is gg
            tracer_z1 = ccl.NumberCountsTracer(cosmo_fid, bias=(z_cent, b_z), \
                                               dndz=(z_cent, dndz_z_single_i[i,:]),mag_bias=None, \
                                               has_rsd=False)
            tracer_z2 = ccl.NumberCountsTracer(cosmo_fid, bias=(z_cent, b_z), \
                                               dndz=(z_cent, dndz_z_single_j[j,:]),mag_bias=None, \
                                               has_rsd=False)

            cl_gg = ccl.angular_cl(cosmo_fid, tracer_z1, tracer_z2, ell, p_of_k_a=pk_gg)
            cl_gg_no = cl_gg + 0.
            CL_ALL[(N_ell*c):(N_ell*c)+N_ell] = cl_gg_no

        
        elif t_i*2+t_j == 1: # this is gs
            tracer_z1 = ccl.NumberCountsTracer(cosmo_fid, bias=(z_cent, b_z), \
                                               dndz=(z_cent, dndz_z_single_i[i,:]),mag_bias=None, \
                                               has_rsd=False)
            tracer_z2 = ccl.WeakLensingTracer(cosmo_fid, dndz=(z_cent, dndz_z_single_j[j,:]))
            cl_gs = ccl.angular_cl(cosmo_fid, tracer_z1, tracer_z2, ell, p_of_k_a=pk_gm)
            cl_gs_no = cl_gs
            CL_ALL[(N_ell*c):(N_ell*c)+N_ell] = cl_gs_no

        elif t_i*2+t_j == 3: # this is ss
            
            tracer_z1 = ccl.WeakLensingTracer(cosmo_fid, dndz=(z_cent, dndz_z_single_i[i,:]))
            tracer_z2 = ccl.WeakLensingTracer(cosmo_fid, dndz=(z_cent, dndz_z_single_j[j,:]))
            cl_ss = ccl.angular_cl(cosmo_fid, tracer_z1, tracer_z2, ell, p_of_k_a=pk_mm)
            cl_ss_no = cl_ss + 0.
            CL_ALL[(N_ell*c):(N_ell*c)+N_ell] = cl_ss_no
                
    return CL_ALL
            

# =====================================================
#       SUMMON curly C (comment out for fastness)
# =====================================================
'''
# generate the shapes
N_many = N_zsamples # THIS IS 100 TIMES N_ZSAMPLES_THEO
z_many = z_s_cents
Delta_z_s = np.mean(np.diff(z_s_cents_theo))


# Compute the many samples of the shapes - linear and nearest
D_i_many = np.zeros((N_zsamples_theo,N_many))
D_i_many_near = np.zeros((N_zsamples_theo,N_many))

for i in range(N_zsamples_theo):
    D_i_many[i,:] = D_i(z_many,z_s_cents_theo[i],Delta_z_s)
    D_i_many_near[i,:] = D_i_near(z_many,z_s_cents_theo[i],Delta_z_s)
    plt.plot(z_many, D_i_many[i,:], label='linear z = %f'%(z_s_cents_theo[i]))
    plt.plot(z_many, D_i_many_near[i,:], label='nearest z = %f'%(z_s_cents_theo[i]))


plt.legend()
plt.xlabel("z", fontsize=14)
plt.ylabel("p(z)", fontsize=14)
plt.savefig("mD.png")
plt.close()

# CURLY C INDEPENDENT OF REDSHIFT BIN
# When computing, set the N_tomo_single to 1
N_tomo_single = 1

N_elm = (2*N_tomo_single+1)*N_tomo_single*N_ell # = 30 for gg, gs, ss
mat_C = np.zeros((N_zsamples_theo,N_zsamples_theo,N_elm))
# NOTE THAT curly C, i.e. mat_C is not symmetric

for i in range(N_zsamples_theo):
    for j in range(N_zsamples_theo):
        print(i,j)
        # In the function the dndz parameters are updated to the D_i_many guys
        Cl_many = compute_mCs(params,i_zs=i,j_zs=j)
        print(Cl_many.sum())
        # record matrix entries for this choise of zsample bins i and j
        mat_C[i,j,:] = Cl_many 
        
# Save curly C matrix
np.save("mat_C_"+str(tomo)+"_"+str(zsam)+".npy",mat_C)
'''
# ___________________________________
#  LOADING CURLY C AND ADDING NOISE
# ___________________________________

def compute_fast_Cls(dndz_z_curr,mat_cC,b_z_curr=np.zeros(5),ell=ells,compute_ders=False,compute_2nd_ders=False,compute_cov=False,plot_for_sanity=False):
    # After obtaining the mCs we can now do simple linalg to get the Cls
    # check if it is the correct shape
    if np.sum(b_z_curr) == 0: complex_bz = False
    else: complex_bz = True
    if dndz_z_curr.shape[0] != N_tomo: dndz_z_curr = dndz_z_curr.reshape(N_tomo,N_zsamples_theo)
    
    # correlation number for all types gg gs ss and all combinations of redshift
    # THIS N_TOMO IS NOW THE FULL NUMBER OF TOMOGRAPHIC BINS
    tot_corr = N_ell*(N_tomo*(2*N_tomo+1))
    
    # make Cl_fast of size N_ell*N_tomo*(2N_tomo+1) as this is the tot no. of corrs
    Cl_fast_all = np.zeros(tot_corr)
    # We would have the same number of parameters for dndz and bz
    if complex_bz == False:
        b_z_curr = np.ones_like(dndz_z_curr)
        dCl_fast_all = np.zeros((N_tomo*N_zsamples_theo,tot_corr))
    else:
        if b_z_curr.shape[0] != N_tomo: b_z_curr = b_z_curr.reshape(N_tomo,N_zsamples_theo)
        dCl_fast_all = np.zeros((2*N_tomo*N_zsamples_theo,tot_corr))

    ddCl_fast_all = np.zeros((N_tomo*N_zsamples_theo,N_tomo*N_zsamples_theo,tot_corr))
    temp = np.arange(2*N_tomo)
    temp = np.vstack((temp,temp)).T
    combs = np.array(list(combinations(range(2*N_tomo),2)))
    all_combos = np.vstack((temp,combs))

    for c, comb in enumerate(all_combos):
        i_tomo = comb[0]%N_tomo # first redshift bin
        j_tomo = comb[1]%N_tomo # second redshift bin
        t_i = comb[0]//N_tomo # tracer type 0 means g and 1 means s
        t_j = comb[1]//N_tomo # tracer type 0 means g and 1 means s
    
        # Noise term
        if (i_tomo == j_tomo):
            # number density of galaxies
            N_gal = N_gal_bin[i_tomo]
            n_gal = N_gal/area_COSMOS # in rad^-2 
            # computing noise
            noise_gal = 1./n_gal
            noise_shape = sigma_e2[i_tomo]/n_gal
        else:
            noise_gal = 0.
            noise_shape = 0.

        # for each combination c, the correlation CL has N_ell = 10 entries
        CL = np.zeros(N_ell)
        bias_mat = np.ones_like(b_mat)
        ni = dndz_z_curr[i_tomo,:].reshape(1,N_zsamples_theo)
        nj = dndz_z_curr[j_tomo,:].reshape(N_zsamples_theo,1)
        
              
        if t_i*2+t_j == 0: # this is gg
            type_xy = 0 # type_xy is such BECAUSE THE ORDER IN mat_cC[.,.,k] is
            # gg, ss, gs (from comb=(0,0),(1,1),(0,1) and c = 0,1,2)
            noise = noise_gal
            if complex_bz == False: bias_mat = b2_mat; di = ni; dj = nj
            else:
                bi = b_z_curr[i_tomo,:].reshape(1,N_zsamples_theo)
                bj = b_z_curr[j_tomo,:].reshape(N_zsamples_theo,1)
                bi_f = np.ones_like(bi); bj_f = np.ones_like(bj);
                ni_f = np.ones_like(bi); nj_f = np.ones_like(bj);

                # used for Cl computation and no complex bias
                di = ni*bi; dj = nj*bj
                
        if t_i*2+t_j == 1: # this is gs
            type_xy = 2 # type_xy is such BECAUSE THE ORDER IN mat_cC[.,.,k] is
            # gg, ss, gs (from comb=(0,0),(1,1),(0,1) and c = 0,1,2)
            noise = 0.
            if complex_bz == False: bias_mat = b_mat; di = ni; dj = nj
            else:
                bi = b_z_curr[i_tomo,:].reshape(1,N_zsamples_theo)
                bj = np.ones((N_zsamples_theo,1))
                bi_f = np.ones_like(bi); bj_f = np.zeros_like(bj);
                ni_f = np.ones_like(bi); nj_f = np.ones_like(bj);

                # used for Cl computation and no complex bias
                di = ni*bi; dj = nj
            
                
        if t_i*2+t_j == 3: # this is ss
            type_xy = 1 # type_xy is such BECAUSE THE ORDER IN mat_cC[.,.,k] is
            # gg, ss, gs (from comb=(0,0),(1,1),(0,1) and c = 0,1,2)
            noise = noise_shape
            if complex_bz == False: bias_mat = nob_mat; di = ni; dj = nj
            else:
                bi = np.ones((1,N_zsamples_theo))
                bj = np.ones((N_zsamples_theo,1))
                bi_f = np.zeros_like(bi); bj_f = np.zeros_like(bj);
                ni_f = np.ones_like(bi); nj_f = np.ones_like(bj);

                # used for Cl computation and no complex bias
                di = ni; dj = nj
                
        for k in range(N_ell):
            # Here we compute the Cls analytically using the curly C matrix # bias mat is ones for complex bias
            matC_k = mat_cC[:,:,N_ell*type_xy+k]*bias_mat
            
            CL[k] = np.dot(np.dot(di,matC_k),dj)
            if (compute_ders == True and complex_bz==False):
                if (i_tomo == j_tomo):
                    dCl_fast_all[i_tomo*N_zsamples_theo:(i_tomo+1)*N_zsamples_theo,(N_ell*c)+k] = (np.dot(matC_k,dj)).T+(np.dot(di,matC_k)) # can be dj.T
                else:
                    dCl_fast_all[i_tomo*N_zsamples_theo:(i_tomo+1)*N_zsamples_theo,(N_ell*c)+k] = (np.dot(matC_k,dj)).T
                    dCl_fast_all[j_tomo*N_zsamples_theo:(j_tomo+1)*N_zsamples_theo,(N_ell*c)+k] = (np.dot(di,matC_k))
                print(dCl_fast_all.shape)
                print("we here")

            if (compute_ders == True and complex_bz==True):
                # n der: ba na_f Caj bj nj + bi ni Cia na_f ba
                # b der: bi_f ni Caj bj nj + bj nj Cja na ba_f 
                if (i_tomo == j_tomo):
                    dCl_fast_all[i_tomo*N_zsamples_theo:(i_tomo+1)*N_zsamples_theo,(N_ell*c)+k] = np.dot(matC_k,bj*nj).T*(ni_f*bi)+np.dot(bi*ni,matC_k)*(nj_f*bj).T
                    dCl_fast_all[N_tomo*N_zsamples_theo+i_tomo*N_zsamples_theo:N_tomo*N_zsamples_theo+(i_tomo+1)*N_zsamples_theo,(N_ell*c)+k] = np.dot(matC_k,bj*nj).T*(ni*bi_f)+np.dot(bi*ni,matC_k)*(nj*bj_f).T
                else:
                    dCl_fast_all[i_tomo*N_zsamples_theo:(i_tomo+1)*N_zsamples_theo,(N_ell*c)+k] = np.dot(matC_k,bj*nj).T*(ni_f*bi)
                    dCl_fast_all[N_tomo*N_zsamples_theo+i_tomo*N_zsamples_theo:N_tomo*N_zsamples_theo+(i_tomo+1)*N_zsamples_theo,(N_ell*c)+k] = np.dot(matC_k,bj*nj).T*(ni*bi_f)
                    dCl_fast_all[j_tomo*N_zsamples_theo:(j_tomo+1)*N_zsamples_theo,(N_ell*c)+k] = np.dot(bi*ni,matC_k)*(nj_f*bj).T
                    dCl_fast_all[N_tomo*N_zsamples_theo+j_tomo*N_zsamples_theo:N_tomo*N_zsamples_theo+(j_tomo+1)*N_zsamples_theo,(N_ell*c)+k] = np.dot(bi*ni,matC_k)*(nj*bj_f).T
                    
            if (compute_2nd_ders == True and complex_bz==False): # NOT WORKING PERFECTLY
                if (i_tomo == j_tomo):
                    ddCl_fast_all[i_tomo*N_zsamples_theo:(i_tomo+1)*N_zsamples_theo,i_tomo*N_zsamples_theo:(i_tomo+1)*N_zsamples_theo,(N_ell*c)+k] = matC_k*2
                else:
                    ddCl_fast_all[i_tomo*N_zsamples_theo:(i_tomo+1)*N_zsamples_theo,j_tomo*N_zsamples_theo:(j_tomo+1)*N_zsamples_theo,(N_ell*c)+k] = matC_k
                    ddCl_fast_all[j_tomo*N_zsamples_theo:(j_tomo+1)*N_zsamples_theo,i_tomo*N_zsamples_theo:(i_tomo+1)*N_zsamples_theo,(N_ell*c)+k] = matC_k
            if (compute_2nd_ders == True and complex_bz==True): # NOT WORKING PERFECTLY # I think the ders are nor symmetric ,ab=/=,ba
                # C^ij,b^a n^b der: bf^a_m na_m C_mn b^b_n nf^b_n delta^ia delta^jb + b^b_m nf^b_m C_mn bf^a_n na_n delta^ib delta^ja
                # + b^i_m n^i_m C_mn bf^a_n nf^a_n delta^ab delta^ia delta^jb + bf^a_m nf^a_m C_mn b^j_n n^j_n delta^ab delta^ia delta^jb
                if (i_tomo == j_tomo):
                    # nn
                    ddCl_fast_all[i_tomo*N_zsamples_theo:(i_tomo+1)*N_zsamples_theo,i_tomo*N_zsamples_theo:(i_tomo+1)*N_zsamples_theo,(N_ell*c)+k] = (bi_f*ni_f).T*matC_k*2
                    # bb
                    ddCl_fast_all[N_tomo*N_zsamples_theo+i_tomo*N_zsamples_theo:N_tomo*N_zsamples_theo+(i_tomo+1)*N_zsamples_theo,N_tomo*N_zsamples_theo+i_tomo*N_zsamples_theo:N_tomo*N_zsamples_theo+(i_tomo+1)*N_zsamples_theo,(N_ell*c)+k] = matC_k*2
                else:
                    # nn
                    ddCl_fast_all[i_tomo*N_zsamples_theo:(i_tomo+1)*N_zsamples_theo,j_tomo*N_zsamples_theo:(j_tomo+1)*N_zsamples_theo,(N_ell*c)+k] = (bi*ni_f).T*matC_k*(bj*nj_f).T
                    # bb
                    ddCl_fast_all[N_tomo*N_zsamples_theo+j_tomo*N_zsamples_theo:N_tomo*N_zsamples_theo+(j_tomo+1)*N_zsamples_theo,N_tomo*N_zsamples_theo+i_tomo*N_zsamples_theo:N_tomo*N_zsamples_theo+(i_tomo+1)*N_zsamples_theo,(N_ell*c)+k] = (bi_f*ni).T*matC_k*(bj_f*nj).T
                    # bn
                    ddCl_fast_all[N_tomo*N_zsamples_theo+j_tomo*N_zsamples_theo:N_tomo*N_zsamples_theo+(j_tomo+1)*N_zsamples_theo,i_tomo*N_zsamples_theo:(i_tomo+1)*N_zsamples_theo,(N_ell*c)+k] = (bi_f*ni).T*matC_k*(bj*nj_f).T
                    # nb
                    ddCl_fast_all[j_tomo*N_zsamples_theo:(j_tomo+1)*N_zsamples_theo,N_tomo*N_zsamples_theo+i_tomo*N_zsamples_theo:N_tomo*N_zsamples_theo+(i_tomo+1)*N_zsamples_theo,(N_ell*c)+k] = (bi*ni_f).T*matC_k*(bj*nj_f).T
        # Finally add noise depending on type of correlation
        CL += noise
        # This is the usual, proven way of recording the Cls
        Cl_fast_all[(N_ell*c):(N_ell*c)+N_ell] = CL

    if (compute_ders == False and compute_2nd_ders == False and compute_cov == False):
        return Cl_fast_all
    elif (compute_ders == True and compute_2nd_ders == False and compute_cov == False):
        return Cl_fast_all, dCl_fast_all
    elif (compute_ders == False and compute_2nd_ders == True and compute_cov == False):
        return Cl_fast_all, ddCl_fast_all
    elif (compute_cov == False):
        return Cl_fast_all, Cl_fast_all, ddCl_fast_all

    print(len(Cl_fast_all))
    
    Cov_fast_all = np.zeros((len(Cl_fast_all),len(Cl_fast_all)))    
    # COMPUTE COVARIANCE MATRIX 
    for c_A, comb_A in enumerate(all_combos):
        for c_B, comb_B in enumerate(all_combos):
            i = comb_A[0]%N_tomo # first redshift bin
            j = comb_A[1]%N_tomo # second redshift bin
            m = comb_B[0]%N_tomo # first redshift bin
            n = comb_B[1]%N_tomo # second redshift bin                        
            
            c_im = np.argmax(np.product(np.sort(np.array([comb_A[0],comb_B[0]]))==all_combos,axis=1))
            c_jn = np.argmax(np.product(np.sort(np.array([comb_A[1],comb_B[1]]))==all_combos,axis=1))
            c_in = np.argmax(np.product(np.sort(np.array([comb_A[0],comb_B[1]]))==all_combos,axis=1))
            c_jm = np.argmax(np.product(np.sort(np.array([comb_A[1],comb_B[0]]))==all_combos,axis=1))

            # PAIRS A,B ARE (ti,tj),(tm,tn) at same ell
            #cov(ij,mn) = im,jn + in,jm                    
            C_im = Cl_fast_all[(c_im*N_ell):(c_im*N_ell)+N_ell]
            C_jn = Cl_fast_all[(c_jn*N_ell):(c_jn*N_ell)+N_ell]
            C_in = Cl_fast_all[(c_in*N_ell):(c_in*N_ell)+N_ell]
            C_jm = Cl_fast_all[(c_jm*N_ell):(c_jm*N_ell)+N_ell]

            
            # Knox formula
            Cov_ijmn = (C_im*C_jn+C_in*C_jm)/((2*ell+1.)*delta_ell*f_sky)
            
            if plot_for_sanity == True:
                if (c_A == c_B):
                    print(i,j,'=',m,n)
                    print(c_A)
                    Cl_err = np.sqrt(Cov_ijmn)

                    t_i = comb_A[0]//N_tomo
                    t_j = comb_A[1]//N_tomo
                    print(N_tomo*i+j+1)
                    
                    plt.subplot(N_tomo, N_tomo, N_tomo*i+j+1)
                    plt.title("z=%f x z=%f"%(z_bin_cents[i],z_bin_cents[j]))
                    c_ij = np.argmax(np.product(np.sort(np.array([comb_A[0],comb_A[1]]))==all_combos,axis=1))
                    C_ij = Cl_fast_all[(c_ij*N_ell):(c_ij*N_ell)+N_ell]
                    # maybe add legend
                    (_,caps,eb)=plt.errorbar(ell,C_ij,yerr=Cl_err,lw=2.,ls='-',capsize=5,label=str(t_i*2+t_j)) 
                    plt.legend()
                    plt.xscale('log')
                    plt.yscale('log')



            Cov_fast_all[(N_ell*c_A):(N_ell*c_A)+\
                    N_ell,(N_ell*c_B):(N_ell*c_B)+N_ell] = np.diag(Cov_ijmn)
            Cov_fast_all[(N_ell*c_B):(N_ell*c_B)+\
                    N_ell,(N_ell*c_A):(N_ell*c_A)+N_ell] = np.diag(Cov_ijmn)


            

    evals,evecs = la.eig(Cov_fast_all)
    
    if (is_pos_def(Cov_fast_all) != True): print("Covariance is not positive definite!"); exit(0)

    if (compute_ders == False and compute_2nd_ders == False):
        return Cl_fast_all, Cov_fast_all
    elif (compute_ders == True and compute_2nd_ders == False):
        return Cl_fast_all, dCl_fast_all, Cov_fast_all
    elif (compute_ders == False and compute_2nd_ders == True):
        return Cl_fast_all, ddCl_fast_all, Cov_fast_all
    else:
        return Cl_fast_all, Cl_fast_all, ddCl_fast_all, Cov_fast_all
    


# Load the curly C
mat_C = np.load("mat_C_"+str(tomo)+"_"+str(zsam)+".npy")


#_____________________________________________
#            TESTING B_Z DERS
#_____________________________________________

# NUMERICAL DERIVATIVES
def compute_num_derivs(ind_j,dval):
    dndz_z = dndz_data_theo.copy()
    b_z = bz_z.copy()
    k = ind_j % N_zsamples_theo # gives sample
    i = ind_j // N_zsamples_theo # gives tomo

    print('tomo_i, zsam_k = ',i,k)
    dndz_ik = dndz_z[i,k]
    dndz_z[i,k] = dndz_ik+dval
    clp = compute_fast_Cls(dndz_z,mat_C,b_z)
    dndz_z[i,k] = dndz_ik-dval
    clm = compute_fast_Cls(dndz_z,mat_C,b_z)
    return (clp-clm)/(2*dval)

def compute_num_derivs_bz(ind_j,dval):
    dndz_z = dndz_data_theo.copy()
    b_z = bz_z.copy()
    k = ind_j % N_zsamples_theo # gives sample
    i = ind_j // N_zsamples_theo # gives tomo

    print('tomo_i, zsam_k = ',i,k)
    bz_ik = b_z[i,k]
    b_z[i,k] = bz_ik+dval
    clp = compute_fast_Cls(dndz_z,mat_C,b_z)
    b_z[i,k] = bz_ik-dval
    clm = compute_fast_Cls(dndz_z,mat_C,b_z)
    return (clp-clm)/(2*dval)

def compute_num_2nd_derivs(ind_j1,ind_j2,dval):
    dndz_z = dndz_data_theo.copy()
    k1 = ind_j1 % N_zsamples_theo # gives sample
    i1 = ind_j1 // N_zsamples_theo # gives tomo
    k2 = ind_j2 % N_zsamples_theo # gives sample
    i2 = ind_j2 // N_zsamples_theo # gives tomo

    # original value
    dndz_ik1 = dndz_z[i1,k1]
    
    dndz_z[i1,k1] = dndz_ik1+dval
    dndz_ik2 = dndz_z[i2,k2]
    dndz_z[i2,k2] = dndz_ik2+dval
    clp1p2 = compute_fast_Cls(dndz_z,mat_C)
    dndz_z[i2,k2] = dndz_ik2-dval
    clp1m2 = compute_fast_Cls(dndz_z,mat_C)

    dndz_z[i1,k1] = dndz_ik1-dval
    dndz_ik2 = dndz_z[i2,k2]
    dndz_z[i2,k2] = dndz_ik2+dval
    clm1p2 = compute_fast_Cls(dndz_z,mat_C)
    dndz_z[i2,k2] = dndz_ik2-dval
    clm1m2 = compute_fast_Cls(dndz_z,mat_C)

    return ((clp1p2-clp1m2)-(clm1p2-clm1m2))/(4.*dval**2)

mat_C = np.load("mat_C_"+str(tomo)+"_"+str(zsam)+".npy")
temp = np.arange(2*N_tomo)
temp = np.vstack((temp,temp)).T
combs = np.array(list(combinations(range(2*N_tomo),2)))
all_combos = np.vstack((temp,combs))

dndz_z = dndz_data_theo.copy()
bz_z = bz_data_theo.copy()

# Computing the Cls
Cl_fast = compute_fast_Cls(dndz_z,mat_C,bz_z)

# PLOTTING 
fig = plt.figure(figsize=(30,25))
for c, comb in enumerate(all_combos):
    i_tomo = comb[0]%N_tomo # first redshift bin
    j_tomo = comb[1]%N_tomo # second redshift bin
    t_i = comb[0]//N_tomo # tracer type 0 means g and 1 means s
    t_j = comb[1]//N_tomo # tracer type 0 means g and 1 means s
    
    Cf = Cl_fast[(c*N_ell):(c*N_ell)+N_ell]
    Ct = Cl_true[(c*N_ell):(c*N_ell)+N_ell]
    
    if (t_i*2+t_j == 0): int_t='gg'; c='red'
    if (t_i*2+t_j == 1): int_t='gs'; c='purple'
    if (t_i*2+t_j == 3): int_t='ss'; c='blue'
    # maybe add legend and make sure colors are the same for each type

    plt.subplot(N_tomo, N_tomo, N_tomo*i_tomo+j_tomo+1)                                                                                          
    plt.title("z=%f x z=%f"%(z_bin_cents[i_tomo],z_bin_cents[j_tomo]))                                                                           
    plt.plot(ells,Cf,lw=2.,ls='--',color=c,label=int_t+' fast')
    plt.plot(ells,Ct,lw=2.,ls='-',color=c,label=int_t+' true',alpha=0.4)

    print(np.mean(1-Cf/Ct))
    
    plt.legend()                                                                                                                       
    plt.xscale('log')                                                                                                                  
    plt.yscale('log')
plt.savefig("Cl_all.pdf")

complex_bz = True
if complex_bz == False:
    Cl_fast, dCldp_fast_alpha = compute_fast_Cls(dndz_z,mat_C,b_z,compute_ders=True)
    dCldp_num_alpha = np.zeros((N_tomo*N_zsamples_theo,int(N_tomo*(2*N_tomo+1)*N_ell)))
else:
    Cl_fast, dCldp_fast_alpha = compute_fast_Cls(dndz_z,mat_C,bz_z,compute_ders=True)
    dCldp_num_alpha = np.zeros((2*N_tomo*N_zsamples_theo,int(N_tomo*(2*N_tomo+1)*N_ell)))
for j in np.arange(N_tomo*N_zsamples_theo):
    dCldp_num_alpha[j,:] = compute_num_derivs(j,dndz_dval)
if complex_bz == True:
    for j in np.arange(N_tomo*N_zsamples_theo):
        dCldp_num_alpha[N_tomo*N_zsamples_theo+j,:] = compute_num_derivs_bz(j,bz_dval)
        
#MORE PLOTS
if complex_bz == False: N_pars = N_tomo*N_zsamples_theo
else: N_pars = N_tomo*N_zsamples_theo*2

for N_plot in range(N_pars):
    fig = plt.figure(figsize=(30,25))
    bz = N_plot//(N_tomo*N_zsamples_theo) # if 0 then we are not varying the bz's
    N_tz = N_plot%(N_tomo*N_zsamples_theo)
        
    for c, comb in enumerate(all_combos):
        i_tomo = comb[0]%N_tomo # first redshift bin
        j_tomo = comb[1]%N_tomo # second redshift bin
        t_i = comb[0]//N_tomo # tracer type 0 means g and 1 means s
        t_j = comb[1]//N_tomo # tracer type 0 means g and 1 means s
        
        
        dCt = dCldp_num_alpha[N_plot,(c*N_ell):(c*N_ell)+N_ell]
        dCf = dCldp_fast_alpha[N_plot,(c*N_ell):(c*N_ell)+N_ell]
        if (t_i*2+t_j == 0): int_t='gg'; c='red'
        if (t_i*2+t_j == 1): int_t='gs'; c='purple'
        if (t_i*2+t_j == 3): int_t='ss'; c='blue'
        # maybe add legend and make sure colors are the same for each type
        
        plt.subplot(N_tomo, N_tomo, N_tomo*i_tomo+j_tomo+1)                                                                                          
        plt.title("z=%f x z=%f"%(z_bin_cents[i_tomo],z_bin_cents[j_tomo]))                                                                           
        plt.plot(ells,np.abs(dCf),lw=2.,ls='--',color=c,label=int_t+' fast')
        plt.plot(ells,dCt,lw=2.,ls='-',color=c,label=int_t+' num',alpha=0.4)
        print("diff = ",np.mean(1-dCf/dCt))
        print("dCf = ",np.mean(dCf))
        print("dCt = ",np.mean(dCt))
        plt.legend()                                                                           
        plt.xscale('log')
        plt.yscale('log')
        
    tomo = N_tz//N_zsamples_theo
    zsamp = N_tz%N_zsamples_theo
    if bz == 0:
        plt.savefig("dCl_all_n_tomo_"+str(tomo)+"_zsamp_"+str(zsamp)+".pdf")
    else: plt.savefig("dCl_all_b_tomo_"+str(tomo)+"_zsamp_"+str(zsamp)+".pdf")
