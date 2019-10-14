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
from scipy.optimize import minimize
from scipy.optimize import curve_fit
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

def compute_Cls(par,hod_par=hod_params,z_cent=z_s_cents,N_gal_sample=N_gal_bin,k_arr=k_ar,z_arr=z_ar,a_arr=a_ar,ell=ells,
                compute_cov=False, compute_inv_cov=False,plot_for_sanity=False,powerspec='halofit',simple_bias=True):
    
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
    b_z = np.array([par['b_%02d'%k]['val'] for k in range(N_zsamples)])
    
    # load the dndz parameters
    dndz_z = np.zeros((N_tomo,N_zsamples))
    for i_z in range(N_tomo):
        dndz_z[i_z,:] = np.array([par['dndz_%02d_%02d'%(i_z,k)]['val'] for k in range(N_zsamples)])
    

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
            tracer_z1 = ccl.NumberCountsTracer(cosmo_fid, bias=(z_cent, b_z), \
                                               dndz=(z_cent, dndz_z[i,:]),mag_bias=None, \
                                               has_rsd=False)
            tracer_z2 = ccl.NumberCountsTracer(cosmo_fid, bias=(z_cent, b_z), \
                                               dndz=(z_cent, dndz_z[j,:]),mag_bias=None, \
                                               has_rsd=False)

            cl_gg = ccl.angular_cl(cosmo_fid, tracer_z1, tracer_z2, ell, p_of_k_a=pk_gg)
            cl_gg_no = cl_gg + noise_gal

            CL_ALL[(N_ell*c):(N_ell*c)+N_ell] = cl_gg_no
        elif t_i*2+t_j == 1: # this is gs
            
            tracer_z1 = ccl.NumberCountsTracer(cosmo_fid, bias=(z_cent, b_z), \
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

    if compute_inv_cov == False and compute_cov == False:
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


    if compute_cov:
        return COV_ALL

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

# matrices used for the fast calculation
b_mat = np.repeat(b_zsamples_theo.reshape(N_zsamples_theo,1),N_zsamples_theo,axis=1)
b2_mat = b_mat.T*b_mat # yes this is what i need
nob_mat = np.ones((N_zsamples_theo,N_zsamples_theo))

# interpolating to nearest to pass to code
if interp == 'nearest':
    f = interp1d(np.append(z_s_cents_theo,np.array([z_s_cents[0],z_s_cents[-1]])),np.append(b_zsamples_theo,np.array([b_zsamples_theo[0],b_zsamples_theo[-1]])),kind='nearest',bounds_error=0,fill_value=0.)
else:
    print("STICK TO NEAREST FOR NOW")

b_zsamples = f(z_s_cents)
plt.plot(z_s_cents,b_zsamples)
plt.plot(z_s_cents_theo,b_zsamples_theo)
plt.savefig('biases.png')
plt.close()

for i in np.arange(N_tomo):
    for k in np.arange(N_zsamples):
        params['b_%02d'%k]  = {'val':b_zsamples[k]  ,'dval':0.02 ,'label':'$b_%02d$'%k ,'isfree':False}
        params['dndz_%02d_%02d'%(i,k)]  = {'val':dndz_data[i,k] ,'dval':0.1 ,'label':'dndz_%02d_%02d'%(i,k),'isfree':False}

Cl_true = compute_Cls(params)
# add noise
Cov=compute_Cls(params,compute_cov=True)
R = np.random.normal(0.,1.,N_ell*N_tomo*(2*N_tomo+1))  
X = la.cholesky(Cov).T                                                                      
Nl = np.dot(X,R)                                                                            
# adding noise                                                                                   
Cl_true += Nl

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

    print (tmp0)
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

cov_fname="mat_C_"+str(tomo)+"_"+str(zsam)+".npy"
if not os.path.isfile(cov_fname):
    print ("Generating ",cov_fname)
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

# ___________________________________
#  LOADING CURLY C AND ADDING NOISE
# ___________________________________

def compute_fast_Cls(dndz_z_curr,mat_cC,ell=ells,compute_ders=False,compute_2nd_ders=False,compute_cov=False,plot_for_sanity=False):
    # After obtaining the mCs we can now do simple linalg to get the Cls
    # check if it is the correct shape
    if dndz_z_curr.shape[0] != N_tomo: dndz_z_curr = dndz_z_curr.reshape(N_tomo,N_zsamples_theo)
    
    # correlation number for all types gg gs ss and all combinations of redshift
    # THIS N_TOMO IS NOW THE FULL NUMBER OF TOMOGRAPHIC BINS
    tot_corr = N_ell*(N_tomo*(2*N_tomo+1))
    
    # make Cl_fast of size N_ell*N_tomo*(2N_tomo+1) as this is the tot no. of corrs
    Cl_fast_all = np.zeros(tot_corr)
    dCl_fast_all = np.zeros((N_tomo*N_zsamples_theo,tot_corr))
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
        for k in range(N_ell):
            if t_i*2+t_j == 0: # this is gg
                type_xy = 0 # type_xy is such BECAUSE THE ORDER IN mat_cC[.,.,k] is
                # gg, ss, gs (from comb=(0,0),(1,1),(0,1) and c = 0,1,2)
                noise = noise_gal
                bias_mat = b2_mat
            if t_i*2+t_j == 1: # this is gs
                type_xy = 2 # type_xy is such BECAUSE THE ORDER IN mat_cC[.,.,k] is
                # gg, ss, gs (from comb=(0,0),(1,1),(0,1) and c = 0,1,2)
                noise = 0.
                bias_mat = b_mat # first row b0,b0,b0; second row b1,b1,b1
            if t_i*2+t_j == 3: # this is ss
                type_xy = 1 # type_xy is such BECAUSE THE ORDER IN mat_cC[.,.,k] is
                # gg, ss, gs (from comb=(0,0),(1,1),(0,1) and c = 0,1,2)
                noise = noise_shape
                bias_mat = nob_mat
            # Here we compute the Cls analytically using the curly C matrix
            di = dndz_z_curr[i_tomo,:].reshape(1,N_zsamples_theo)
            dj = dndz_z_curr[j_tomo,:].reshape(N_zsamples_theo,1)
            matC_k = mat_cC[:,:,N_ell*type_xy+k]*bias_mat
            
            CL[k] = np.dot(np.dot(di,matC_k),dj)
            if (compute_ders == True):
                if (i_tomo == j_tomo):
                    dCl_fast_all[i_tomo*N_zsamples_theo:(i_tomo+1)*N_zsamples_theo,(N_ell*c)+k] = (np.dot(matC_k,dj)).T+(np.dot(di,matC_k)) # can be dj.T
                else:
                    dCl_fast_all[i_tomo*N_zsamples_theo:(i_tomo+1)*N_zsamples_theo,(N_ell*c)+k] = (np.dot(matC_k,dj)).T
                    dCl_fast_all[j_tomo*N_zsamples_theo:(j_tomo+1)*N_zsamples_theo,(N_ell*c)+k] = (np.dot(di,matC_k))
            if (compute_2nd_ders == True): # NOT WORKING PERFECTLY
                if (i_tomo == j_tomo):
                    ddCl_fast_all[i_tomo*N_zsamples_theo:(i_tomo+1)*N_zsamples_theo,i_tomo*N_zsamples_theo:(i_tomo+1)*N_zsamples_theo,(N_ell*c)+k] = matC_k+matC_k.T
                else:
                    ddCl_fast_all[i_tomo*N_zsamples_theo:(i_tomo+1)*N_zsamples_theo,j_tomo*N_zsamples_theo:(j_tomo+1)*N_zsamples_theo,(N_ell*c)+k] = matC_k
                    ddCl_fast_all[j_tomo*N_zsamples_theo:(j_tomo+1)*N_zsamples_theo,i_tomo*N_zsamples_theo:(i_tomo+1)*N_zsamples_theo,(N_ell*c)+k] = matC_k#.T
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

# ____________________________________________
#                 INITIAL GUESS
# ____________________________________________


# Might wanna use the other Cl_true which does not use the approximation
cl_true, dCldp_true, Cov_true = compute_fast_Cls(dndz_data_theo,mat_C,compute_ders=True,compute_2nd_ders=False,compute_cov=True)
iCov_true = la.inv(Cov_true)

# In case you ever need fisher
fisher = np.dot(dCldp_true,np.dot(iCov_true,dCldp_true.T))
N_elm = len(Cl_true)
Cl_true = Cl_true.reshape(N_elm,1)

# Initiatl guess
full_x = dndz_data_theo.flatten().copy()
gauss_guess = 1
if gauss_guess == 1:
    for i in range(N_tomo):
        dndz_guess = gaussian(z_s_cents_theo,z_bin_cents[i],0.2)
        dndz_guess /= np.sum(dndz_guess)
        full_x[i*N_zsamples_theo:(i+1)*N_zsamples_theo] = dndz_guess
full_x0 = full_x.copy()
# ____________________________________________
#                 NEWTON-RAPHSON
# ____________________________________________


for s in range(steps):    
    print("Delta_dndz = ",np.sqrt(np.sum((dndz_data_theo.flatten()-full_x)**2)))

    # compute the Cls and their derivatives
    Cl_fast, dCldp_fast, Cov_fast = compute_fast_Cls(full_x,mat_C,compute_ders=True,compute_2nd_ders=False,compute_cov=True)

    Cl_fast = Cl_fast.reshape(N_elm,1)

    iCov_fast = la.inv(Cov_fast)

    Delta_Cl = Cl_fast-Cl_true

    # extra regularization term
    # TESTING
    n_diff = (full_x).reshape(N_zsamples_theo*N_tomo,1)
    Reg_V = np.dot(D,n_diff)-1./N_tomo
    Reg_A = D
    reg = 0#1.e14
    
    print("sum_dndz = ",np.sum(full_x))
    print("chi2_reg = ",np.dot(full_x.T,np.dot(Reg_A,full_x))-2*np.sum(full_x)/N_tomo+1)
    
    A = np.dot(dCldp_fast,np.dot(iCov_fast,dCldp_fast.T)) + reg*Reg_A
    V = np.dot(dCldp_fast,np.dot(iCov_fast,Delta_Cl)) + reg*Reg_V
    
    chi2 = np.dot(Delta_Cl.T,np.dot(iCov_fast,Delta_Cl))[0][0]
    print("chi2 = ",chi2)
    print("__________________________")
    iA = la.inv(A)
    
    full_x += -np.dot(iA,V).flatten()


print("dndz_true = ",(dndz_data_theo.flatten()))
print("dndz_answer = ",(full_x))
print("dndz_guess = ",(full_x0))

with open('results.txt','w') as f:
    for a,b,c in zip(dndz_data_theo.flatten(), full_x, full_x0):
        f.write("%f %f %f \n"%(a,b,c))
        
