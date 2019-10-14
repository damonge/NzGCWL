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

# ___________________________________
#             PARAMETERS
# ___________________________________

# what interpolation scheme do we want
# OUR EQUATIONS SHOULD WORK FOR NEAREST AND LINEAR
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
N_tomo = int(tomo) #5#7#10#13
N_zsamples_theo = int(zsam) #5#10#10#13

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
    return nz, z_bins, ngal

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

   
    dndz_this, z_edges_theo, N_gal_this = \
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
        # EXTRAPOLATION OUTSIDE RANGE -- DAVID SAYS NOT DONE FOR PYCCL
        f = interp1d(np.append(0.5*(z_edges_theo[:-1]+z_edges_theo[1:]),np.array([z_s_cents[0],z_s_cents[-1]])),np.append(dndz_theo_fn,np.array([dndz_this[0],dndz_this[-1]])),kind='cubic',fill_value=0)#'extrapolate')
        # NO EXTRAPOLATION
        #f = interp1d(0.5*(z_edges_theo[:-1]+z_edges_theo[1:]),dndz_theo_fn,kind='cubic',bounds_error=0,fill_value=0.)#(dndz_theo_fn[0],dndz_theo_fn[-1]))
    elif interp == 'log': # NOT USED IN THIS CODE
        f = interp1d(np.append(0.5*(z_edges_theo[:-1]+z_edges_theo[1:]),np.array([z_s_cents[0],z_s_cents[-1]])),np.append(np.log10(dndz_theo_fn),np.log10(np.array([dndz_this[0],dndz_this[-1]]))),kind='cubic',fill_value='extrapolate')  
    elif interp == 'nearest':
        #f = interp1d(0.5*(z_edges_theo[:-1]+z_edges_theo[1:]),dndz_theo_fn,kind='nearest',bounds_error=0,fill_value=0.)
        #f = interp1d(0.5*(z_edges_theo[:-1]+z_edges_theo[1:]),dndz_theo_fn,kind='nearest',bounds_error=0,fill_value=(dndz_theo_fn[0],dndz_theo_fn[-1]))
        f = interp1d(np.append(0.5*(z_edges_theo[:-1]+z_edges_theo[1:]),np.array([z_s_cents[0],z_s_cents[-1]])),np.append(dndz_theo_fn,np.array([dndz_theo_fn[0],dndz_theo_fn[-1]])),kind='nearest',bounds_error=0,fill_value=0.)
        
    elif interp == 'linear':
        #f = interp1d(0.5*(z_edges_theo[:-1]+z_edges_theo[1:]),dndz_theo_fn,kind='linear',bounds_error=0,fill_value=0.)
        f = interp1d(np.append(0.5*(z_edges_theo[:-1]+z_edges_theo[1:]),np.array([z_s_cents[0],z_s_cents[-1]])),np.append(dndz_theo_fn,np.array([dndz_theo_fn[0],dndz_theo_fn[-1]])),kind='linear',bounds_error=0,fill_value=0.)
        
    if interp == 'log': dndz_data[i,:] = 10**f(z_s_cents)
    else: dndz_data[i,:] = f(z_s_cents)

    # Normalization not necessary
    #sum_dndz = np.sum(dndz_data[i,:]*(z_s_cents[1]-z_s_cents[0])) # equals 1
    #dndz_data[i,:] = dndz_data[i,:]/sum_dndz # DOESN'T MATTER SINCE PYCCL NORMALIZES
    
    # record discrete dndzs
    dndz_data_theo[i,:] = dndz_this

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
    b_z = np.array([par['b_%02d'%k]['val'] for k in range(N_zsamples)])
    #b_z=np.ones(N_zsamples)
    '''
    # I am currently commenting this out so that b_z = ones
    a_cent = 1./(1.+z_cent)
    b_z = 0.95/ccl.growth_factor(cosmo_fid,a_cent)
    '''
    
    
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

# matrices used for the fast calculation
b_mat = np.repeat(b_zsamples_theo.reshape(N_zsamples_theo,1),N_zsamples_theo,axis=1)
b2_mat = b_mat.T*b_mat
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
D1 = np.zeros((N_zsamples_theo*N_tomo,N_zsamples_theo*N_tomo))
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

D0 /= sigma0**2.
D1 /= sigma1**2.


D = D1#+D0


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
#                           curly C comment out for fastness
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
np.save("mat_C.npy",mat_C)
'''
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
            if (compute_2nd_ders == True):
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
mat_C = np.load("mat_C.npy")

# ____________________________________________
#                 INITIAL GUESS
# ____________________________________________

x0 = np.zeros(N_tomo*(N_zsamples_theo-1))
full_x0 = np.zeros(N_tomo*N_zsamples_theo)

for i in range(N_tomo):
    if False:#True:
        x0[i*(N_zsamples_theo-1):(i+1)*(N_zsamples_theo-1)] = dndz_data_theo[i,:-1]#+0.01 # TESTING
        continue
    dndz_this = gaussian(z_s_cents_theo,z_bin_cents[i],0.2)
    dndz_this /= np.sum(dndz_this)#*(z_s_edges_theo[1]-z_s_edges_theo[0]))
    x0[i*(N_zsamples_theo-1):(i+1)*(N_zsamples_theo-1)] = dndz_this[:-1]
    

for i in range(N_tomo):
    full_x0[i*(N_zsamples_theo):(i+1)*(N_zsamples_theo)-1] = x0[i*(N_zsamples_theo-1):(i+1)*(N_zsamples_theo-1)]
    sum_dndz = np.sum(x0[i*(N_zsamples_theo-1):(i+1)*(N_zsamples_theo-1)])#*(z_s_edges_theo[1]-z_s_edges_theo[0]))
    full_x0[(i+1)*(N_zsamples_theo)-1] = (1-sum_dndz)#/(z_s_edges_theo[1]-z_s_edges_theo[0])

# ____________________________________________
#                 NEWTON-RAPHSON
# ____________________________________________

full_x = np.zeros(N_tomo*N_zsamples_theo)
x = x0.copy()
for i in range(N_tomo):
    full_x[i*(N_zsamples_theo):(i+1)*(N_zsamples_theo)-1] = x[i*(N_zsamples_theo-1):(i+1)*(N_zsamples_theo-1)]
    sum_dndz = np.sum(x[i*(N_zsamples_theo-1):(i+1)*(N_zsamples_theo-1)])#*(z_s_edges_theo[1]-z_s_edges_theo[0]))
    full_x[(i+1)*(N_zsamples_theo)-1] = (1-sum_dndz)#/(z_edges_theo[1]-z_edges_theo[0])

print("Delta_dndz = ",np.sum((dndz_data_theo.flatten()-full_x)**2))

# FIX NOT SURE

# compute the Cls and their derivatives
Cl_fast, dCldp_fast, Cov_fast = compute_fast_Cls(full_x,mat_C,compute_ders=True,compute_2nd_ders=False,compute_cov=True)
N_elm = len(Cl_fast)
Cl_fast = Cl_fast.reshape(N_elm,1)
Cl_true = Cl_true.reshape(N_elm,1)

iCov_fast = la.inv(Cov_fast)

Delta_Cl = Cl_fast-Cl_true

A = np.dot(dCldp_fast,np.dot(iCov_fast,dCldp_fast.T))
V = np.dot(dCldp_fast,np.dot(iCov_fast,Delta_Cl))

chi2 = np.dot(Delta_Cl.T,np.dot(iCov_fast,Delta_Cl))#[0][0]
print(chi2)

iA = la.inv(A)
full_x += -np.dot(iA,V).flatten()#-np.dot(iA,V).flatten()

print("Delta_dndz = ",np.sum((dndz_data_theo.flatten()-full_x)**2))

for i in range(N_tomo):
    sum_dndz = np.sum(full_x[i*(N_zsamples_theo):(i+1)*(N_zsamples_theo)])#*(z_s_edges_theo[1]-z_s_edges_theo[0]))
    full_x[i*(N_zsamples_theo):(i+1)*(N_zsamples_theo)] /= sum_dndz
    sum_dndz = np.sum(full_x[i*(N_zsamples_theo):(i+1)*(N_zsamples_theo)])
    print(sum_dndz)

sum_dndz = np.sum(dndz_this*(z_edges_theo[1]-z_edges_theo[0]))

Cl_fast, dCldp_fast, Cov_fast = compute_fast_Cls(full_x,mat_C,compute_ders=True,compute_2nd_ders=False,compute_cov=True)
N_elm = len(Cl_fast)
Cl_fast = Cl_fast.reshape(N_elm,1)

iCov_fast = la.inv(Cov_fast)
Delta_Cl = Cl_true-Cl_fast
chi2 = np.dot(Delta_Cl.T,np.dot(iCov_fast,Delta_Cl))#[0][0]
print(chi2)
quit()
# Matches

print("A = ",A[:5,:5])
print("F = ",fisher[:5,:5])
print("A-F = ",np.mean(A-fisher))

# NUMERICAL DERIVATIVES
def compute_num_derivs(ind_j,dval):
    dndz_z = dndz_data_theo.copy()
    k = ind_j % N_zsamples_theo # gives sample
    i = ind_j // N_zsamples_theo # gives tomo

    print('tomo_i, zsam_k = ',i,k)
    dndz_ik = dndz_z[i,k]
    dndz_z[i,k] = dndz_ik+dval
    clp = compute_fast_Cls(dndz_z,mat_C)
    dndz_z[i,k] = dndz_ik-dval
    clm = compute_fast_Cls(dndz_z,mat_C)
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


mat_C = np.load("mat_C.npy")
dndz_dval = 0.02
temp = np.arange(2*N_tomo)
temp = np.vstack((temp,temp)).T
combs = np.array(list(combinations(range(2*N_tomo),2)))
all_combos = np.vstack((temp,combs))

dndz_z = dndz_data_theo.copy()

'''
# TESTING
Cl_fast = compute_fast_Cls(dndz_z,mat_C)

# PLOTTING TESTING
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


dCldp_fast_alpha = compute_fast_Cls(dndz_z,mat_C,compute_ders=True)
dCldp_num_alpha = np.zeros((N_tomo*N_zsamples_theo,int(N_tomo*(2*N_tomo+1)*N_ell)))
for j in np.arange(N_tomo*N_zsamples_theo):
    dCldp_num_alpha[j,:] = compute_num_derivs(j,dndz_dval)
    j += 1

#MORE PLOTS

for N_plot in range(N_tomo*N_zsamples_theo):
    fig = plt.figure(figsize=(30,25))
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
        
    tomo = N_plot//N_zsamples_theo
    zsamp = N_plot%N_zsamples_theo
    plt.savefig("dCl_all_tomo_"+str(tomo)+"_zsamp_"+str(zsamp)+".pdf")

'''

ddCldp_fast = compute_fast_Cls(dndz_z,mat_C,compute_ders=False,compute_2nd_ders=True)
ddCldp_num = np.zeros((N_tomo*N_zsamples_theo,N_tomo*N_zsamples_theo,int(N_tomo*(2*N_tomo+1)*N_ell)))
for j1 in range(N_tomo*N_zsamples_theo):
    for j2 in range(N_tomo*N_zsamples_theo):
        if j2>j1: continue
        ddCldp_num[j1,j2,:] = compute_num_2nd_derivs(j1,j2,dndz_dval)
        ddCldp_num[j2,j1,:] = ddCldp_num[j1,j2,:]


for N_plot1 in range(N_tomo*N_zsamples_theo):
    for N_plot2 in range(N_tomo*N_zsamples_theo):
        #if N_plot2 > N_plot1: continue
        zsamp1 = N_plot1%N_zsamples_theo
        zsamp2 = N_plot2%N_zsamples_theo
        tomo1 = N_plot1//N_zsamples_theo
        tomo2 = N_plot2//N_zsamples_theo
        for c, comb in enumerate(all_combos):
            i_tomo = comb[0]%N_tomo # first redshift bin
            j_tomo = comb[1]%N_tomo # second redshift bin
            t_i = comb[0]//N_tomo # tracer type 0 means g and 1 means s
            t_j = comb[1]//N_tomo # tracer type 0 means g and 1 means s

            ddCt = ddCldp_num[N_plot1,N_plot2,(c*N_ell):(c*N_ell)+N_ell]
            ddCf = ddCldp_fast[N_plot1,N_plot2,(c*N_ell):(c*N_ell)+N_ell]
            
            if (t_i*2+t_j == 0): int_t='gg'; c='red'
            if (t_i*2+t_j == 1): int_t='gs'; c='purple'
            if (t_i*2+t_j == 3): int_t='ss'; c='blue'
            
            if tomo1 == tomo2: print("tomo1=2 = "+str(tomo1)+", zsamp1,zsamp2",zsamp1,zsamp2,", type = ",int_t)
            
            else: print("tomo1,2 = ",tomo1,tomo2,"zsamp1,zsamp2",zsamp1,zsamp2,", type = ",int_t)
            #if tomo1 == tomo2:
            print("diff = ",np.mean(1-ddCf/ddCt))
            print("ddCf = ",np.mean(ddCf))
            print("ddCt = ",np.mean(ddCt))
            print("--------------")
quit()
#MORE PLOTS

for N_plot in range(N_tomo*N_zsamples_theo):
    fig = plt.figure(figsize=(30,25))
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
        
    tomo = N_plot//N_zsamples_theo
    zsamp = N_plot%N_zsamples_theo
    plt.savefig("dCl_all_tomo_"+str(tomo)+"_zsamp_"+str(zsamp)+".pdf")

quit()

dCldp_alpha=np.zeros((npar_vary,int(N_tomo*(2*N_tomo+1)*N_ell)))
def fisher(x):
    #Compute derivatives of the data vector
    print("WHAT")
    for i,nam in enumerate(sorted(params_vary)):
        print("Parameter ",nam,", which is number ",(i+1),"out of ",npar_vary)
        dCldp_alpha[i,:]=compute_derivs(params,nam)
    #Compute Fisher matrix, covariance and correlation matrix
    Cov = compute_Cls(params,compute_cov=True)
    ICov = la.inv(Cov)
    fisher = np.dot(dCldp_alpha,np.dot(ICov,dCldp_alpha.T))
    return fisher

def fun(x):
    print("x-x0 = ",np.sum(x-x0))
    x = x.reshape(N_tomo,N_zsamples)
    for i in np.arange(N_tomo):
        for k in np.arange(N_zsamples):
            params['dndz_%02d_%02d'%(i,k)]['val'] = x[i,k]
    print(x.shape)
    Cl_theo = compute_Cls(params)
    Cov = compute_Cls(params,compute_cov=True)
    ICov = la.inv(Cov)
    Delta_Cl = (Cl_theo-Cl_true).reshape(1,len(Cl_theo))
    chi2 = np.dot(np.dot(Delta_Cl,ICov),Delta_Cl.T)[0][0]
    print(chi2)
    return chi2











quit()


# =======================================
#            CHI2
# =======================================

def fun(x):
    chi2_s = np.zeros(N_tomo)
    print("sum(x_diff) = ",np.sum(x-x0))
    
    for i in range(N_tomo):
        dndz_this = x[i*(N_zsamples_theo-1):(i+1)*(N_zsamples_theo-1)]
        sum_dndz = np.sum(dndz_this*(z_edges_theo[1]-z_edges_theo[0]))
        
        if raw == True:
            for k in range(N_zsamples-1):
                params['dndz_%02d_%02d'%(i,k)]['val'] = dndz_this[k]
            params['dndz_%02d_%02d'%(i,N_zsamples-1)]['val'] = (1-sum_dndz)/(z_edges_theo[1]-z_edges_theo[0])
            chi2_s[i] = 1.e11*(np.sum(dndz_this*(z_s_edges_theo[1]-z_s_edges_theo[0]))-1)
            
            continue
        mean = np.sum(z_s_cents_theo*dndz_this)/np.sum(dndz_this)           
        sigma = np.sqrt(np.sum(dndz_this*(z_s_cents_theo-mean)**2)/np.sum(dndz_this))
        print("sigma = ",sigma)
        if sigma < 0.01: sigma = 0.2
        if np.isnan(sigma) == True: sigma = 0.2
        #popt,pcov = curve_fit(gaussian,z_s_cents_theo,dndz_this,p0=[mean,sigma])
        #g_fit = gaussian(z_s_cents,*popt)
        g_fit = gaussian(z_s_cents,mean,sigma)
        # TESTING
        #g_fit *= sum_dndz/np.sum(g_fit*(z_s_edges[1]-z_s_edges[0]))
        for k in range(N_zsamples):
            params['dndz_%02d_%02d'%(i,k)]['val'] = g_fit[k]
        chi2_s[i] = 1.e11*(np.sum(dndz_this*(z_s_edges_theo[1]-z_s_edges_theo[0]))-1)
    #Cov, Cl_theo = compute_Cls(params)
    Cl_theo = compute_Cls(params,compute_only_cl=True)
    #lam, R = la.eig(Cov)
    #ICov = la.inv(Cov)
    Delta_Cl = (Cl_theo-Cl_true).reshape(len(Cl_theo),1)
    #rDCl = np.dot(R.T,Delta_Cl) # and formula is R L R.T = Cov
    #chi2 = (rDCl.flatten()/np.sqrt(lam)).real
    chi2 = np.dot(np.dot(Delta_Cl.T,ICov),Delta_Cl)[0][0]
    #print("sum res^2 = ",np.sum(chi2**2,axis=0))
    print("chi^2 = ",(chi2))
    return chi2

const = "cons = ("
for i in range(N_tomo): const += "{'type': 'eq', 'fun': lambda x:  np.sum(x["+str(i)+"*N_zsamples:"+str(i)+"*N_zsamples+N_zsamples])-1},"
const += ")"
exec(const)


x0 = np.zeros(N_tomo*(N_zsamples_theo-1))
full_x0 = np.zeros(N_tomo*(N_zsamples_theo))
chi2_s = np.zeros(N_tomo)
for i in range(N_tomo):
    if True:
        x0[i*(N_zsamples_theo-1):(i+1)*(N_zsamples_theo-1)] = dndz_data_theo[i,:-1]#+0.01 # TESTING
        continue
    dndz_this = gaussian(z_s_cents_theo,z_bin_cents[i],0.2)
    dndz_this /= np.sum(dndz_this*(z_s_edges_theo[1]-z_s_edges_theo[0]))
    x0[i*(N_zsamples_theo-1):(i+1)*(N_zsamples_theo-1)] = dndz_this[:-1]
    chi2_s[i] = (np.sum(dndz_this*(z_s_edges_theo[1]-z_s_edges_theo[0]))-1)

#x0 = np.array([ 1.55187484e+00,  2.20285102e-01,  6.87258597e-04,  1.38896539e+00, -4.01761932e-02,  2.64249287e-01])
for i in range(N_tomo):
    full_x0[i*(N_zsamples_theo):(i+1)*(N_zsamples_theo)-1] = x0[i*(N_zsamples_theo-1):(i+1)*(N_zsamples_theo-1)]
    sum_dndz = np.sum(x0[i*(N_zsamples_theo-1):(i+1)*(N_zsamples_theo-1)]*(z_s_edges_theo[1]-z_s_edges_theo[0]))
    full_x0[(i+1)*(N_zsamples_theo)-1] = (1-sum_dndz)/(z_edges_theo[1]-z_edges_theo[0])
print(full_x0)
print(np.sum(chi2_s)); # starting with 0 for no noise and 1000 for noise
print(fun(x0));quit()


# =======================================
#            NOISE
# =======================================
Cov = compute_Cls(params,compute_cov=True)
X = la.cholesky(Cov).T

N_draws = 100
Nl_all = np.zeros((N_ell*N_tomo*(2*N_tomo+1),N_draws))

for i in range(N_draws):
    R = np.random.normal(0.,1.,N_ell*N_tomo*(2*N_tomo+1))
    Nl = np.dot(X,R)
    Nl_all[:,i] = Nl

Cl_true += Nl

'''
Cl_theo = compute_Cls(params)
# TESTING
fig = plt.figure(figsize=(30,25))
ICov = compute_Cls(params,compute_inv_cov=True)
# TESTING
plt.savefig("Cl_err.pdf");
plt.close()
quit()
chi2 = np.dot(np.dot(Delta_Cl,ICov),Delta_Cl.T)[0][0]
print(chi2)
'''
