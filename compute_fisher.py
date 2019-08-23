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
import hod 
reload(hod)
import hod_funcs_evol_fit
reload(hod_funcs_evol_fit)
import matplotlib.pyplot as plt
from itertools import combinations
import scipy.linalg as la

# Specify where to save things
direc = '/users/boryanah/HSC/figs/'
# User chooses how many tomographic bins and z samples they want
tomo = sys.argv[1]
zsam = sys.argv[2]

# constants
deg_to_rad = np.pi/180.
arcmin_to_rad = (1./60.*np.pi/180.)

# redshift parameters
# range for the zsamples
z_ini_sample = 0.
z_end_sample = 2.
# range for the tomographic bins
z_ini_bin = 0.
z_end_bin = 2.
# number of tomo bins and zsamples
N_tomo = int(tomo) #5#7#10#13
N_zsamples = int(zsam) #5#10
# edges and centers of the zsamples
z_s_edges = np.linspace(z_ini_sample,z_end_sample,N_zsamples+1)
z_s_cents = (z_s_edges[1:]+z_s_edges[:-1])*.5
# edges and centers of the zsamples
z_bin_edges = np.linspace(z_ini_bin,z_end_bin,N_tomo+1)
z_bin_cents = (z_bin_edges[1:]+z_bin_edges[:-1])*.5
# ellipticity std
sigma_e2 = np.ones(N_tomo)*(.4**2)

# name pf saved file
name = 'fisher_hod_dndz_'+str(N_tomo)+'_'+str(N_zsamples)

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

# Power spectrum parameters
k_ar = np.logspace(-4.3, 3, 1000)
z_ar = np.linspace(0., 3., 50)[::-1]
a_ar = 1./(1. + z_ar)

# HSC table 1 parameters -- very irrelevant at this point and not used
z_range = np.array([0.3,0.6,0.9,1.2,1.5])
N_gal = np.array([2842635,2848777,2103995,1185335])
n_g = np.array([5.9,5.9,4.3,2.4])
n_geff_H12 = np.array([5.5,5.5,4.2,2.4])
n_geff_C13 = np.array([5.4,5.3,3.8,2.])
e2_rms_sq = np.array([0.394,0.395,0.404,0.409])
e2psig2_rms_sq = np.array([0.411,0.415,.430,.447])

# ___________________________________
#          END OF PARAMETERS

# Read catalog with overlap with COSMOS
cat=fits.open("/users/boryanah/repos/NzGCWL/data/cosmos_weights.fits")[1].data

# is matrix symmetric
def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

# is matrix positive definite
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

    return nz, z_bins, ngal


# This is how you get a redshift distribution for galaxies
# with z_ini < z_ephor_ab < z_end The redshift distribution will
# be sampled in 10 bins between in the range 0 < z_true < 2.
dndz_data = np.zeros((N_tomo,N_zsamples))
N_gal_bin = np.zeros(N_tomo)

for i in range(N_tomo):
    # select tomographic bin
    z_bin_ini = z_bin_edges[i]
    z_bin_end = z_bin_edges[i+1]
    z_bin_mid = 0.5*(z_bin_ini+z_bin_end)

    # get dndz's for that tomo bin with edges and total number of galaxies in it
    dndz_this, z_edges, N_gal_this = get_nz_from_photoz_bins(zp_code='pz_best_eab',# Photo-z code
                                                        zp_ini=z_bin_ini, zp_end=z_bin_end, # Bin edges
                                                        zt_edges=(z_ini_sample, z_end_sample),# Sampling range
                                                        zt_nbins=N_zsamples)# Number of samples


    dndz_data[i,:] = dndz_this    
    N_gal_bin[i] = N_gal_this
    
    plt.plot(0.5*(z_edges[:-1]+z_edges[1:]), dndz_this, 'o-', label='z = %f'%(z_bin_mid))

plt.legend()
plt.xlabel("z", fontsize=14)
plt.ylabel("p(z)", fontsize=14)    
plt.savefig('dndz.png')
plt.close()

# print gals in each tomo bin to remind yourself we are not alone
print(N_gal_bin)

# algorithm for computing Cls
def compute_Cls(par,z_cent=z_s_cents,N_gal_sample=N_gal_bin,k_arr=k_ar,z_arr=z_ar,a_arr=a_ar,ell=ells,compute_inv_cov=False,plot_for_sanity=False,powerspec='halofit',simple_bias=False):
    # HOD parameters -- don't need to be called every time so can be taken out
    hod_par = {
    'sigm_0': 0.4,
    'sigm_1': 0.,
    'alpha_0': 1.0,
    'alpha_1': 0.,
    'fc_0': 1.,
    'fc_1': 0.,
    'lmmin_0': par['lmmin_0']['val'],
    'lmmin_1': par['lmmin_1']['val'],
    'm0_0': par['m0_0']['val'],
    'm0_1': par['m0_1']['val'],
    'm1_0': par['m1_0']['val'],
    'm1_1': par['m1_1']['val']}
    
    if powerspec == 'halofit':
        # Compute matter pk using halofit -- preferred
        pk_mm_arr = np.array([ccl.nonlin_matter_power(cosmo_fid, k_arr, a) for a in a_arr])
    elif powerspec == 'halomodel':
        # Alternatively use halo model
        pk_mm_arr = np.array([ccl.halomodel.halomodel_matter_power(cosmo_fid, k_arr, a) for a in a_arr])
        
    if simple_bias == True:
        # Using bias parameters
        # Pk_gg = b^2 Pmm and Pk_gm = b Pmm; however, this is taken into account when def tracers
        pk_gg_arr = pk_mm_arr.copy() 
        pk_gm_arr = pk_mm_arr.copy()
        # load bias parameter
        b_z = np.array([par['b_%02d'%k]['val'] for k in range(N_zsamples)])
    else:
        # Using the HOD model
        # load HOD and compute profile
        hodpars = hod_funcs_evol_fit.HODParams(hod_par, islogm0=True, islogm1=True)
        hodprof = hod.HODProfile(cosmo_fid, hodpars.lmminf, hodpars.sigmf,\
                                 hodpars.fcf, hodpars.m0f, hodpars.m1f, hodpars.alphaf)
        # Compute galaxy power spectrum using halofit
        pk_gg_arr = np.array([hodprof.pk(k_arr, a_arr[i]) for i in range(a_arr.shape[0])])
        # Compute galaxy-matter pk using halofit
        pk_gm_arr = np.array([hodprof.pk_gm(k_arr, a_arr[i]) for i in range(a_arr.shape[0])])
        # Here we don't vary the bias parameters
        b_z = np.ones(N_zsamples) 
        
    # Create pk2D objects for interpolation
    pk_mm = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_arr), pk_arr=np.log(pk_mm_arr), is_logp=True)
    pk_gg = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_arr), pk_arr=np.log(pk_gg_arr), is_logp=True)
    pk_gm = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_arr), pk_arr=np.log(pk_gm_arr), is_logp=True)


    # Assume a functional form of 0.95/D(a) for the bias parameters (notice how this plays in
    # the case of varying b(z)) it doesn't matter a lot in this case for we take finite diff
    a_cent = 1./(1.+z_cent)
    b_z *= 0.95/ccl.growth_factor(cosmo_fid,a_cent)
    
    # load the dndz parameters
    dndz_z = np.zeros((N_tomo,N_zsamples))
    for i_z in range(N_tomo):
        dndz_z[i_z,:] = np.array([par['dndz_%02d_%02d'%(i_z,k)]['val'] for k in range(N_zsamples)])
    
    # Number of total correlation elements for all of gg gs and ss variants
    tot_corr = N_ell*(N_tomo*(2*N_tomo+1))
    
    # make CL_all of size N_ell*N_tomo*(2N_tomo+1)
    CL_ALL = np.zeros(tot_corr)

    # this lists all possible combinations of all gg gs and ss's for a given number of tomo bins
    # here is how it works: say we have 3 tomo bins, then we make an array of 2x3 = 6 numbers
    # from 0 through 5; where 0 through 2 correspond to galaxy tracer (g) and 3 through 5 to
    # shear tracer (s). 
    # Our final list all_combos consists of every possible pair between
    # tracer 1 which can be any of 3 tomos and either g or s and tracer 2 which can also be
    # any of 3 tomos and either g or s. The way I do this is by first listing in temp
    # the 6 pairs making up all the tracer 1 and tracer 2 combs where they are either
    # both g or both s at the same tomographic redshift, i.e. temp is (0,0), (1,1) ... (5,5)
    # then I use the function combinations which finds all unique combinations between
    # two lists i.e. (0,1),(0,2)...(0,5),(1,2)...(1,5),(2,3)...(2,5),(3,4),(3,5),(4,5)
    # and call this list combs. Finally I combine the two in all_combos.
    # Note: combs does not have (1,0) for it contributes the same info as (0,1)
    # example pair (2,4) in this example corresponds to tracer 1 being galaxies in third tomographic
    # bin and tracer 2 being shear in second tomographic bin, in short (2,4)=(g2,s1)
    temp = np.arange(2*N_tomo)
    temp = np.vstack((temp,temp)).T
    combs = np.array(list(combinations(range(2*N_tomo),2)))
    all_combos = np.vstack((temp,combs))

    
    for c, comb in enumerate(all_combos):
        # comb is the list pair, e.g. (2,5) while c is its order no. (b/n 0 and Ntomo(2Ntomo+1)-1)
        i = comb[0]%N_tomo # redshift bin of tracer 1 (b/n 0 and N_tomo-1 regardless of g or s)
        j = comb[1]%N_tomo # redshift bin of tracer 2 (b/n 0 and N_tomo-1 regardless of g or s)
        t_i = comb[0]//N_tomo # tracer type (0 or 1): 0 means g and 1 means s
        t_j = comb[1]//N_tomo # tracer type (0 or 1): 0 means g and 1 means s
        
        # Adding NOISE to only the diagonal elements
        if (i == j):
            # number density of galaxies - use area cosmos for these are the gals that overlap
            N_gal = N_gal_sample[i]
            n_gal = N_gal/area_COSMOS # in rad^-2
            
            # Adding noise
            noise_gal = 1./n_gal
            noise_shape = sigma_e2[i]/n_gal
        else:
            noise_gal = 0.
            noise_shape = 0.
        
        # Now create corresponding Cls with pk2D objects matched to pk
        if t_i*2+t_j == 0: # this is gg (notice that this can only be 2*0+0 = 0)
            tracer_z1 = ccl.NumberCountsTracer(cosmo_fid, bias=(z_cent, b_z), \
                                               dndz=(z_cent, dndz_z[i,:]),mag_bias=None, \
                                               has_rsd=False)
            tracer_z2 = ccl.NumberCountsTracer(cosmo_fid, bias=(z_cent, b_z), \
                                               dndz=(z_cent, dndz_z[j,:]),mag_bias=None, \
                                               has_rsd=False)
            cl_gg = ccl.angular_cl(cosmo_fid, tracer_z1, tracer_z2, ell, p_of_k_a=pk_gg)
            cl_gg_no = cl_gg + noise_gal

            CL_ALL[(N_ell*c):(N_ell*c)+N_ell] = cl_gg_no
        elif t_i*2+t_j == 1: # this is gs (notice that this can only be 2*0+1 = 1, never 2*1+0=2)
            tracer_z1 = ccl.NumberCountsTracer(cosmo_fid, bias=(z_cent, b_z), \
                                               dndz=(z_cent, dndz_z[i,:]),mag_bias=None, \
                                               has_rsd=False)
            tracer_z2 = ccl.WeakLensingTracer(cosmo_fid, dndz=(z_cent, dndz_z[j,:]))
            cl_gs = ccl.angular_cl(cosmo_fid, tracer_z1, tracer_z2, ell, p_of_k_a=pk_gm)
            cl_gs_no = cl_gs

            CL_ALL[(N_ell*c):(N_ell*c)+N_ell] = cl_gs_no
            
        elif t_i*2+t_j == 3: # this is ss (notice that this can only be 2*1+1 = 3)
            tracer_z1 = ccl.WeakLensingTracer(cosmo_fid, dndz=(z_cent, dndz_z[i,:]))
            tracer_z2 = ccl.WeakLensingTracer(cosmo_fid, dndz=(z_cent, dndz_z[j,:]))
            cl_ss = ccl.angular_cl(cosmo_fid, tracer_z1, tracer_z2, ell, p_of_k_a=pk_mm)
            cl_ss_no = cl_ss + noise_shape
            
            CL_ALL[(N_ell*c):(N_ell*c)+N_ell] = cl_ss_no

        # check whether things make sense
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

    # if we don't want the inverse covariance matrix
    if compute_inv_cov == False:
        print(len(CL_ALL))
        return CL_ALL

    # print out numbers to make sure there is internet connection
    print(len(CL_ALL))
    print(np.sum(CL_ALL>0))

    # initiate covariance matrix into the trade
    COV_ALL = np.zeros((len(CL_ALL),len(CL_ALL)))    
    # COMPUTE COVARIANCE MATRIX: Cov(A,B)
    for c_A, comb_A in enumerate(all_combos):
        for c_B, comb_B in enumerate(all_combos):
            # pair A=(ti,tj), pair B=(tm,tn) at same ell, where t stands for either g or s
            i = comb_A[0]%N_tomo # redshift bin of tracer A_1 (i.e. along rows of Cov)
            j = comb_A[1]%N_tomo # redshift bin of tracer A_2 (i.e. along rows of Cov)
            m = comb_B[0]%N_tomo # redshift bin of tracer B_1 (i.e. along columns of Cov)
            n = comb_B[1]%N_tomo # redshift bin of tracer B_2 (i.e. along columns of Cov)

            # These are the pairs (their order number in all_combos) we need for Knox
            # cov(ij,mn) = im,jn + in,jm
            c_im = np.argmax(np.product(np.sort(np.array([comb_A[0],comb_B[0]]))==all_combos,axis=1))
            c_jn = np.argmax(np.product(np.sort(np.array([comb_A[1],comb_B[1]]))==all_combos,axis=1))
            c_in = np.argmax(np.product(np.sort(np.array([comb_A[0],comb_B[1]]))==all_combos,axis=1))
            c_jm = np.argmax(np.product(np.sort(np.array([comb_A[1],comb_B[0]]))==all_combos,axis=1))
            # retrieving their Cls
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
                    # maybe add legend and make sure colors are the same for each type
                    (_,caps,eb)=plt.errorbar(ell,C_ij,yerr=Cl_err,lw=2.,ls='-',capsize=5,label=str(t_i*2+t_j)) 
                    plt.legend()
                    plt.xscale('log')
                    plt.yscale('log')
                    #plt.ylim([1.e-12,1.e-5])

            # fill in the diagonals of the cov matrix and make use of its symmetry Cov(A,B)=Cov(B,A)
            COV_ALL[(N_ell*c_A):(N_ell*c_A)+\
                    N_ell,(N_ell*c_B):(N_ell*c_B)+N_ell] = np.diag(Cov_ijmn)
            COV_ALL[(N_ell*c_B):(N_ell*c_B)+\
                    N_ell,(N_ell*c_A):(N_ell*c_A)+N_ell] = np.diag(Cov_ijmn)

    # check matrix is positive definite
    if (is_pos_def(COV_ALL) != True): print("Covariance is not positive definite!"); exit(0)
    # compute and return inverse
    ICOV_ALL = la.inv(COV_ALL)
    return ICOV_ALL

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

# initiate cosmology object
cosmo_fid = ccl.Cosmology(**FID_COSMO_PARAMS)

# biasing initial value
a_cent = 1./(1.+z_s_cents)
b_z = 0.95/ccl.growth_factor(cosmo_fid,a_cent)

# HOD parameters
params['lmmin_0'] = {'val':3.71   ,'dval':0.02 ,'label':'$\\l_{m,min,0}$','isfree':True}
params['lmmin_1'] = {'val':9.99  ,'dval':0.02 ,'label':'$\\l_{m,min,1}$','isfree':True}
params['m0_0'] = {'val':1.28   ,'dval':0.02 ,'label':'$\\m_{0,0}$','isfree':True}
params['m0_1'] = {'val':10.34  ,'dval':0.02 ,'label':'$\\m_{0,1}$','isfree':True}
params['m1_0'] = {'val':7.08  ,'dval':0.02 ,'label':'$\\m_{1,0}$','isfree':True}
params['m1_1'] = {'val':9.34  ,'dval':0.02 ,'label':'$\\m_{1,1}$','isfree':True}

for i in np.arange(N_tomo):
    for k in np.arange(N_zsamples):
        params['b_%02d'%k]  = {'val':b_z[k]  ,'dval':0.02 ,'label':'$b_%02d$'%k ,'isfree':False}
        params['dndz_%02d_%02d'%(i,k)]  = {'val':dndz_data[i,k] ,'dval':0.1 ,'label':'dndz_%02d_%02d'%(i,k),'isfree':True}

# how many params
npar=len(params)
# which params do we vary
params_vary={}
for p in sorted(params.iterkeys()):
    if params[p]['isfree']:
        params_vary[p] = params[p]
# how many do we vary
npar_vary=len(params_vary)


# depending on what we vary we want to print out just the dndz values
# so we need to offset the params_vary dictionary if varying b(z) or hod params
if params['lmmin_0']['isfree'] == True:
    offset = 0; offset_end = -6
elif params['b_%02d'%0]['isfree'] == True:
    offset = N_zsamples; offset_end = npar_vary
else:
    offset = 0; offset_end = npar_vary
for i in range(N_tomo):
    nam = sorted(params_vary)[offset+N_zsamples*i:offset+N_zsamples*i+N_zsamples][0][:-3]
    print(nam)

# Compute Cls for given params
#CL_TOTAL = compute_Cls(params)

# test by plotting - uncomment if plot_for_sanity is True (could be cleaner)
#fig = plt.figure(figsize=(30,25))
ICov_CL_TOTAL = compute_Cls(params,compute_inv_cov=True)
# test by plotting
#plt.savefig("Cl_err.pdf");
#plt.close()

# Compute numerical derivatives of the data vector
dCldp_alpha=np.zeros([npar_vary,int(N_tomo*(2*N_tomo+1)*N_ell)])
def compute_derivs(pars,parname):
    pars_here=copy.deepcopy(pars)
    pars_here[parname]['val']=pars[parname]['val']+pars[parname]['dval']
    clp=compute_Cls(pars_here)
    pars_here[parname]['val']=pars[parname]['val']-pars[parname]['dval']
    clm=compute_Cls(pars_here)
    return (clp-clm)/(2*pars[parname]['dval'])

# compute derivs with resp to all params that are being varied
for i,nam in enumerate(sorted(params_vary)):
    print("Parameter ",nam,", which is number ",(i+1),"out of ",npar_vary)
    dCldp_alpha[i,:]=compute_derivs(params,nam)

# compute Fisher matrix
fisher = np.dot(dCldp_alpha,np.dot(ICov_CL_TOTAL,dCldp_alpha.T))
np.save(name+'.npy',fisher)
    
# requiring sum dndz = const by adding diagonal square blocks of 1's times large number
fisher_sumfix = np.zeros((N_zsamples*N_tomo,N_zsamples*N_tomo))
ones = np.ones((N_zsamples,N_zsamples))*1.e11
for i in range(N_tomo):                     
    fisher_sumfix[i*N_zsamples:i*N_zsamples+N_zsamples,i*N_zsamples:i*N_zsamples+N_zsamples] = ones

# importantly we add this only to the dndz part of the fisher matrix
fisher[offset:offset_end,offset:offset_end] += fisher_sumfix

# check if fisher is positive definite
if (is_pos_def(fisher) != True): print("Fisher is not positive definite!"); exit(0)

# plot fisher and hope for the best
plt.imshow(fisher,interpolation="nearest")
plt.colorbar()
plt.savefig(direc+name+'.png')
plt.close()

# compute inverse fisher
inv_fisher = la.inv(fisher)

# compute evals cause computer needs to be kept busy
evals,evecs = la.eig(fisher)
print(evals)

# check if inv fisher is positive definite
if (is_pos_def(inv_fisher) != True): print("Inverse Fisher is not positive definite!"); exit(0)

# normalize inv fisher matrix to show fractional difference
for i,nam_i in enumerate(sorted(params_vary)):
    for j,nam_j in enumerate(sorted(params_vary)):
        #print("Parameter ",nam_i,", which is number ",(i+1),"out of ",npar_vary)
        #print("Parameter ",nam_j,", which is number ",(j+1),"out of ",npar_vary)
        inv_fisher[i,j] /= (params_vary[nam_i]['val']*params_vary[nam_j]['val'])
# save inv fisher
np.save('inv_'+name+'.npy',inv_fisher)

# get only the marginalized errors on each element
inv_F_alpha_alpha = np.diag(inv_fisher)
sigma_alpha = np.sqrt(inv_F_alpha_alpha[offset:offset_end])

# plot a block of inverse fisher for the dndz's in each tomographic bin cause why not
for i in range(N_tomo):
    nam = sorted(params_vary)[offset+N_zsamples*i:offset+N_zsamples*i+N_zsamples][0][:-3]
    print(nam)
    print(sigma_alpha[N_zsamples*i:N_zsamples*i+N_zsamples])   

    plt.imshow(inv_fisher[offset+N_zsamples*i:offset+N_zsamples*i+N_zsamples,offset+N_zsamples*i:offset+N_zsamples*i+N_zsamples],interpolation="nearest")
    plt.colorbar()
    plt.savefig(direc+'inv_'+name+'_block_'+nam+'.png')
    plt.close()
    
plt.imshow(inv_fisher,interpolation="nearest")
plt.colorbar()
plt.savefig(direc+'inv_'+name+'.png')
plt.close()
