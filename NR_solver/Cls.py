import numpy as np
import pyccl as ccl
import hod 
import hod_funcs_evol_fit
from itertools import combinations

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def compute_Cls(dndz_z,b_z,z_cent,N_gal_sample,ell,a_arr,k_arr,sigma_e2,area_overlap,powerspec='halofit',simple_bias=True):

    # choose a scheme for evaluating the matter power spectrum -- halofit is recommended
    if powerspec == 'halofit':
        # Compute matter pk using halofit
        pk_mm_arr = np.array([ccl.nonlin_matter_power(cosmo_fid, k_arr, a) for a in a_arr])
    elif powerspec == 'halomodel':
        # Alternatively use halo model
        pk_mm_arr = np.array([ccl.halomodel.halomodel_matter_power(cosmo_fid, k_arr, a) for a in a_arr])
        

    # choose a scheme for evaluating the galaxy power spectrum between simple bias and HOD model
    if simple_bias == True:
        # Set the power spectra equal to the matter PS -- this changes later in the code
        pk_gg_arr = pk_mm_arr.copy()
        pk_gm_arr = pk_mm_arr.copy()
    else:
        # HOD parameter values
        hod_par = {'sigm_0': 0.4,'sigm_1': 0.,'alpha_0': 1.0,'alpha_1': 0.,'fc_0': 1.,'fc_1': 0.,\
                  'lmmin_0': 3.71,'lmmin_1': 9.99,'m0_0': 1.28,'m0_1': 10.34,'m1_0': 7.08,'m1_1': 9.34}
        # Evolve HOD parameters and create an HOD profile
        hodpars = hod_funcs_evol_fit.HODParams(hod_par, islogm0=True, islogm1=True)
        hodprof = hod.HODProfile(cosmo_fid, hodpars.lmminf, hodpars.sigmf,\
                                 hodpars.fcf, hodpars.m0f, hodpars.m1f, hodpars.alphaf)
        # Compute galaxy pk using halofit
        pk_gg_arr = np.array([hodprof.pk(k_arr, a_arr[i]) for i in range(a_arr.shape[0])])
        # Compute galaxy-matter pk using halofit
        pk_gm_arr = np.array([hodprof.pk_gm(k_arr, a_arr[i]) for i in range(a_arr.shape[0])])
        # Wipe b_z since HOD is adopted
        b_z[:,:] = 1.
        
    # Create pk2D objects for interpolation
    pk_mm = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_arr), pk_arr=np.log(pk_mm_arr), is_logp=True)
    pk_gg = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_arr), pk_arr=np.log(pk_gg_arr), is_logp=True)
    pk_gm = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_arr), pk_arr=np.log(pk_gm_arr), is_logp=True)
    
    # number of indep correlation functions per type gg gs ss
    N_tot_corr = N_ell*(N_tomo*(2*N_tomo+1))
    
    # make C_ell of size N_ell*N_tomo*(2N_tomo+1)
    C_ell = np.zeros(N_tot_corr)
    
    for c, comb in enumerate(all_combos):
        i = comb[0]%N_tomo # first redshift bin
        j = comb[1]%N_tomo # second redshift bin
        t_i = comb[0]//N_tomo # tracer type 0 means g and 1 means s
        t_j = comb[1]//N_tomo # tracer type 0 means g and 1 means s
        
        # Noise
        if (i == j):
            # number density of galaxies
            N_gal = N_gal_sample[i]            
            n_gal = N_gal/area_overlap # in rad^-2
            
            # Adding noise
            noise_gal = 1./n_gal
            noise_shape = sigma_e2[i]/n_gal
        else:
            noise_gal = 0.
            noise_shape = 0.
        
        # Now create corresponding Cls with pk2D objects matched to pk
        # this is gg
        if t_i*2+t_j == 0:  
            tracer_z1 = ccl.NumberCountsTracer(cosmo_fid, bias=(z_cent, b_z[i,:]), \
                                               dndz=(z_cent, dndz_z[i,:]),mag_bias=None, \
                                               has_rsd=False)
            tracer_z2 = ccl.NumberCountsTracer(cosmo_fid, bias=(z_cent, b_z[j,:]), \
                                               dndz=(z_cent, dndz_z[j,:]),mag_bias=None, \
                                               has_rsd=False)

            cl_gg = ccl.angular_cl(cosmo_fid, tracer_z1, tracer_z2, ell, p_of_k_a=pk_gg)
            cl_gg_no = cl_gg + noise_gal

            C_ell[(N_ell*c):(N_ell*c)+N_ell] = cl_gg_no
        
        # this is gs
        elif t_i*2+t_j == 1:
            tracer_z1 = ccl.NumberCountsTracer(cosmo_fid, bias=(z_cent, b_z[i,:]), \
                                               dndz=(z_cent, dndz_z[i,:]),mag_bias=None, \
                                               has_rsd=False)
            tracer_z2 = ccl.WeakLensingTracer(cosmo_fid, dndz=(z_cent, dndz_z[j,:]))
            cl_gs = ccl.angular_cl(cosmo_fid, tracer_z1, tracer_z2, ell, p_of_k_a=pk_gm)
            cl_gs_no = cl_gs

            C_ell[(N_ell*c):(N_ell*c)+N_ell] = cl_gs_no

        # this is ss
        elif t_i*2+t_j == 3:   
            tracer_z1 = ccl.WeakLensingTracer(cosmo_fid, dndz=(z_cent, dndz_z[i,:]))
            tracer_z2 = ccl.WeakLensingTracer(cosmo_fid, dndz=(z_cent, dndz_z[j,:]))
            cl_ss = ccl.angular_cl(cosmo_fid, tracer_z1, tracer_z2, ell, p_of_k_a=pk_mm)
            cl_ss_no = cl_ss + noise_shape

            C_ell[(N_ell*c):(N_ell*c)+N_ell] = cl_ss_no

    print(len(C_ell))
    return C_ell

def compute_inv_Cov(C_ell,N_tomo,ell,f_sky):
    # number of ells 
    N_ell = len(ell)
    delta_ell = ell[1]-ell[0]

    # List all redshift bin combinations
    temp = np.arange(2*N_tomo)
    temp = np.vstack((temp,temp)).T
    combs = np.array(list(combinations(range(2*N_tomo),2)))
    all_combos = np.vstack((temp,combs))
    
    Cov_ell = np.zeros((len(C_ell),len(C_ell)))    
    # Compute covariance matrix
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
            C_im = C_ell[(c_im*N_ell):(c_im*N_ell)+N_ell]
            C_jn = C_ell[(c_jn*N_ell):(c_jn*N_ell)+N_ell]
            C_in = C_ell[(c_in*N_ell):(c_in*N_ell)+N_ell]
            C_jm = C_ell[(c_jm*N_ell):(c_jm*N_ell)+N_ell]

            
           # Knox formula
            Cov_ijmn = (C_im*C_jn+C_in*C_jm)/((2*ell+1.)*delta_ell*f_sky)

            # fill all values
            Cov_ell[(N_ell*c_A):(N_ell*c_A)+\
                    N_ell,(N_ell*c_B):(N_ell*c_B)+N_ell] = np.diag(Cov_ijmn)
            Cov_ell[(N_ell*c_B):(N_ell*c_B)+\
                    N_ell,(N_ell*c_A):(N_ell*c_A)+N_ell] = np.diag(Cov_ijmn)


    # Check it is positive definite
    assert(is_pos_def(Cov_ell),"Covariance is not positive definite!")

    return Cov_ell 