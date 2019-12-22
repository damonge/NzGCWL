import numpy as np
import pyccl as ccl
from itertools import combinations

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
def compute_mCs(cosmo_fid,z_cent_theo,z_cent,a_arr,k_arr,ell,i_zs,j_zs): 
    # bias parameters should be set to 1
    N_zsamples = len(z_cent)
    N_zsamples_theo = len(z_cent_theo)
    b_z = np.ones(N_zsamples)
    N_tomo_single = 1
    N_ell = len(ell)

    # generate the shapes
    N_many = N_zsamples # THIS IS 100 TIMES N_ZSAMPLES_THEO
    z_many = z_cent
    Delta_z_s = np.mean(np.diff(z_cent_theo))
    D_i_many_near = np.zeros((N_zsamples_theo,N_many))

    for i in range(N_zsamples_theo):
        D_i_many_near[i,:] = D_i_near(z_many,z_cent_theo[i],Delta_z_s)

    # Compute matter pk using halofit
    pk_mm_arr = np.array([ccl.nonlin_matter_power(cosmo_fid, k_arr, a) for a in a_arr])
    pk_gg_arr = pk_mm_arr.copy() # because we later include galaxy bias
    pk_gm_arr = pk_mm_arr.copy() # because we later include galaxy bias

    # Create pk2D objects for interpolation
    pk_mm = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_arr), pk_arr=np.log(pk_mm_arr), is_logp=True)
    pk_gg = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_arr), pk_arr=np.log(pk_gg_arr), is_logp=True)
    pk_gm = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_arr), pk_arr=np.log(pk_gm_arr), is_logp=True)
    
    # load the dndz parameters for curly C matrix element i_zs,j_zs
    dndz_z_single_i = np.zeros((N_tomo_single,N_zsamples))
    dndz_z_single_j = np.zeros((N_tomo_single,N_zsamples))
    for i_z in range(N_tomo_single):
        dndz_z_single_i[i_z,:] = D_i_many_near[i_zs,:]
        dndz_z_single_j[i_z,:] = D_i_many_near[j_zs,:]
        
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
            cl_gs_no = cl_gs + 0.
            CL_ALL[(N_ell*c):(N_ell*c)+N_ell] = cl_gs_no

        elif t_i*2+t_j == 3: # this is ss
            tracer_z1 = ccl.WeakLensingTracer(cosmo_fid, dndz=(z_cent, dndz_z_single_i[i,:]))
            tracer_z2 = ccl.WeakLensingTracer(cosmo_fid, dndz=(z_cent, dndz_z_single_j[j,:]))
            cl_ss = ccl.angular_cl(cosmo_fid, tracer_z1, tracer_z2, ell, p_of_k_a=pk_mm)
            cl_ss_no = cl_ss + 0.
            CL_ALL[(N_ell*c):(N_ell*c)+N_ell] = cl_ss_no
                
    return CL_ALL
            
