import numpy as np
from itertools import combinations

def compute_fast_Cls(dndz_z_curr,mat_cC,b_z_curr,N_gal_sample,ell,sigma_e2,area_overlap,compute_2nd_ders=False):
    # After obtaining the mCs we can now do simple linalg to get the Cls
    # check if it is the correct shape
    if dndz_z_curr.shape[0] != N_tomo: dndz_z_curr = dndz_z_curr.reshape(N_tomo,N_zsamples_theo)
    if b_z_curr.shape[0] != N_tomo: b_z_curr = b_z_curr.reshape(N_tomo,N_zsamples_theo)
    # number of ell elements
    N_ell = len(ell)
    # correlation number for all types gg gs ss and all combinations of redshift
    tot_corr = N_ell*(N_tomo*(2*N_tomo+1))
    
    # make Cl_fast of size N_ell*N_tomo*(2N_tomo+1) as this is the tot no. of corrs
    Cl_fast_all = np.zeros(tot_corr)
    # We would have the same number of parameters for dndz and bz
    dCl_fast_all = np.zeros((2*N_tomo*N_zsamples_theo,tot_corr))
    ddCl_fast_all = np.zeros((N_tomo*N_zsamples_theo,N_tomo*N_zsamples_theo,tot_corr))

    # all combinations
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
            N_gal = N_gal_sample[i_tomo]
            n_gal = N_gal/area_overlap # in rad^-2 
            # computing noise
            noise_gal = 1./n_gal
            noise_shape = sigma_e2[i_tomo]/n_gal
        else:
            noise_gal = 0.
            noise_shape = 0.

        # for each combination c, the correlation C_ell has N_ell = 10 entries
        C_ell = np.zeros(N_ell)
        bias_mat = np.ones_like(b_mat)
        ni = dndz_z_curr[i_tomo,:].reshape(1,N_zsamples_theo)
        nj = dndz_z_curr[j_tomo,:].reshape(N_zsamples_theo,1)
        
        # bi/ni refers to the first redshift while bj/nj to the second
        # the notation _f shows that this is the first derivative of
        # the quantity, i.e. bi_f is dbi/dbi = 1
        # di is the combination of bi and ni
        if t_i*2+t_j == 0: # this is gg
            type_xy = 0 # type_xy is such BECAUSE THE ORDER IN mat_cC[.,.,k] is
            # gg, ss, gs (from comb=(0,0),(1,1),(0,1) and c = 0,1,2)
            noise = noise_gal
            
            bi = b_z_curr[i_tomo,:].reshape(1,N_zsamples_theo)
            bj = b_z_curr[j_tomo,:].reshape(N_zsamples_theo,1)
            bi_f = np.ones_like(bi); bj_f = np.ones_like(bj);
            ni_f = np.ones_like(bi); nj_f = np.ones_like(bj);
                
            di = ni*bi; dj = nj*bj
            
        if t_i*2+t_j == 1: # this is gs
            type_xy = 2 # type_xy is such BECAUSE THE ORDER IN mat_cC[.,.,k] is
            # gg, ss, gs (from comb=(0,0),(1,1),(0,1) and c = 0,1,2)
            noise = 0.
            bi = b_z_curr[i_tomo,:].reshape(1,N_zsamples_theo)
            bj = np.ones((N_zsamples_theo,1))
            bi_f = np.ones_like(bi); bj_f = np.zeros_like(bj);
            ni_f = np.ones_like(bi); nj_f = np.ones_like(bj);
                
            di = ni*bi; dj = nj
            
            
        if t_i*2+t_j == 3: # this is ss
            type_xy = 1 # type_xy is such BECAUSE THE ORDER IN mat_cC[.,.,k] is
            # gg, ss, gs (from comb=(0,0),(1,1),(0,1) and c = 0,1,2)
            noise = noise_shape
            bi = np.ones((1,N_zsamples_theo))
            bj = np.ones((N_zsamples_theo,1))
            bi_f = np.zeros_like(bi); bj_f = np.zeros_like(bj);
            ni_f = np.ones_like(bi); nj_f = np.ones_like(bj);
                
            di = ni; dj = nj
                
        for k in range(N_ell):
            # Here we compute the Cls analytically using the curly C matrix at wavemode k
            matC_k = mat_cC[:,:,N_ell*type_xy+k]*bias_mat
            
            C_ell[k] = np.dot(np.dot(di,matC_k),dj)
            
            if (compute_ders == True and complex_bz==True):
                # n_a der: ba na_f Caj bj nj + bi ni Cia na_f ba
                # b_a der: ba_f na Caj bj nj + bi ni Cia na ba_f
                if (i_tomo == j_tomo):
                    dCl_fast_all[i_tomo*N_zsamples_theo:(i_tomo+1)*N_zsamples_theo,(N_ell*c)+k] = np.dot(matC_k,bj*nj).T*(ni_f*bi)+np.dot(bi*ni,matC_k)*(nj_f*bj).T
                    dCl_fast_all[N_tomo*N_zsamples_theo+i_tomo*N_zsamples_theo:N_tomo*N_zsamples_theo+(i_tomo+1)*N_zsamples_theo,(N_ell*c)+k] = np.dot(matC_k,bj*nj).T*(ni*bi_f)+np.dot(bi*ni,matC_k)*(nj*bj_f).T
                else:
                    dCl_fast_all[i_tomo*N_zsamples_theo:(i_tomo+1)*N_zsamples_theo,(N_ell*c)+k] = np.dot(matC_k,bj*nj).T*(ni_f*bi)
                    dCl_fast_all[N_tomo*N_zsamples_theo+i_tomo*N_zsamples_theo:N_tomo*N_zsamples_theo+(i_tomo+1)*N_zsamples_theo,(N_ell*c)+k] = np.dot(matC_k,bj*nj).T*(ni*bi_f)
                    dCl_fast_all[j_tomo*N_zsamples_theo:(j_tomo+1)*N_zsamples_theo,(N_ell*c)+k] = np.dot(bi*ni,matC_k)*(nj_f*bj).T
                    dCl_fast_all[N_tomo*N_zsamples_theo+j_tomo*N_zsamples_theo:N_tomo*N_zsamples_theo+(j_tomo+1)*N_zsamples_theo,(N_ell*c)+k] = np.dot(bi*ni,matC_k)*(nj*bj_f).T
                    
            if (compute_2nd_ders == True): # NOT WORKING PERFECTLY
                if (i_tomo == j_tomo):
                    ddCl_fast_all[i_tomo*N_zsamples_theo:(i_tomo+1)*N_zsamples_theo,i_tomo*N_zsamples_theo:(i_tomo+1)*N_zsamples_theo,(N_ell*c)+k] = matC_k*2.
                else:
                    ddCl_fast_all[i_tomo*N_zsamples_theo:(i_tomo+1)*N_zsamples_theo,j_tomo*N_zsamples_theo:(j_tomo+1)*N_zsamples_theo,(N_ell*c)+k] = matC_k
                    ddCl_fast_all[j_tomo*N_zsamples_theo:(j_tomo+1)*N_zsamples_theo,i_tomo*N_zsamples_theo:(i_tomo+1)*N_zsamples_theo,(N_ell*c)+k] = matC_k
        # Finally add noise depending on type of correlation
        C_ell += noise
        # This is the usual, proven way of recording the Cls
        Cl_fast_all[(N_ell*c):(N_ell*c)+N_ell] = C_ell
    

    # We now compute the covariance matrix
    Cov_fast_all = np.zeros((len(Cl_fast_all),len(Cl_fast_all)))    
    # Knox formula
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

            Cov_fast_all[(N_ell*c_A):(N_ell*c_A)+\
                    N_ell,(N_ell*c_B):(N_ell*c_B)+N_ell] = np.diag(Cov_ijmn)
            Cov_fast_all[(N_ell*c_B):(N_ell*c_B)+\
                    N_ell,(N_ell*c_A):(N_ell*c_A)+N_ell] = np.diag(Cov_ijmn)


       
    if (is_pos_def(Cov_fast_all) != True): print("Covariance is not positive definite!"); exit(0)

    return Cl_fast_all, dCl_fast_all, Cov_fast_all