import numpy as np
import scipy.linalg as la
from fast_Cls import compute_fast_Cls

def adaptable_nr(Cl_true,dndz_data_theo,bz_data_theo,full_x,mat_C,D,N_tomo,N_zsamples_theo,N_gal_sample,ell,sigma_e2,area_overlap,f_sky,steps,vary_only=False):
    # Initial step-size
    alpha_ini = 1.
    alpha = alpha_ini
    for s in range(steps):    
        dndz_this = full_x[:(N_zsamples_theo*N_tomo)]
        bz_this = full_x[(N_zsamples_theo*N_tomo):2*(N_zsamples_theo*N_tomo)]
        
        print("Delta_dndz = ",np.sqrt(np.sum((dndz_data_theo.flatten()-dndz_this)**2)))
        print("Delta_bz = ",np.sqrt(np.sum((bz_data_theo.flatten()-(bz_this))**2)))
        
        # compute the Cls and their derivatives analytically
        Cl_fast, dCldp_fast, Cov_fast = compute_fast_Cls(dndz_this,mat_C,bz_this,N_gal_sample,ell,sigma_e2,area_overlap,f_sky)
        N_elm = len(Cl_fast)
        Cl_fast = Cl_fast.reshape(N_elm,1)
        iCov_fast = la.inv(Cov_fast)
        Delta_Cl = Cl_fast-Cl_true

        # regularization params
        n_diff = (dndz_this).reshape(N_zsamples_theo*N_tomo,1)
        
        Reg_V = np.dot(D,n_diff)#-1./N_tomo#TESTING # a bit better
        Reg_A = D
        reg = 1. # originally we used to vary this
        
        print("sum_dndz = ",np.sum(dndz_this))
        print("sum_bz = ",np.sum((bz_this)))    
        
        # compute chi2 for this try
        if s == 0: chi2_old = 1.e50; chi2_min = 1.e50
        else: chi2_old = chi2
        if chi2_old < chi2_min: chi2_min = chi2_old
        chi2 = np.dot(Delta_Cl.T,np.dot(iCov_fast,Delta_Cl))[0][0]
        print("chi2 = ",chi2)
        print("__________________________")
            
        # Regularization for keeping sum fixed and derivatives smooth
        # Reg matrix A
        R_full_A = np.zeros((2*N_zsamples_theo*N_tomo,2*N_zsamples_theo*N_tomo))
        R_full_A[:N_zsamples_theo*N_tomo,:N_zsamples_theo*N_tomo] = reg*Reg_A
        # Reg vector V
        R_full_V = np.zeros((2*N_zsamples_theo*N_tomo,1))
        R_full_V[:N_zsamples_theo*N_tomo] = reg*Reg_V

        # TESTING Regularization for b(z) -- may need to change the factors
        R_full_A[N_zsamples_theo*N_tomo:N_zsamples_theo*N_tomo*2,N_zsamples_theo*N_tomo:N_zsamples_theo*N_tomo*2] = reg*Reg_A
        R_full_V[N_zsamples_theo*N_tomo:N_zsamples_theo*N_tomo*2] = reg*Reg_V
        
        # original version without second derivative
        A = np.dot(dCldp_fast,np.dot(iCov_fast,dCldp_fast.T)) + R_full_A
        V = np.dot(dCldp_fast,np.dot(iCov_fast,Delta_Cl)) + R_full_V 

        # fixing dndz and varying only 'bz' or vice versa or varying everything
        if vary_only == 'bz': select1 = 1; select2 = 2
        if vary_only == 'dndz': select1 = 0; select2 = 1
        if vary_only == False: select1 = 0; select2 = 2
        A = A[select1*N_zsamples_theo*N_tomo:select2*N_zsamples_theo*N_tomo,select1*N_zsamples_theo*N_tomo:select2*N_zsamples_theo*N_tomo]
        V = V[select1*N_zsamples_theo*N_tomo:select2*N_zsamples_theo*N_tomo]    
        iA = la.inv(A)

        # Compute the next step
        step_nr = np.dot(iA,V).flatten()
        if alpha < alpha_ini: alpha = alpha_ini
        full_x[select1*N_zsamples_theo*N_tomo:select2*N_zsamples_theo*N_tomo] += -alpha*step_nr

        print("chi2_min, chi2 = ",chi2_min, chi2)
        # if new try is worse than the previous
        while (chi2_min-1.e-6 < chi2):
            print("In the loop")
            # go back to where it was
            full_x[select1*N_zsamples_theo*N_tomo:select2*N_zsamples_theo*N_tomo] += alpha*step_nr
            alpha /= 2.
            # take half step
            full_x[select1*N_zsamples_theo*N_tomo:select2*N_zsamples_theo*N_tomo] += -alpha*step_nr

            # extract the parameter values
            dndz_this = full_x[:(N_zsamples_theo*N_tomo)]
            bz_this = full_x[(N_zsamples_theo*N_tomo):2*(N_zsamples_theo*N_tomo)]
        
            # compute the Cls and their derivatives analytically
            Cl_fast, dCldp_fast, Cov_fast = compute_fast_Cls(dndz_this,mat_C,bz_this,N_gal_sample,ell,sigma_e2,area_overlap,f_sky)
            Cl_fast = Cl_fast.reshape(N_elm,1)
            iCov_fast = la.inv(Cov_fast)
            Delta_Cl = Cl_fast-Cl_true

            chi2 = np.dot(Delta_Cl.T,np.dot(iCov_fast,Delta_Cl))[0][0]
            if alpha < 0.01: break

    return dndz_this, bz_this
