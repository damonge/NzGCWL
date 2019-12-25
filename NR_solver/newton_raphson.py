import numpy as np
import scipy.linalg as la
from fast_Cls import compute_fast_Cls


def compute_chi2(Cl_fast,Cl_true,iCov_fast):
    N_elm = len(Cl_fast)
    Cl_fast = Cl_fast.reshape(N_elm,1)
    Delta_Cl = Cl_fast-Cl_true
    chi2 = np.dot(Delta_Cl.T,np.dot(iCov_fast,Delta_Cl))[0][0]
    print("chi2 = ",chi2)
    return chi2

def obtain_R_AV(dndz_this,D,len_par):
    # Regularization for keeping sum fixed and derivatives smooth
    Reg_V = np.dot(D,dndz_this.reshape(len_par,1))#-1./N_tomo#TESTING # a bit better
    Reg_A = D
        
    # Reg matrix A
    R_A = np.zeros((2*len_par,2*len_par))
    R_A[:len_par,:len_par] = Reg_A
    # Reg vector V
    R_V = np.zeros((2*len_par,1))
    R_V[:len_par] = Reg_V

    # TESTING Regularization for b(z) -- may need to change the factors
    R_A[len_par:len_par*2,len_par:len_par*2] = Reg_A
    R_V[len_par:len_par*2] = Reg_V

    return R_A, R_V

def obtain_AV(dCldp_fast,Cl_fast,Cl_true,iCov_fast,R_A,R_V,select1,select2):
    N_elm = len(Cl_fast)
    Cl_fast = Cl_fast.reshape(N_elm,1)
    Delta_Cl = Cl_fast-Cl_true
    
    # We neglect the second derivative with repect to Cl
    A = np.dot(dCldp_fast,np.dot(iCov_fast,dCldp_fast.T)) + R_A
    V = np.dot(dCldp_fast,np.dot(iCov_fast,Delta_Cl)) + R_V  
    
    A = A[select1:select2,select1:select2]
    V = V[select1:select2]
    
    return A, V


def prior_bz_AV(bz,bz0,prior_bz):
    if not prior_bz: return 0., 0. 
    sigma = .8#1.
    I = np.eye(len(bz))
    A = I/sigma**2
    V = (bz-bz0)/sigma**2
    return A, V

def adaptable_nr(Cl_true,dndz_data_theo,bz_data_theo,full_x,mat_C,D,N_tomo,N_zsamples_theo,N_gal_sample,ell,sigma_e2,area_overlap,f_sky,steps,vary_only=False):

    # do we want to have a prior on bz
    prior_bz = 1#False
    
    # number of parameters
    len_par = N_zsamples_theo*N_tomo

    # number of elements
    N_elm = len(Cl_true)
    
    # which parameters are we varying
    if vary_only == 'bz': select1 = 1; select2 = 2
    if vary_only == 'dndz': select1 = 0; select2 = 1
    if vary_only == False: select1 = 0; select2 = 2
    select1 *= len_par; select2 *= len_par

    # Initialize the chi2 value
    chi2 = 1.e50; chi2_min = 1.e50
    
    # Initial step-size
    alpha_ini = 1.
    alpha = alpha_ini
    for s in range(steps):
        print("______________  Step = ",s,"______________")

        # extract the parameters
        dndz_this = full_x[:len_par]; bz_this = full_x[len_par:2*len_par]
        
        # compute the Cls and their derivatives analytically
        Cl_fast, dCldp_fast, Cov_fast = compute_fast_Cls(dndz_this,mat_C,bz_this,N_gal_sample,
                                                         ell,sigma_e2,area_overlap,f_sky)
        iCov_fast = la.inv(Cov_fast)

        print("Delta_dndz = ",np.sqrt(np.sum((dndz_data_theo.flatten()-dndz_this)**2)))
        print("Delta_bz = ",np.sqrt(np.sum((bz_data_theo.flatten()-(bz_this))**2)))
        print("sum_dndz = ",np.sum(dndz_this))
        print("sum_bz = ",np.sum((bz_this)))    
        
        if chi2 < chi2_min:
            chi2_min = chi2; dndz_ans = dndz_this.copy(); bz_ans = bz_this.copy()
            iCov_ans = iCov_fast; dCldp_ans = dCldp_fast 
        # compute chi2 for this try
        chi2 = compute_chi2(Cl_fast,Cl_true,iCov_fast)

        # if new try is worse than the previous
        while (chi2_min-1.e-6 < chi2):
            print("In the loop")
            # go back to where the smaller value of chi2 was
            full_x[select1:select2] += alpha*step_nr
            alpha /= 2.

            # take half step instead
            full_x[select1:select2] += -alpha*step_nr

            # extract the parameter values
            dndz_this = full_x[:len_par]; bz_this = full_x[len_par:2*len_par]
        
            # compute the Cls and their derivatives analytically
            Cl_fast, dCldp_fast, Cov_fast = compute_fast_Cls(dndz_this,mat_C,bz_this,N_gal_sample,ell,sigma_e2,area_overlap,f_sky)
            iCov_fast = la.inv(Cov_fast)

            chi2 = compute_chi2(Cl_fast,Cl_true,iCov_fast)
            if alpha < 0.01: break

        # regularization matrix and vector    
        R_A, R_V = obtain_R_AV(dndz_this,D,len_par)

        # prior on bz
        a, v = prior_bz_AV(bz_this,bz_data_theo.flatten(),prior_bz)
        print(a.shape)
        print(v.shape)
        print(R_V.shape)
        
        R_A[len_par:2*len_par,len_par:2*len_par] += a
        R_V[len_par:2*len_par,0] += v
        
        A, V = obtain_AV(dCldp_fast,Cl_fast,Cl_true,iCov_fast,R_A,R_V,select1,select2)
        
        iA = la.inv(A)
            
        # Compute the next step
        step_nr = np.dot(iA,V).flatten()
        if alpha < alpha_ini: alpha = alpha_ini
        full_x[select1:select2] += -alpha*step_nr

    return dndz_ans, bz_ans, dCldp_ans, iCov_ans
