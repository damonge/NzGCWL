import numpy as np
import scipy.linalg as la

from dndz import summon_dndz
from bz import summon_bz
from Cls import compute_Cls, compute_Cov
from curly_C import compute_mCs
from fast_Cls import compute_fast_Cls
from regularization import summon_D, gaussian
from newton_raphson import adaptable_nr


# number of tomo bins and zsamples
N_tomo = 7
N_zsamples_theo = 20
catalog_dir = "../data/cosmos_weights.fits"
N_zsamples = N_zsamples_theo*10
z_ini = 0.
z_end = 2.

# redshift parameters
z_s_edges = np.linspace(z_ini,z_end,N_zsamples+1)
z_s_cents = (z_s_edges[1:]+z_s_edges[:-1])*.5
z_s_edges_theo = np.linspace(z_ini,z_end,N_zsamples_theo+1)
z_s_cents_theo = (z_s_edges_theo[1:]+z_s_edges_theo[:-1])*.5
z_bin_edges = np.linspace(z_ini,z_end,N_tomo+1)
z_bin_cents = (z_bin_edges[1:]+z_bin_edges[:-1])*.5

# ellipticity noise of gals in HSC
sigma_e2 = np.ones(N_tomo)*(.4**2)

# Power spectrum parameters
N_ell = 10
ells = np.linspace(100., 2000., N_ell) 
k_ar = np.logspace(-4.3, 3, 1000)
a_ar = (1./(1+np.linspace(0., 3., 50)))[::-1]

# area of samples
area_COSMOS_HSC = 1.7*np.pi/180.**2 # in sq rad
f_sky_HSC = 100./41253.

N_gal_bin, dndz_data_theo, dndz_data = summon_dndz(z_bin_edges,N_zsamples_theo,N_zsamples,catalog_dir)
bz_data_theo, bz_data, cosmo_fid = summon_bz(N_tomo,z_s_cents_theo,z_s_cents)


cl_fname = "Cltrue_"+str(tomo)+"_"+str(zsam)+"_0.npy"
if not os.path.isfile(cl_fname):
    print ("Computing Cl_true")
    Cl_true = compute_Cls(dndz_data, bz_data, z_s_cents,N_gal_bin,ells,a_ar,k_ar,sigma_e2,area_COSMOS_HSC)
    print ("done")
    np.save(cl_fname,Cl_true)
else:
    Cl_true = np.load(cl_fname)

# add noise
add_noise = False#True # TESTING
if (add_noise):
    cln_fname = "Clnoise_"+str(tomo)+"_"+str(zsam)+"_0.npy"
    if not os.path.isfile(cln_fname):
        print ("computing cov for noise")
        Cov = compute_Cov(Cl_all,N_tomo,ells,f_sky_HSC)
        print ("done")
        np.save(cln_fname,Cov)
    else:
        Cov = np.load(cln_fname)
    R = np.random.normal(0.,1.,N_ell*N_tomo*(2*N_tomo+1))  
    X = la.cholesky(Cov).T                                                                      
    Nl = np.dot(X,R)                                                                            
    # adding noise                                                             
    Cl_true += 0.2*Nl
    # 0.2 to reduce noise

Delta_z_bin = np.mean(np.diff(z_bin_edges))
s = .1 #TESTING maybe change for bz
D = summon_D(N_tomo,N_zsamples_theo,Delta_z_bin,s,first=True,second=True,sum=True)

# curly C!
cov_fname="mat_C_"+str(N_tomo)+"_"+str(N_zsamples_theo)+".npy"
if not os.path.isfile(cov_fname):
    print ("Generating ",cov_fname)
    N_tomo_single = 1
    N_elm = (2*N_tomo_single+1)*N_tomo_single*N_ell # = 30 for gg, gs, ss
    mat_C = np.zeros((N_zsamples_theo,N_zsamples_theo,N_elm))
    
    # NOTE THAT curly C, i.e. mat_C is not symmetric
    for i in range(N_zsamples_theo):
        for j in range(N_zsamples_theo):
            print(i,j)
            # In the function the dndz parameters are updated to the D_i_many guys and bz is 0
            Cl_many = compute_mCs(cosmo_fid,i_zs=i,j_zs=j,z_s_cents_theo,z_s_cents,a_ar,k_ar,ells)
            print(Cl_many.sum())
            # record matrix entries for this choice of zsample bins i and j
            mat_C[i,j,:] = Cl_many 
        
    # Save curly C matrix
    np.save("mat_C_"+str(N_tomo)+"_"+str(N_zsamples_theo)+".npy",mat_C)

# Load the curly C
mat_C = np.load("mat_C_"+str(tomo)+"_"+str(zsam)+".npy")

'''
# Can either use Cl_true with the approximation or the original
compute_fast_Cls(dndz_data_theo,mat_C,bz_data_theo,N_gal_bin,ells,sigma_e2,area_COSMOS,compute_2nd_ders=False)
'''
# number of power spectrum elements
N_elm = len(Cl_true)
Cl_true = Cl_true.reshape(N_elm,1)

# Initial Guess for the Newton-Raphson method
# bz guess
bz_guess = bz_data_theo.flatten()+0.1*np.random.randn(N_zsamples_theo*N_tomo)#TESTING

# full parameter vector
full_x = np.hstack((dndz_data_theo.flatten().copy(),bz_guess))

# dndz guess
gauss_guess = 1
if gauss_guess == 1:
    for i in range(N_tomo):
        dndz_guess = gaussian(z_s_cents_theo,z_bin_cents[i],0.2)
        dndz_guess /= np.sum(dndz_guess)
        full_x[i*N_zsamples_theo:(i+1)*N_zsamples_theo] = dndz_guess

# Initial full guess
full_x0 = full_x.copy()

# TODO check whether this works, do weak prior on bz, make sure the adaptable step works

dndz_answer, bz_answer = adaptable_nr(full_x,mat_C,D,N_tomo,N_zsamples_theo,steps=15,N_gal_bin,ells,sigma_e2,area_COSMOS,vary_only=False)

print("dndz_true = ",(dndz_data_theo.flatten()))
print("dndz_answer = ",(dndz_answer))
print("dndz_guess = ",(full_x0[:N_zsamples_theo*N_tomo]))
print("__________________________________________")
print("bz_true = ",(bz_data_theo.flatten()))
print("bz_answer = ",((bz_answer)))
print("bz_guess = ",((full_x0[N_zsamples_theo*N_tomo:2*N_zsamples_theo*N_tomo])))

with open('results_dndz_bz_true_answer_initial.txt','w') as f:
    for a,b,c in zip(np.hstack((dndz_data_theo.flatten(),bz_data_theo.flatten())), full_x, full_x0):
        f.write("%f %f %f \n"%(a,b,c))
