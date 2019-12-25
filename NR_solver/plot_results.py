import matplotlib.pyplot as plt
import numpy as np
import sys
import scipy.linalg as la

#spec = '_noise_7_20'
spec = ''
tru, ans, ini = np.loadtxt("results_dndz_bz_true_answer_initial"+spec+".txt",unpack=True)

N_tomo = int(sys.argv[1])
N_zsamples = int(sys.argv[2])
exponent = int(sys.argv[3])
par_len = N_zsamples*N_tomo

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

# redshift parameters
z_ini_sample = 0.
z_end_sample = 2.
z_ini_bin = 0.
z_end_bin = 2.

z_s_edges = np.linspace(z_ini_sample,z_end_sample,N_zsamples+1)
z_s_cents = (z_s_edges[1:]+z_s_edges[:-1])*.5
z_bin_edges = np.linspace(z_ini_bin,z_end_bin,N_tomo+1)
z_bin_cents = (z_bin_edges[1:]+z_bin_edges[:-1])*.5

# compute the error bars from the fisher matrix
fisher = np.load("fisher"+spec+".npy")#[:N_zsamples*N_tomo,:N_zsamples*N_tomo]# TESTING

# check if inv fisher is positive definite
if (is_pos_def(fisher) != True): print("Fisher is not positive definite!"); exit(0)

# marginalize the b parameters
fisher_sumfix = np.zeros((N_zsamples*N_tomo,N_zsamples*N_tomo))
ones = np.ones((N_zsamples,N_zsamples))*10.**(exponent)
for i in range(N_tomo):                     
    fisher_sumfix[i*N_zsamples:i*N_zsamples+N_zsamples,i*N_zsamples:i*N_zsamples+N_zsamples] = ones

offset = 0*par_len
offset_end = 1*par_len
# importantly we add this only to the dndz part of the fisher matrix
fisher[offset:offset_end,offset:offset_end] += fisher_sumfix

# compute inverse fisher
inv_fisher = la.inv(fisher)

# check if inv fisher is positive definite
#if (is_pos_def(inv_fisher) != True): print("Inverse Fisher is not positive definite!"); exit(0)

# get only the marginalized errors on each element
inv_F_alpha_alpha = np.diag(inv_fisher)
sig = np.sqrt(inv_F_alpha_alpha)

plt.figure(figsize=(33,4))
print("why you so effing slow")
for i in range(N_tomo):
    t_t = tru[N_zsamples*i:N_zsamples*i+N_zsamples]
    a_t = ans[N_zsamples*i:N_zsamples*i+N_zsamples]
    i_t = ini[N_zsamples*i:N_zsamples*i+N_zsamples]
    s_t = sig[N_zsamples*i:N_zsamples*i+N_zsamples]

    z_bin_ini = z_bin_edges[i]
    z_bin_end = z_bin_edges[i+1]
    z_bin_mid = 0.5*(z_bin_ini+z_bin_end)

    plt.subplot(1,N_tomo,i+1)

    plt.title('dndz constraints with noise, z = %f'%(z_bin_mid))
    plt.plot(0.5*(z_s_edges[:-1]+z_s_edges[1:]), t_t, label='Truth')
    plt.errorbar(0.5*(z_s_edges[:-1]+z_s_edges[1:]), a_t, yerr=s_t, label='Answer')
    print(s_t)
    plt.plot(0.5*(z_s_edges[:-1]+z_s_edges[1:]), i_t, label='Init. Guess')
    plt.ylim([-.1,1])
plt.legend()
plt.savefig('figs/dndz_reg_tomo_'+str(N_tomo)+'_zsam_'+str(N_zsamples)+'.png')
plt.close()


plt.figure(figsize=(33,4))
print("why you so effing slow")
for i in range(N_tomo,2*N_tomo):
    t_t = tru[N_zsamples*i:N_zsamples*i+N_zsamples]
    a_t = ans[N_zsamples*i:N_zsamples*i+N_zsamples]
    i_t = ini[N_zsamples*i:N_zsamples*i+N_zsamples]
    s_t = sig[N_zsamples*i:N_zsamples*i+N_zsamples]
    
    z_bin_ini = z_bin_edges[i-N_tomo]
    z_bin_end = z_bin_edges[i+1-N_tomo]
    z_bin_mid = 0.5*(z_bin_ini+z_bin_end)

    plt.subplot(1,N_tomo,i+1-N_tomo)

    plt.title('bz constraints with noise, z = %f'%(z_bin_mid))
    plt.plot(0.5*(z_s_edges[:-1]+z_s_edges[1:]), t_t, label='Truth')
    plt.errorbar(0.5*(z_s_edges[:-1]+z_s_edges[1:]), a_t, yerr=s_t, label='Answer')
    plt.plot(0.5*(z_s_edges[:-1]+z_s_edges[1:]), i_t, label='Init. Guess')
    plt.ylim([0,5])
plt.legend()
plt.savefig('figs/bz_reg_100_tomo_'+str(N_tomo)+'_zsam_'+str(N_zsamples)+'.png')
plt.close()
