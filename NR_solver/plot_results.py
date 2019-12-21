import matplotlib.pyplot as plt
import numpy as np
import sys 
tru, ans, ini = np.loadtxt("results_dndz_bz_true_answer_initial_noise_7_20.txt",unpack=True)

tomo = sys.argv[1]
zsam = sys.argv[2]

# redshift parameters
z_ini_sample = 0.
z_end_sample = 2.
z_ini_bin = 0.
z_end_bin = 2.
N_tomo = int(tomo)
N_zsamples = int(zsam)

z_s_edges = np.linspace(z_ini_sample,z_end_sample,N_zsamples+1)
z_s_cents = (z_s_edges[1:]+z_s_edges[:-1])*.5
z_bin_edges = np.linspace(z_ini_bin,z_end_bin,N_tomo+1)
z_bin_cents = (z_bin_edges[1:]+z_bin_edges[:-1])*.5

plt.figure(figsize=(33,4))
print("why you so slow")
for i in range(N_tomo):
    t_t = tru[N_zsamples*i:N_zsamples*i+N_zsamples]
    a_t = ans[N_zsamples*i:N_zsamples*i+N_zsamples]
    i_t = ini[N_zsamples*i:N_zsamples*i+N_zsamples]

    z_bin_ini = z_bin_edges[i]
    z_bin_end = z_bin_edges[i+1]
    z_bin_mid = 0.5*(z_bin_ini+z_bin_end)

    plt.subplot(1,N_tomo,i+1)

    plt.title('dndz constraints with noise, z = %f'%(z_bin_mid))
    plt.plot(0.5*(z_s_edges[:-1]+z_s_edges[1:]), t_t, label='Truth')
    plt.plot(0.5*(z_s_edges[:-1]+z_s_edges[1:]), a_t, label='Answer')
    plt.plot(0.5*(z_s_edges[:-1]+z_s_edges[1:]), i_t, label='Init. Guess')
plt.legend()
plt.savefig('figs/dndz_reg_100_tomo_'+str(N_tomo)+'_zsam_'+str(N_zsamples)+'.pdf')
plt.close()

plt.figure(figsize=(33,4))
print("why you so slow")
for i in range(N_tomo,2*N_tomo):
    t_t = tru[N_zsamples*i:N_zsamples*i+N_zsamples]
    a_t = ans[N_zsamples*i:N_zsamples*i+N_zsamples]
    i_t = ini[N_zsamples*i:N_zsamples*i+N_zsamples]

    z_bin_ini = z_bin_edges[i-N_tomo]
    z_bin_end = z_bin_edges[i+1-N_tomo]
    z_bin_mid = 0.5*(z_bin_ini+z_bin_end)

    plt.subplot(1,N_tomo,i+1-N_tomo)

    plt.title('bz constraints with noise, z = %f'%(z_bin_mid))
    plt.plot(0.5*(z_s_edges[:-1]+z_s_edges[1:]), t_t, label='Truth')
    plt.plot(0.5*(z_s_edges[:-1]+z_s_edges[1:]), a_t, label='Answer')
    plt.plot(0.5*(z_s_edges[:-1]+z_s_edges[1:]), i_t, label='Init. Guess')
plt.legend()
plt.savefig('figs/bz_reg_100_tomo_'+str(N_tomo)+'_zsam_'+str(N_zsamples)+'.pdf')
plt.close()
