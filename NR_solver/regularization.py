import numpy as np

# 1d gaussian
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def regularizator(N_tomo,N_zsamples_theo,Delta_z_bin,corr=0.1,first,second,sum):
    # first and second derivatives
    D0 = np.zeros((N_zsamples_theo*N_tomo,N_zsamples_theo*N_tomo))
    D1 = np.zeros((N_zsamples_theo*N_tomo,N_zsamples_theo*N_tomo))
    D2 = np.zeros((N_zsamples_theo*N_tomo,N_zsamples_theo*N_tomo))
    theo = dndz_data_theo.flatten()
    for i in range(N_tomo):
        D0[i*N_zsamples_theo:(i+1)*N_zsamples_theo,i*N_zsamples_theo:(i+1)*N_zsamples_theo] += np.ones((N_zsamples_theo,N_zsamples_theo))
        for j in range (0,N_zsamples_theo-1):
            lam = np.zeros(N_zsamples_theo*N_tomo)
            lam[i*N_zsamples_theo+j]=1
            lam[i*N_zsamples_theo+j+1]=-1
            D1+=np.outer(lam,lam)
        for j in range (1,N_zsamples_theo-1):
            lam = np.zeros(N_zsamples_theo*N_tomo)
            lam[i*N_zsamples_theo+j-1]=-1
            lam[i*N_zsamples_theo+j]=2
            lam[i*N_zsamples_theo+j+1]=-1
            D2+=np.outer(lam,lam)
            

    S = Delta_z_bin
    s = corr

    sigma1 = (Delta_z_bin**2/(10.*S*s))
    sigma2 = (Delta_z_bin**4/(10.*S*s**2))
    sigma1sq = sigma1**2
    sigma2sq = sigma2**2

    D0 /= N_tomo**2#TESTING ask Anze
    D1 /= (sigma1sq)
    D2 /= (sigma2sq)

    if first == False: D1 *= 0
    if second == False: D2 *= 0
    if sum == False: D0 *= 0

    D = D0+D1+D2
