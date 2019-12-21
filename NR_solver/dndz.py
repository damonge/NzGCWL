import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d

# Lorentzian function
def lorentz(x, A, mu, sig):
    return A/(1+(np.abs(x-mu)/(0.5*sig))**2.7)

def get_nz_from_photoz_bins(zp_code,zp_ini,zp_end,zt_edges,zt_nbins):
    # Select galaxies in photo-z bin
    sel = (cat[zp_code] <= zp_end) & (cat[zp_code] > zp_ini)

    # Effective number of galaxies
    ngal = len(cat) * np.sum(cat['weight'][sel])/np.sum(cat['weight'])
    # Mean spectroscopic redshift and standard deviation
    mean = np.sum(photo_zs*weights)/np.sum(weights)
    sigma = np.sqrt(np.sum(photo_zs**2*weights)/np.sum(weights)-mean**2)
    
    # return the spectroscopic redshifts and the weights
    photo_zs = cat['PHOTOZ'][sel]                                                                          
    weights = cat['weight'][sel]

    # Make a normalized histogram
    nz, z_bins=np.histogram(photo_zs,                    # 30-band photo-zs
                            bins=zt_nbins,               # Number of bins
                            range=zt_edges,              # Range in z_true
                            weights=weights,             # Color-space weights
                            density=True)


    return nz, z_bins, ngal, mean, sigma

# N_zsamples_theo refers to the number of true samples taken for each tomographic bin
# N_tomo is the number of tomographic bins  into which we split the data
def summon_dndz(z_bin_edges,N_zsamples_theo,N_zsamples,cat_dir):
    # Tomographic bins
    N_tomo = len(z_bin_edges)-1
    # zsamples are only used when running pyccl to mitigate effects of spline by
    # enforcing a step-like function (nearest) when feeding it to pyccl
    z_ini_sample = z_bin_edges[0]
    z_end_sample = z_bin_edges[-1]
    z_s_edges = np.linspace(z_ini_sample,z_end_sample,N_zsamples+1)
    z_s_cents = (z_s_edges[1:]+z_s_edges[:-1])*.5

    # Read catalog # REMOVE TESTING
    cat = fits.open(cat_dir)[1].data

    # initiating the arrays
    dndz_data = np.zeros((N_tomo,N_zsamples))
    dndz_data_theo = np.zeros((N_tomo,N_zsamples_theo))
    N_gal_bin = np.zeros(N_tomo)

    # Get a redshift distribution for galaxies
    # with z_bin_ini < z_ephor_ab < z_bin_end
    # fill the dndz_data values with the interpolated samples
    # dndz_data_theo values with the discrete samples
    for i in range(N_tomo):
        z_bin_ini = z_bin_edges[i]
        z_bin_end = z_bin_edges[i+1]
        z_bin_mid = 0.5*(z_bin_ini+z_bin_end)        

       
        dndz_this, z_edges_theo, N_gal_this, mean, sigma = \
                        get_nz_from_photoz_bins(zp_code='pz_best_eab',# Photo-z code
                                                zp_ini=z_bin_ini, zp_end=z_bin_end, # Bin edges
                                                zt_edges=(z_ini_sample, z_end_sample),          # Sampling range
                                                zt_nbins=N_zsamples_theo)         # Number of samples


        # Centers of the zsamples theo # maybe change TESTING
        # This is effectively the mid-values of np.linspace(0,2,N_zsamples_theo+1)
        z_cents_theo = 0.5*(z_edges_theo[:-1]+z_edges_theo[1:])        
        # area under the curve (must be 1)
        #sum_dndz = np.sum(dndz_this*(z_edges_theo[1]-z_edges_theo[0])) # equals 1

        # Important normalization which matches the quantitative code
        dndz_this *= (z_edges_theo[1]-z_edges_theo[0]) 
        
        # leftmost and rightmost edges of the samples
        z_out = np.array([z_s_cents[0],z_s_cents[-1]])
        dndz_out = np.array([dndz_this[0],dndz_this[-1]])


        # interpolating from N_zsamples_theo points
        f = interp1d(np.append(z_cents_theo,zout),\
            np.append(dndz_this,dndz_out),kind='nearest',bounds_error=0,fill_value=0.)
            
        # record discrete dndzs
        dndz_data_theo[i,:] = dndz_this

        # record the interpolated dndzs
        dndz_data[i,:] = f(z_s_cents)

        # Record number of galaxies
        N_gal_bin[i] = N_gal_this
        
    # Sanity checks
    print(N_gal_bin)
    return N_gal_bin, dndz_data_theo, dndz_data
