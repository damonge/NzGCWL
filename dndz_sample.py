import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

# Read catalog
cat=fits.open("data/cosmos_weights.fits")[1].data

def get_nz_from_photoz_bins(zp_code,zp_ini,zp_end,zt_edges,zt_nbins):
    # Select galaxies in photo-z bin
    sel=(cat[zp_code]<=zp_end) & (cat[zp_code]>zp_ini)

    # Effective number of galaxies
    ngal=len(cat) * np.sum(cat['weight'][sel])/np.sum(cat['weight'])

    # Make a normalized histogram
    nz,z_bins=np.histogram(cat['PHOTOZ'][sel],          # 30-band photo-zs
                           bins=zt_nbins,               # Number of bins
                           range=zt_edges,              # Range in z_true
                           weights=cat['weight'][sel],  # Color-space weights
                           density=True)

    return nz, z_bins, ngal


# This is how you get a redshift distribution for galaxies
# with 0.8 < z_ephor_ab < 1.0. The redshift distribution will
# be sampled in 10 bins between in the range 0 < z_true < 2.
z_ini = 0.8
z_end = 1.0
dndz, z_edges, ngal = get_nz_from_photoz_bins(zp_code='pz_best_eab',      # Photo-z code
                                              zp_ini=z_ini, zp_end=z_end, # Bin edges
                                              zt_edges=(0., 2.),          # Sampling range
                                              zt_nbins=10)                # Number of samples
plt.figure()
plt.title(" %.1lf < z_ph < %.1lf, %.1lf sources N(z)" % (z_ini, z_end, ngal))
plt.plot(0.5*(z_edges[:-1]+z_edges[1:]), dndz)
plt.xlabel("z", fontsize=14)
plt.ylabel("p(z)", fontsize=14)
plt.show()
