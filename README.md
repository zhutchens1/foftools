# foftools - galaxy group identification using the friends-of-friends algorithm.


This package was written to include a variety of computational tools for performing friends-of-friends group identification analyses, especially as related to the RESOLVE/ECO surveys (Kannappan et al. 2011, Moffett et al. 2015).

## Using foftools
There are two key functions  in the package:
1. the `fof.fast_fof` function, an implementation of the friends-of-friends algorithm (Huchra & Geller, 1982; Berlind et al. 2006).
2. the `fof.fast_pfof` function, an implementation of the probability friends-of-friends algorithm (Liu et al. 2009).

### The Friends-of-Friends Algorithm
We sort galaxe following the process described in Berlind et al. (2006). In this approach, two galaxies are considered friends if the perpendicular and line-of-sight comoving distances between them are each less than a characteristic linking length. The perpendicular and line-of-sight linking lengths are products of the mean separation between galaxies with the perpendicular and line-of-sight linking constants. Application of this algorithm requires computing transverse sky-distances between galaxies

The choice of linking constant is dependent on survey setup and scientific goals. As in Eckert et al. (2017), groups identified in the RESOLVE and ECO surveys use linking constants that are optimized for the study of galaxy environment, `bperp=0.07` and `blos=1.1`, as in Duarte & Mamon (2014).

Our implementation of the FoF algorithm is the `fof.fast_fof` function. It can be called as:
```
fof.fast_fof(ra, dec, cz, bperp, blos, s)
```
Here `ra`, `dec`, and `cz` are iterables that represent the coordinates of galaxies to be included in the group-finding. The `bperp` and `blos` parametersare the dimensionless linking factors that optimize the mean separation between galaxies, `s`, to identify groups. The `fast_fof` function will return a NumPy array containing a group identification number for every input galaxy.


**Note: The FoF algorithm will run helper functions for calculating the perpendicular and line-of-sight comoving distances between galaxies. The `foftools` package defaults to a LambdaCDM cosmology with H0 = 100.0, OmegaM = 0.3, and OmegaDE = 0.7. These can be modified in the source.**

### The Probability Friends-of-Friends Algorithm

The package includes an implementation of the probability friends-of-friends algorithm (Liu et al. 2008) for volume-limited data sets. This algorithm presents a modification to the traditional friends-of-friends linking criteria. In PFoF, galaxies are modeled with probability distribution functions, and satisfaction of the line-of-sight linking criterion is given probabilistically. Our implementation models the galaxy PDFs with Gaussian distributions whose centroids are the galaxies' measured redshifts and whose standard deviations are the errorbars on the galaxies' redshifts. The routine can be called from
```
fof.fast_pfof(ra, dec, cz, czerr, perpll, losll, Pth)
``` 
This function follows the same input scheme as `foftools.fast_fof`, but requires an additional input array `czerr` that represents the error bars on individual galaxy redshift velocities. The `perpll` and `losll` are the on-sky and line-of-sight linking lengths in Mpc/h and km/s respectively. The last parameter, `Pth`, is the threshold probability used to construct to a group catalog from PFoF's friendship probabilities.

The PFoF algorithm is very computationally-expensive; the supporting functions require Numba's `@njit` decorator to improve integration speed for galaxy friendship probabilities.


## References
- Kannappan, Sheila, et al. "The RESOLVE Survey: REsolved Spectroscopy Of a Local VolumE." Bulletin of the American Astronomical Society. Vol. 43. 2011.
- Moffett, Amanda J., et al. "ECO and RESOLVE: Galaxy Disk Growth in Environmental Context." The Astrophysical Journal 812.2 (2015): 89.
- Eckert, Kathleen D., et al. "RESOLVE and ECO: The Halo Mass-dependent Shape of Galaxy Stellar and Baryonic Mass Functions." The Astrophysical Journal 824.2 (2016): 124.
- Berlind, Andreas A., et al. "Percolation galaxy groups and clusters in the SDSS redshift survey: identification, catalogs, and the multiplicity function." The Astrophysical Journal Supplement Series 167.1 (2006): 1.
- Liu, Hauyu Baobab, et al. "A new galaxy group finding algorithm: Probability friends-of-friends." The Astrophysical Journal 681.2 (2008): 1046.
