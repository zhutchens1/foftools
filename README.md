# foftools - galaxy group identification using the friends-of-friends algorithm.


This package was written to include a variety of computational tools for performing friends-of-friends group identification analyses, especially as related to the RESOLVE/ECO surveys (Kannappan et al. 2011, Moffett et al. 2015). Thus, the following tools are best suited in applications to volume-limited surveys with order 100 - 10,000 sample galaxies. For RESOLVE and ECO, analyses of the friends-of-friends group finding is reported in Eckert et al. 2016.

## Using foftools
There are four key pieces of code in the package:
1. the `fof.galaxy` class, for storing galaxies and their properties and associated methods
2. the `fof.fast_fof` function, an implementation of the friends-of-friends algorithm.
3. the `fof.fast_pfof` function, an implementation of the probability friends-of-friends algorithm.
4. the `fof.group` class, for storing groups and their properties and associated methods


### Galaxies
The galaxies class allows for the storing of entire galaxies and their properties into a single data type.
Suppose we have a galaxy `test01` with RA and declination (51, -29) degrees, and with absolute magnitude -18.0 and redshift 5000 km/s. We can then initialize the galaxy to `x` as
```
import foftools as fof
x = fof.galaxy("test01", 51, -29, 5000, -18.0)
```

The FOF algorithm is sensitive to RA and declination values, so we encourage users to implement higher-precision coordinate values for group-finding purposes. With the class instance, we can also pass a boolean flag to each galaxy for various purposes under the property `fof.galaxy.fl`. The group ID number for a galaxy defaults to zero, as it assumes that galaxies have not yet been sorted into groups, but this can also be initialized through the `fof.galaxy.groupID` property. The initialization attributes are:
- `fof.galaxy.ra`: right-ascension in degrees
- `fof.galaxy.dec`: declination in degrees
- `fof.galaxy.cz`: redshift velocity of target in km/s
- `fof.galaxy.mag`: absolute magnitude of galaxy
- `fof.galaxy.fl`: general purpose 1/0 flag, default None.
- `fof.galaxy.groupID`: galaxy group ID number, defaults to zero.
- `fof.galaxy.logMgas`: logarithmic gas mass of galaxy, default None.
- `fof.galaxy.logMstar`: logarithmic stellar mass of galaxy, default None.
- `fof.galaxy.czerr`: 1-sigma errorbar on `cz` value. Needed for probability FoF group finding. 

The defined class attributes are:
-  `fof.galaxy.phi`: the azimuthal coordinate of the galaxy (in radians) iin a spherical-polar coordinate system.
-  `fof.galaxy.theta`: the polar coordinate of the galaxy (in radians) in a spherical-polar coordinate system.
- `fof.galaxy.x`: the angular x-component of the galaxy's sky location.
- `fof.galaxy.y`: the angular y-component of the galaxy's sky location.
- `fof.galaxy.z`: the angular z-component of the galaxy's sky location.
- `fof.galaxy.comovingdist`: line-of-sight comoving distance to the galaxy in Mpc/h. Assumes H0=100, OmegaM=0.3, OmegaDE=0.7.

The galaxy class also includes the following methods:
- `fof.galaxy.get_cz()`: return the cz value of the galaxy.
- `fof.galaxy.set_cz(cz)`: redefine the velocity of the galaxy to value `cz`. 
- `fof.galaxy.get_groupID()`: return the groupID of the galaxy. The convention of the code is that a group ID of zero indicates the galaxy is not yet associated to any group whatsoever (i.e., meaning it has also not been explicitly identified as a single-galaxy group).
- `fof.galaxy.set_groupID(groupID)`: redefine the group ID of the galaxy to the value of `groupID`.
- `fof.galaxy.get_logMbary()`: return the galaxy's logarithmic baryonic mass (gas + stellar mass). To use this, you will need to have provided values to the `galaxy.logMstar` and `galaxy.logMgas` attributes.

### The Friends-of-Friends Algorithm
We sort groups of galaxies following the process described in Berlind et al. (2006). In this approach, two galaxies are considered friends if the perpendicular and line-of-sight comoving distances between them are each less than a characteristic linking length. The perpendicular and line-of-sight linking lengths are products of the mean separation between galaxies with the perpendicular and line-of-sight linking factors.

The choice of linking factor is dependent on survey and can be optimized for different statistical purposes. For example, a flux-limited galaxy survey may require linking factors that are redshift-dependent (cf. Liu et al. 2008). However, since RESOLVE and ECO are volume-limited, our code is written to assume the linking factors to be completely constant. However, the `foftools` module can be easily modified to meet other purposes.

Our implementation of the FoF algorithm is the `fof.fast_fof` function. It can be called as
```
fof.fast_fof(ra, dec, cz, bperp, blos, s)
```
Here `ra`, `dec`, and `cz` are iterables that represent the coordinates of galaxies to be included in the group-finding. The `bperp` and `blos` parametersare the dimensionless linking factors that optimize the mean separation between galaxies, `s`, to identify groups. The `fast_fof` function will return a NumPy array containing a group identification number for every input galaxy. This can be converted into a sequence of galaxy objects using the `fof.arrays_to_galaxies` function.

*The previously used `foftools.galaxy_fof` function is now deprecated.* This algorithm is completely vectorized using NumPy broadcasting, providing enormous performance gains over the previous version. 

**Note: The FoF algorithm will run helper functions for calculating the perpendicular and line-of-sight comoving distances between galaxies. The `foftools` package defaults to a LambdaCDM cosmology with H0 = 100.0, OmegaM = 0.3, and OmegaDE = 0.7. These can be modified in the source.**

### The Probability Friends-of-Friends Algorithm

The package includes an implementation of the probability friends-of-friends algorithm (Liu et al. 2008) for volume-limited data sets. This algorithm presents a modification to the traditional friends-of-friends linking criteria. In PFoF, galaxies are modeled with probability distribution functions, and satisfaction of the line-of-sight linking criterion is given probabilistically. Our implementation models the galaxy PDFs with Gaussian distributions whose centroids are the galaxies' measured redshifts and whose standard deviations are the errorbars on the galaxies' redshifts. Thus, each galaxy to be grouped needs to have its `galaxy.czerr` attribute defined. The routine can be called from
```
fof.prob_fof(ra, dec, cz, czerr, perpll, losll, Pth)
``` 
This function follows the same input scheme as `foftools.fast_fof`, but requires an additional input array `czerr` that represents the error bars on individual galaxy redshift velocities. The `perpll` and `losll` are the on-sky and line-of-sight linking lengths in Mpc/h and km/s respectively. The last parameter, `Pth`, is the threshold probability used to construct to a group catalog from PFoF's friendship probabilities.

### Groups

The package also includes a class `fof.group` for working with groups of galaxies. It initialized merely by a groupID number, e.g., `y = fof.group(551)`. This group will be empty, so we can add members by using the `fof.group.add_member` method. For example, consider a series of galaxies with unique group ID number 551. We can create a group through something like:
```
grp551 = fof.group(551)

for g in list_of_galaxies:
    if g.groupID == 551:
        grp551.add_member(g)
```
There is a built-in function to do this; see `fof.find_group` and `fof.galaxy_to_group_arr` in the next section.


The only other attributes of the group class are `fof.group.members`, a list of galaxies in that group, and `fof.group.n`, the size of the group in number of galaxies. To add a member to the group, always use `group.add_member(...)` and **do not use `fof.group.members.append(...)` as this will not update the `group.n` property.**

Other methods of the group class include:
- `fof.group.get_skycoords()`: return the central (phi, theta) values for the group.
- `fof.group.get_cen_cz()`: return the central redshift (cz) value for the group.
- `fof.group.get_total_mag()`: return the group total absolute magnitude.
- `fof.group.get_proj_radius()`: return the projected radius of the group.
- `fof.group.get_cz_disp()`: return the velocity dispersion of the group's members.
- `fof.group.get_int_logMstar()`: return the group-integrated logarithmic stellar mass (logM_total).
- `fof.group.get_int_logMbary()`: return the group-integrated logarithmic baryonic mass (gas + stellar mass).
- `fof.to_df()`: return the group's members as a pandas dataframe. Additonally, save the group's members to CSV by specifying a path to `savename` (default `None`).


### Other functions
For more information on these, import the module and use the Python `?` tool.

- `fof.linking_length(g1, g2, b_perp, b_para, s)`: test whether two galaxies are friends. Return True/False.
- `fof.reset_groups(gxs)`: Revert all galaxies in a list to have group ID = 0.
- `fof.within_velocity_range(czmin, czmax, *gxs)`: check whether any element in a list of galaxies is outside a bounding redshift range. 
- `fof.find_group(gxs, grpID)`: return a group of specified ID number from a list of galaxies.
- `fof.galaxy_to_group_arr(gxs)`: convert a list of galaxies, `gxs`, into a list of groups.
- `fof.count_singular(grps)`: count the number of single-galaxy groups in an array `grps`.
- `fof.count_nonsingular(grps)`: count the number of nonsingular groups in an array `grps`.
- `fof.print_count(grps)`: print out basic counting numbers for a list of groups.
- `fof.catalog_to_txt(grparr, savename)`: write an array of groups into a catalog text file. 
- `fof.array_to_pandas(grparr, savename=None)`: Return (and/or save) a pandas dataframe constructed from a list of groups.


## References
- Kannappan, Sheila, et al. "The RESOLVE Survey: REsolved Spectroscopy Of a Local VolumE." Bulletin of the American Astronomical Society. Vol. 43. 2011.
- Moffett, Amanda J., et al. "ECO and RESOLVE: Galaxy Disk Growth in Environmental Context." The Astrophysical Journal 812.2 (2015): 89.
- Eckert, Kathleen D., et al. "RESOLVE and ECO: The Halo Mass-dependent Shape of Galaxy Stellar and Baryonic Mass Functions." The Astrophysical Journal 824.2 (2016): 124.
- Berlind, Andreas A., et al. "Percolation galaxy groups and clusters in the SDSS redshift survey: identification, catalogs, and the multiplicity function." The Astrophysical Journal Supplement Series 167.1 (2006): 1.
- Liu, Hauyu Baobab, et al. "A new galaxy group finding algorithm: Probability friends-of-friends." The Astrophysical Journal 681.2 (2008): 1046.
