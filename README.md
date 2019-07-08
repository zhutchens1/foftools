# foftools - galaxy group identification using the friends-of-friends algorithm.


This package was written to include a variety of computational tools for performing friends-of-friends group identification analyses, especially as related to the RESOLVE/ECO surveys (Kannappan et al. 2011, Moffett et al. 2015). Thus, the following tools are best suited in applications to volume-limited surveys with order 100 - 10,000 sample galaxies. For RESOLVE and ECO, analyses of the friends-of-friends group finding is reported in Eckert et al. 2016.

## Using foftools
There are three key pieces of code in the package:
1. the `fof.galaxy` class, for storing galaxies and their properties and associated methods
2. the `fof.galaxy_fof` function, an implementation of the friends-of-friends algorithm for a list of galaxies.
3. the `fof.group` class, for storing groups and their properties and associated methods


### Galaxies
The galaxies class allows for the storing of entire galaxies and their properties into a single data type.
Suppose we have a galaxy `test01` with RA and declination (195, 28) degrees, and with absolute magnitude -18.0 and redshift 5000 km/s. We can then initialize the galaxy to `x` as
```
import foftools as fof
x = fof.galaxy("test01", 195, 28, 5000, -18.0)
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
We sort groups of galaxies following the process described in Berlind et al. (2006). In this approach, two galaxies are considered friends if the perpendicular and line-of-sight comoving distances between them are each less than a characteristic linking length. The perpendicular and line-of-sight linking lengths are products of the mean separation between galaxies with the perpendicular and line-of-sight linking faictors.

The choice of linking factor is dependent on survey and can be optimized for different statistical purposes. For example, a flux-limited galaxy survey may require linking factors that are redshift-dependent (cf. Liu et al. 2008). However, since RESOLVE and ECO are volume-limited, our code is written to assume the linking factors to be completely constant. However, the `foftools` module can be easily modified to meet other purposes.

Our implementation of the FoF algorithm is the `fof.galaxy_fof` function. It can be called as
```
fof.galaxy_fof(gxs, bperp, blos, s)
```
where `bperp` is the perpendicular linking factor, `blos` is the line-of-sight linking factor, and `s` is the mean separation, each as described above. The argument `gxs` is a list of galaxies (instances of the `fof.galaxy` class) on which to perform the group-finding. **Therefore, the `gxs` array must be prepared in advance of using the FoF algorithm to meet the luminosity-floor and bounding redshift values required by the sample.** The function returns nothing; instead, it identifies unique groups of galaxies and assigns each galaxy the appropriate group ID number. Afterwards, it distinguishes those galaxies that were not found to be in nonsingular groups and gives them a unique ID number. These are called "single-galaxy groups." After the code is finished, it will print a confirmation, and each galaxy in `gxs` will have a non-zero group ID number.

**Note: The FoF algorithm will run helper functions for calculating the perpendicular and line-of-sight comoving distances between galaxies. The `foftools` package defaults to a LambdaCDM cosmology with H0 = 100.0, OmegaM = 0.3, and OmegaDE = 0.7. These can be modified in the source.**

### The Probability Friends-of-Friends Algorithm

The package includes an implementation of the probability friends-of-friends algorithm (Liu et al. 2008) for volume-limited data sets. This algorithm presents a modification to the traditional friends-of-friends linking criteria. In PFoF, galaxies are modeled with probability distribution functions, and satisfaction of the line-of-sight linking criterion is given probabilistically. Our implementation models the galaxy PDFs with Gaussian distributions whose centroids are the galaxies' measured redshifts and whose standard deviations are the errorbars on the galaxies' redshifts. Thus, each galaxy to be grouped needs to have its `galaxy.czerr` attribute defined. The routine can be called from
```
fof.prob_fof(gxs, bperp, blos, s, pth)
``` 
where `gxs` is a list of galaxy instances with redshift errorbars, `bperp` is the perpendicular linking constant, `blos` is the line-of-sight linking constant, and `s` is the mean separation between targets in the volume-limited data set. The additional argument `pth` is a threshold or "cutoff" probability that is used to determine whether the friendship probability between a pair of galaxies is sufficient for group membership. 

This function is written in the same conventions as `fof.galaxy_fof(...)`. That is, it resets all galaxies to have group ID, and, in determining the groups, will set each galaxy's group ID (`fof.galaxy.groupID`) to the appropriate value. Single galaxy-groups are given their own unique, nonzero group ID number.

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
