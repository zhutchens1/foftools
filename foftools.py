# -*- coding: utf-8 -*-
"""
package 'foftools'
@author: Zackary L. Hutchens - UNC-CH
v3.0 August 15, 2019.

This package contains classes and functions for performing galaxy group
identification using the friends-of-friends (FoF) and probability friends-of-friends
(PFoF) algorithms, as well class structures for working with galaxies and groups as
their own data types.
"""

# Needed packages
import numpy as np
import pandas as pd
import itertools
from scipy.integrate import quad
from math import erf
import scipy.stats as stats
from copy import deepcopy


__versioninfo__ = "FoFtools version 3.0, October 2019."

from astropy.cosmology import LambdaCDM
cosmo = LambdaCDM(H0=100.0, Om0=0.3, Ode0=0.7) # this puts everything in "per h" units.
SPEED_OF_LIGHT = 3.00E+05 # km/s

#####################################################################
#####################################################################
#####################################################################

class galaxy(object):
    """ 
    -----------------
    class `galaxy`
    A python class for a galaxy data type, which stores information such as the
    name, position, redshift, and magnitude.

    Initialization parameters:
        name (str):             name of the galaxy
        ra (float):             right-acension of galaxy in degrees
        dec (float):            declination of galaxy in degrees
        cz (float):             local group-corrected velocity in km/s
        mag (float):            SDSS r-band absolute magnitude of the galaxy.
        fl (bool):              Boolean flag for galaxy (default None). Can be used to note whether or a RESOLVE A/B galaxy is above the luminosity floor.
        groupID (int):          groupID number of the galaxy (default 0).
        logMgas (float):        logarithmic gas mass of the galaxy (default None.)
        logMstar (float):       logarithmic stellar mass of the galaxy (default None.)
        czerr (float):          1-sigma uncertainty in cz measurement `cz.` Needed for probability FoF (default None).
        assoc (int):            1/0 flag indicating whether galaxy was added to its group by association (default None, see `foftools.associate_galaxies`).
        

    Properties:
        phi (float):            azimuthal angle (ra) in radians
        theta (float):          polar angle (90 deg - dec) in radians
        x,y,z (float):          Cartesian-like angular coordinates in the sky.
        comovingdist (float):   line-of-sight comoving distance to the galaxy in Mpc/h. (Assumes H0 = 100, OmegaM = 0.3, OmegaDE = 0.7.)

    Methods:
        get_groupID:    returns group ID number
        set_groupID:    sets group ID number
        get_cz:         returns local group-corrected velocity [km/s]
        set_cz:         sets local group-corrected velocity [km/s] and associated comoving distance [Mpc/h]
        get_logMbary:   return the logarithmic baryonic mass of the galaxy (stellar + gas).
    -----------------
    """
    def __init__(self, name, ra, dec, cz, mag, fl=None, groupID=0, logMstar=None, logMgas=None, czerr=None, assoc=None, **kwargs):
        # Basic properties
        self.name = name
        self.ra = np.float128(ra) # degrees
        self.dec = np.float128(dec) # degrees
        self.cz = cz
        self.mag = mag
        self.fl = fl
        self.groupID = groupID
        self.logMstar = logMstar
        self.logMgas = logMgas
        self.czerr = czerr
        self.assoc = assoc
        self.__dict__.update(kwargs)

        # Spherical coordinates
        self.phi = np.float128(self.ra*(np.pi/180.0)) # rad
        self.theta = np.float128(np.pi/2.0 - self.dec*(np.pi/180)) # rad
        

        # x, y, z coordinates in the sky
        self.x = np.sin(self.theta)*np.cos(self.phi)
        self.y = np.sin(self.theta)*np.sin(self.phi)
        self.z = np.cos(self.theta)

        z = self.cz / SPEED_OF_LIGHT
        self.comovingdist = cosmo.comoving_distance(np.float64(z)).value
        self.transverse_comovingdist = cosmo.comoving_transverse_distance(np.float64(z)).value

    def get_groupID(self):
        """
        Return the group ID of the galaxy.
        Arguments: None
        Returns: group ID (int)
        """
        return self.groupID
    
    def set_groupID(self, groupID):
        """
        Set the group ID number to a specified value.
        Arguments: groupID (int)
        Return: None
        """
        self.groupID = groupID

    def get_cz(self):
        """
        Return the cz velocity of the galaxy.
        Arguments: None
        Returns: cz value (float)
        """
        return self.cz

    def set_cz(self, cz):
        """
        Set the cz of the galaxy to a specified value.
        Arguments: cz (float)
        Returns: None
        """
        c = SPEED_OF_LIGHT
        self.cz = cz
        z = cz / c
        self.comovingdist = cosmo.comoving_distance(np.float64(z)).value

    def get_logMbary(self):
        try:
            return np.log10(10**self.logMstar + 10**self.logMgas)
        except ValueError:
            print("Check that logMstar and logMgas attributes are provided to class instance")

    # Allow Python to print galaxy data.
    def __repr__(self):
        return "Name: {}\t RA:{:0.2}\t Dec:{:0.2}\t cz={:0.4} km/s \t Mag:{:0.3}\t grpID={}".format(*[self.name, self.ra, self.dec, self.cz, self.mag, self.groupID])
    
    def __str__(self):
        return self.__repr__()

    
class group(object):
    """
    -----------------
    class `group`
    A Python class for a galaxy group.

    Initialaization parameters:
        groupID (int):      unique group ID number
    
    Properties:
        members (arr):      array of all group members. Each group member needs to be expressed as an instance of the galaxy class (see above for `foftools.galaxy` class.)
        n (int):            number of members, defined as len(self.members).
    
    Methods:
        add_member:         appends a galaxy to self.members.
        get_skycoords:      return (phi, theta) values (in rad) the group center.
        get_cen_cz:         return central velocity of group (km/s).
        get_total_mag:      return group-integrated absolute magntiude.
        get_proj_radius:    return projected radius of galaxy in Mpc/h.
        get_cz_disp:        return cz dispersion computed from all group members in km/s.
        get_int_logMstar:   return group-integrated logarithmic stellar mass.
        get_int_logMbary:   return group-integrated logarithmic baryonic mass.
        to_df               obtain the members of the group as a pandas dataframe. Gives the option to save to csv by specifying a path to `savename`.
        
    WARNING: Always use the the group.add_member() method to add galaxies to the group.
    Doing, e.g., group.members.append(...) will not update the group.n property.
    -----------------
    """
    def __init__(self, groupID, **kwargs):
        self.groupID = groupID
        
        self.members = []
        self.n = len(self.members)
        
        self.__dict__.update(kwargs)

    def add_member(self, glxy):
        """
        Add a galaxy to the list of group members.
        Arguments: glxy (type galaxy)
        Returns: None. 
        """
        if isinstance(glxy, galaxy):
            self.members.append(glxy)
            self.n = len(self.members)
        else:
            raise ValueError("argument must be instance of galaxy class")
        
    
    def __repr__(self):
        if self.n == 1:
            return "ID {}: N={}\t RA={:0.4}\t Dec={:0.4}\t cz={:0.4E} km/s\t M={:0.3}\t R_proj={:0.4}".format(int(self.groupID),
                            self.n, self.get_skycoords()[0],self.get_skycoords()[1], self.get_cen_cz(),
                            self.get_total_mag(), 0.0)         
        else:
            return "ID {}: N={}\t RA={:0.4}\t Dec={:0.4}\t cz={:0.4E} km/s\t M={:0.3}\t R_proj={:0.4}".format(int(self.groupID),
                            self.n, self.get_skycoords()[0],self.get_skycoords()[1], self.get_cen_cz(),
                            self.get_total_mag(), self.get_proj_radius())
    
    def __str__(self):
        return self.__repr__()
        
    
    def get_skycoords(self):
        """ 
        Get the cental sky spherical-polar coordinates  of the group.
        Arguments: None
        Returns: tuple containing phi, theta values in degrees
        """
        xcen = 0.
        ycen = 0.
        zcen = 0.
        for g in self.members:
            xcen += g.cz * g.x
            ycen += g.cz * g.y
            zcen += g.cz * g.z
        xcen /= len(self.members)
        ycen /= len(self.members)
        zcen /= len(self.members)
        
        czcen = np.sqrt(xcen**2 + ycen**2 + zcen**2)
        
        thetacen = np.arcsin(zcen/czcen)*(180.0/np.pi)
        
        if (ycen >=0 and xcen >=0):
            phicor = 0.0
        elif (ycen < 0 and xcen < 0):
            phicor = 180.0
        elif (ycen >= 0 and xcen < 0):
            phicor = 180.0
        elif (ycen < 0 and xcen >=0):
            phicor = 360.0
        elif (xcen==0 and ycen==0):
            print("Warning: xcen=0 and ycen=0 for group {}".format(self.groupID))
            
        phicen = np.arctan(ycen/xcen)*(180.0/np.pi) + phicor
        
        return (phicen, thetacen) # in degrees
    
    
    def get_cen_cz(self):
        """
        Get the central redshift velocity of the group, c*z_central.
        Arguments: None
        Returns: czcen (type float)
        """
        xcen = 0
        ycen = 0
        zcen = 0
        for g in self.members:
            xcen += g.cz * g.x
            ycen += g.cz * g.y
            zcen += g.cz * g.z
        xcen /= len(self.members)
        ycen /= len(self.members)
        zcen /= len(self.members)
        
        czcen = np.sqrt(xcen**2 + ycen**2 + zcen**2)
        return czcen
        
    def get_total_mag(self):
        """
        Obtain the group-integrated total magnitude.
        Arguments: None
        Return: M (type float)
        """
        M = 0
        for g in self.members:
            M += 10.0**(-0.4 * g.mag)
            
        M = -2.5*np.log10(M)
        return M
    
    def get_proj_radius(self):
        """
        Return the projected radius of the group. For single-galaxy groups, return 0.0.
        Arguments: None
        Return: projected radius (type float)
        """
        Rproj = 0.
        
        if self.n <= 1:
            return 0.0

        for g in self.members:
            
            # Get RA/Dec in degrees
            (phicen, thetacen) = self.get_skycoords()
            
            # Convert to radians
            phicen = phicen * np.pi/180.0
            thetacen = thetacen * np.pi/180.0
            
            # Get Rproj
            cosDpsi = np.cos(thetacen)*np.cos(np.pi/2 - g.theta)+np.sin(thetacen)*np.sin(np.pi/2 - g.theta)*np.cos((phicen - g.phi))
            sinDpsi = np.sqrt(1-cosDpsi**2)
           
            if cosDpsi > 1:
                print(self)

            rp = sinDpsi * g.cz/(100.)
            Rproj += rp**2
            
        if self.n > 1:
            return np.sqrt(Rproj/float(self.n))
        else:
            return 0.

    
    def get_cz_disp(self):
        """
        Obtain the velocity dispersion of the group of galaxies. For single-galaxy groups, return 0.0.
        Arguments: None
        Return: cz_disp (type float)
        """
        # velocity dispersion
        c = 299800 # km/s
        Dz2 = 0
        for g in self.members:
            Dz2 += (g.cz - self.get_cen_cz())**2
        
        if self.n > 1:
            cz_disp = np.sqrt(Dz2/(self.n-1))/(1.+self.get_cen_cz()/c)
            return cz_disp
        else:
            return 0.
    
    def get_int_logMstar(self):
        """
        Return the group-integrated stellar mass.
        Arguments: None
        Output: group-integrated log stellar mass, logM_total (float).
        """
        sumv = 0.
        try:
            for g in self.members:
                sumv += 10.0**(g.logMstar)
            return np.log10(sumv)
        except:
            raise AttributeError("galaxies need logMstar attributes")

    def get_int_logMbary(self):
        """
        Return the group-integrated baryonic (gas+stellar) mass.
        Arguments: None
        Output: group-integrated log baryonic mass, logM_bary_total (float).
        """
        sumv = 0.
        try:
            for g in self.members:
                sumv += 10.0**(g.get_logMbary())
            return np.log10(sumv)
        except:
            raise AttributeError("galaxies need both logMstar and logMgas attributes")


    def to_df(self, savename=None):
        """
        Output the group's member to a pandas dataframe and save to a file (optional).
        Arguments: savename (default None): name of file to save the dataframe. If None, no file will be saved.
        Returns: df (type pandas.DataFrame): dataframe containing list of group members with name, coordinates, magnitude, cz, and groupID number.
        """
        table = []
        for g in self.members:
            table.append([g.name, g.ra, g.dec, g.cz, g.mag, g.groupID])
            
        df = pd.DataFrame(table)
        
        # Save if requested
        try:
            if (savename is not None):
                df.to_csv(savename, index=False)
            else:
                pass
        except:
            raise IOError("File {} is invalid".format(savename))
        
        df.columns=['name', 'radeg', 'dedeg', 'cz', 'absrmag', 'grp']
        # return the dataframe
        return df
        
        
        
#########################################################################
#########################################################################
#########################################################################
#########################################################################
            
"""
The following entries to this package are helper functions needed for using the
prinicpal friends-of-friends algorithms.
"""
   
def ang_sep(x, glxy):
    """Compute the angular separation bewteen two galaxies using the Haversine formula.
    Arguments: x (type galaxy), glxy (type galaxy).
    Returns: angular separation of two galaxies in the sky in radians (type float)
    """
    y = 2*np.arcsin(np.sqrt(np.sin((glxy.theta-x.theta)/2.0)**2.0 + np.sin(glxy.theta)*np.sin(x.theta)*np.sin((glxy.phi - x.phi)/2.0)**2.0))
    havtheta = np.float128(y)
    return havtheta
    
    
def d_perp(x, glxy):
    """Compute the perpendicular separation between galaxies (Berlind et al. 2006).
    Arguments: x (type galaxy), glxy (type galaxy).
    Returns: Dperp in Mpc/h (type float)
    """
    half_angle = ang_sep(x, glxy) / 2.0
    g1_transvcm = x.transverse_comovingdist * half_angle
    g2_transvcm = glxy.transverse_comovingdist * half_angle
    return (g1_transvcm + g2_transvcm)

def d_los(x, glxy):
    """Compute the line-of-sight separation between galaxies (Berlind et al. 2006).
    Arguments: x (type galaxy), glxy (type galaxy).
    Returns: Dlos in Mpc/h (type float)
    """
    return np.abs(x.comovingdist - glxy.comovingdist)
    

def reset_groups(gxs):
    """Revert all galaxies to groupID = 0.
    Arguments: gxs (array-like), containing elements only of type `galaxy`.
    Returns: None. Resets all elements of gxs to have groupID = 0.
    """
    for g in gxs:
        try:
            g.set_groupID(0)
        except:
            raise TypeError("element g is not type galaxy")
            
            
def within_velocity_range(czmin, czmax, *gxs):
    """Check if a galaxy(s) is within a given velocity range.
    Arguments: 
        czmin (type float)
        czmax (type float)
        gxs (arr-like, elements: type galaxy).
    Returns: x (bool). If any of the elements of gxs are outside the specified range, it returns false. Otherwise, return true.
    """
    for g in gxs:
        if ((g.cz <= czmin) or (g.cz >= czmax)):
            x = False
            break
        else:
            x = True
    return x

def in_same_group(g1, g2):
    """
    Check if two galaxies are in the same identified group.
    Arguments:
        g1, g2 (type galaxy): galaxies to compare
    Return:
        Bool indicating whether galaxies belong to same group.
    """
    if (g1.groupID == g2.groupID) and (g1.groupID != 0) and (g2.groupID != 0):
        return True
    else:
        return False



####################################################################
####################################################################
####################################################################
# Conventional FoF    

def linking_length(g1, g2, b_perp, b_para, s):
    """Determine whether two galaxies are within the linking length.
    Arguments: 
        g1, g2 (type galaxy): two galaxies to compare.
        b_perp (float): perpendicular linking constant
        b_para (float): parallel/line-of-sight linking constant
        s (float): mean separation between galaxies
    Returns: Bool. Indicates whether the two galaxies are within a linking length apart.
    """
    DPERP = d_perp(g1, g2)
    DLOS = d_los(g1, g2)
    
    if ((DPERP <= b_perp*s) and (DLOS <= b_para*s)):
        return True
    else:
        return False

def galaxy_fof(gxs, bperp, blos, s, printConf=True):
    """
    -----------------
    Perform a friends-of-friends group finding on an array of galaxies,
    based on the method of Berlind et al. 2006. For volume-limited samples, the 
    luminosity floor needs to be implemented prior to using this function, so that
    gxs contains only galaxies brigther than the magnitude floor and within the 
    desired redshift range.
    Arguments:
        gxs (iterable, elements: type galaxy): array of galaxies to group with FOF
        bperp (float or callable): perpendicular linking constant; for ECO, use 0.07. If callable, must be a only a function of redshift `z`
        blos (float or callable): line-of-sight linking constant; for ECO, use 1.1. If callable, must be a only a function of redshift `z`
        s (float or callable): mean separation between galaxies. Calculated as n**(-1/3), where n = N/V. If callable, must be a only a function of redshift `z`.
        printConf (bool, default True): if True, print a confirmation that FOF is complete.
    
    Returns:
        None. The algorithm creates unique group ID numbers for every identified
        group, including "single-galaxy groups." It sends these ID's to the galaxy.groupID
        attribute for every element in the gxs argument.
    -----------------
    """
    # Establish counter
    grpindex = 1
    
    # Reset the group array
    reset_groups(gxs)
    
    # Check if s is scalar or callable
    if np.isscalar(s):
        sepf = lambda x: s
    elif callable(s):
        sepf = s
    else:
        raise ValueError('Argument `s` must be scalar or callable. If callable, s must be only a function of z.')
        
    # Check if bperp, blos are scalar or callble
    if np.isscalar(bperp):
        bperpf = lambda x: bperp
    elif callable(bperp):
        bperpf = bperp
    else:
        raise ValueError('Argument `bperp` must be scalar or callable. If callable, s must be only a function of z.')
        
    if np.isscalar(blos):
        blosf = lambda x: blos
    elif callable(blos):
        blosf = blos
    else:
         raise ValueError('Argument `blos` must be scalar or callable. If callable, s must be only a function of z.')
    
    # Identify the groups
    c = SPEED_OF_LIGHT
    for g1, g2 in itertools.product(gxs, gxs):
        if (g1.name != g2.name):
            if linking_length(g1, g2, bperpf(g1.cz/c), blosf(g1.cz/c), sepf(g1.cz/c)):
                    
                # Case 1: Galaxies already in the same group
                if ((g1.groupID == g2.groupID) and (g1.groupID != 0) and (g2.groupID != 0)):
                    pass

                # Case 2: g1 is already in a group, but g2 is not. Put g2 in the group of g1.
                elif ((g1.groupID != g2.groupID) and (g1.groupID != 0) and (g2.groupID == 0)):
                    g2.set_groupID(g1.groupID)
                    
                # Case 3: g2 is already in a group, but g1 is not. Put g1 in the group of g2.
                elif ((g1.groupID != g2.groupID) and (g1.groupID == 0) and (g2.groupID != 0)):
                    g1.set_groupID(g2.groupID)
                    
                # Case 4: Neither of them are in a group. Make a new group.
                elif ((g1.groupID == 0) and (g2.groupID == 0)):
                    g2.set_groupID(grpindex)
                    g1.set_groupID(grpindex)
                    grpindex += 1
                # Case 5: g2 and g1 are already identified into groups, but they are still
                # within a linking length of one another. In that case, we need to link their
                # groups together.
                elif ((g1.groupID != g2.groupID) and (g1.groupID !=0) and (g2.groupID != 0)):
                    keeping_id = g1.groupID
                    equiv_id = g2.groupID
                    for g in gxs:
                        if g.groupID == equiv_id:
                            g.set_groupID(keeping_id)   
                else:
                    # Should be nothing here!
                    print('WARNING: These galaxies failed the group-finding algorithm.')
                    print(g1)
                    print(g2)
            else:
                pass
        else:
            pass
    # The group finding for nonsingular groups is done. Now, just need to make all
    # which were not idenfied into single galaxy groups.
    for i,g in enumerate(gxs):
        if g.groupID == 0:
            g.set_groupID(grpindex + i)
    if printConf:        
        print("FOF Group finding complete.")




####################################################################
####################################################################
####################################################################
# Eke et al. FoF
def eke_linking_length(g1, g2, b_perp, b_para, s):
    """
    Determine whether two galaxies are friends using the linking length criteria of 
    Eke et al. 2004, MNRAS 348, 866.
    """
    c = SPEED_OF_LIGHT
    
    #print(type(b_perp), type(b_para), type(s))
    cond1 = np.abs(g1.comovingdist - g2.comovingdist) <= (b_para(g1.cz/c)*s(g1.cz/c) + b_para(g2.cz/c)*s(g2.cz/c))
    angle = ang_sep(g1, g2) # in rad
    cond2 = angle <= 0.5*(b_perp(g1.cz/c)*s(g1.cz/c)/g1.comovingdist + b_perp(g2.cz/c)*s(g2.cz/c)/g2.comovingdist)  
    
    return (cond1 and cond2)

def eke_fof(gxs, bperp, blos, s, printConf=True):
    """
    -----------------
    Perform a friends-of-friends group finding on an array of galaxies,
    based on the method of Eke et al (2004). For volume-limited samples, the 
    luminosity floor needs to be implemented prior to using this function, so that
    gxs contains only galaxies brigther than the magnitude floor and within the 
    desired redshift range.
    Arguments:
        gxs (iterable, elements: type galaxy): array of galaxies to group with FOF
        bperp (float or callable): perpendicular linking constant; for ECO, use 0.07. If callable, must be a only a function of redshift `z`
        blos (float or callable): line-of-sight linking constant; for ECO, use 1.1. If callable, must be a only a function of redshift `z`
        s (float or callable): mean separation between galaxies. Calculated as n**(-1/3), where n = N/V. If callable, must be a only a function of redshift `z`.
        printConf (bool, default True): if True, print a confirmation that FOF is complete.
    
    Returns:
        None. The algorithm creates unique group ID numbers for every identified
        group, including "single-galaxy groups." It sends these ID's to the galaxy.groupID
        attribute for every element in the gxs argument.
    -----------------
    """
    # Establish counter
    grpindex = 1
    
    # Reset the group array
    reset_groups(gxs)
    
    # Check if s is scalar or callable
    if np.isscalar(s):
        sepf = lambda x: s
    elif callable(s):
        sepf = s
    else:
        raise ValueError('Argument `s` must be scalar or callable. If callable, s must be only a function of z.')
        
    # Check if bperp, blos are scalar or callble
    if np.isscalar(bperp):
        bperpf = lambda x: bperp
    elif callable(bperp):
        bperpf = bperp
    else:
        raise ValueError('Argument `bperp` must be scalar or callable. If callable, s must be only a function of z.')
        
    if np.isscalar(blos):
        blosf = lambda x: blos
    elif callable(blos):
        blosf = blos
    else:
         raise ValueError('Argument `blos` must be scalar or callable. If callable, s must be only a function of z.')
    
    # Identify the groups
    c = SPEED_OF_LIGHT
    for g1, g2 in itertools.product(gxs, gxs):
        if (g1.name != g2.name):
            if eke_linking_length(g1, g2, bperpf, blosf, sepf): # add arguments back in
                    
                # Case 1: Galaxies already in the same group
                if ((g1.groupID == g2.groupID) and (g1.groupID != 0) and (g2.groupID != 0)):
                    pass

                # Case 2: g1 is already in a group, but g2 is not. Put g2 in the group of g1.
                elif ((g1.groupID != g2.groupID) and (g1.groupID != 0) and (g2.groupID == 0)):
                    g2.set_groupID(g1.groupID)
                    
                # Case 3: g2 is already in a group, but g1 is not. Put g1 in the group of g2.
                elif ((g1.groupID != g2.groupID) and (g1.groupID == 0) and (g2.groupID != 0)):
                    g1.set_groupID(g2.groupID)
                    
                # Case 4: Neither of them are in a group. Make a new group.
                elif ((g1.groupID == 0) and (g2.groupID == 0)):
                    g2.set_groupID(grpindex)
                    g1.set_groupID(grpindex)
                    grpindex += 1
                # Case 5: g2 and g1 are already identified into groups, but they are still
                # within a linking length of one another. In that case, we need to link their
                # groups together.
                elif ((g1.groupID != g2.groupID) and (g1.groupID !=0) and (g2.groupID != 0)):
                    keeping_id = g1.groupID
                    equiv_id = g2.groupID
                    for g in gxs:
                        if g.groupID == equiv_id:
                            g.set_groupID(keeping_id)   
                else:
                    # Should be nothing here!
                    print('WARNING: These galaxies failed the group-finding algorithm.')
                    print(g1)
                    print(g2)
            else:
                pass
        else:
            pass
    
    # The group finding for nonsingular groups is done. Now, just need to make all
    # which were not idenfied into single galaxy groups.
    for i,g in enumerate(gxs):
        if g.groupID == 0:
            g.set_groupID(grpindex + i)
    if printConf:        
        print("Eke FOF Group finding complete.")

###################################################################
####################################################################
####################################################################
# Probability Friends-of-friends

def gauss(x, mu, sigma):
    """
    Gaussian function.
    Arguments:
        x - dynamic variable
        mu - centroid of distribution
        sigma - standard error of distribution
    Returns:
        PDF value evaluated at `x`
    """
    return 1/((2*np.pi)**0.5 * sigma) * np.exp(-1 * 0.5 * ((x-mu)/sigma) * ((x-mu)/sigma))

def prob_linking_length(g1, g2, perpll, losll, Pth, return_val=False):
    """
    -----------------
    Determine whether two galaxies satisfy the linking-length condition for a probability FOF approach. Two galaxies are considered 
    friends if bperp*s <= dist_perp AND
                
                prob (|z2 - z1| <= dist_los) = \int_0^\infty dz G_1(z) \int_{z-dist_los}^{z+dist_los} G_2(z')dz' >= P_{th},
                
    where P_{th} is the threshold probability level, and where G_1 and G_2 are the probability distribution functions for galaxies #1 and #2.
    Here we assume that G1 and G2 take the form of a Gaussian distribution with their centroids defined by the measured redshifts, and their
    sigmas taken as the redshift error bar.
    
    Note: The above integral is computed using `scipy.integrate.quad`, which is an extension of the algorithm `QUADPACK` from Fortran 77 library.
    When error bars are small, the Gaussian approaches a Delta distribution. Since the region of interest is much smaller than the integration window,
    we divide the integral into three pieces: up to -5*sigma of G1, over the distribution, and then 5*sigma of G1 to z -> infinity. We approximate
    infinite redshift to be z~10 to save computation time.
    Arguments:
        g1, g2 (type fof.galaxy): two galaxies to compare.
        perpll (type float): perpendicular linking length in Mpc/h, usually computed as bperp * s, where s is the mean separation betwen galaxies in the search volume.
        losll (type float): line-of-sight linking length, in km/s, usually computed as blos*s * H/c, where s is the mean separation between galaxies in the search volume.
        Pth (float): threshold probability constituting group membership. Argument must be on interval (0,1).
        return_val (bool, default False): If True, return the integrated probability, rather than the boolean P > Pth.
    Returns:
        Boolean value indicating whether the two galaxies satisfy the linking-length condition.
        If return_val==True, return total integrated probability, P.
    -----------------
    """
    # Line of sight linking condition
    c = SPEED_OF_LIGHT
    VL = losll/SPEED_OF_LIGHT
    
    # Write the inside function F(z)
    #F = lambda z: 0.5*erf((z+VL-g2.cz/c)/((2**0.5)*g2.czerr/c)) - 0.5*erf((z-VL-g2.cz/c)/((2**0.5)*g2.czerr/c)) 
    # Write down integrand I(z), integrate in pieces to get probability
    I = lambda z:  gauss(z, g1.cz/c, g1.czerr/c) * (0.5*erf((z+VL-g2.cz/c)/((2**0.5)*g2.czerr/c)) - 0.5*erf((z-VL-g2.cz/c)/((2**0.5)*g2.czerr/c))) 

    #val1 = quad(I, 0, g1.cz/c - 3*g1.czerr/c)
    #val2 = quad(I, g1.cz/c -3 * g1.czerr/c, g1.cz/c + 3*g1.czerr/c)
    #val3 = quad(I, g1.cz/c + 3*g1.czerr/c, np.inf)

    val = quad(I, 0, 100, points=np.float64([g1.cz/c-3*g1.czerr/c, g1.cz/c, g1.cz/c+3*g1.czerr/c]))
    val = val[0]
    #print(val)
    if return_val: 
        return val
    
    #print(val)
    # if both conditions satisifed, return True.
    if ((d_perp(g1, g2) <= perpll) and (val > Pth)):
        return True
    else:
        return False


def prob_fof(gxs, perpll, losll, Pth, printConf=True):
    """
    -----------------
    Conduct a probability friends-of-friends (PFoF) analysis on a list of galaxies using the method of Liu et al. (2008).
    In the PFoF approach, two galaxies are considered friends if bperp * s <= dist_perp and 
    
         prob (|z2 - z1| <= dist_los) = \int_0^\infty dz G_1(z) \int_{z-dist_los}^{z+dist_los} G_2(z')dz' >= P_{th},
                
    where P_{th} is the threshold probability level, and where G_1 and G_2 are the probability distribution functions for galaxies #1 and #2.
    Here we assume that G1 and G2 take the form of a Gaussian distribution with their centroids defined by the measured redshifts, and their
    sigmas taken as the redshift error bar. This implementation is suited for analyses of galaxies belonging to volume-limited data sets,
    or generally those in which the linking lengths bear no redshift-dependence. This function also requires that each galaxy in `gxs` has a 
    unique name at `galaxy.name.`
    
    Arguments:
        gxs (iterable, elements: type fof.galaxy): list of galaxies within which to identify to groups.
        bperp (float or callable): perpendicular linking constant
        blos (float or callable): line-of-sight linking constant
        s (float or callable): mean-separation between galaxies
        Pth (float): threshold probability for what constitutes to friendship between a given pair of galaxies. As Pth -> 1, the group catalog approaches
            that of conventional friends-of-friends.
        printConf (bool, default True): bool indicating whether to print a written confirmation that group finding is complete. Default True.
    Returns:
        None. The algorithm assigns a unique, nonzero group ID number to every galaxy in `gxs.`
    -----------------
    """
    grpindex = 1
    reset_groups(gxs)
    
    # Check if perpll is scalar or callable
    if np.isscalar(perpll):
        perpf = lambda x: perpll
    elif callable(perpll):
        perpf = perpll
    else:
        raise ValueError('Argument `perpll` must be scalar or callable. If callable, s must be only a function of z.')
        
    # Check if losll is scalar or callable
    if np.isscalar(losll):
        losf = lambda x: losll
    elif callable(losll):
        losf = losll
    else:
        raise ValueError('Argument `losll` must be scalar or callable. If callable, s must be only a function of z.')
        
    
    c = SPEED_OF_LIGHT
    for g1, g2 in itertools.product(gxs, gxs):
            if (g1.name != g2.name) and (not in_same_group(g1, g2)): 
                if prob_linking_length(g1, g2, perpf(g1.cz/c), losf(g1.cz/c), Pth): 
                        
                    # Case 1: galaxies already in same group
                    if ((g1.groupID == g2.groupID) and (g1.groupID != 0) and (g2.groupID !=0)):
                        pass
                    # Case 2: g1 is in already in a group, but g2 is not. Put g2 in the group of g1.
                    elif ((g1.groupID != g2.groupID) and (g1.groupID != 0) and (g2.groupID == 0)):
                        g2.set_groupID(g1.groupID)
                    # Case 3: g2 is already in a group, but g1 is not. Put g1 in the group of g2.
                    elif ((g1.groupID != g2.groupID) and (g1.groupID == 0) and (g2.groupID != 0)):
                        g1.set_groupID(g2.groupID)
                    # Case 4: Neither in a group; make a new group.
                    elif ((g1.groupID == 0) and (g2.groupID == 0)):
                        g2.set_groupID(grpindex)
                        g1.set_groupID(grpindex)
                        grpindex += 1
                    # Case 5: g2 and g1 are already identified into groups, but they are still satisfying the linking length. Merge their groups.
                    elif ((g1.groupID != g2.groupID) and (g1.groupID != 0) and (g2.groupID != 0)):
                        keeping_id = g1.groupID
                        equiv_id = g2.groupID
                        for g in gxs:
                            if g.groupID == equiv_id:
                                g.set_groupID(keeping_id)
                    else:
                        # Should never get here.
                        print("WARNING: These galaxies failed the PFOF algorithm!")
                        print(g1.name, g2.name)
    # The group finding for nonsingular groups is now done. Now, just make all 
    # which were not identified into single-galaxy groups.
    for i,g in enumerate(gxs):
        if g.groupID == 0:
            g.set_groupID(grpindex + i)
    if printConf:
        print("PFoF group-finding complete.")
        
    return gxs
        
####################################################################
####################################################################
####################################################################
# Tools for faint galaxy association to FoF groups
        
def set_assoc_zero(gxs):
    '''
    Set the association case (bool) of each galaxy to zero.
    '''
    for g in gxs:
        g.assoc = 0
        
def largerof(a, b):
    return max([a,b])



def faint_assoc(idgxs, faintgxs, grpids, rvir, vdisp, losll, ret_groups=True):
    """
    ----------
    Associate faint galaxies below a specified floor back into groups on the condition of
    Eckert et al. (2016). This algorithm will search for faint galaxies that can be associated to
    friends-of-friends dark matter halos, storing this information to a property `foftools.galaxy.assoc.`
    The current key for this parameter is:
        1   faint galaxy, associated to a FoF halo.
        0   bright galaxy, brigther than luminosity floor, already in a singular or nonsingular FoF halo.
       -1   faint galaxy that was unable to associate to an existing FoF halo, now forming a single-galaxy group below the floor.
        
    
    Arguments:
        idgxs (iterable; elements: type galaxy) galaxies with identified group numbers, to be associated against.
        faintgxs (iterable; elements: type galaxy) list of galaxies that are fainter than the magnitude floor.
        grpids (iterable; elements: int) list of grpids output from HAM.
        rvir (iterable, elements: float) virial radii from HAM
        sigma (iterable, elements: float) velocity dispersion from HAM
        losll (float or callable): line-of-sight linking length in Mpc/h
    Returns:
        GROUPS (iterable, elements: type group): groups with faint galaxies associated.
    ----------
    """
    if np.isscalar(losll):
        losllf = lambda x: losll
    elif callable(losll):
        losllf = losll
    else:
        raise ValueError("Argument `losll` must be scalar or callable.")
    
    set_assoc_zero(idgxs)
    set_assoc_zero(faintgxs)
    c = SPEED_OF_LIGHT
    
    idtargets = deepcopy(idgxs)
    fainttargets = deepcopy(faintgxs)
    
    assert max(grpids)==max(idgxs, key=lambda g:g.groupID).groupID, "Argument `grpids` has different ID's than `idgxs`"
    
    for i,idv in enumerate(grpids):
        grp = find_group(idtargets, idv)
        Vdisp = vdisp[i]
        
        cra, cdec = grp.get_skycoords()
        grpcenterfake = galaxy(name='grpcenter', ra=cra, dec=cdec, cz=grp.get_cen_cz(), mag=grp.get_total_mag())
        
        for fg in fainttargets:
            Rp = d_perp(grpcenterfake, fg)
            DeltaV = (fg.cz - grpcenterfake.cz)
            tempratio = Rp/rvir[i]
            condition = ((tempratio < 1) and (DeltaV < largerof(losllf(grpcenterfake.cz/c)*cosmo.H(grpcenterfake.cz/c).value, 3*Vdisp)))
            
            if fg.name=='rf0691' and idv==46:
                pass
                #print(tempratio, DeltaV, 3*Vdisp, losll*100, condition)
            
            if condition and (fg.assoc):
                # this means that two groups are competing for a galaxy.
                if tempratio < fg.radiusratio:
                    fg.radisratio = tempratio
                    fg.set_groupID(idv)
                    fg.assoc=1
                else:
                    pass
                    
            elif condition and (not fg.assoc):
                fg.radiusratio = tempratio
                fg.set_groupID(idv)
                fg.assoc = 1
            elif (not condition):
                pass
            else:
                print("WARNING: This faint galaxy failed the association algorithm", fg)
                
    maxidv = max(grpids)
    i=100
    for fg in fainttargets:
        if not fg.assoc:
            fg.set_groupID(maxidv+i)
            fg.assoc = -1 
            i += 1
    
    outgxs = list(np.concatenate([idtargets, fainttargets]))
    if ret_groups:
        return galaxy_to_group_arr(outgxs)
    else:
        return outgxs
        
    

####################################################################
####################################################################
####################################################################

# Functions to organize groups + galaxies
def find_group(gxs, grpID):
    """Return a group from an array of galaxies for a given ID number.
    Arguments:
        gxs (array-like, elements: type galaxy): array containing list of galaxies with unique group ID numbers
        grpID (type int): group ID used to identify galaxies in that group
    Returns:
        x (type group): group of galaxies at the specified group ID number.
    """
    x = group(grpID)
    
    for g in gxs:
        if g.groupID == grpID:
            x.add_member(g)
    return x

def galaxy_to_group_arr(gxs):
    """
    Obtain a list of groups from an list of galaxies.
    Arguments: 
        galaxy_arr (array-like, elements: type galaxy): array containing list of galaxies with unique group ID numbers.
    Returns:
        group_arr (list, elements: type group): array containing organized groups, initialized using the groupIDs
        from galaxy_arr's elements.
    """
    # Get an array of unique group ID's
    grpids = []
    for g in gxs:
        grpids.append(g.groupID)
    grpids = np.unique(grpids)
    
    # Create a new group for every unique ID, and add members with that ID.
    group_arr = []
    for val in grpids:
        temp = group(val)
        for g in gxs:
            if g.groupID == val:
                temp.add_member(g)
        group_arr.append(temp)
    
    return group_arr



def count_singular(grps):
    """
    Count the number of single-galaxy groups in an array of groups.
    Arguments:
        grps (arr, elements: type group): array of galaxy groups.
    Returns:
        count (int): Number of groups containing only one galaxy.
    
    """
    try:
        count = 0
        for g in grps:
            if g.n == 1:
                count += 1
        return count
    except:
        raise TypeError("Check that elements of argument are type `fof.group`.")
    
    
def count_nonsingular(grps):
    """
    Count the number of nonsingular galaxy groups in an array of groups.
    Arguments:
        grps (arr, elements: type group): array of galaxy groups.
    Returns:
        count (int): Number of groups containing more than one galaxy.
    """
    try:
        count = 0
        for g in grps:
            if g.n > 1:
                count += 1
        return count
    except:
        raise TypeError("Check that elements of argument are type `fof.group`.")
        

def print_count(grps):
    """
    Print the number of singular, nonsingular, and total groups.
    Arguments:
        grps (arr, elements: type group): array of galaxy groups.
    Returns:
        None. Prints counts.
    """
    x = count_singular(grps)
    y = count_nonsingular(grps)
    print("Singular = {a}\nNonsingular: {b}\nTotal Groups: {c}".format(a=x, b=y, c=x+y))
    



####################################################################
####################################################################
####################################################################
    

# IO functions

    
def catalog_to_txt(gxs, savename):
    """Write the group catalog to text file. The function formats the text file by printing
    a group (id, # members, and central properties) and then printing every member galaxy
    of that group (ra, dec, magnitude, etc.)
    Arguments:
        savename (str): file destination
        grparr (arr, elements: type group): list of groups to output.
    Returns: none. Saves file to specified directory.
    
    """
    pass # under construction!




def array_to_pandas(grparr, savename=None):
    """Write a group array to a pandas dataframe/csv format. This output includes
    only the overall properties of the groups, but contains no information about
    the galaxies in each group.
    Arguments:
        grparr (arr, elements: type group): list of groups to output.
        savename (str): file destination
    Returns: None.Saves file to specified directory.
    """
    table = []
    for G in grparr:
        ra, dec = G.get_skycoords()
        if G.n == 1:
            czdisp = rproj = 0
        else:
            czdisp = G.get_cz_disp()
            rproj = G.get_proj_radius()
        
        table.append([G.groupID, G.n, ra, dec, G.get_cen_cz(), G.get_total_mag(), czdisp, rproj])
    
    df = pd.DataFrame(table)
    df.columns = ["GroupID", "N", "RA", "DEC", "cz", "mag", "cz_disp", "Rproj"]
    
    if (savename is not None):
        try:
            df.to_csv(savename, index=False)
        except IOError:
            print("Invalid file destination")
            
    return df



def arrays_to_galaxies(names, ras, decs, czs, mags, mag_floor=99, czmin=-1, czmax=99999999):
    """
    Form a series of arrays into a list of galaxies. Does not currently support reading in of
    external group ID numbers.
    Arguments:
        names (iterable, elements: string): names of galaxies
        ras (iterable, elements: float): RA's of each galaxy in decimal degrees
        decs (iterable, elements: float) Dec's of each galaxy in decimal degrees
        czs (iterable, elements: float) velocities of each galaxy in km/s.
        mags (iterable, elements: float): absolute magnitude of each galaxy.
        mag_floor (float, default 99): magnitude floor on which to choose galaxies added to list.
    Returns:
        gxs (list, elements: type galaxy) list of galaxies formed of input arrays.
    """
    names = np.array(names)
    ras = np.array(ras)
    decs = np.array(decs)
    czs = np.array(czs)
    mags = np.array(mags)
    gxs = []
    for i,e in enumerate(names):
        if mags[i] <= mag_floor:
            gxs.append(galaxy(e, ras[i], decs[i], czs[i], mags[i]))
    return gxs


def groups_to_haminput(grps, savename):
    """
    Output a group catalog in a format compatible with the HAM code of A. Berlind.
    Arguments:
        grps (iterable, elements: type group): list of groups to ouput to file.
        savename (string): path to which the group catalog will be saved.
    """
    f = open(savename, 'w')
    for G in grps:
        rav, decv = G.get_skycoords()
        f.write("G\t{a}\t{b}\t{c}\t{d}\t{e}\t{f}\t{g}\t{h}\n".format(a=int(G.groupID),b=rav, c=decv, d=G.get_cen_cz(), e=G.n, f=G.get_cz_disp(), g=G.get_proj_radius(), h=G.get_total_mag()))

    f.close()

####################################################################
####################################################################
####################################################################
# Other tools
    
def inpointing(sequence, center_ra, center_dec, ldm0, D):
    """
    ---------------
    Determine if a list of targets is with a given single-dish pointing. The dish diameter
    is `D` and the pointing is centered on the rest wavelength `ldm0`.  
    Arguments:
        sequence (iterable, elements: type `galaxy` or `tuple`): list of targets to check. This argument can either be a list
                    galaxy instances, or it could be a list of tuples (RA, Dec).
        center_ra (float): RA of field center in decimal degrees.
        center_dec (float): Declination of field center in decimal degrees.
        ldm0 (float): rest-wavelength of pointing in meters (e.g., for HI emission use 0.21 m for the 21cm line.)
        D (float): diameter of dish in meters.
    Returns:
        outseq (iterable): targets in the field, data types matching the elements `sequence.`
    ---------------
    """
    outseq = []
    if isinstance(sequence[0], galaxy):
        for gal in sequence:
            ra, dec = gal.ra, gal.dec
            theta = 1.22 * ldm0 * (1 + gal.cz/SPEED_OF_LIGHT) /  D
            R = theta/2.0 * 180/np.pi
            
            f = np.pi/180.0
            cosr = np.sin(dec*f)*np.sin(center_dec*f) + np.cos(dec*f)*np.cos(center_dec*f)*np.cos((ra-center_ra)*f)
            r = np.arccos(cosr) * 180/np.pi
            
            if r <= R:
                outseq.append(gal)
            else:
                pass
        return outseq
    elif len(sequence[0])==2 and np.isscalar(sequence[0][0]):
        for (ra, dec) in sequence:
            theta = 1.22 * ldm0 * (1 + gal.cz/SPEED_OF_LIGHT) /  D
            R = theta/2.0 * 180/np.pi
            
            f = np.pi/180.0
            cosr = np.sin(dec*f)*np.sin(center_dec*f) + np.cos(dec*f)*np.cos(center_dec*f)*np.cos((ra-center_ra)*f)
            r = np.arccos(cosr) * 180/np.pi
            
            if r <= R:
                outseq.append(gal)
            else:
                pass
        return outseq
            
    else:
        raise ValueError("The elements of `sequence` must all be `galaxy` or `tuple` of length 2 (RA/Dec pair).")
    
    
    
