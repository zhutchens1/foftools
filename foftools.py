# -*- coding: utf-8 -*-
"""
@package 'foftools'
@author: Zackary L. Hutchens, UNC Chapel Hill

This package contains classes and functions for performing galaxy group
identification using the friends-of-friends (FoF) and probability friends-of-friends
(PFoF) algorithms. Additionally, it includes functions for group association of faint
galaxies (Eckert+ 2016), as well as related tools such as functions for computing
group-integrated quantities (like luminosity or stellar mass).

The previous version of foftools, 3.3 (28 Feb 2020) is now historic and been renamed
`objectbasedfoftools.py` within the git repository. The new version, 4.0, provides
the same tools with a performance-based approach that prioritizes numpy/njit optimization
over readability/convenience of python classes, which require itertions throughout
the many necessary algorithms for group finding. 
"""

import numpy as np
import pandas as pd
import itertools
from scipy.integrate import quad
from scipy.interpolate import interp1d
from math import erf
from copy import deepcopy
import math
import time
import warnings
from numba import njit

__versioninfo__ = "foftools version 4.0 (previous version 3.3 now labeled `objectbasedfoftools.py`)"

from astropy.cosmology import LambdaCDM
cosmo = LambdaCDM(H0=100.0, Om0=0.3, Ode0=0.7) # this puts everything in "per h" units.
SPEED_OF_LIGHT = 3.00E+05 # km/s


# -------------------------------------------------------------- #
#  friends-of-friends (FOF) algorithm
# -------------------------------------------------------------- #
def fast_fof(ra, dec, cz, bperp, blos, s, printConf=True):
    """
    -----------
    Compute group membership from galaxies' equatorial coordinates using a friends-of-friends algorithm,
    based on the method of Berlind et al. 2006. This algorithm is designed to identify groups
    in volume-limited catalogs down to a common magnitude floor. All input arrays (RA, Dec, cz)
    must have already been selected to be above the group-finding floor.
    
    Arguments:
        ra (iterable): list of right-ascesnsion coordinates of galaxies in decimal degrees.
        dec (iterable): list of declination coordinates of galaxies in decimal degrees.
        cz (iterable): line-of-sight recessional velocities of galaxies in km/s.
        bperp (scalar): linking proportion for the on-sky plane (use 0.07 for RESOLVE/ECO)
        blos (scalar): linking proportion for line-of-sight component (use 1.1 for RESOLVE/ECO)
        s (scalar): mean separation of galaxies above floor in volume-limited catalog.
        printConf (bool, default True): bool indicating whether to print confirmation at the end.
    Returns:
        grpid (np.array): list containing unique group ID numbers for each target in the input coordinates.
                The list will have shape len(ra).
    -----------
    """
    t1 = time.time()
    Ngalaxies = len(ra)
    ra = np.float64(ra)
    dec = np.float64(dec)
    cz = np.float64(cz)
    assert (len(ra)==len(dec) and len(dec)==len(cz)),"RA/Dec/cz arrays must equivalent length."

    #check if s is adaptive or fixed
    if np.isscalar(s):
        adaptive=False
    else:
        adaptive=True
        s = np.float64(s)[:,None] # convert to column vector

    phi = (ra * np.pi/180.)
    theta =(np.pi/2. - dec*(np.pi/180.))
    transv_cmvgdist = (cosmo.comoving_transverse_distance(cz/SPEED_OF_LIGHT).value)
    los_cmvgdist = (cosmo.comoving_distance(cz/SPEED_OF_LIGHT).value)
    friendship = np.zeros((Ngalaxies, Ngalaxies))

    # Compute on-sky and line-of-sight distance between galaxy pairs
    column_phi = phi[:, None]
    column_theta = theta[:, None]
    half_angle = np.arcsin((np.sin((column_theta-theta)/2.0)**2.0 + np.sin(column_theta)*np.sin(theta)*np.sin((column_phi-phi)/2.0)**2.0)**0.5)
    
    # Compute on-sky perpendicular distance
    column_transv_cmvgdist = transv_cmvgdist[:, None]
    dperp = (column_transv_cmvgdist + transv_cmvgdist) * (half_angle) # In Mpc/h
    
    # Compute line-of-sight distances
    dlos = np.abs(los_cmvgdist - los_cmvgdist[:, None])
    
    # Compute friendship - fixed case
    if not adaptive:
        index = np.where(np.logical_and(dlos<=blos*s, dperp<=bperp*s))
        friendship[index]=1
        assert np.all(np.abs(friendship-friendship.T) < 1e-8), "Friendship matrix must be symmetric."
    
        if printConf:
            print('FoF complete in {a:0.4f} s'.format(a=time.time()-t1))
    # compute friendship - adaptive case
    elif adaptive:
        index = np.where(np.logical_and(dlos-blos*s<=0, dperp-bperp*s<=0))
        friendship[index]=1
        if printConf:
            print('FoF complete in {a:0.4f} s'.format(a=time.time()-t1))        

    return collapse_friendship_matrix(friendship)


# -------------------------------------------------------------- #
# probability friends-of-friends algorithm                       #
# -------------------------------------------------------------- #
@njit
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
    return 1/(math.sqrt(2*np.pi) * sigma) * math.exp(-1 * 0.5 * ((x-mu)/sigma) * ((x-mu)/sigma))

@njit
def pfof_integral(z, czi, czerri, czj, czerrj, VL):
    c=SPEED_OF_LIGHT
    return gauss(z, czi/c, czerri/c) * (0.5*math.erf((z+VL-czj/c)/((2**0.5)*czerrj/c)) - 0.5*math.erf((z-VL-czj/c)/((2**0.5)*czerrj/c)))


def fast_pfof(ra, dec, cz, czerr, perpll, losll, Pth, printConf=True):
    """
    -----
    Compute group membership from galaxies' equatorial  coordinates using a probabilitiy
    friends-of-friends (PFoF) algorithm, based on the method of Liu et al. 2008. PFoF is
    a variant of FoF (see `foftools.fast_fof`, Berlind+2006), which treats galaxies as Gaussian
    probability distributions, allowing group membership selection to account for the 
    redshift errors of photometric redshift measurements. 
    
    Arguments:
        ra (iterable): list of right-ascesnsion coordinates of galaxies in decimal degrees.
        dec (iterable): list of declination coordinates of galaxies in decimal degrees.
        cz (iterable): line-of-sight recessional velocities of galaxies in km/s.
        czerr (iterable): errors on redshifts of galaxies in km/s.
        perpll (float): perpendicular linking length in Mpc/h. 
        losll (float): line-of-sight linking length in km/s.
        Pth (float): Threshold probability from which to construct the group catalog. If None, the
            function will return a NxN matrix of friendship probabilities.
        printConf (bool, default True): bool indicating whether to print confirmation at the end.
    Returns:
        grpid (np.array): list containing unique group ID numbers for each target in the input coordinates.
                The list will have shape len(ra).
    -----
    """
    print('you know.... you could speed this up more if check for transverse friendship before integrating...')
    t1 = time.time()
    Ngalaxies = len(ra)
    ra = np.float32(ra)
    dec = np.float32(dec)
    cz = np.float32(cz)
    czerr = np.float32(czerr)
    assert (len(ra)==len(dec) and len(dec)==len(cz)),"RA/Dec/cz arrays must equivalent length."

    phi = (ra * np.pi/180.)
    theta = (np.pi/2. - dec*(np.pi/180.))
    transv_cmvgdist = (cosmo.comoving_transverse_distance(cz/SPEED_OF_LIGHT).value)
    friendship = np.zeros((Ngalaxies, Ngalaxies))

    # Compute on-sky perpendicular distance
    column_phi = phi[:, None]
    column_theta = theta[:, None]
    half_angle = np.arcsin((np.sin((column_theta-theta)/2.0)**2.0 + np.sin(column_theta)*np.sin(theta)*np.sin((column_phi-phi)/2.0)**2.0)**0.5)
    column_transv_cmvgdist = transv_cmvgdist[:, None]
    dperp = (column_transv_cmvgdist + transv_cmvgdist) * half_angle # In Mpc/h
    
    # Compute line-of-sight probabilities
    prob_dlos=np.zeros((Ngalaxies, Ngalaxies))
    c=SPEED_OF_LIGHT
    VL = losll/c
    for i in range(0,Ngalaxies):
        for j in range(0, i+1):
            if j<i:
                val = quad(pfof_integral, 0, 100, args=(cz[i], czerr[i], cz[j], czerr[j], VL),\
                           points=np.float64([cz[i]/c-5*czerr[i]/c,cz[i]/c-3*czerr[i]/c, cz[i]/c, cz[i]/c+3*czerr[i]/c, cz[i]/c+5*czerr[i]/c]),\
                            wvar=cz[i]/c)
                prob_dlos[i][j]=val[0]
                prob_dlos[j][i]=val[0]
            elif i==j:
                prob_dlos[i][j]=1
    
    # Produce friendship matrix and return groups
    index = np.where(np.logical_and(prob_dlos>Pth, dperp<=perpll))
    friendship[index]=1
    assert np.all(np.abs(friendship-friendship.T) < 1e-8), "Friendship matrix must be symmetric."
    
    if printConf:
        print('PFoF complete in {a:0.4f} s'.format(a=time.time()-t1))
    return collapse_friendship_matrix(friendship)


# -------------------------------------------------------------- #
# algorithms for extracting group catalog from FOF friendship    #
# -------------------------------------------------------------- #
def collapse_friendship_matrix(friendship_matrix):
    """
    ----
    Collapse a friendship matrix resultant of a FoF computation into an array of
    unique group numbers. 
    
    Arguments:
        friendship_matrix (iterable): iterable of shape (N, N) where N is the number of targets.
            Each element (i,j) of the matrix should represent the galaxy i and galaxy j are friends,
            as determined by the FoF linking length.
    Returns:
        grpid (iterable): 1-D array of size N containing unique group ID numbers for every target.
    ----
    """
    friendship_matrix=np.array(friendship_matrix)
    Ngalaxies = len(friendship_matrix[0])
    grpid = np.zeros(Ngalaxies)
    grpnumber = 1
    
    for row_num,row in enumerate(friendship_matrix):
        if not grpid[row_num]:
            group_indices = get_group_ind(friendship_matrix, row_num, visited=[row_num])
            grpid[group_indices]=grpnumber
            grpnumber+=1 
    return grpid
        
def get_group_ind(matrix, active_row_num, visited):
    """
    ----
    Recursive algorithm to form a tree of indices from a friendship matrix row. Similar 
    to the common depth-first search tree-finding algorithm, but enabling identification
    of isolated nodes and no backtracking up the resultant trees' edges. 
    
    Example: Consider a group formed of the indices [10,12,133,53], but not all are 
    connected to one another.
                
                10 ++++ 12
                +
               133 ++++ 53
    
    The function `collapse_friendship_matrix` begins when 10 is the active row number. This algorithm
    searches for friends of #10, which are #12 and #133. Then it *visits* the #12 and #133 galaxies
    recursively, finding their friends also. It adds 12 and 133 to the visited array, noting that
    #10 - #12's lone friend - has already been visited. It then finds #53 as a friend of #133,
    but again notes that #53's only friend it has been visited. It then returns the array
    visited=[10, 12, 133, 53], which form the FoF group we desired to find.
    
    Arguments:
        matrix (iterable): iterable of shape (N, N) where N is the number of targets.
            Each element (i,j) of the matrix should represent the galaxy i and galaxy j are friends,
            as determined from the FoF linking lengths.
        active_row_num (int): row number to start the recursive row searching.
        visited (int): array containing group members that have already been visited. The recursion
            ends if all friends have been visited. In the initial call, use visited=[active_row_num].
    ----
    """
    friends_of_active = np.where(matrix[active_row_num])
    for friend_ind in [k for k in friends_of_active[0] if k not in visited]:
        visited.append(friend_ind)
        visited = get_group_ind(matrix, friend_ind, visited)
    return visited




# -------------------------------------------------------------- #
#  functions for galaxy association to existing groups
# -------------------------------------------------------------- #

def fast_faint_assoc(faintra, faintdec, faintcz, grpra, grpdec, grpcz, grpid, radius_boundary, velocity_boundary, losll=-1):
    """
    Associate galaxies to a group catalog based on given radius and velocity boundaries, based on a method
    similar to that presented in Eckert+ 2016. 

    Parameters
    ----------
    faintra : iterable
        Right-ascension of faint galaxies in degrees.
    faintdec : iterable
        Declination of faint galaxies in degrees.
    faintcz : iterable
        Redhshift velocities of faint galaxies in km/s.
    grpra : iterable
        Right-ascension of group centers in degrees.
    grpdec : iterable
        Declination of group centers in degrees. Length matches `grpra`.
    grpcz : iterable
        Redshift velocity of group center in km/s. Length matches `grpra`.
    grpid : iterable
        group ID of each FoF group (i.e., from `foftools.fast_fof`.) Length matches `grpra`.
    radius_boundary : iterable
        Radius within which to search for faint galaxies around FoF groups. Length matches `grpra`.
    velocity_boundary : iterable
        Velocity from group center within which to search for faint galaxies around FoF groups. Length matches `grpra`.
    losll : scalar, default -1
         Line-of-sight linking length in km/s. If losll>0, then associations are made with the *larger of* the velocity
         boundary of the LOS linking length. If losll=-1 (default), use only the velocity boundary to associate.

    Returns
    -------
    assoc_grpid : iterable
        group ID of every faint galaxy. Length matches `faintra`.
    assoc_flag : iterable
        association flag for every galaxy (see function description). Length matches `faintra`.
    """
    velocity_boundary=np.asarray(velocity_boundary)
    radius_boundary=np.asarray(radius_boundary)
    Nfaint = len(faintra)
    assoc_grpid = np.zeros(Nfaint).astype(int)
    assoc_flag = np.zeros(Nfaint).astype(int)
    radius_ratio=np.zeros(Nfaint)

    # resize group coordinates to be the # of groups, not # galaxies
    junk, uniqind = np.unique(grpid, return_index=True)
    grpra = grpra[uniqind]
    grpdec = grpdec[uniqind]
    grpcz = grpcz[uniqind]
    grpid = grpid[uniqind]
    velocity_boundary=velocity_boundary[uniqind]
    radius_boundary=radius_boundary[uniqind]

    # Redefine velocity boundary array to take larger of (dV, linking length)
    if losll>0:
        velocity_boundary[np.where(velocity_boundary<=losll)]=losll

    # Make Nfaints x Ngroups grids for transverse/LOS distances from group centers
    faintphi = (faintra * np.pi/180.)[:,None]
    fainttheta = (np.pi/2. - faintdec*(np.pi/180.))[:,None]
    faint_cmvg = (cosmo.comoving_transverse_distance(faintcz/SPEED_OF_LIGHT).value)[:, None]
    grpphi = (grpra * np.pi/180.)
    grptheta = (np.pi/2. - grpdec*(np.pi/180.))
    grp_cmvg = cosmo.comoving_transverse_distance(grpcz/SPEED_OF_LIGHT).value

    half_angle = np.arcsin((np.sin((fainttheta-grptheta)/2.0)**2.0 + np.sin(fainttheta)*np.sin(grptheta)*np.sin((faintphi-grpphi)/2.0)**2.0)**0.5)
    Rp = (faint_cmvg + grp_cmvg) * np.sin(half_angle)
    DeltaV = np.abs(faintcz[:,None] - grpcz)

    for gg in range(0,len(grpid)):
        for fg in range(0,Nfaint):
            tempratio = Rp[fg][gg]/radius_boundary[gg]
            condition=((tempratio<1) and (DeltaV[fg][gg]<velocity_boundary[gg]))
            # multiple groups competing (has already been associated before)
            if condition and assoc_flag[fg]:
                # multiple grps competing to associate galaxy
                if tempratio<radius_ratio[fg]:
                    radius_ratio[fg]=tempratio
                    assoc_grpid[fg]=grpid[gg]
                    assoc_flag[fg]=1
                else:
                    pass
            # galaxy not associated yet - go ahead and do it
            elif condition and (not assoc_flag[fg]):
                radius_ratio[fg]=tempratio
                assoc_grpid[fg]=grpid[gg]
                assoc_flag[fg]=1
            # condition not met
            elif (not condition):
                pass
            else:
                print("Galaxy failed association algorithm at index {}".format(fg))

    # assign group ID numbers to galaxies that didn't associate
    still_isolated = np.where(assoc_grpid==0)
    assoc_grpid[still_isolated]=np.arange(np.max(grpid)+1, np.max(grpid)+1+len(still_isolated[0]), 1)
    assoc_flag[still_isolated]=-1
    return assoc_grpid, assoc_flag

# -------------------------------------------------------------- #
# functions for computing properties of groups                   #
# -------------------------------------------------------------- #

def group_skycoords(galaxyra, galaxydec, galaxycz, galaxygrpid):
    """
    -----
    Obtain a list of group centers (RA/Dec/cz) given a list of galaxy coordinates (equatorial)
    and their corresponding group ID numbers.
    
    Inputs (all same length)
       galaxyra : 1D iterable,  list of galaxy RA values in decimal degrees
       galaxydec : 1D iterable, list of galaxy dec values in decimal degrees
       galaxycz : 1D iterable, list of galaxy cz values in km/s
       galaxygrpid : 1D iterable, group ID number for every galaxy in previous arguments.
    
    Outputs (all shape match `galaxyra`)
       groupra : RA in decimal degrees of galaxy i's group center.
       groupdec : Declination in decimal degrees of galaxy i's group center.
       groupcz : Redshift velocity in km/s of galaxy i's group center.
    
    Note: the FoF code of AA Berlind uses theta_i = declination, with theta_cen = 
    the central declination. This version uses theta_i = pi/2-dec, with some trig functions
    changed so that the output *matches* that of Berlind's FoF code (my "deccen" is the same as
    his "thetacen", to be exact.)
    -----
    """
    # Prepare cartesian coordinates of input galaxies
    ngalaxies = len(galaxyra)
    galaxyphi = galaxyra * np.pi/180.
    galaxytheta = np.pi/2. - galaxydec*np.pi/180.
    galaxyx = np.sin(galaxytheta)*np.cos(galaxyphi)
    galaxyy = np.sin(galaxytheta)*np.sin(galaxyphi)
    galaxyz = np.cos(galaxytheta)
    # Prepare output arrays
    uniqidnumbers = np.unique(galaxygrpid)
    groupra = np.zeros(ngalaxies)
    groupdec = np.zeros(ngalaxies)
    groupcz = np.zeros(ngalaxies)
    for i,uid in enumerate(uniqidnumbers):
        sel=np.where(galaxygrpid==uid)
        nmembers = len(galaxygrpid[sel])
        xcen=np.sum(galaxycz[sel]*galaxyx[sel])/nmembers
        ycen=np.sum(galaxycz[sel]*galaxyy[sel])/nmembers
        zcen=np.sum(galaxycz[sel]*galaxyz[sel])/nmembers
        czcen = np.sqrt(xcen**2 + ycen**2 + zcen**2)
        deccen = np.arcsin(zcen/czcen)*180.0/np.pi # degrees
        if (ycen >=0 and xcen >=0):
            phicor = 0.0
        elif (ycen < 0 and xcen < 0):
            phicor = 180.0
        elif (ycen >= 0 and xcen < 0):
            phicor = 180.0
        elif (ycen < 0 and xcen >=0):
            phicor = 360.0
        elif (xcen==0 and ycen==0):
            print("Warning: xcen=0 and ycen=0 for group {}".format(galaxygrpid[i]))
        # set up phicorrection and return phicen.
        racen=np.arctan(ycen/xcen)*(180/np.pi)+phicor # in degrees
        # set values at each element in the array that belongs to the group under iteration
        groupra[sel] = racen # in degrees
        groupdec[sel] = deccen # in degrees
        groupcz[sel] = czcen
    return groupra, groupdec, groupcz


def get_rproj_czdisp(galaxyra, galaxydec, galaxycz, galaxygrpid, HUBBLE_CONST=70.):
    """
    Compute the observational projected radius, in Mpc/h, and the observational
    velocity dispersion, in km/s, for a galaxy group catalog. Input should match
    the # of galaxies, and the output will as well. Based on FoF4 code of Berlind+ 
    2006.
    
    Parameters
    ----------
    galaxyra : iterable
        Right-ascension of grouped galaxies in decimal degrees.
    galaxydec : iterable
        Declination of grouped galaxies in decimal degrees.
    galaxycz : iterable
        Redshift velocity (cz) of grouped galaxies in km/s.
    galaxygrpid : iterable
        Group ID numbers of grouped galaxies, shape should match `galaxyra`.

    Returns
    -------
    rproj : np.array, shape matches `galaxyra`
        For element index i, projected radius of galaxy group to which galaxy i belongs, in Mpc/h.
    vdisp : np.array, shape matches `galaxyra`
        For element index i, velocity dispersion of galaxy group to which galaxy i belongs, in km/s.

    """
    galaxyra=np.asarray(galaxyra)
    galaxydec=np.asarray(galaxydec)
    galaxycz=np.asarray(galaxycz)
    galaxygrpid=np.asarray(galaxygrpid)
    rproj=np.zeros(len(galaxyra))
    vdisp=np.zeros(len(galaxyra))
    grpra, grpdec, grpcz = group_skycoords(galaxyra, galaxydec, galaxycz, galaxygrpid)
    grpra = grpra*np.pi/180. #convert  everything to radians
    galaxyra=galaxyra*np.pi/180.
    galaxydec=galaxydec*np.pi/180.
    grpdec = grpdec*np.pi/180.
    uniqid = np.unique(galaxygrpid)
    cspeed=299800 # km/s
    for uid in uniqid:
        sel = np.where(galaxygrpid==uid)
        nmembers=len(sel[0])
        if nmembers==1:
            rproj[sel]=0.
            vdisp[sel]=0.
        else:
            phicen=grpra[sel][0]
            thetacen=grpdec[sel][0]
            cosDpsi=np.cos(thetacen)*np.cos(galaxydec[sel])+np.sin(thetacen)*np.sin(galaxydec[sel])*np.cos((phicen - galaxyra[sel]))
            sinDpsi=np.sqrt(1-cosDpsi**2)
            rp=sinDpsi*galaxycz[sel]/HUBBLE_CONST
            rproj[sel]=np.sqrt(np.sum(rp**2)/len(sel[0]))
            czcen = grpcz[sel][0]
            Dz2 = np.sum((galaxycz[sel]-czcen)**2.0)
            vdisp[sel]=np.sqrt(Dz2/(nmembers-1))/(1.+czcen/cspeed)
    return rproj, vdisp


def multiplicity_function(grpids, return_by_galaxy=False):
    """
    Return counts for binning based on group ID numbers.

    Parameters
    ----------
    grpids : iterable
        List of group ID numbers. Length must match # galaxies.
    Returns
    -------
    occurences : list
        Number of galaxies in each galaxy group (length matches # groups).
    """
    grpids=np.asarray(grpids)
    uniqid = np.unique(grpids)
    if return_by_galaxy:
        grpn_by_gal=np.zeros(len(grpids)).astype(int)
        for idv in grpids:
            sel = np.where(grpids==idv)
            grpn_by_gal[sel]=len(sel[0])
        return grpn_by_gal
    else:
        occurences=[]
        for uid in uniqid:
            sel = np.where(grpids==uid)
            occurences.append(len(grpids[sel]))
        return occurences


def get_grprproj_e17(galra, galdec, galcz, galgrpid, h):
    """
    Credit: Ella Castelloe for original python code 
    Compute the observational group projected radius from Eckert et al. (2017). Adapted from
    Katie's IDL code, which was used to calculate grprproj for FoF groups in RESOLVE.
    75% radius of all members

    Parameters
    --------------------
    galra : iterable
       RA of input galaxies in decimal degrees
    galdec : iterable
       Declination of input galaxies in decimal degrees
    galcz : iterable
       Observed local group-corrected radial velocities of input galaxies (km/s)
    galgrpid : iterable
       Group ID numbers for input galaxies, length matches `galra`.

    Returns
    --------------------
    grprproj : np.array
        Group projected radii in Mpc/h, length matches `galra`.

    """
    rproj75dist = np.zeros(len(galgrpid))
    uniqgrpid = np.unique(galgrpid)
    for uid in uniqgrpid:
        galsel = np.where(galgrpid==uid)
        if len(galsel[0])> 2:
            ras = np.array(galra[galsel])*np.pi/180 #in radians
            decs = np.array(galdec[galsel])*np.pi/180 #in radians
            czs = np.array(galcz[galsel])
            grpn = len(galsel[0]) 
            H0 = 100*h #km/s/Mpc
            grpdec = np.mean(decs)
            grpra = np.mean(ras)
            grpcz = np.mean(czs)
            theta = 2*np.arcsin(np.sqrt((np.sin((decs-grpdec)/2))**2 + np.cos(decs)*np.cos(grpdec)*(np.sin((ras-grpra)/2)**2)))
            theta = np.array(theta)
            rproj = theta*grpcz/H0
            sortorder = np.argsort(rproj)
            rprojsrt_y = rproj[sortorder]
            rprojval_x = np.arange(0 , grpn)/(grpn-1.) #array from 0 to 1, use for interpolation
            #print(rprojsrt, rprojval, grpn)
            f = interp1d(rprojval_x, rprojsrt_y) 
            rproj75val = f(0.75)
        else:
            rproj75val = 0.0
            #rprojval_x = 0.0
            #rprojsrt_y = 0.0
        rproj75dist[galsel]=rproj75val
    return rproj75dist

def getmhoffset(delta1, delta2, borc1, borc2, cc):
    """
    Credit: Ella Castelloe & Katie Eckert
    Adapted from Katie's code, using eqns from "Sample Variance Considerations for Cluster Surveys," Hu & Kravtsov (2003) ApJ, 584, 702
    (astro-ph/0203169)
    delta1 is overdensity of input, delta2 is overdensity of output -- for mock, delta1 = 200
    borc = 1 if wrt background density, borc = 0 if wrt critical density
    cc is concentration of halo- use cc=6
    """
    if borc1 == 0:
        delta1 = delta1/0.3
    if borc2 == 0:
        delta2 = delta2/0.3
    xin = 1./cc
    f_1overc = (xin)**3. * (np.log(1. + (1./xin)) - (1. + xin)**(-1.))
    f1 = delta1/delta2 * f_1overc
    a1=0.5116
    a2=-0.4283
    a3=-3.13e-3
    a4=-3.52e-5
    p = a2 + a3*np.log(f1) + a4*(np.log(f1))**2.
    x_f1 = (a1*f1**(2.*p) + (3./4.)**2.)**(-1./2.) + 2.*f1
    r2overr1 = x_f1
    m1overm2 = (delta1/delta2) * (1./r2overr1)**3. * (1./cc)**3.
    return m1overm2

def get_central_flag(galquantity, galgrpid):
    """
    Produce 1/0 flag indicating central/satellite for a galaxy
    group dataset.

    Parameters
    -------------------
    galquantity : np.array
       Quantity by which to select centrals. If (galquantity>0).all(), the central is taken as the maximum galquantity in each group.
       If (galquantity<0).all(), the quantity is assumed to be an abs. magnitude, and the central is the minimum quanity in each group.
    galgrpid : np.array
       Group ID number for each galaxy.

    Returns
    -------------------
    cflag : np.array
       1/0 flag indicating central/satellite status.
    """
    cflag = np.zeros(len(galquantity))
    uniqgrpid = np.unique(galgrpid)
    if ((galquantity>-1).all()):
       centralfn = np.max
    if ((galquantity<0).all()):
       centralfn = np.min
    for uid in uniqgrpid:
        galsel = np.where(galgrpid==uid)
        centralmass = centralfn(galquantity[galsel])
        centralsel = np.where((galgrpid==uid) & (galquantity==centralmass))
        cflag[centralsel]=1.
        satsel = np.where((galgrpid==uid) & (galquantity!=centralmass))
        cflag[satsel]=0.
    return cflag


def get_outermost_galradius(galra, galdec, galcz, galgrpid):
    """
    Get the radius, in arcseconds, to the outermost gruop galaxy.
    
    Parameters
    -------------------
    galra : np.array
        RA in decimal degrees of grouped galaxies.
    galdec : np.array
        Dec in decimal degrees of grouped galaxies.
    galgrprpid : np.array
        Group ID number of input galaxies.

    Returns
    --------------------
    radius : np.array
        Radius to outermost member of galaxy's group (length = # galaxies = len(galra)). Units: arcseconds.
    """
    radius = np.zeros(len(galra))
    grpra, grpdec, grpcz = group_skycoords(galra, galdec, galcz, galgrpid)
    #angsep = angular_separation(galra, galdec, grpra, grpdec)  # arcsec
    angsep = np.sqrt((grpra-galra)**2. + (grpdec-galdec)**2.)*(np.pi/180.)
    rproj = (galcz+grpcz)/70. * np.sin(angsep/2.)
    for uid in np.unique(galgrpid):
        galsel = np.where(galgrpid==uid)
        radius[galsel] = np.max(angsep[galsel])
        #radius[galsel] = np.max(rproj[galsel])
    #return radius*206265/(grpcz/70.)
    return radius*206265

def angular_separation(ra1,dec1,ra2,dec2):
    """
    Compute the angular separation bewteen two lists of galaxies using the Haversine formula.
    
    Parameters
    ------------
    ra1, dec1, ra2, dec2 : array-like
       Lists of right-ascension and declination values for input targets, in decimal degrees. 
    
    Returns
    ------------
    angle : np.array
       Array containing the angular separations between coordinates in list #1 and list #2, as above.
       Return value expressed in radians, NOT decimal degrees.
    """
    phi1 = ra1*np.pi/180.
    phi2 = ra2*np.pi/180.
    theta1 = np.pi/2. - dec1*np.pi/180.
    theta2 = np.pi/2. - dec2*np.pi/180.
    return 2*np.arcsin(np.sqrt(np.sin((theta2-theta1)/2.0)**2.0 + np.sin(theta1)*np.sin(theta2)*np.sin((phi2 - phi1)/2.0)**2.0))

def sepmodel(x, a, b, c, d, e):
    #return np.abs(a)*np.exp(-1*np.abs(b)*x + c)+d
    #return a*(x**3)+b*(x**2)+c*x+d
    return a*(x**4)+b*(x**3)+c*(x**2)+(d*x)+e

def giantmodel(x, a, b):
    return np.abs(a)*np.log(np.abs(b)*x+1)

