#!/usr/bin/env python3

'''
A package for verifying modeled auroral precipitation against SSUSI
observations.
'''



#imports
import os
import sys
#sys.path.append('/home/mxbui/')

import supermag
from supermag import supermag_api

from hapiclient import hapi
from hapiplot import hapiplot

from pyitm.fileio.util import read_all_files

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
from cartopy.feature.nightshade import Nightshade

import numpy as np
import pandas as pd
import xarray as xr
import scipy as spy

import datetime as dt 
from datetime import datetime

import pickle
import apexpy

import spacepy
from spacepy.pybats import gitm
from spacepy import coordinates as coord
from spacepy.time import Ticktock
from spacepy.datamodel import dmarray, SpaceData
from spacepy.pybats import rim
from spacepy.plot import set_target


class PrecipFile(SpaceData):
    '''
    PrecipFile is a SpaceData dictionary
    that handles a precipitation file for event comparison.

    Attributes
    ==========
    time : 1D array 
        time series of the observation / simulation
    - coordinates (magnetic and geographic)
    - energy flux
    - average energy

    Meta-Attributes
    ===============
    These are the attributes for the overall precipitation file. 

    start_date, end_date : datetime.datetime
        desired time range of data
    datalabel, datapath : string
        data label and path to the data directory

    Example
    =======
    Here's how to instantiate a PrecipFile. 
    You can copy-paste below and replace with your desired stuff.
    ---
    # Step 1: Choose your start and end dates as datetime objects
    start_date = datetime.datetime(year, month, day, hour, minute, second)
    end_date   = start_date + datetime.timedelta(days=2.5)

    # Step 2: Specify your data source 
    datalabel = 'Data Label'
    datapath = '/home/you/yourdata/'

    # Step 3: Plug in all of the above to instantiate a PrecipFile. 
    precipfile = ssusi_verif.PrecipFile(start_date, end_date, datalabel, datapath)

    '''

    def __init__(self, start_date, end_date, datalabel, datapath, *args, **kwargs):
        super(PrecipFile, self).__init__(*args, **kwargs)  # Init as SpaceData.

        # time range of data
        self.attrs['start_date'] = start_date 
        self.attrs['end_date']  = end_date

        # description of data
        self.attrs['datalabel'] = datalabel
        self.attrs['datapath'] = datapath

    def calc_hp(self, window):
        '''
        Calculate hemispheric power.

        Parameters
        ==========
        glat, glon : 2D np.array of floats
            Geographic latitudes and longitudes [degrees]
        eflux : 2D np.array of floats 
            Energy flux [mW/m^2]    

        Other Parameters
        ================
        eflux_type : str
            type of energy flux (e.g. total, diffuse, etc.)
        window : str
            'swath' or 'total'

        Returns
        =======
        hp : float
            Hemispheric power [GW] calculated for one auroral oval event.
        
        Examples
        ========
        >>> import spacepy.pybats.bats as pbs
        >>> mhd = pbs.Bats2d('spacepy-code/spacepy/pybats/slice2d_species.out')
        >>> pbs._calc_ndens(mhd)
        '''

        glat = self['glat']
        glon = self['glon']
        eflux = self['eflux']
        r_E = 6378.0 * 1000
        r = r_E + self['alt']*1000

        integrand = np.zeros((len(eflux[:,0])-1,len(eflux[0,:])-1))

        for j in range(0,len(eflux[:,0])-1):
            for i in range(0,len(eflux[0,:])-1):
                r = r_E + 110*1000
                del_glat = np.abs(glat[j,i+1]-glat[j,i])*np.pi/180                           
                del_glon = np.abs(glon[j+1,i]-glon[j,i])*np.pi/180
                if 0.0 <= eflux[j,i]:
                    integrand[j,i] = eflux[j,i] * r**2 * np.sin(90-glat[j,i]*np.pi/180) * del_glat * del_glon                 # convert ergs/s/cm^2 to W/m^2

        self[f'hp_{window}'] = np.sum(integrand)/(1.0e9)

        pass


    def add_hemi_plot(self, coord='mag', target=None, loc=111):
        '''
        Add a hemispheric plot figure.

        Parameters
        ==========
        coord : str, defaults to 'mag'
           Set coordinate system to use, either 'mag' or 'geo'.

        Other Parameters
        ================

        Returns
        =======
        fig : Matplotlib Figure object
            The figure on which the plot is made.
        ax : Matplotlib Axes object
            The axes on which the plot is made.

        Examples
        ========
        >>> import spacepy.pybats.bats as pbs
        >>> mhd = pbs.Bats2d('spacepy-code/spacepy/pybats/slice2d_species.out')
        >>> pbs._calc_ndens(mhd)
        '''
        pass

class SwmfPrecip(PrecipFile):
    '''
    Subclass for handling swmf output.
    '''

    def __init__(self, filename, *args, **kwargs):
        super(PrecipFile, self).__init__(*args, **kwargs)  # Init as SpaceData.

        self.attrs['file'] = filename

        data = rim.Iono(filename)

        self['avee'] = data['n_ave']

    def calc_hp(self):
        pass