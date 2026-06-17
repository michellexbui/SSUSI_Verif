#!/usr/bin/env python3

'''
A package for verifying modeled auroral precipitation against SSUSI observations.

Stand-Alone Functions
=====================
dir_exist : None
    checks if a path to a directory exists. 

Classes
=======

PrecipFile : SpaceDate dict 
    handles precipitation from a model or SSUSI observation

    Functions
    ==========
    calc_HP
        calculates hemispheric power
    id_boundaries
        identifies poleward and equatorward boundaries
    id_PB
        identfies poleward boundary
    id_EB
        identifies equatorward boundary
    saveas_pickle
        saves PrecipFile as a hard-copy pickle
    add_hemiplot
        creates a hemispheric plot in either magnetic or geographic coords

    Subclasses
    ==========
    SSUSIPrecip : SpaceData dict
        handles SSUSI observations 
    2DGELPrecip : SpaceData dict
        handles model data in a 2DGEL.bin format
    SWMFPrecip : SpaceData dict
        handles model data in SWMF.idl format

TimeSeriesComparisons : SpaceData dict
    processes multiple PrecipFiles for timeseries comparisons

    Functions
    ==========
    metricsSummary : dict, summaryplot.png
        yields a suite of accuracy metrics (Bias, RMSE, correlation fits)
        as well as a summary plot
    linFit_all : SpaceData dict
        tests a linear fit between all points of a model and observation
        yields a dict containing slope, yint, R
    linFit_lim : slope, yint, R
        tests a linear fit between a limited range of points of a model and observation
        yields a dict containing slope, yint, R


'''



# imports
# =======
import os
import sys

import supermag
from supermag import supermag_api

from hapiclient import hapi
from hapiplot import hapiplot

from pyitm.fileio.util import read_all_files

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature.nightshade as cfn

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


def dir_exist(path):
    '''
    dir_exist checks if a path to a directory exists. 
    If the directory does not exist, then the function creates that directory.
    Just a small check so the code doesn't explode ya know?
    
    Parameters
    ==========
    path : str    
        Path to desired directory

    Returns
    =======
    None

    Example
    =======
    Here's how to use dir_exist to check where to dump your files.
    You can copy-paste below and replace with your desired stuff.
    ---
    path = 'figures/energyflux/20100405/
    dir_exist(path)

    '''
    
    if not os.path.isdir(path):
        os.makedirs(path)
        print(f'Created dir: {path}')
    else:
        print(f'Dir exists: {path}')

def add_circle_boundary(ax):
    '''
    Forces a circular boundary for a Cartopy map.
    Takes the reference code below as a useful function.

    Reference: Cartopy always circular stereo example
    https://scitools.org.uk/cartopy/docs/v0.15/examples/always_circular_stereo.html 

    Parameters
    ==========
    ax : Matplotlib axis object

    Returns
    =======
    None

    Examples
    ========
    Here's how to use add_circle_boundary to enforce a circular cartopy map.
    You can copy-paste below and replace with your desired stuff.
    ---
    # create a figure in polar
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(1,2,1, projection=ccrs.NorthPolarStereo())

    # add plot content here

    # finalize figure
    ax.set_title('Title of the Plot')
    ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())
    add_circle_boundary(ax)

    '''

    theta = np.linspace(0, 2*np.pi, 100) 
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    return


class PrecipFile(SpaceData):
    '''
    PrecipFile is a SpaceData dictionary
    that handles a precipitation file for event comparison.

    Attributes
    ==========
    Every PrecipFile should have this information gathered for processing.

    time : 1D np.array 
        time series of the observation / simulation in datetime.datetime
    mlat, mlon, mlt : 2D np.array 
        magnetic coordinates [º]
    glat, glon : 2D np.array
        geographic coordinates [º]
    avee : 2D np.array
        Average energy [keV]
    eflux : 2D np.array
        Energy flux [mW/m^2]

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


    def add_hemiplot(self, coord='mag', target=None, loc=111):
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

class SSUSIPrecip(PrecipFile):
    '''
    SSUSIPrecip is a subclass of PrecipFile
    for handling SSUSI data.
    '''

    def __init__(self, start_date, end_date, datalabel, datapath, *args, **kwargs):
        super(PrecipFile, self).__init__(*args, **kwargs)  # Init as SpaceData.

        # time range of data
        self.attrs['start_date'] = start_date 
        self.attrs['end_date']  = end_date

        # description of data
        self.attrs['datalabel'] = datalabel
        self.attrs['datapath'] = datapath


    def find_SSUSI_path(date_str, sat_name, sourcename):
    '''
    Download SSUSI EDR (Environmental Data Record) aurora data from 'cdaweb' public server if run anywhere 
    or 'mia' server if run locally on mia
    
    Downloaded '.nc' files are located in uplodat/{date_str}/ 

    Parameters
    ----------
    date_str : str   
        Desired date as a string, formatted as 'YYYYMMDD' 
        where   YYYY is the four-digit year, 
                MM is a zero-padded month, and 
                DD is a zero-padded day
    sat_name : str
        Desired satellite name ('f17', 'f18', etc.) as a string.
        See here for data availability: https://docs.google.com/spreadsheets/d/1QyxeKCH3AZUILgoSASgSuJIv4_F9cR3c1689ooKb8dk/edit?gid=0#gid=0 
    sourcename : str
        Desired data source. 
        If 'cdaweb', download 'nc' from CDAWeb public server and save to uplodat/{date_str}
        If running locally on 'mia', data can be sourced from mia's backup repo
        
    Returns
    -------
    path_to_dir : str
        Path to SSUSI EDR aurora data .nc files
    
    Examples
    --------
    # source the EDR aurora data .nc files
    dirpath = find_SSUSI_path(date_str,sat_name,sourcename)

    '''

    # year and day-of-year
    year = date_str[0:4]   
    datetime_Ymd = dt.datetime.strptime(date_str, '%Y%m%d')
    datetime_doy = datetime_Ymd.timetuple().tm_yday
    doy = f'{datetime_doy:03d}'

    if sourcename == 'mia':
        # SSUSI directory path
        path_to_dir = f'/backup/Data/ssusi/data/ssusi.jhuapl.edu/dataN/{sat_name}/apl/edr-aur/{year}/{doy}/'
    elif sourcename == 'cdaweb':
        # CHECK IF uplodat/ and uplodat/{date_str}/ exists
        dir_exist('uplodat/')
        dir_exist(f'uplodat/{date_str}')
        dir_exist(f'uplodat/{date_str}/{sat_name}')

        # then upload data
        urlstr = f'https://cdaweb.gsfc.nasa.gov/pub/data/dmsp/dmsp{sat_name}/ssusi/data/edr-aurora/{year}/{doy}/'
        # wget index.html 
        os.system(f'wget -P uplodat/{date_str}/{sat_name}/ {urlstr}') 

        # read index.html for filenames
        filename = f'uplodat/{date_str}/{sat_name}/index.html' ; files = []
        with open(filename, 'r', encoding='utf-8') as f:
            html_content = f.read()

        lines = html_content.splitlines() 
        for eachline in lines:
            index_start = eachline.find('dmsp')
            index_end = eachline.find('.nc')

            if len(eachline[index_start:index_end+3]) > 0:
                files.append(eachline[index_start:index_end+3])

        # wget files into uplodat/{date_str}
        for file in files:
            os.system(f'wget -P uplodat/{date_str}/{sat_name}/ {urlstr}{file}')
        
        # success u have the files!
        path_to_dir = f'uplodat/{date_str}/{sat_name}/'

    return path_to_dir

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