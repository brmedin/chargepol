"""
This script defines parameters, have functions and code 
to predict charge layer polarity from flashes.

It uses VHF Lightning Mapping Array Level 2 HDF5 files obtained from lmatools as input. 
Please refer to lmatools to convert LMA Level 1 data to Level 2.

Change parameters in the beggining of the code (lines 29-62) accordingly.

Usage: python chargepol.py

Output: NetCDF file with polarity of a charge layer, time of a charge layer in seconds after 0 UTC, 
charge layer bottom altitude in km, charge layer width in km, east-west distance from network center in km, 
south-north distance from network center in km.

Please refer to Medina et al., 2021, AGU ESS
Or contact bruno.medina@colorado.edu

Created on 04/23/2020
"""

# Import packages:
import numpy as np
import glob
import h5py
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Define parameters:
# Directory where LMA level 2 HDF5 files are saved:
direct = '/home/user/lma/level2/'

# file path for output
filename_output = 'chargepol.csv'

# LMA network center (lat, lon):
netw_center = np.array([-31.7,-64.1]) 

# Maximum range in km from network center:
max_range = 100 

# Minimum number of sources for a flash:  
nsou = 20 

# Minimum duration of a preliminary breakdown, in ms:
min_pb_dur = 2 

# Maximum duration of a preliminary breakdown, in ms:
max_pb_dur = 10 

# Minimum number of preliminary breakdown sources:
min_pb_sou = 4 

# Minimum preliminary breakdown vertical speed in km/milissecond (e.g., use 0.05 for 5*10^4 m/s):
min_ver_speed = 0.05 

# Maximum MSE for the linear regression fit to the preliminary breakdown sources:
max_mse = 0.25

# Percentile interval to define charge layers (e.g., use [0.1,0.9] for 10th-90th percentile interval):
perc_interv = np.array([0.1,0.9]) 

def read_lma(file, netw_center, max_range, nsou):

    """
    Read LMA level 2 data.

    Parameters
    ---------
    file : str
        .h5 file to be read.
    netw_center : numpy array 
        Latitude and longitude of network center.
    max_range : float
        Maximum range from the network center to search for flashes, in km.
    nsou : int
        Minimum number of sources for a flash.

    Returns
    -------
    lma_x: 
        West-east distance of sources from network center, in km
    lma_y: 
        South-north distance of sources from network center, in km
    lma_z: 
        Altitude of sources above MSL, in km
    lma_t: 
        Time of sources in seconds after 00 UTC
    lma_flid: 
        Flash ID of sources
    flid: 
        Flash ID of flashes within max_range and with number of sources > nsou 
    flx: 
        West-east distance of flash centroid from network center, in km
    fly:
        South-north distance of flash centroid from network center, in km
    """
              
    # Read LMA data:    
    print(f"Reading: {file}")
    data = h5py.File(file)
    len_file = int(len(file))
    hhmm = file[len_file-20:len_file-16]
    date = file.split("_")[-3]
    hhmm = file.split("_")[-2]
    if len(list(data['events'].keys())) == 1:
        events = np.array(data['events'][list(data['events'].keys())[0]])
        flashes = np.array(data['flashes'][list(data['events'].keys())[0]])
    else:
        events = np.array(data['events']['LMA_'+date+'_'+hhmm+'_600'])
        flashes = np.array(data['flashes']['LMA_'+date+'_'+hhmm+'_600'])
     
    # Save Events: 
    lma_lat = []
    lma_lon = []
    lma_z = []
    lma_t = []
    lma_flid = []

    for b in range(0,int(events.size)): 
        lma_z.append([events[b][0]])
        lma_flid.append([events[b][3]])
        lma_lat.append([events[b][4]])
        lma_lon.append([events[b][5]])
        lma_t.append([events[b][9]])

    lma_lat = np.array(lma_lat, np.dtype(float))
    lma_lon = np.array(lma_lon, np.dtype(float))
    lma_z = np.array(lma_z, np.dtype(float))
    lma_t = np.array(lma_t, np.dtype(float))
    lma_flid = np.array(lma_flid, np.dtype(float))

    # Convert LMA lat,lon to x,y relative to the center of nework 
    Re = 6356 # Earth radius in km
    # Longitude distance from source to network center multipl by 1deg latitude distance
    lma_x = (lma_lon - netw_center[1]) * (2*np.pi*Re*np.cos(lma_lat*np.pi/180)/360) 
    # Latitude distance from source to network center multipl by 1deg longitude distance
    lma_y = (lma_lat - netw_center[0]) * (2*np.pi*Re/360)
    # Convert altitude height to km
    lma_z = lma_z/1000. 

    # Save Flashes:
    flid = []
    flx = []
    fly = []
    
    for c in range(0,int(flashes.size)): 
    
        flcent_x = (flashes[c][3] - netw_center[1]) * (2*np.pi*Re*np.cos(flashes[c][2]*np.pi/180)/360) 
        flcent_y = (flashes[c][2] - netw_center[0]) * (2*np.pi*Re/360)
        flcent_xy = ( flcent_x**(2.) + flcent_y**(2.) )**(0.5)
    
        # Flashes within max_range and with more than nsou sources:
        if flcent_xy < max_range and flashes[c][10] > nsou:
            flid.append([flashes[c][5]])
            flx.append([flcent_x])
            fly.append([flcent_y])
    
    flid = np.array(flid, np.dtype(float))
    flx = np.array(flx, np.dtype(float))
    fly = np.array(fly, np.dtype(float))

    return lma_x, lma_y, lma_z, lma_t, lma_flid, flid, flx, fly

def regression_pb(lma_t_fl, lma_z_fl, min_t, max_pb_dur):

    """
    Linear regression fit to the preliminary breakdown sources.

    Parameters
    ---------
    lma_t_fl : float
        Time of sources in seconds after 00 UTC.
    lma_z_fl : float 
        Altitude of sources above MSL, in km.
    min_t : float
        Time of the first source of a flash in seconds after 00 UTC.
    max_pb_dur : int
        Maximum duration of a preliminary breakdown, in ms.

    Returns
    -------
    ch_hgt_thresh, pb_vert_speed, mse: 
        Linear regression intercept (charge height threshold), in km
    pb_vert_speed: 
        Linear regression coefficient (preliminary breakdown vertical speed), in km/ms
    mse: 
        Regression mean squared error
    """
    
    # PB sources:
    whpb = np.where((1000.*(lma_t_fl-min_t) <= max_pb_dur ) & (lma_z_fl <= 20))
    x = lma_t_fl[whpb] 
    x = 1000.*(x-min_t)
    y = lma_z_fl[whpb]
    x=x.transpose()
    x = x.reshape(x.shape[0], 1)
    x=x.tolist()
    
    # Linear regression:
    model = LinearRegression()
    
    # Minimum number of PB sources and minimum PB duration conditions are applied here:
    if y.size >=min_pb_sou and np.max(x)>min_pb_dur:
        model.fit(x, y)
        y_pred = model.predict(x)       
        # Mean squared error:
        mse = mean_squared_error(y, y_pred)
    else:
        model.coef_ = 0
        model.intercept_ = 0
        mse = 2.0
        
    pb_vert_speed = model.coef_
    ch_hgt_thresh = model.intercept_
    
    return ch_hgt_thresh, pb_vert_speed, mse 

def write_output(filename_output, pos_time, pos_zmin, pos_zwid, pos_flax, pos_flay, neg_time, neg_zmin, neg_zwid, neg_flax, neg_flay):
     
    """
    Write output to a .nc file.

    Parameters
    ---------
    filename_output : str
        File name to write output.
    pos_time : float 
        Positive charge layer time, in seconds after 00 UTC.
    pos_zmin : float
        Positive charge layer bottom height, in km above MSL.
    pos_zwid : float
        Positive charge layer width, in km.
    pos_flax : float
        Positive charge layer west-east distance to network center, in km.
    pos_flay : float
        Positive charge layer south-north distance to network center, in km.
    neg_time : float
        Negative charge layer time, in seconds after 00 UTC.
    neg_zmin : float
        Negative charge layer bottom height, in km above MSL.
    neg_zwid : float
        Negative charge layer width, in km.
    neg_flax : float
        Negative charge layer west-east distance to network center, in km.
    neg_flay : float
        Negative charge layer south-north distance to network center, in km.

    """
    
    pos_time = np.array(pos_time, np.dtype(float))
    pos_zmin = np.array(pos_zmin, np.dtype(float))
    pos_zwid = np.array(pos_zwid, np.dtype(float))
    pos_flax = np.array(pos_flax, np.dtype(float))
    pos_flay = np.array(pos_flay, np.dtype(float))
    neg_time = np.array(neg_time, np.dtype(float))
    neg_zmin = np.array(neg_zmin, np.dtype(float))
    neg_zwid = np.array(neg_zwid, np.dtype(float))
    neg_flax = np.array(neg_flax, np.dtype(float))
    neg_flay = np.array(neg_flay, np.dtype(float))

    Re = 6356
    pos_lat = ((180*pos_flay)/(np.pi*Re))+netw_center[0]
    pos_lon = ((360*pos_flax)/(2*np.pi*Re*np.cos(np.pi*pos_lat/180)))+netw_center[1]
    neg_lat = ((180*neg_flay)/(np.pi*Re))+netw_center[0]
    neg_lon = ((360*neg_flax)/(2*np.pi*Re*np.cos(np.pi*neg_lat/180)))+netw_center[1]

    e = open(filename_output, 'a')
    e.write("# Generated by ChargePol\n")
    e.write("#charge (pos/neg), time (UT sec), bottom height of layer (km), depth of layer (km), x distance from LMA center (km), y distance from LMA center (km), longitude, latitude\n")
    e.write("charge,time,zmin,zwidth,x,y,lon,lat\n")
    for f in np.arange(0,len(pos_time)):
        e.write("pos,%f,%f,%f,%f,%f,%f,%f\n" % (pos_time[f], pos_zmin[f], pos_zwid[f], pos_flax[f], pos_flay[f], pos_lon[f], pos_lat[f]))

    for g in np.arange(0,len(neg_time)):
        e.write("neg,%f,%f,%f,%f,%f,%f,%f\n" % (neg_time[g], neg_zmin[g], neg_zwid[g], neg_flax[g], neg_flay[g], neg_lon[g], neg_lat[g]))
        
    e.close()

# All .h5 files in directory:
filenames = glob.glob(direct+'*.flash.h5') 
filenames = sorted(filenames)

# Define variables:
pos_zmin = []
pos_zwid = []
pos_time = []
pos_flax = []
pos_flay = []
neg_zmin = []
neg_zwid = []
neg_time = []
neg_flax = []
neg_flay = []

# Loop on each LMA level 2 .h5 file
for i in range(0,len(filenames)): 

    file = filenames[i]
       
    # Read LMA data:    
    lma_x, lma_y, lma_z, lma_t, lma_flid, flid, flx, fly = read_lma(file, netw_center, max_range, nsou)
    
    # Loop for each flash:    
    for j in range(0,flid.size):
    
        # LMA flash ID:
        flashid = flid[j] 
    
        # Sources associated to a flash ID:
        ind = np.where(lma_flid==flashid)
        lma_x_fl = lma_x[ind]
        lma_y_fl = lma_y[ind]
        lma_z_fl = lma_z[ind]
        lma_t_fl = lma_t[ind]
        lma_flid_fl = lma_flid[ind]
        
        # Time of first source of a flash:
        min_t = np.min(lma_t_fl)

        # Calculate linear regression on PB sources:
        ch_hgt_thresh, pb_vert_speed, mse = regression_pb(lma_t_fl, lma_z_fl, min_t, max_pb_dur)
                
        # Non-PB sources sources (after 10 ms):
        whch = np.where((1000.*(lma_t_fl-min_t) > max_pb_dur ) & (lma_z_fl <= 20))
        zch = lma_z_fl[whch]
        tch = lma_t_fl[whch]
                
        # If flash passes vertical speed and MSE conditions:
        if np.abs(pb_vert_speed) > min_ver_speed and mse < max_mse:
        
            # Non-PB sources above and below charge height threshold (CHT):
            wh_uplay = np.where(zch >= ch_hgt_thresh)
            wh_lwlay = np.where(zch < ch_hgt_thresh)
                
            # Upward PB:
            if np.sign(pb_vert_speed) > 0:
                
                # Positive layer above CHT, negative below CHT
                pos_sour = zch[wh_uplay]
                neg_sour = zch[wh_lwlay]
                pos_t = tch[wh_uplay]
                neg_t = tch[wh_lwlay]
                
                # If candidate sources for positive layer are found:                
                if pos_sour.size > 0:
              
                    # Get sources that define positive charge layer, percentile interval on altitudes
                    pos_qua = np.quantile(pos_sour, [perc_interv[0], perc_interv[1]]) 
                    whqua = np.where((pos_sour > pos_qua[0]) & (pos_sour < pos_qua[1] ))
                    pos_sour = pos_sour[whqua] 
                    pos_t = pos_t[whqua] 
                    if pos_sour.size > 0:
                        pos_zmin.append([np.min(pos_sour)])
                        pos_zwid.append([np.max(pos_sour)-np.min(pos_sour)])
                        pos_time.append([np.min(pos_t)])
                        pos_flax.append([flx[j]])
                        pos_flay.append([fly[j]])
                    
                # If candidate sources for negative layer are found: 
                if neg_sour.size > 0:                
                              
                    # Get sources that define negative charge layer, percentile interval on altitudes:
                    neg_qua = np.quantile(neg_sour, [perc_interv[0], perc_interv[1]]) 
                    wnqua = np.where((neg_sour > neg_qua[0]) & (neg_sour < neg_qua[1] ))
                    neg_sour = neg_sour[wnqua] 
                    neg_t = neg_t[wnqua] 
                    if neg_sour.size > 0:
                        neg_zmin.append([np.min(neg_sour)])
                        neg_zwid.append([np.max(neg_sour)-np.min(neg_sour)])
                        neg_time.append([np.min(neg_t)])
                        neg_flax.append([flx[j]])
                        neg_flay.append([fly[j]])
                 
            # Downward PB:
            else:
            
                # Negative layer above CHT, positive below CHT:
                pos_sour = zch[wh_lwlay]
                neg_sour = zch[wh_uplay]
                pos_t = tch[wh_lwlay]
                neg_t = tch[wh_uplay]
                
                # If candidate sources for positive layer are found:  
                if pos_sour.size > 0:
              
                    # Get sources that define positive charge layer, percentile interval on altitudes:
                    pos_qua = np.quantile(pos_sour, [perc_interv[0], perc_interv[1]]) 
                    whqua = np.where((pos_sour > pos_qua[0]) & (pos_sour < pos_qua[1] ))
                    pos_sour = pos_sour[whqua] 
                    pos_t = pos_t[whqua] 
                    if pos_sour.size > 0:
                        pos_zmin.append([np.min(pos_sour)])
                        pos_zwid.append([np.max(pos_sour)-np.min(pos_sour)])
                        pos_time.append([np.min(pos_t)])
                        pos_flax.append([flx[j]])
                        pos_flay.append([fly[j]])
                
                # If candidate sources for negative layer are found:                 
                if neg_sour.size > 0:
                                
                    # Get sources that define negative charge layer, percentile interval on altitudes:
                    neg_qua = np.quantile(neg_sour, [perc_interv[0], perc_interv[1]]) 
                    wnqua = np.where((neg_sour > neg_qua[0]) & (neg_sour < neg_qua[1] ))
                    neg_sour = neg_sour[wnqua] 
                    neg_t = neg_t[wnqua] 
                    if neg_sour.size > 0:
                        neg_zmin.append([np.min(neg_sour)])
                        neg_zwid.append([np.max(neg_sour)-np.min(neg_sour)])
                        neg_time.append([np.min(neg_t)])
                        neg_flax.append([flx[j]])
                        neg_flay.append([fly[j]])
                    
# Save charge layers output:
write_out = write_output(filename_output, pos_time, pos_zmin, pos_zwid, pos_flax, pos_flay, neg_time, neg_zmin, neg_zwid, neg_flax, neg_flay  )

print('Done')
