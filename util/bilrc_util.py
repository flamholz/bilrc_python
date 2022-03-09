#!/usr/bin/python

import numpy as np
import struct

def read_ns1_data(fname):
    """
    Read binary timecourse data from ns1 into memory.
    
    TODO: update to return an object that contains all the data.
    
    Args:
        fname: the filename to read from.
       
    Returns:
        A 2-tuple (T, Y) where T is the timepoints and Y are the values recorded.
    """
    with open(fname, 'rb') as binfile:
        # '>' denotes big-endian byte format, which is what Jay uses
        header_tuple = struct.unpack('>iiffiiiii', binfile.read(9*4))
        EXP, TB, PTPS, VOLTS, LAM_EX, LAB_OB, SHOTS, DIM, PTS = header_tuple
        #print('EXP, TB, PTPS, VOLTS, LAM_EX, LAB_OB, SHOTS, DIM, PTS')
        #print(header_tuple)
        
        DATA = np.zeros((2, PTS))
        pts_fmt = '>' + 'i'*PTS
        for j in range(2):
            DATA[j, :] = struct.unpack(pts_fmt, binfile.read(PTS*4))
        
        EXOFF, BLOFF = struct.unpack('>ii', binfile.read(2*4))
        
        # TODO: read the comment here. It's ascii til the EOF
    
    DIG = 2
    if TB == 0 or TB == 2:
        DIG = 1
    
    xinc = 1/PTPS
    T = np.linspace(0, xinc*(PTS), num=(PTS))
    TZ = 0.05*T.max()
    T = (T-TZ).copy()
    
    if DIG == 1:
        VDATA = ((DATA - (SHOTS*.127))*VOLTS/128)
    else:
        VDATA = DATA*VOLTS/32768.0 
    
    exoff_av = EXOFF / SHOTS
    bloff_av = BLOFF / SHOTS
    
    if EXP == 0:
        if DIG == 1:
            Y = -(VDATA[0,:]+EXOFF)+(VDATA[1,:]+BLOFF)
        else:
            Y = ((VDATA[0,:]+EXOFF)-(VDATA[1,:]+BLOFF)) / SHOTS
    else:
        if DIG == 1:
            Y = -np.log10((VDATA[0,:] + EXOFF) / (VDATA[1,:] + BLOFF))
        else:
            Y = -np.log10((VDATA[0,:] + EXOFF) / (VDATA[1,:] + BLOFF))
            
    return T, Y


def logtime_r2(t, y, ppd):
    """
    Convert y=f(t) data from linear in time to logarithmic in time.
    
    Args:
        t: is the input time vector, linearly spaced
        y: is the input vector of y values
        ppd: number of points per decade for the output
        
    Returns:
        A 3-tuple (tout, yout, wt) where tout and yout are logarithimically-spaced
        versions of t and y and wt is a vector of weights giving the number of points
        averaged for each yout value.
    """
    zt = len(t)
    zy = len(y)
    assert zt == zy
    
    # Find the index of t = 0 by taking the index where t^2 is minimum.
    indzero = np.argmin(np.power(t,2))
    if t[indzero] < 0:
        indzero += 1
    
    # tmin is minimum nonzero value of t after start.
    tmin = t[indzero]
    tmax = np.max(t)
    if tmin == 0:
        tmin = t[indzero+1]
        
    ltmin = np.log10(tmin)
    ltmax = np.log10(tmax)

    tt = np.arange(ltmin, ltmax, 1/(2*ppd))
    tt = np.power(10, tt)
    ztt = tt.size
    
    # perform resampling from indzero to end, forward in time
    icount, jcount = indzero, 0
    tout, yout, wt = np.zeros(ztt), np.zeros(ztt), np.zeros(ztt)    
    for i in np.arange(1, ztt, 2):
        # accumulate points until we reach the end of the interval
        while icount < zt and t[icount] < tt[i]:
            tout[jcount] = tout[jcount] + t[icount]
            yout[jcount] = yout[jcount] + y[icount]
            wt[jcount] += 1
            icount += 1
    
        # If we accumulated data points, then average by the number of points. 
        if wt[jcount] > 0:
            tout[jcount] = tout[jcount] / wt[jcount];
            yout[jcount] = yout[jcount] / wt[jcount];
            jcount += 1
            
    # Purposely allocated too much space at the start. Trim zeroes from the end. 
    yout = np.trim_zeros(yout, 'b')
    tout = tout[:yout.size]
    wt = wt[:yout.size]

    # If we started at the beginning, then we are done.
    if indzero == 0:
        return (tout, yout, wt)

    # If not, perform resampling from indzero backwards in time. 
    tmp_t, tmp_y = -t[indzero-1::-1], y[indzero-1::-1]
    tmp_zt = len(tmp_t)
    
    icount, jcount = 0, 0
    tmp_tout, tmp_yout, tmp_wt = np.zeros(ztt), np.zeros(ztt), np.zeros(ztt)    
    for i in np.arange(1, ztt, 2):
        while icount < tmp_zt and tmp_t[icount] < tt[i]:
            tmp_tout[jcount] = tmp_tout[jcount] + tmp_t[icount]
            tmp_yout[jcount] = tmp_yout[jcount] + tmp_y[icount]
            tmp_wt[jcount] += 1
            icount += 1
        if tmp_wt[jcount] > 0:
            tmp_tout[jcount] = tmp_tout[jcount] / tmp_wt[jcount];
            tmp_yout[jcount] = tmp_yout[jcount] / tmp_wt[jcount];
            jcount += 1
            
    # Purposely allocated too much space at the start. Trim zeroes from the end. 
    tmp_yout = np.trim_zeros(tmp_yout, 'b')
    tmp_tout = tmp_tout[:tmp_yout.size]
    tmp_wt = tmp_wt[:tmp_yout.size]
    
    # Concat results and return
    return (np.concatenate([-tmp_tout[::-1], tout]), 
            np.concatenate([tmp_yout[::-1], yout]), 
            np.concatenate([tmp_wt[::-1], wt]))
