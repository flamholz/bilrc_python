#!/usr/bin/python

"""bilrc_util.py: utilities for parsing data from the Caltech BI Laser Resource Center.

Translated from MATLAB scripts by Jay Winkler with help/input from Jay. 
"""

__author__      = "Avi I. Flamholz"

import numpy as np
import struct
import pandas as pd

class NS1Data:
    """Container for data read from a binary file from NS1."""
    
    def __init__(self, T, Y, EXP, TB, PTPS, VOLTS,
                 LAM_EX, LAM_OB, SHOTS, DIM, PTS,
                 exoff_av, bloff_av,
                 comment=None, fname=None):
        """Initialize NS1Data object representing a single output file from
        nanosecond1 at the Caltech BI Laser Resource Center.
        
        Args:
            T: timepoints.
            Y: raw timeseries data.
            EXP: Experiment Type. EXP=0, luminescence; EXP=2, transient absorption
            TB: Time base - not used in parsing
            PTPS: number of timepoints per second. 
            VOLTS: Full scale voltage digitizer input range = +/- VOLTS
            LAM_EX: excitation wavelength.
            LAM_OB: observation wavelength.
            SHOTS: number of shots accumulated in the file. 
            DIM: ?
            PTS: total number of points.
            exoff_av: average EXOFF, see notes below. 
            bloff_av: average BLOFF, see notes below. 
            comment: comment in the file about the run.
            fname: filename from which the data came.
        """
        self.T = T
        self.Y = Y
        self.Ynorm = Y/Y.max()
        self.EXP = EXP
        self.TB = TB
        self.PTPS = PTPS
        self.VOLTS = VOLTS
        self.LAM_EX = LAM_EX
        self.LAM_OB = LAM_OB
        self.SHOTS = SHOTS
        self.DIM = DIM
        self.PTS = PTS
        self.bloff_av = bloff_av
        self.exoff_av = exoff_av
        self.comment = comment
        self.fname = fname
       
    def log_time(self, ppd=100):
        """Return a copy of self with logarithmically-spaced timepoints.
        
        Args:
            ppd: points per decile, see logtime_r2().
        """
        # Default resampling/averaging of the data to have logarithmically spaced timepoints
        # with 100 points per decade. This provides additional averaging and is preferred for 
        # fitting multi-exponential processes with very different timescales since the raw
        # data (evenly-spaced timepoints) intrinsically overweights long-timescales.
        resampled_t, resampled_y, _wts = logtime_r2(self.T, self.Y, 100)
        
        self_params = dict(vars(self))
        self_params.pop('T')
        self_params.pop('Y')
        self_params.pop('Ynorm')
        return NS1Data(resampled_t, resampled_y, **self_params)
            
    def __str__(self):
        fmt = '<NS1Data excitation={0}nm emission={1}nm shots={2} points={3} points_per_second={4} fname="{5}"">'
        return fmt.format(self.LAM_EX, self.LAM_OB,
                          self.SHOTS, self.PTS,
                          self.PTPS, self.fname)
    
    def to_series(self):
        """Returns Y=X as a Pandas series"""
        idx = pd.Index(data=self.T, name='time_s')
        return pd.Series(data=self.Y, index=idx, name='intensity')
        

def read_ns1_data(fname):
    """
    Read binary timecourse data from ns1 into memory.
    
    Notes from Jay Winkler about parameters used here: 
    EXP: Experiment Type. EXP=0, luminescence; EXP=2, transient absorption
    TB:  Time base - parameter is not used in parsing
    PTPS: number of time points per second.
    VOLTS: Full scale voltage digitizer input range = +/- VOLTS
    LAM_EX: excitation wavelength.
    LAM_OB: observation wavelength.
    SHOTS: total number of laser shots.
    DIM: Unused in the code.
    PTS: Total number of data points.
    EXOFF: In transient absorption, the PMT amplifier applies an offset voltage
    to the signal, driving it to 0 V before the laser fires. That offset voltage
    for excitation cycles is stored in EXOFF. To get true voltage, this value
    is added back to the recorded data.
    BLOFF: In transient absorption, the PMT amplifier applies an offset voltage
    to the signal, driving it to 0 V before the laser fires. That offset voltage
    for blank cycles is stored in BLOFF. To get true voltage, this value is added
    back to the recorded data.
        
    Args:
        fname: the filename to read from.
       
    Returns:
        A 2-tuple (T, Y) where T is the timepoints and Y are the values recorded.
    """
    with open(fname, 'rb') as binfile:
        # '>' denotes big-endian byte format, which is what Jay uses
        header_tuple = struct.unpack('>iiffiiiii', binfile.read(9*4))
        EXP, TB, PTPS, VOLTS, LAM_EX, LAM_OB, SHOTS, DIM, PTS = header_tuple
        
        #for n,v in zip('EXP,TB,PTPS,VOLTS,LAM_EX,LAM_OB,SHOTS,DIM,PTS'.split(','), header_tuple):
        #    print(n, '=', v)
                
        DATA = np.zeros((2, PTS))
        pts_fmt = '>' + 'i'*PTS
        for j in range(2):
            DATA[j, :] = struct.unpack(pts_fmt, binfile.read(PTS*4))
        
        EXOFF, BLOFF = struct.unpack('>ii', binfile.read(2*4))
        
        # Rest of the file is ASCII comment - 1 byte per char
        # TODO: not sure this is right... double check on a file with a real comment.
        bin_comment = binfile.read()
        comment_len = len(bin_comment)
        comment_format = '>{0:d}s'.format(comment_len)
        comment = struct.unpack(comment_format, bin_comment)[0]
    
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
    
    # Subtract off the constant offset - i.e. blank the data
    # NB: This assumes that the first 100 points are pre-shot data.
    Y -= Y[:100].mean()
    
    return NS1Data(T, Y, *header_tuple, exoff_av, bloff_av, comment, fname)


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
