# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:19:06 2019

@author: duqueh
"""

"""DCCA function modified from https://gist.github.com/jaimeide/a9cba18192ee904307298bd110c28b14"""

import numpy as np
from numpy.matlib import repmat
import pandas as pd



def sliding_window(xx,k):
    # Function to generate boxes given dataset(xx) and box size (k)
    import numpy as np
    # generate indexes. O(1) way of doing it :)
    idx = np.arange(k)[None, :]+np.arange(len(xx)-k+1)[:, None]
    return xx[idx],idx


def GetDCCACoefficients (series1, series2, minimumTimeScale, maximumTimeScale):

    # Plot
    cdata = np.array([series1,series2]).T
#    plt.plot(cdata)
#    plt.title('Sample time series')
#    plt.legend(['$x_1$','$x_2$'])
#    plt.show()
#    plt.clf()


    # Define
    nsamples,nvars = cdata.shape

    # Cummulative sum after removing mean
    cdata = cdata-cdata.mean(axis=0)
    xx = np.cumsum(cdata,axis=0)

    kList = []
    DCCAList = []

    for k in range(minimumTimeScale, maximumTimeScale):
        F2_dfa_x = np.zeros(nvars)
        allxdif = []
        for ivar in range(nvars): # do for all vars
            xx_swin , idx = sliding_window(xx[:,ivar],k)
            nwin = xx_swin.shape[0]
            b1, b0 = np.polyfit(np.arange(k),xx_swin.T,deg=1) # linear fit
            #x_hat = [[b1[i]*j+b0[i] for j in range(k)] for i in range(nwin)] # slow version
            x_hatx = repmat(b1,k,1).T*repmat(range(k),nwin,1) + repmat(b0,k,1).T
            # Store differences to the linear fit
            xdif = xx_swin-x_hatx
            allxdif.append(xdif)
            # Eq.4
            F2_dfa_x[ivar] = (xdif**2).mean()


        # Get the DCCA matrix
        dcca = np.zeros([nvars,nvars])
        for i in range(nvars): # do for all vars
            for j in range(nvars): # do for all vars
                # Eq.5 and 6
                F2_dcca = (allxdif[i]*allxdif[j]).mean()
                # Eq.1: DCCA
                dcca[i,j] = F2_dcca / np.sqrt(F2_dfa_x[i] * F2_dfa_x[j])

    
        kList.append(k)
        print(kList)
        DCCAList.append(dcca[0,1])
    
    print(dict(zip(kList, DCCAList)))
    return dict(zip(kList, DCCAList))

