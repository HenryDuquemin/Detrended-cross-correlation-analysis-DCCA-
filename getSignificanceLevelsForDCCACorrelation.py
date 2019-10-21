# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 15:05:44 2019

@author: duqueh
"""

import time

import numpy as np
import matplotlib.pyplot as plt
from numpy.matlib import repmat
import pandas as pd
from scipy import stats  
from random import gauss


# Return sliding windows
def sliding_window(xx,k):
    # Function to generate boxes given dataset(xx) and box size (k)
    import numpy as np
    # generate indexes. O(1) way of doing it :)
    idx = np.arange(k)[None, :]+np.arange(len(xx)-k+1)[:, None]
    return xx[idx],idx


def getSignificanceLevelsForDCCACorrelation(numberOfRandomSeries, lengthOfSeries, minimumTimeScale, maximumTimeScale):
    
    ListOfDCCAList = []
    for x in range(0, numberOfRandomSeries):
        print(x)
    
        S1 = np.random.normal(loc=0, scale = 1, size = lengthOfSeries)
        print(len(S1))
        S2 = np.random.normal(loc=0, scale = 1, size = lengthOfSeries)
        
    
        x1 = S1
        x2 = S2
    
        # Plot
        
        cdata = np.array([x1,x2]).T
    
        # Define
        nsamples,nvars = cdata.shape
    
        # Cummulative sum after removing mean
        cdata = cdata-cdata.mean(axis=0)
        xx = np.cumsum(cdata,axis=0)
    
        kList = []
        dccaList = []
    
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
    
            #print(dcca)
            #print(dcca[0,1])
            #print(k)
        
            kList.append(k)
            dccaList.append(dcca[0,1])
    
    
        #print(kList)
        #print(dccaList)
        
        ListOfDCCAList.append(dccaList)
        
    
    print(len(ListOfDCCAList))
    
    #Initialize final output data frame
    FinalOutput = pd.DataFrame(columns = ["k", "mean", "stdDev", "NinetyPercentConfidenceIntervalLower", "NinetyPercentConfidenceIntervalUpper", "NinetyFivePercentConfidenceIntervalLower", "NinetyFivePercentConfidenceIntervalUpper", "NinetyNinePercentConfidenceIntervalLower", "NinetyNinePercentConfidenceIntervalUpper"])
    
    
    
    for IndexToGet in range(0, len(ListOfDCCAList[0])):
        AllK5Values = []
        #iterate through each individual list of DCCA components
        for y in range(0, len(ListOfDCCAList)): 
            AllK5Values.append(ListOfDCCAList[y][IndexToGet])
    
    
        mu, sigma = stats.norm.fit(AllK5Values) # get mean and standard deviation  
        
        print("The value of the mean is " + str(mu))
        print("The standard deviation is " + str(sigma))
        kValue = IndexToGet + minimumTimeScale #Index to get starts from 0 so need to add min time scale to get correct k
        
        lower, upper = -1, 1    
    
        NinetyPercentConfidenceInterval = stats.truncnorm.interval(0.90,
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        
        NinetyFivePercentConfidenceInterval = stats.truncnorm.interval(0.95,
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        
        NinetyNinePercentConfidenceInterval = stats.truncnorm.interval(0.99,
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        
        tempDF = pd.DataFrame([np.array([kValue, mu, sigma, NinetyPercentConfidenceInterval[0], NinetyPercentConfidenceInterval[1], NinetyFivePercentConfidenceInterval[0], NinetyFivePercentConfidenceInterval[1], NinetyNinePercentConfidenceInterval[0], NinetyNinePercentConfidenceInterval[1]])],
                                  columns = ["k", "mean", "stdDev", "NinetyPercentConfidenceIntervalLower", "NinetyPercentConfidenceIntervalUpper", "NinetyFivePercentConfidenceIntervalLower", "NinetyFivePercentConfidenceIntervalUpper", "NinetyNinePercentConfidenceIntervalLower", "NinetyNinePercentConfidenceIntervalUpper"])
        FinalOutput = FinalOutput.append(tempDF, ignore_index = True)
        
    
    
    print(FinalOutput)
    return FinalOutput

getSignificanceLevelsForDCCACorrelation(100, 30, 6, 20)