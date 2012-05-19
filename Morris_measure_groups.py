# -*- coding: utf-8 -*-
"""
Created on Sat Mar 03 15:10:59 2012

@author: VHOEYS
python adaptation to Calculate Elemenetary effects

http://sensitivity-analysis.jrc.ec.europa.eu/software/index.htm
"""

import numpy as np
import matplotlib.pyplot as plt

def Morris_Measure_Groups(NumFact, Sample, OutFact, Output, p=4, Group=[]):
    '''
    % [SAmeas, OutMatrix] = Morris_Measure_Groups(NumFact, Sample, Output, p, Group)
    %
    % Given the Morris sample matrix, the output values and the group matrix compute the Morris measures
    % -------------------------------------------------------------------------
    % INPUTS
    % -------------------------------------------------------------------------
    % Group [NumFactor, NumGroups] := Matrix describing the groups. 
    % Each column represents one group. 
    % The element of each column are zero if the factor is not in the
    % group. Otherwise it is 1.
    
    % Sample := Matrix of the Morris sampled trajectories 
    Factch  :=Matrix with the factor changings as specified in Morris_sampling
    
    % Output := Matrix of the output(s) values in correspondence of each point
    % of each trajectory
    
    % k = Number of factors
    % -------------------------------------------------------------------------
    % OUTPUTS 
    % OutMatrix (NumFactor*NumOutputs, 3)= [Mu*, Mu, StDev]
    % for each output it gives the three measures of each factor
    % -------------------------------------------------------------------------
    '''
        
    try:
        NumGroups = Group.shape[1]
        print '%d Groups are used' %NumGroups
    except:
        NumGroups = 0
        print 'No Groups are used'
    
    Delt = p/(2.*(p-1.))

    if NumGroups<>0:
        sizea=NumGroups
        sizeb=sizea+1
        GroupMat=Group
        GroupMat = GroupMat.transpose()
        print NumGroups
    else:
        sizea = NumFact
        sizeb=sizea+1
        
    r = Sample.shape[0]/(sizea+1)
    
    try:
        NumOutp = Output.shape[1]
    except:
        NumOutp = 1
        Output=Output.reshape((Output.size,1))
        
    
    # For each Output
    if NumGroups == 0:
        OutMatrix=np.zeros((NumOutp*NumFact,3)) #for every output: every factor is a line, columns are mu*,mu and std
    else:
        OutMatrix=np.zeros((NumOutp*NumFact,1)) #for every output: every factor is a line, column is mu*
        
    SAmeas_out=np.zeros((NumOutp*NumFact,r))
    
    for k in range(NumOutp):   
        OutValues=Output[:,k] 
        
        #For each trajectory
        SAmeas=np.zeros((NumFact,r)) #vorm afhankelijk maken van group of niet...
        for i in range(r):
            # For each step j in the trajectory
            # Read the orientation matrix fact for the r-th sampling
            # Read the corresponding output values
            # read the line of changing factors
            
            Single_Sample = Sample[i*(sizeb):i*(sizeb)+(sizeb),:]
            Single_OutValues = OutValues[i*(sizeb):i*(sizeb)+(sizeb)]
            Single_Facts = OutFact[i*(sizeb):i*(sizeb)+(sizeb)] #gives factor in change (or group)
            
            A = (Single_Sample[1:sizeb,:]-Single_Sample[:sizea,:]).transpose()
            Delta=A[np.where(A)] #AAN TE PASSEN?
            
            print A
            print Delta
            print Single_Facts
            
            # For each point of the fixed trajectory compute the values of the Morris function. 
            for j in range(sizea):
                if NumGroups <> 0:  #work with groups
                    Auxfind=A[:,j]
                    Change_factor = np.where(np.abs(Auxfind)>1e-010)[0]
                    for gk in Change_factor:
                        SAmeas[gk,i] = np.abs((Single_OutValues[j] - Single_OutValues[j+1])/Delt)   #nog niet volledig goe
                    
                else:
                    if Delta[j]> 0.0:
                        SAmeas[int(Single_Facts[j]),i] = (Single_OutValues[j+1] - Single_OutValues[j])/Delt                        
                    else:
                        SAmeas[int(Single_Facts[j]),i] = (Single_OutValues[j] - Single_OutValues[j+1])/Delt
        
        
        # Compute Mu AbsMu and StDev
        if np.isnan(SAmeas).any():
            AbsMu=np.zeros(NumFact)
            Stdev=np.zeros(NumFact)
            Mu=np.zeros(NumFact)
            
            for j in range(NumFact):
                SAm=SAmeas[j,:]
                SAm=SAm[~np.isnan(SAm)]
                rr=np.float(SAm.size)
                AbsMu[j] = np.sum(np.abs(SAm))/rr
                if NumGroups == 0:
                    Mu[j] = SAm.mean()
                    Stdev[j] = np.std(SAm, dtype=np.float64,ddof=1) #ddof: /N-1 instead of /N
        else:
            AbsMu = np.sum(np.abs(SAmeas),axis=1)/r
            if NumGroups == 0:
                Mu = SAmeas.mean(axis=1)
                Stdev = np.std(SAmeas, dtype=np.float64, ddof=1,axis=1) #ddof: /N-1 instead of /N
            else:
                Stdev=np.zeros(NumFact)
                Mu=np.zeros(NumFact)               
        
        OutMatrix[k*NumFact:k*NumFact+NumFact,0]=AbsMu
        if NumGroups == 0:
            OutMatrix[k*NumFact:k*NumFact+NumFact,1]=Mu
            OutMatrix[k*NumFact:k*NumFact+NumFact,2]=Stdev
        
        SAmeas_out[k*NumFact:k*NumFact+NumFact,:]=SAmeas    
    
    
    return SAmeas_out, OutMatrix
    

def plotbar(Outmatrix,factornames=[]):
    ind = np.arange(Outmatrix.shape[0])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    width = 0.35    
    ax.bar(ind, Outmatrix[:,0],width)
    
    
    ax.set_ylabel(r'$\mu$*')
    ax.set_xlabel(r'Factors')
    ax.set_xticks(ind+width/2)
    if len(factornames)>0:
        ax.set_xticklabels( factornames )
        
    
    














