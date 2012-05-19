# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 16:56:37 2012

@author: VHOEYS

python adaptation of the Morris Sampling to calculate Elementary effects

http://sensitivity-analysis.jrc.ec.europa.eu/software/index.htm
"""

import numpy as np

def  Sampling_Function_2(p, k, r, UB, LB, GroupMat=np.array([])):
    '''
    Python version of the Morris sampling function:
        
    %[Outmatrix, OutFact] = Sampling_Function_2(p, k, r, UB, LB, GroupMat)
    %   Inputs: k (1,1)                      := number of factors examined
    %                                           In case the groups are chosen the number of factors is stores in NumFact and
    %                                           sizea becomes the number of created groups. 
    %           NumFact (1,1)                := number of factors examined in the case when groups are chosen
    %	    	 r (1,1)                      := sample size  
    %           p (1,1)                      := number of intervals considered in [0, 1]
    %           UB(sizea,1)                  := Upper Bound for each factor in list or array
    %           LB(sizea,1)                  := Lower Bound for each factor in list or array
    %           GroupNumber(1,1)             := Number of groups (eventually 0)
    %           GroupMat(NumFact,GroupNumber):= Array which describes the chosen groups. Each column represents a group and its elements 
    %                                           are set to 1 in correspondence of the factors that belong to the fixed group. All
    %                                           the other elements are zero.
    %   Local Variables:  
    %	    	 sizeb (1,1)         := sizea+1
    %           sizec (1,1)         := 1
    %           randmult (sizea,1)  := vector of random +1 and -1  
    %           perm_e(1,sizea)     := vector of sizea random permutated indeces    
    %           fact(sizea)         := vector containing the factor varied within each traj
    % 	      DDo(sizea,sizea)    := D*       in Morris, 1991   
    %	      A(sizeb,sizea)      := Jk+1,k   in Morris, 1991
    %	      B(sizeb,sizea)      := B        in Morris, 1991
    %	      Po(sizea,sizea)     := P*       in Morris, 1991
    %           Bo(sizeb,sizea)     := B*       in Morris, 1991
    %	      Ao(sizeb,sizec)     := Jk+1,1   in Morris, 1991
    %	      xo(sizec,sizea)     := x*       in Morris, 1991 (starting point for the trajectory)
    %           In(sizeb,sizea)     := for each loop orientation matrix. It corresponds to a trajectory
    %                                  of k step in the parameter space and it provides a single elementary
    %                                  effect per factor 
    %           MyInt()
    %           Fact(sizea,1)       := for each loop vector indicating which factor or group of factors has been changed 
    %                                  in each step of the trajectory
    %           AuxMat(sizeb,sizea) := Delta*0.5*((2*B - A) * DD0 + A) in Morris, 1991. The AuxMat is used as in Morris design
    %                                  for single factor analysis, while it constitutes an intermediate step for the group analysis.
    %
    %	Output: Outmatrix(sizeb*r, sizea) := for the entire sample size computed In(i,j) matrices
    %           OutFact(sizea*r,1)        := for the entire sample size computed Fact(i,1) vectors
    %           
    %   Note: B0 is constructed as in Morris design when groups are not considered. When groups are considered the routine
    %         follows the following steps:
    %           1- Creation of P0 and DD0 matrices defined in Morris for the groups. This means that the dimensions of these
    %              2 matrices are (GroupNumber,GroupNumber).
    %           2- Creation of AuxMat matrix with (GroupNumber+1,GroupNumber) elements.
    %           3- Definition of GroupB0 starting from AuxMat, GroupMat and P0.
    %           4- The final B0 for groups is obtained as [ones(sizeb,1)*x0' + GroupB0]. The P0 permutation is present in GroupB0
    %              and it's not necessary to permute the matrix (ones(sizeb,1)*x0') because it's already randomly created. 
    %   Reference:
    %   A. Saltelli, K. Chan, E.M. Scott, "Sensitivity Analysis" on page 68 ss
    %
    %   F. Campolongo, J. Cariboni, JRC - IPSC Ispra, Varese, IT
    %   Last Update: 15 November 2005 by J.Cariboni
    '''
    
    # Parameters and initialisation of the output matrix
    sizea = k
    Delta = p/(2.*(p-1.))
    NumFact = sizea  
    if GroupMat.shape[0]==GroupMat.size:
        Groupnumber=0
    else:
        Groupnumber = GroupMat.shape[1]    #size(GroupMat,2)
        sizea = GroupMat.shape[1]
    
    sizeb = sizea + 1
    #    sizec = 1
    
    Outmatrix = np.zeros(((sizea+1)*r,NumFact))  
    OutFact = np.zeros(((sizea+1)*r,1))    
    # For each i generate a trajectory  
    for i in range(r):
        Fact=np.zeros(sizea+1)
        # Construct DD0  
        DD0 = np.matrix(np.diagflat(np.sign(np.random.random(k)*2-1)))
        
        # Construct B (lower triangular)
        B = np.matrix(np.tri((sizeb), sizea,k=-1, dtype=int))

        # Construct A0, A
        A0 = np.ones((sizeb,1))
        A = np.ones((sizeb,NumFact))
        
        # Construct the permutation matrix P0. In each column of P0 one randomly chosen element equals 1
        # while all the others equal zero. 
        # P0 tells the order in which order factors are changed in each   
        # Note that P0 is then used reading it by rows.         
        I = np.matrix(np.eye(sizea))
        P0 = I[:,np.random.permutation(sizea)]
        
        # When groups are present the random permutation is done only on B. The effect is the same since 
        # the added part (A0*x0') is completely random.         
        if Groupnumber <> 0:
            B = B * (np.matrix(GroupMat)*P0.transpose()).transpose()
            
        # Compute AuxMat both for single factors and groups analysis. For Single factors analysis
        # AuxMat is added to (A0*X0) and then permutated through P0. When groups are active AuxMat is
        # used to build GroupB0. AuxMat is created considering DD0. If the element on DD0 diagonal
        # is 1 then AuxMat will start with zero and add Delta. If the element on DD0 diagonal is -1 
        # then DD0 will start Delta and goes to zero.
        AuxMat = Delta* 0.5 *((2*B - A) * DD0 + A)  
        
        #----------------------------------------------------------------------
        # a --> Define the random vector x0 for the factors. Note that x0 takes value in the hypercube
        # [0,...,1-Delta]*[0,...,1-Delta]*[0,...,1-Delta]*[0,...,1-Delta]         
        xset=np.arange(0.0,1.0-Delta,1.0/(p-1))
        x0 = np.matrix(xset.take(list(np.ceil(np.random.random(k)*np.floor(p/2))-1)))  #.transpose()
        
        #----------------------------------------------------------------------
        # b --> Compute the matrix B*, here indicated as B0. Each row in B0 is a
        # trajectory for Morris Calculations. The dimension  of B0 is (Numfactors+1,Numfactors)    
        if Groupnumber <> 0: 
            B0 = (A0*x0 + AuxMat)
        else:
            B0 = (A0*x0 + AuxMat)*P0
        
        #----------------------------------------------------------------------
        # c --> Compute values in the original intervals
        # B0 has values x(i,j) in [0, 1/(p -1), 2/(p -1), ... , 1].
        # To obtain values in the original intervals [LB, UB] we compute
        # LB(j) + x(i,j)*(UB(j)-LB(j))
        In=np.tile(LB, (sizeb,1)) + np.array(B0)*np.tile((UB-LB), (sizeb,1)) #array!! ???? 
        
        # Create the Factor vector. Each component of this vector indicate which factor or group of factor
        # has been changed in each step of the trajectory.        
        for j in range(sizea):
            Fact[j] = np.where(P0[j,:])[1]
        Fact[sizea] = int(-1)  #Enkel om vorm logisch te houden. of Fact kleiner maken      
        
        #append the create traject to the others      
        Outmatrix[i*(sizea+1):i*(sizea+1)+(sizea+1),:]=np.array(In)
        OutFact[i*(sizea+1):i*(sizea+1)+(sizea+1)]=np.array(Fact).reshape((sizea+1,1))
#        print Outmatrix
#        print OutFact
    
    return Outmatrix, OutFact
