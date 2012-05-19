# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 18:07:03 2012

@author: VHOEYS

python adaptation of the Morris Sampling to optimize the sampling and Calculate Elemenetary effects

http://sensitivity-analysis.jrc.ec.europa.eu/software/index.htm
"""

from Morris_sampling import *
import matplotlib.pyplot as plt

def Optimized_Groups(NumFact,LB,UB,N=500,p=4,r=10,GroupMat=np.array([]),Diagnostic=0):
    '''
    % [OptMatrix, OptOutVec] = Optimized_Groups(NumFact,N,p,r,GroupMat,Diagnostic)
    %
    % Optimization in the choice of trajectories for the Morris experiment
    % clear all
    
    % Inputs
    % N:= [1,1]         Total number of trajectories
    % p:= [1,1]         Number of levels
    % r:= [1,1]         Final number of optimal trjectories
    % NumFact:= [1,1]   Number of factors
    % LB:= [NumFact,1]  Lower bound of the uniform distribution for each factor
    % UB:= [NumFact,1]  Upper bound of the uniform distribution for each factor
    % GroupMat:=[NumFact,NumGroups] Matrix describing the groups. Each column represents a group and its elements 
    %                               are set to 1 in correspondence of the factors that belong to the fixed group. All
    %                               the other elements are zero.
    % Diagnostic:= [1,1]            Boolean 1=plot the histograms and compute the
    %                               efficiency of the samplign or not, 0
    %                             otherwise
    '''
    
    LBt = np.zeros(NumFact)
    UBt = np.ones(NumFact)
    
    OutMatrix, OutFact = Sampling_Function_2(p, NumFact, N, UBt, LBt, GroupMat)     #Version with Groups
    
    try:
        Groupnumber = GroupMat.shape[1]
    except:
        Groupnumber = 0
        
    
    if Groupnumber<>0:
        sizeb = Groupnumber +1
    else:
        sizeb = NumFact +1
    
    Dist = np.zeros((N,N))
    Diff_Traj = np.arange(0.0,N,1.0)
    
    # Compute the distance between all pair of trajectories (sum of the distances between points)
    # The distance matrix is a matrix N*N
    # The distance is defined as the sum of the distances between all pairs of points
    # if the two trajectories differ, 0 otherwise
    for j in range(N):   #combine all trajectories: eg N=3: 0&1; 0&2; 1&2 (is not dependent from sequence)
        for z in range(j+1,N):
            MyDist = np.zeros((sizeb,sizeb))
            for i in range(sizeb):
                for k in range(sizeb):
                    MyDist[i,k] = (np.sum((OutMatrix[sizeb*(j)+i,:]-OutMatrix[sizeb*(z)+k,:])**2))**0.5 #indices aan te passen
            if np.where(MyDist==0)[0].size == sizeb:
                # Same trajectory. If the number of zeros in Dist matrix is equal to 
                # (NumFact+1) then the trajectory is a replica. In fact (NumFact+1) is the maximum numebr of 
                # points that two trajectories can have in common
                Dist[j,z] = 0.
                Dist[z,j] = 0.
                
                # Memorise the replicated trajectory
                Diff_Traj[z] = -1.  #the z value identifies the duplicate 
            else:
                # Define the distance between two trajectories as 
                # the minimum distance among their points  
                Dist[j,z] = np.sum(MyDist)
                Dist[z,j] = np.sum(MyDist)
    
    #prepare array with excluded duplicates (alternative would be deleting rows)
    dupli=np.where(Diff_Traj==-1)[0].size
    New_OutMatrix = np.zeros(((sizeb)*(N-dupli),NumFact)) 
    New_OutFact = np.zeros(((sizeb)*(N-dupli),1))
                         
    # Eliminate replicated trajectories in the sampled matrix
    ID=0
    for i in range(N):
        if Diff_Traj[i]<>-1.:
            New_OutMatrix[ID*sizeb:ID*sizeb+sizeb,:] = OutMatrix[i*(sizeb) : i*(sizeb) + sizeb,:]
            New_OutFact[ID*sizeb:ID*sizeb+sizeb,:] = OutFact[i*(sizeb) : i*(sizeb) + sizeb,:]
            ID+=1
            
    # Select in the distance matrix only the rows and columns of different trajectories    
    Dist_Diff = Dist[np.where(Diff_Traj<>-1)[0],:] #moet 2D matrix zijn... wis rijen ipv hou bij
    Dist_Diff = Dist_Diff[:,np.where(Diff_Traj<>-1)[0]] #moet 2D matrix zijn... wis rijen ipv hou bij
    #    Dist_Diff = np.delete(Dist_Diff,np.where(Diff_Traj==-1.)[0])
    New_N = np.size(np.where(Diff_Traj<>-1)[0])
    
    # Select the optimal set of trajectories
    Traj_Vec = np.zeros((New_N, r))
    OptDist = np.zeros((New_N, r))
    for m in range(New_N):                  #each row in Traj_Vec
        Traj_Vec[m,0]=m  

        for z in range(1,r):              #elements in columns after first   
            Max_New_Dist_Diff = 0.0
            
            for j in range(New_N):
                # Check that trajectory j is not already in
                Is_done = False
                for h in range(z):
                    if j == Traj_Vec[m,h]:
                        Is_done=True
                
                if Is_done==False:
                    New_Dist_Diff = 0.0
                    
                    #compute distance
                    for k in range(z):                       
                        New_Dist_Diff = New_Dist_Diff + (Dist_Diff[Traj_Vec[m, k],j])**2                         
                    
                    # Check if the distance is greater than the old one
                    if New_Dist_Diff**0.5 > Max_New_Dist_Diff:
                        Max_New_Dist_Diff = New_Dist_Diff**0.5
                        Pippo = j
                        
            # Set the new trajectory
            Traj_Vec[m,z] = Pippo
            OptDist[m,z] = Max_New_Dist_Diff
           
    # Construct optimal matrix
    SumOptDist = np.sum(OptDist, axis=1)
    # Find the maximum distance
    Pluto = np.where(SumOptDist == np.max(SumOptDist))[0]
    Opt_Traj_Vec = Traj_Vec[Pluto[0],:]
    
    OptMatrix = np.zeros(((sizeb)*r,NumFact))  
    OptOutVec = np.zeros(((sizeb)*r,1))    
    
    for k in range(r):       
        OptMatrix[k*(sizeb):k*(sizeb)+(sizeb),:]= New_OutMatrix[(sizeb)*(Opt_Traj_Vec[k]):(sizeb)*(Opt_Traj_Vec[k]) + sizeb,:] 
        OptOutVec[k*(sizeb):k*(sizeb)+(sizeb)]= New_OutFact[(sizeb)*(Opt_Traj_Vec[k]):(sizeb)*(Opt_Traj_Vec[k])+ sizeb,:]
    
    #----------------------------------------------------------------------
    # Compute values in the original intervals
    # Optmatrix has values x(i,j) in [0, 1/(p -1), 2/(p -1), ... , 1].
    # To obtain values in the original intervals [LB, UB] we compute
    # LB(j) + x(i,j)*(UB(j)-LB(j))
    OptMatrix_b = OptMatrix.copy()
    OptMatrix=np.tile(LB, (sizeb*r,1)) + OptMatrix*np.tile((UB-LB), (sizeb*r,1))    
    
    if Diagnostic==True:
        # Clean the trajectories from repetitions and plot the histograms
        hplot=np.zeros((2*r,NumFact))
        
        for i in range(NumFact):
            for j in range(r):
                # select the first value of the factor
                hplot[j*2,i] = OptMatrix_b[j*sizeb,i]
                
                # search the second value 
                for ii in range(1,sizeb):
                    if OptMatrix_b[j*sizeb+ii,i] <> OptMatrix_b[j*sizeb,i]:
                        kk = 1       
                        hplot[j*2+kk,i] = OptMatrix_b[j*sizeb+ii,i]
        
        fig=plt.figure()
        fig.suptitle('New Strategy')
        DimPlots = np.round(NumFact/2)
        for i in range(NumFact):
            ax=fig.add_subplot(DimPlots,2,i)
            ax.hist(hplot[:,i],p)
        
        # Plot the histogram for the original samplng strategy
        # Select the matrix
        OrigSample = OutMatrix[:r*(sizeb),:]
        print OrigSample
        Orihplot = np.zeros((2*r,NumFact))
        print Orihplot
        
        for i in range(NumFact):
            for j in range(r):
                # select the first value of the factor
                Orihplot[j*2,i] = OrigSample[j*sizeb,i]
                
                # search the second value 
                for ii in range(1,sizeb):
                    if OrigSample[j*sizeb+ii,i] <> OrigSample[j*sizeb,i]:
                        kk = 1       
                        Orihplot[j*2+kk,i] = OrigSample[j*sizeb+ii,i]        
            
        fig=plt.figure()
        fig.suptitle('Old Strategy')
        DimPlots = np.round(NumFact/2)
        for i in range(NumFact):
            ax=fig.add_subplot(DimPlots,2,i)
            ax.hist(Orihplot[:,i],p) 
            #        plt.title('Old Strategy')
        print 'hplotten'
        print hplot
        
        # Measure the quality of the sampling strategy
        levels=np.arange(0.0,1.1,1.0/(p-1))
        NumSPoint=np.zeros((NumFact,p))
        NumSOrigPoint=np.zeros((NumFact,p))
        for i in range(NumFact):
            for j in range(p):
                # For each factor and each level count the number of times the factor is on the level
                #This for the new and original sampling
                NumSPoint[i,j] = np.where(np.abs(hplot[:,i]-np.tile(levels[j], hplot.shape[0]))<1e-5)[0].size
                NumSOrigPoint[i,j] = np.where(np.abs(Orihplot[:,i]-np.tile(levels[j], Orihplot.shape[0]))<1e-5)[0].size
                
        # The optimal sampling has values uniformly distributed across the levels
        OptSampl = 2.*r/p
        QualMeasure = 0.
        QualOriMeasure = 0.
        for i in range(NumFact):
            for j in range(p):
                QualMeasure = QualMeasure + np.abs(NumSPoint[i,j]-OptSampl)
                QualOriMeasure = QualOriMeasure + np.abs(NumSOrigPoint[i,j]-OptSampl)      
        
        QualMeasure = 1. - QualMeasure/(OptSampl*p*NumFact)
        QualOriMeasure = 1. - QualOriMeasure/(OptSampl*p*NumFact)
        
        print 'The quality of the sampling strategy changed from %f with the old strategy to %f for the optimized strategy' %(QualOriMeasure,QualMeasure)
            
    return OptMatrix, OptOutVec
