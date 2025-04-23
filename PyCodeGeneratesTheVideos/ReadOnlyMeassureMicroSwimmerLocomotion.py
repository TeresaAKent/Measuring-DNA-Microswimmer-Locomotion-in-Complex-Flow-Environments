# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 18:18:52 2025

@author: tkent
"""

# Load Packages


from __future__ import print_function
import sys
import cv2
from random import randint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib as mpl
import csv
from PIL import Image
import matplotlib.cm as cm



def AnalyzeRawData(FinalData, FinalDataM, FinalDataMag, FieldData, HeaderRows,ExclusionRadius):
    # Analyze Fiducials Net Motion
    ## It is important all the arrays are the same size
    EndPoint= np.min([np.size(FinalData,0), np.size(FinalDataM,0), np.size(FinalDataMag,0), np.size(FieldData,0)])-10
    
    # Non-Magnetic Fiducials
    # Convert posistion data to relative motion
    Motion = np.array(FinalData[:EndPoint,1:]-FinalData[HeaderRows,1:], dtype= float)
    # Break motion in to magnitudes directions and x and y vectors
    Motion2 = np.sqrt( np.array(Motion[:,0::2]**2 +Motion[:,1::2]**2,dtype = np.int32))
    MotionX = Motion[:,0::2]
    MotionY = Motion[:,1::2]
   
    # Swimmer Motion
    # Convert posistion data to relative motion
    MotionM = np.array(FinalDataM[:EndPoint,1:]-FinalDataM[HeaderRows,1:], dtype = float)
    # Break motion in to magnitudes directions and x and y vectors
    MotionMX = MotionM[:,0::2]
    MotionMY = MotionM[:,1::2]
    Motion2M = np.sqrt( np.array(MotionM[:,0::2]**2 +MotionM[:,1::2]**2,dtype = np.int32))
   
    # Magnetic Fiducials
    # TConvert posistion data to relative motion
    MotionMag = np.array(FinalDataMag[:EndPoint,1:]-FinalDataMag[HeaderRows,1:], dtype = float)
    # Break motion in to magnitudes directions and x and y vectors
    MotionMagX = MotionMag[:,0::2]
    MotionMagY = MotionMag[:,1::2]
    Motion2Mag = np.sqrt( np.array(MotionMag[:,0::2]**2 +MotionMag[:,1::2]**2,dtype = np.int32))
    
    IntraAnal={'time':FinalData[:EndPoint,0],'EndPoint':EndPoint,'Motion2': Motion2, 'MotionX': MotionX, 'MotionY': MotionY,
               'MotionMX':MotionMX, 'MotionMY': MotionMY, 'Motion2M':Motion2M,
               'MotionMagX':MotionMagX, 'MotionMagY':MotionMagY, 'Motion2Mag':Motion2Mag}
    
    #Analyze the Spacial Relationships
    # Find the Distance between Fiducials and Magnetic Swimmers
    DistTot = np.zeros((EndPoint,np.size(Motion2,1),np.size(Motion2M,1)))
    for numSwim in range(np.size(Motion2M,1)):
        DistComp = FinalData[:EndPoint,1:]-np.repeat(FinalDataM[:EndPoint,numSwim*2+1:numSwim*2+3], np.size(Motion2,1), axis=1)
        DistTot[:,:,numSwim] = np.sqrt( np.array(DistComp[:,0::2]**2 +DistComp[:,1::2]**2,dtype = float))
    # If there are multiple microswimmers, we really only care about the closest one
    DistTot = np.min(DistTot,2)
    
    # Find the Distance between Magnetic Fiducials and Magnetic Swimmers
    DistTotMagM = np.zeros((EndPoint,np.size(Motion2Mag,1),np.size(Motion2M,1)))
    for numSwim in range(np.size(Motion2M,1)):
        DistComp = FinalDataMag[:EndPoint,1:]-np.repeat(FinalDataM[:EndPoint,numSwim*2+1:numSwim*2+3], np.size(Motion2Mag,1), axis=1)
        DistTotMagM[:,:,numSwim] = np.sqrt(np.array(DistComp[:,0::2]**2 +DistComp[:,1::2]**2,dtype = float))
    DistTotMagM = np.min(DistTotMagM,2)
    
    # Find the Distance between Magnetic Fiducials and non magnetic Fiducials
    DistTotMF = np.zeros((EndPoint,np.size(Motion2,1),np.size(Motion2Mag,1)))
    for numMag in range(np.size(Motion2Mag,1)):
        DistComp = FinalData[:EndPoint,1:]-np.repeat(FinalDataMag[:EndPoint,numMag*2+1:numMag*2+3], np.size(Motion2,1), axis=1)
        DistTotMF[:,:,numMag] = np.sqrt( np.array(DistComp[:,0::2]**2 +DistComp[:,1::2]**2,dtype = float))
    DistTotMF = np.min(DistTotMF,2)
    
    Spacial={'DistTot':DistTot, 'DistTotMagM': DistTotMagM, 'DistTotMF': DistTotMF}
    
    # Method 1: Net Method
    # Find the median motion of each particle type in x and y 
    
    # Non-magnetic fiducial
    NetFlowX = np.median(MotionX,1)
    NetFlowY = np.median(MotionY,1)
    # MicroSwimmer
    SwimmerX = np.median(MotionMX,1)
    SwimmerY = np.median(MotionMY,1)
    # MagneticFiducial
    MagnetX = np.median(MotionMagX,1)
    MagnetY = np.median(MotionMagY,1)
    #Save the Data From Algorithm 1 (uncomment the next 3 lines if you want this file)
    #NetCSVFileName="{}Median.csv".format(OutputfileName)
    #MedianArray = np.stack((FinalData[:EndPoint,0], NetFlowX,NetFlowY,SwimmerX[:,0],SwimmerY,MagnetX,MagnetY))
    #np.savetxt(NetCSVFileName, MedianArray,delimiter=',', header="Time [s], Fiducial X, Fiducial Y, Swimmer X, Swimmer Y, MagnetX, MagnetY")
    
    # Method 2: Only consider fiducails a specific distance from the microswimmer
    # This is because the swimmer messes up the flow field
    # Here we consider a symmetric distance in pixels of the radius below
    # Non-Magnetic Fiducials
    AlgFlowX = np.zeros_like(NetFlowX)
    AlgFlowY = np.zeros_like(NetFlowX)
    
    for time in range(EndPoint):
        NotTooCloseToSwimmer = np.argwhere(DistTot[time,:]>ExclusionRadius)
        NotTooCloseToMagnet = np.argwhere(DistTotMF[time,:]>ExclusionRadius)    
        GoodFiducials = NotTooCloseToSwimmer
        if np.size(GoodFiducials)<1:
            GoodFiducials=np.argmax(DistTot[time,:])
        # We considered if the magnetic fiducial had an affect on the non-magnetic fiducial but found no evidence
        # GoodFiducials = np.intersect1d(NotTooCloseToSwimmer, NotTooCloseToMagnet)
        XDataTemp = MotionX[time,GoodFiducials]
        YDataTemp = MotionY[time,GoodFiducials]
        AlgFlowX[time] = np.median(XDataTemp)
        AlgFlowY[time] = np.median(YDataTemp)
    #print(GoodFiducials)
    
        
    
    # MagneticFiducials
    AlgGradX = np.zeros_like(MagnetX)
    AlgGradY = np.zeros_like(MagnetX)
    for time in range(EndPoint):
        GoodMagFiducials = np.argwhere(DistTotMagM[time,:]>ExclusionRadius)
        if np.size(GoodMagFiducials)<1:
            GoodMagFiducials=np.argmax(DistTotMagM[time,:])
        XDataTemp = MotionMagX[time,GoodMagFiducials]
        YDataTemp = MotionMagY[time,GoodMagFiducials]
        AlgGradX[time] = np.median(XDataTemp)
        AlgGradY[time] = np.median(YDataTemp)
    
    # Method 3: Gives higher power to the fiducials further from swimmers
    # We found this unnecesarry but in a more complex system with more components it may prove fruitful
    
    #Non-magnetic Fiducial
    MinDistance = np.minimum(DistTot,DistTotMF)
    FlowVelocityX = np.sum(MinDistance*Motion[:,0::2],1)/np.sum(MinDistance,1)
    FlowVelocityY = np.sum(MinDistance*Motion[:,1::2],1)/np.sum(MinDistance,1)
    #Magnetic Fiducial
    GradVelocityX = np.sum(DistTotMagM*MotionMag[:,0::2],1)/np.sum(DistTotMagM,1)
    GradVelocityY = np.sum(DistTotMagM*MotionMag[:,1::2],1)/np.sum(DistTotMagM,1)
    
    AnalCompNoMag = {'time':FinalData[:EndPoint,0], 'NetFlowX': NetFlowX, 'NetFlowY':NetFlowY, 
                   'AlgFlowX':AlgFlowX, 'AlgFlowY': AlgFlowY, 
                   'FlowVelocityX':FlowVelocityX, 'FlowVelocityY': FlowVelocityY}
    
    AnalCompMag = {'time':FinalData[:EndPoint,0], 'MagnetX': MagnetX, 'MagnetY':MagnetY, 
                   'AlgGradX':AlgGradX, 'AlgGradY': AlgGradY, 
                   'GradVelocityX':GradVelocityX, 'GradVelocityY': GradVelocityY}
    
    SafeFiducials = {'GoodFiducials':GoodFiducials, 'GoodMagFiducials':GoodMagFiducials}
    
    return IntraAnal, Spacial, AnalCompNoMag, AnalCompMag, SafeFiducials

def get_cmap_colors(cmap_name, n):
  """Returns n colors from the specified colormap."""
  cmap = cm.get_cmap(cmap_name, n)
  return [cmap(i) for i in range(n)]

def FigArrayPrep(CompAnalMethod, AnalCompNoMag, AnalCompMag,n,Sn):
    #Flow Predicted By the Fiducials (Change the name if you want to use a different Method)
    if CompAnalMethod=='ExclusionRemoval':
        FNetX = np.repeat(np.reshape(AnalCompNoMag['AlgFlowX'],(-1,1)),n,1)
        FNetY = np.repeat(np.reshape(AnalCompNoMag['AlgFlowY'],(-1,1)),n,1)
        
        #Same as above but the size of the swimmer
        FNetX2 = np.repeat(np.reshape(AnalCompNoMag['AlgFlowX'],(-1,1)),Sn,1)
        FNetY2 = np.repeat(np.reshape(AnalCompNoMag['AlgFlowY'],(-1,1)),Sn,1)
        
        # Flow Predicted By the Magnetic Spheres
        MNetX = np.repeat(np.reshape(AnalCompMag['AlgGradX'],(-1,1)),Sn,1)
        MNetY = np.repeat(np.reshape(AnalCompMag['AlgGradY'],(-1,1)),Sn,1)
    elif CompAnalMethod == 'Mean':
        FNetX = np.repeat(np.reshape(AnalCompNoMag['NetFlowX'],(-1,1)),n,1)
        FNetY = np.repeat(np.reshape(AnalCompNoMag['NetFlowY'],(-1,1)),n,1)
        
        #Same as above but the size of the swimmer
        FNetX2 = np.repeat(np.reshape(AnalCompNoMag['NetFlowX'],(-1,1)),Sn,1)
        FNetY2 = np.repeat(np.reshape(AnalCompNoMag['NetFlowY'],(-1,1)),Sn,1)
        
        # Flow Predicted By the Magnetic Spheres
        MNetX = np.repeat(np.reshape(AnalCompMag['MagnetX'],(-1,1)),Sn,1)
        MNetY = np.repeat(np.reshape(AnalCompMag['MagnetY'],(-1,1)),Sn,1)
    else:
        FNetX = np.repeat(np.reshape(AnalCompNoMag['FlowVelocityX'],(-1,1)),n,1)
        FNetY = np.repeat(np.reshape(AnalCompNoMag['FlowVelocityY'],(-1,1)),n,1)
        
        #Same as above but the size of the swimmer
        FNetX2 = np.repeat(np.reshape(AnalCompNoMag['FlowVelocityX'],(-1,1)),Sn,1)
        FNetY2 = np.repeat(np.reshape(AnalCompNoMag['FlowVelocityY'],(-1,1)),Sn,1)
        
        # Flow Predicted By the Magnetic Spheres
        MNetX = np.repeat(np.reshape(AnalCompMag['GradVelocityX'],(-1,1)),Sn,1)
        MNetY = np.repeat(np.reshape(AnalCompMag['GradVelocityX'],(-1,1)),Sn,1)
        
    Net={'FNetX':FNetX, 'FNetY':FNetY, 'FNetX2': FNetX2, 'FNetY2': FNetY2,
         'MNetX': MNetX, 'MNetY':MNetY}
    return Net

def PlotTotalMotion(IntraAnal,FieldData):
    EndPoint = IntraAnal['EndPoint']
    cols = ['{}'.format(col) for col in ['X', 'Y']]
    rows = ['{}'.format(row) for row in ['Non-Magnetic', 'Magnetic Field', 'Magnetic']]
    
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 8))
    
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)
        
    
    fig.suptitle('Total Motion of the Fiducials and Magnetic Swimmer (Black)')
    plt.subplot(3, 2, 3)
    plt.plot(FieldData[:EndPoint,0],FieldData[:EndPoint,1])
    plt.xlabel('Time [ms]')
    plt.ylabel('Magnetic Field [deg]')
    
    plt.subplot(3, 2, 4)
    plt.plot(FieldData[:EndPoint,0],FieldData[:EndPoint,1])
    plt.xlabel('Time [ms]')
    plt.ylabel('Magnetic Field [deg]')
    
    plt.subplot(3, 2, 1)
    plt.plot(IntraAnal['time'],IntraAnal['MotionX'])
    plt.plot(IntraAnal['time'],IntraAnal['MotionMX'],'k')
    #plt.ylim([-15,35])
    plt.xlabel('Time [ms]')
    plt.ylabel('Non-Magnetic Distance in X [pixels]')
    
    plt.subplot(3, 2, 5)
    plt.plot(IntraAnal['time'],IntraAnal['MotionMagX'])
    plt.plot(IntraAnal['time'],IntraAnal['MotionMX'],'k')
    #plt.ylim([-15,35])
    plt.xlabel('Time [ms]')
    plt.ylabel('Magnetic Distance in X [pixels]')
    
    plt.subplot(3, 2, 2)
    # plt.plot(FinalData[:,0],1000*FlowVelocity)
    # plt.ylim([-10,35])
    plt.plot(IntraAnal['time'],IntraAnal['MotionY'])
    plt.plot(IntraAnal['time'],IntraAnal['MotionMY'],'k')
    plt.xlabel('Time [ms]')
    plt.ylabel('Non-Magnetic Distance in Y [pixels]')
    
    plt.subplot(3, 2, 6)
    # plt.plot(FinalData[:,0],1000*FlowVelocity)
    # plt.ylim([-10,35])
    plt.plot(IntraAnal['time'],IntraAnal['MotionMagY'])
    plt.plot(IntraAnal['time'],IntraAnal['MotionMY'],'k')
    plt.xlabel('Time [ms]')
    plt.ylabel('Magnetic Distance in Y [pixels]')
    plt.show()
    pass

def PlotRelativeToAverageMotion(IntraAnal,NetResults,FieldData):
    EndPoint = IntraAnal['EndPoint']
    cols = ['{}'.format(col) for col in ['X', 'Y']]
    rows = ['{}'.format(row) for row in ['Non-Magnetic', 'Magnetic Field', 'Magnetic']]
    
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 8))
    
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)
        
        
    fig.suptitle('Fiducial Motion Compared to the Flow and Swimmer (Black)')
                 
    plt.subplot(3, 2, 3)
    plt.plot(FieldData[:EndPoint,0],FieldData[:EndPoint,1])
    plt.xlabel('Time [ms]')
    plt.ylabel('Magnetic Field [deg]')
    
    plt.subplot(3, 2, 4)
    plt.plot(FieldData[:EndPoint,0],FieldData[:EndPoint,1])
    plt.xlabel('Time [ms]')
    plt.ylabel('Magnetic Field [deg]')
    
    
    plt.subplot(3, 2, 1)
    plt.plot(IntraAnal['time'],IntraAnal['MotionX']-NetResults['FNetX'])
    plt.plot(IntraAnal['time'].T,IntraAnal['MotionMX']-NetResults['FNetX2'],'k')
    #plt.ylim([-15,35])
    plt.xlabel('Time [ms]')
    plt.ylabel('Non-Magnetic Relative Distance in X [pixels]')
    
    
    plt.subplot(3, 2, 5)
    plt.plot(IntraAnal['time'],IntraAnal['MotionMagX']-NetResults['MNetX'])
    plt.plot(IntraAnal['time'].T,IntraAnal['MotionMX']-NetResults['MNetX'],'k')
    #plt.ylim([-15,35])
    plt.xlabel('Time [ms]')
    plt.ylabel('Magnetic Relative Distance in X [pixels]')
    
    
    
    plt.subplot(3, 2, 2)
    # plt.plot(FinalData[:,0],1000*FlowVelocity)
    # plt.ylim([-10,35])
    plt.plot(IntraAnal['time'],IntraAnal['MotionY']-NetResults['FNetY'])
    plt.plot(IntraAnal['time'].T,(IntraAnal['MotionMY']-NetResults['FNetY2']), 'k')
    plt.xlabel('Time [ms]')
    plt.ylabel('Non-Magnetic Relative Distance in Y [pixels]')
    
    
    plt.subplot(3, 2, 6)
    # plt.plot(FinalData[:,0],1000*FlowVelocity)
    # plt.ylim([-10,35])
    plt.plot(IntraAnal['time'],IntraAnal['MotionMagY']-NetResults['MNetY'])
    plt.plot(IntraAnal['time'].T,(IntraAnal['MotionMY']-NetResults['MNetY']), 'k')
    plt.xlabel('Time [ms]')
    plt.ylabel('Magnetic Relative Distance in Y [pixels]')
    plt.show()
    
    pass

def CalculateNonMagFidAccuracy(IntraAnal, NetResults):
    FiducialErrorX = IntraAnal['MotionX']-NetResults['FNetX']
    FiducialErrorY = IntraAnal['MotionY']-NetResults['FNetY']
    
    
    FiducialToTError = np.sqrt(FiducialErrorX**2+FiducialErrorY**2)
    TotalDistacneTraveled = np.sqrt(IntraAnal['MotionX']**2+IntraAnal['MotionY']**2)
    TotalDistacneTraveled[TotalDistacneTraveled<1]=1
    PercentageVelocityError= FiducialToTError/TotalDistacneTraveled
    
    return PercentageVelocityError

def PlotStartEndVsExpectedEndNonMagFid(pltVal,n,color,FinalData, NetResults, FinalDataM):
    plt.figure()
    # First visualize the spacial posistion of each non-magnetic fiducial and their total motion
    for ii in range (n):
        c= [color[ii]]
        if ii == 0:
            plt.scatter(FinalData[15,1+2*ii], FinalData[15,2+2*ii], color=c, s=35, marker = ".",label='start point')
            plt.scatter(FinalData[pltVal,1+2*ii], FinalData[pltVal,2+2*ii], color=c, s=35, marker = "o", label='end point')
            plt.scatter(FinalData[15,1+2*ii]+NetResults['FNetX'][pltVal,ii], FinalData[15,2+2*ii]+NetResults['FNetY'][pltVal,ii], color=c, s=35, marker = '^', label='predicted end point')
        else: 
            plt.scatter(FinalData[15,1+2*ii], FinalData[15,2+2*ii], color=c, s=35, marker = ".")
            plt.scatter(FinalData[pltVal,1+2*ii], FinalData[pltVal,2+2*ii], color=c, s=35, marker = "o")
            plt.scatter(FinalData[15,1+2*ii]+NetResults['FNetX'][pltVal,ii], FinalData[15,2+2*ii]+NetResults['FNetY'][pltVal,ii], color=c, s=35, marker = '^')
    
    # Also visualize the posistion of the microswimmers
    plt.scatter(FinalDataM[15,1::2], FinalDataM[15,2::2], color='k', s=35, marker = '*', label='micro-swimmer start point')
    plt.scatter(FinalDataM[pltVal,1::2], FinalDataM[pltVal,2::2], color='k', s=35, marker = 'p', label='micro-swimmer end point')
    plt.xlabel('X [pixels]')
    plt.ylabel('Y [pixels]')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    pass

def LookAtPercentErrorNonMagFid(n,color,PercentageVelocityError,convert, time, ExclusionRadius, Spacial):
    #ExclusionRadius is in pixels
    #convert is pixels/micrometer
    plt.figure()
    for i in range (n):
        c = color[i]
        plt.scatter(time/1000,100*PercentageVelocityError[:,i],color=c)
        
    plt.xlabel('Time [s]')
    plt.ylabel('Percent Error [%]')
    plt.ylim(0,100)
    plt.show()
    # Visualization of the distnce between the swimmers and the fiducials overtime
    plt.rcParams.update({'font.size': 10})
    numColumns=3
    fig, axes = plt.subplots(nrows=(n//numColumns+1), ncols=numColumns)
    
    if n < 4:
        for i in range(n):
            axes[i%numColumns].scatter(time/1000,100*PercentageVelocityError[:,i],1, color = color[i])
        #     axes[i//5,i%5].set_ylabel("Error from Estimate [%]")
        #     axes[i//5,i%5].set_xlabel("Time [s]")
            axes[i%numColumns].tick_params(axis="y", labelcolor=color[i])
            axes[i%numColumns].set_ylim(0,100)
            ax2=axes[i%numColumns].twinx()
            ax2.scatter(time/1000,Spacial['DistTot'][:,i]/convert,1,color='k')
        #     ax2.set_ylabel('$Distance from Swimmer$ [$\mu$m]',color ='k')
            ax2.tick_params(axis="y", labelcolor="k")
            ax2.axhline(y=ExclusionRadius/convert, color=[0.5,0.5,0.5], linestyle='--')
            ax2.set_ylim(0,np.max(Spacial['DistTot'])/convert)
    else:
        for i in range(n):
            axes[i//numColumns,i%numColumns].scatter(time/1000,100*PercentageVelocityError[:,i],1, color = color[i])
        #     axes[i//5,i%5].set_ylabel("Error from Estimate [%]")
        #     axes[i//5,i%5].set_xlabel("Time [s]")
            axes[i//numColumns,i%numColumns].tick_params(axis="y", labelcolor=color[i])
            axes[i//numColumns,i%numColumns].set_ylim(0,100)
            ax2=axes[i//numColumns,i%numColumns].twinx()
            ax2.scatter(time/1000,Spacial['DistTot'][:,i]/convert,1,color='k')
        #     ax2.set_ylabel('$Distance from Swimmer$ [$\mu$m]',color ='k')
            ax2.tick_params(axis="y", labelcolor="k")
            ax2.axhline(y=ExclusionRadius/convert, color=[0.5,0.5,0.5], linestyle='--')
            ax2.set_ylim(0,np.max(Spacial['DistTot'])/convert)
    
        
    fig.text(0.5, 0.00, "Time [s]", ha='center')
    fig.text(0.00, 0.5, "Error from Estimate [%]", va='center', rotation='vertical')
    fig.text(1.0, 0.5, '$Distance from Swimmer$ [$\mu$m]', va='center', rotation='vertical')
    plt.tight_layout()
    plt.show()
    pass

def PlotSummaryMotionWithShading(FieldData,SwimmerOrientation,IntraAnal,OscillatingFieldStart,NetResults,convert, n, Mn, SafeFiducials):
    time = IntraAnal['time']
    plt.rcParams.update({'font.size': 10})
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(3.25,6.5))
    
    plt.subplot(5,1,1)
    plt.plot(FieldData[:,0],FieldData[:,1]*180/3.14159,color='k',linewidth=0.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Magnetic Field \n Orientation [$^\circ$]')
    
    plt.subplot(5,1,2)
    plt.plot(SwimmerOrientation[:OscillatingFieldStart,0],SwimmerOrientation[:OscillatingFieldStart,1]*180/3.14159,color = [.443,0,.467],linewidth=0.5)
    plt.plot(SwimmerOrientation[OscillatingFieldStart:,0],SwimmerOrientation[OscillatingFieldStart:,1]*180/3.14159,color = [206.25/255,28.75/255,208.75/255],linewidth=0.5)
    plt.xlabel('Time [s]')
    axes[1].tick_params(axis="y", labelcolor=[.443,0,.467])
    axes[1].set_ylabel('Swimmer\n Orientation [$^\circ$]',color = [.443,0,.467])
    
    
    plt.subplot(5,1,3)
    
    Regs2 = np.sqrt(NetResults['FNetY'][:,0]**2+NetResults['FNetX'][:,0]**2)
    Ms=np.sqrt(NetResults['FNetX'][:,0]**2+NetResults['FNetY'][:,0]**2)
    axes[2].plot(time,Regs2/convert,color = 'b', linewidth=0.5)
    if np.size(SafeFiducials['GoodFiducials'])>1:
        #Regs = np.mean(np.sqrt(IntraAnal['MotionY'][:,SafeFiducials['GoodFiducials']]**2+IntraAnal['MotionX'][:,SafeFiducials['GoodFiducials']]**2),1)
        #RegsX = np.mean(IntraAnal['MotionX'][:,SafeFiducials['GoodFiducials']],1)
        #RegsY = np.mean(IntraAnal['MotionY'][:,SafeFiducials['GoodFiducials']],1)
        #Dir = np.arctan2(RegsY,RegsX)
        RegStd = np.std(np.sqrt(IntraAnal['MotionY'][:,SafeFiducials['GoodFiducials']]**2+
                                IntraAnal['MotionX'][:,SafeFiducials['GoodFiducials']]**2),1)
        axes[2].fill_between(np.linspace(0,np.max(time),np.size(time)), (Regs2.flatten()-RegStd.flatten())/convert, (Regs2.flatten()+RegStd.flatten())/convert, color = 'b', alpha=.25)
    else:
        pass
    axes[2].set_xlabel('Time [s]')
    axes[2].set_ylabel('$\Delta_{ps} $ [$\mu$m]',color = [83/250,19/250,211/250])
    axes[2].tick_params(axis="y", labelcolor=[83/250,19/250,211/250])
    
    
    plt.subplot(5,1,4)
    colorG=np.ones((2,3))*[0,1,0]
    colorG[1,:]=colorG[1,:]*.5
    #Mags = np.sqrt(IntraAnal['MotionMagY']**2+IntraAnal['MotionMagX']**2)
    Regs3 = np.sqrt(NetResults['MNetX'][:,0]**2+NetResults['MNetY'][:,0]**2)-Regs2
    
    axes[3].plot(time,Regs3/convert,color = [0.94,0.03,0.03], linewidth=0.5)
    if Mn>1:
        Reg2Std = np.std(np.sqrt(IntraAnal['MotionMagY'][:,SafeFiducials['GoodMagFiducials']]**2+
                             IntraAnal['MotionMagX'][:,SafeFiducials['GoodMagFiducials']]**2),1)
        axes[3].fill_between(time.astype(float), (Regs3.flatten()-Reg2Std.flatten())/convert, (Regs3.flatten()+Reg2Std.flatten())/convert, color = 'r', alpha =.25)
    else:
        pass
    axes[3].set_xlabel('Time [s]')
    axes[3].set_ylabel('$\Delta_{Mag}-\Delta_{ps}$ [$\mu$m]',color = [240/250,8/250,0/250])
    axes[3].tick_params(axis="y", labelcolor=[240/250,8/250,0/250])
    
    
    plt.subplot(5,1,5)
    #Ms = np.sqrt(IntraAnal['MotionMY']**2+IntraAnal['MotionMX']**2)
    Diff2 = np.sqrt((IntraAnal['MotionMY']-NetResults['MNetY'])**2+
                    (IntraAnal['MotionMX']-NetResults['MNetX'])**2)
    
    axes[4].plot(time[:OscillatingFieldStart],Diff2[:OscillatingFieldStart]/convert,color = [.443,0,.467], linewidth=0.5)
    axes[4].plot(time[OscillatingFieldStart:].flatten(),Diff2[OscillatingFieldStart:]/convert,color = [206.25/255,28.75/255,208.75/255] ,linewidth=0.5)
    axes[4].set_xlabel('Time [s]')
    axes[4].set_ylabel('$\Delta_{Swim}-\Delta_{Mag}$ [$\mu$m]',color = [.443,0,.467])
    axes[4].tick_params(axis="y", labelcolor=[.443,0,.467])
    plt.tight_layout()
    plt.show()

def PlotSwimmerEvaluation(OscillatingFieldStart,FieldData, IntraAnal, convert, n, Mn, NetResults, SafeFiducials):
    time = IntraAnal['time']
    plt.rcParams.update({'font.size': 10})
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6.5,4))
    plt.subplot(2,2,1)
    plt.plot(FieldData[:,0]/1000,FieldData[:,1])
    plt.xlabel('Time [ms]')
    plt.ylabel('Magnetic Field [deg]')
    
    plt.subplot(2,2,3)
    #Ms = np.sqrt(IntraAnal['MotionMY']**2+IntraAnal['MotionMX']**2)
    Diff2 = np.sqrt((IntraAnal['MotionMY']-NetResults['MNetY'])**2+
                    (IntraAnal['MotionMX']-NetResults['MNetX'])**2)
    
    axes[1,0].plot(time[:OscillatingFieldStart],Diff2[:OscillatingFieldStart]/convert,color = [.443,0,.467], linewidth=0.5)
    axes[1,0].plot(time[OscillatingFieldStart:].flatten(),Diff2[OscillatingFieldStart:]/convert,color = [206.25/255,28.75/255,208.75/255] ,linewidth=0.5)
    axes[1,0].set_xlabel('Time [s]')
    axes[1,0].set_ylabel('$\Delta_{Swim}-\Delta_{Mag}$ [$\mu$m]',color = [.443,0,.467])
    axes[1,0].tick_params(axis="y", labelcolor=[.443,0,.467])
    plt.tight_layout()
    
    plt.subplot(1,2,2)
    XAverage=np.median(IntraAnal['MotionX'],1)
    YAverage=np.median(IntraAnal['MotionY'],1)
    plt.plot(XAverage,YAverage,.5,color='b')
    #plt.plot(MotionX[707:,1],MotionY[707:,1],.5,color='b')
    if n>1:
        STDEVX=(np.std(IntraAnal['MotionX'][:,SafeFiducials['GoodFiducials']],1))
        STDEVY=np.std(IntraAnal['MotionY'][:,SafeFiducials['GoodFiducials']],1)
        SummarySTDEV=np.sqrt(np.square(STDEVX)+np.square(STDEVY))
        plt.fill_between(XAverage, (YAverage.flatten()-SummarySTDEV.flatten()), (YAverage.flatten()+SummarySTDEV.flatten()), color=[.25,.25,1], alpha=.5)
    else:
        pass
    
    # plt.plot(MotionMagX[:707,0],MotionMagY[:707,0],.5,color='r')
    # plt.plot(MotionMagX[707:,0],MotionMagY[707:,0],.5,color='r')
    
    
    MagXAverage=np.average(IntraAnal['MotionMagX'],1)
    MagYAverage=np.average(IntraAnal['MotionMagY'],1)
    plt.plot(MagXAverage,MagYAverage,.5,color='r')
    if Mn>1:
        STDEVMagX=(np.std(IntraAnal['MotionMagX'][:,SafeFiducials['GoodMagFiducials']],1))
        STDEVMagY=np.std(IntraAnal['MotionMagY'][:,SafeFiducials['GoodMagFiducials']],1)
        SummarySTDEV=np.sqrt(np.square(STDEVMagX)+np.square(STDEVMagY))
        plt.fill_between(MagXAverage, MagYAverage.flatten()-SummarySTDEV.flatten(), MagYAverage.flatten()+SummarySTDEV.flatten(), color=[1,.25,.25], alpha=.5)
    
    
    plt.plot(IntraAnal['MotionMX'][:OscillatingFieldStart,:],IntraAnal['MotionMY'][:OscillatingFieldStart,:],.5,color=[.443,0,.467])
    plt.plot(IntraAnal['MotionMX'][OscillatingFieldStart:,:],IntraAnal['MotionMY'][OscillatingFieldStart:,:],.5,color=[206.25/255,28.75/255,208.75/255])
    plt.ylabel("$\Delta$ y")
    plt.xlabel("$\Delta$ x")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
            
def MakeStabilizeVideo(videoPath,MovVideoOutputName,divisor, FinalData, FinalDataMag, FinalDataM, IntraAnal, XStabilization, YStabilization,n, Mn, Sn, OverlayCircle):
    # Create a video capture object to read videos
    cap = cv2.VideoCapture(videoPath)
    TotalFrames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    framesPerSecond=cap.get(cv2.CAP_PROP_FPS)
    #framerate=1/framesPerSecond
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    
    # This just copies the video
    success, frame = cap.read()
    OutputIMG=frame
    
    if not success:
      print('Failed to read video')
      sys.exit(1)
    # This portion of the code captures the origional video
    FramesToVideo = np.zeros((np.size(OutputIMG,0),np.size(OutputIMG,1),np.size(OutputIMG,2),int(TotalFrames/divisor+1)),int)
    count = 0
    hold=0
    while cap.isOpened():
      success, frame = cap.read()
      count += divisor # i.e. at 30 fps, this advances one second
      cap.set(cv2.CAP_PROP_POS_FRAMES, count)
      if not success:
        break
      
      FramesToVideo[:,:,:,hold]=frame
      if hold%50==0:
          print('PercentBar',np.round((hold/TotalFrames*divisor/2)*100))
    
     
      # quit on ESC button
      if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
        break
      hold=hold+1
    
    # This portion of the code creates the new video    
    
    # This creates the New video
    # If you want to do this with the magnetic motion change it to M Net instead of FNet
    XNetX = XStabilization
    XNetY = YStabilization
    
    indxX = np.argmax(abs(XNetX[:,0]))
    indxY = np.argmax(abs(XNetY[:,0]))
    MaxMoveX = int(XNetX[indxX,0]+1)
    MaxMoveY = int(XNetY[indxY,0]+1)
    if MaxMoveX>0:
        StartX=abs(MaxMoveX)+10
    else:
        StartX = 10
        
    if MaxMoveY>0:
        StartY = 10
    else:
        StartY=abs(MaxMoveY)+10
    
    video=cv2.VideoWriter(MovVideoOutputName,fourcc,framesPerSecond/divisor,(int(np.size(FramesToVideo,1)+abs(MaxMoveX)+35),int(abs(MaxMoveY)+35+np.size(FramesToVideo,0))))
    
    for j in range(int(np.size(XNetX,0)/divisor)):
        BlankArray = np.zeros((int(np.size(FramesToVideo,0)+abs(MaxMoveY)+35),int(abs(MaxMoveX)+35+np.size(FramesToVideo,1)),3),int)
        #print(np.shape(BlankArray))
        ZeroAxis = int(StartY + XNetY[int(j*divisor),0])
        OneAxis = int(StartX - XNetX[int(j*divisor),0])
        if OverlayCircle == 0:
            BlankArray[ZeroAxis:ZeroAxis+np.size(FramesToVideo,0),OneAxis:OneAxis+np.size(FramesToVideo,1)] = FramesToVideo[:,:,:,j]
            video.write(np.uint8(BlankArray))
        elif OverlayCircle == 1:
            ArrowFrame = np.ascontiguousarray(FramesToVideo[:,:,:,j],dtype=np.uint8)
            i=int((j-2)*divisor)
            for k in range(n):
               cv2.circle(ArrowFrame,(int(FinalData[i,1+2*k]),np.size(ArrowFrame,0)-(int(FinalData[i,2+2*k]))),15,(255,0,0),2)
            for l in range(Mn):
                cv2.circle(ArrowFrame,(int(FinalDataMag[0,1+l*2]+
                                           IntraAnal['MotionMagX'][i,0]),np.size(ArrowFrame,0)-
                                       (int(FinalDataMag[0,2+2*l]+IntraAnal['MotionMagY'][i,0]))),15,(0,0,255),2)
            for c in range(Sn):
                cv2.circle(ArrowFrame,(int(FinalDataM[0,1+2*c]+IntraAnal['MotionMagX'][i,0]),np.size(ArrowFrame,0)-
                                       (int(FinalDataM[0,2+2*c]+IntraAnal['MotionMagY'][i,0]))),15,(255,0,255),2)
        
            BlankArray[ZeroAxis:ZeroAxis+np.size(FramesToVideo,0),OneAxis:OneAxis+np.size(FramesToVideo,1)] =ArrowFrame
            video.write(np.uint8(BlankArray))
            
        
        if j%50==0:
          print('PercentBar2',np.round((0.5+j/TotalFrames*divisor/2)*100))
    
        
          
    video.release()
    cv2.destroyAllWindows()

    pass