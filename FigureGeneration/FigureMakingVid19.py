# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 17:47:06 2024

@author: tkent
"""

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
import scipy
from scipy import stats

# Give an output name
OutputfileName = "DemonstratingMovement"

## Load Fiducials
Locations = "TAEPull_MS19_Fiducials.xlsx"
FinalData = np.array(pd.read_excel(Locations))
FinalData = FinalData[4:,:]
numberOfBoxes=int((np.size(FinalData,1)-1)/2)
FinalDataSizes = np.full_like(FinalData,1)*50

##Load Swimmer Data
Locations = "TAEPull_MS19_Position.xlsx"
FinalDataM = np.array(pd.read_excel(Locations))
FinalDataM = FinalDataM[4:,:]
numberOfBoxes=int((np.size(FinalData,1)-1)/2)
FinalDataSizesM = np.full_like(FinalData,1)*50

##Load Magnetic Fiducial Data
Locations = "TAEPull_MS19_MagFiducials.xlsx"
FinalDataMag = np.array(pd.read_excel(Locations))
FinalDataMag = FinalDataMag[4:,:]
numberOfBoxesMag=int((np.size(FinalData,1)-1)/2)
FinalDataSizesM = np.full_like(FinalData,1)*50

##Load the data about the Magnetic Field
Locations1 = "TAEPull_MS19_Angle.xlsx"
FieldData1 = np.array(pd.read_excel(Locations1))
SwimmerOrientation = FieldData1[4:,:]
#XMinSwim = int(30000/(SwimmerOrientation[2,0]-SwimmerOrientation[1,0]))
#XMaxSwim = int(85000/(SwimmerOrientation[2,0]-FieldData[1,0]))

Locations = "MagField.xlsx"
FieldData = np.array(pd.read_excel(Locations))
FieldData = FieldData[4:,:]
XMinSwim = int(30000/(FieldData[2,0]-FieldData[1,0]))
XMaxSwim = int(85000/(FieldData[2,0]-FieldData[1,0]))

# Analyze Fiducials
EndPoint= np.min([np.size(FinalData,0), np.size(FinalDataM,0), np.size(FinalDataMag,0)])

# This Motion is a relative motion
Motion = np.array(FinalData[:EndPoint,1:]-FinalData[20,1:], dtype= float)
# Break motion in to magnitudes directions and x and y vectors
Motion2 = np.sqrt( np.array(Motion[:,0::2]**2 +Motion[:,1::2]**2,dtype = np.int32))
MotionX = Motion[:,0::2]
MotionY = Motion[:,1::2]
Direction = np.array(np.arctan2(Motion[:,0::2],Motion[:,1::2])*180/3.14159,dtype = np.int32)

# Analyze Magnetic Swimmer

# This Motion is a relative motion
MotionM = np.array(FinalDataM[:EndPoint,1:]-FinalDataM[15,1:], dtype = float)
# Break motion in to magnitudes directions and x and y vectors
MotionMX = np.reshape(MotionM[:,0::2],(-1,1))
MotionMY = np.reshape(MotionM[:,1::2],(-1,1))
Motion2M = np.sqrt( np.array(MotionM[:,0::2]**2 +MotionM[:,1::2]**2,dtype = np.int32))
DirectionM = np.array(np.arctan2(MotionM[:,0::2],MotionM[:,1::2])*180/3.14159,dtype = np.int32)

# Analyze Magnetic Fiducials

# This Motion is a relative motion
MotionMag = np.array(FinalDataMag[:EndPoint,1:]-FinalDataMag[15,1:], dtype = float)
# Break motion in to magnitudes directions and x and y vectors
MotionMagX = MotionMag[:,0::2]
MotionMagY = MotionMag[:,1::2]
Motion2Mag = np.sqrt( np.array(MotionMag[:,0::2]**2 +MotionMag[:,1::2]**2,dtype = np.int32))
DirectionM = np.array(np.arctan2(MotionMag[:,0::2],MotionMag[:,1::2])*180/3.14159,dtype = np.int32)

# Find the Distance between Fiducials and Magnetic Swimmers
DistTot = np.zeros((EndPoint,np.size(Motion2,1),np.size(Motion2M,1)))
for numSwim in range(np.size(Motion2M,1)):
    DistComp = FinalData[:EndPoint,1:]-np.repeat(FinalDataM[:EndPoint,numSwim*2+1:numSwim*2+3], np.size(Motion2,1), axis=1)
    DistTot[:,:,numSwim] = np.sqrt( np.array(DistComp[:,0::2]**2 +DistComp[:,1::2]**2,dtype = float))
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

# First net method  flow x and a flow y 
NetFlowX = np.median(MotionX,1)
NetFlowY = np.median(MotionY,1)

SwimmerX = np.median(MotionMX,1)
SwimmerY = np.median(MotionMY,1)

MagnetX = np.median(MotionMagX,1)
MagnetY = np.median(MotionMagY,1)

NetCSVFileName="{}Median.csv".format(OutputfileName)
MedianArray = np.stack((FinalData[:EndPoint,0], NetFlowX,NetFlowY,SwimmerX,SwimmerY,MagnetX,MagnetY))
np.savetxt(NetCSVFileName, MedianArray,delimiter=',', header="Time [s], Fiducial X, Fiducial Y, Swimmer X, Swimmer Y, MagnetX, MagnetY")


# Second Algorithm ignore data where a non magnetic fiducial is too close to a swimmer
# This is because the swimmer messes up the flow field
AlgFlowX = np.zeros_like(NetFlowX)
AlgFlowY = np.zeros_like(NetFlowX)
HoldGood = np.zeros((np.size(DistTot,0),np.size(MotionX,1)))
for time in range(EndPoint):
    NotTooCloseToSwimmer = np.argwhere(DistTot[time,:]>100)
    NotTooCloseToMagnet = np.argwhere(DistTotMF[time,:]>25)
    GoodFiducials = np.intersect1d(NotTooCloseToSwimmer, NotTooCloseToMagnet)
    GoodFiducials = NotTooCloseToSwimmer
    HoldGood[time,GoodFiducials]=1
    XDataTemp = MotionX[time,GoodFiducials]
    YDataTemp = MotionY[time,GoodFiducials]
    AlgFlowX[time] = np.median(XDataTemp)
    AlgFlowY[time] = np.median(YDataTemp)
print(GoodFiducials)
    
# For the Magnetic Fiducials only the swimmer can affect them
AlgGradX = np.zeros_like(MagnetX)
AlgGradY = np.zeros_like(MagnetX)
for time in range(EndPoint):
    GoodFiducials = np.argwhere(DistTotMagM[time,:]>300)
    XDataTemp = MotionMagX[time,GoodFiducials]
    YDataTemp = MotionMagY[time,GoodFiducials]
    AlgGradX[time] = np.median(XDataTemp)
    AlgGradY[time] = np.median(YDataTemp)
    


# Third Algorithm gives higher power to the fiducials all alone
MinDistance = np.minimum(DistTot,DistTotMF)
FlowVelocityX = np.sum(MinDistance*Motion[:,0::2],1)/np.sum(MinDistance,1)
FlowVelocityY = np.sum(MinDistance*Motion[:,1::2],1)/np.sum(MinDistance,1)

GradVelocityX = np.sum(DistTotMagM*MotionMag[:,0::2],1)/np.sum(DistTotMagM,1)
GradVelocityY = np.sum(DistTotMagM*MotionMag[:,1::2],1)/np.sum(DistTotMagM,1)

NetCSVFileName="{}FlowPredictionAlg.csv".format(OutputfileName)
AlgArray = np.stack((FinalData[:EndPoint,0], NetFlowX,NetFlowY,AlgFlowX,AlgFlowY,FlowVelocityX,FlowVelocityY))
np.savetxt(NetCSVFileName, AlgArray,delimiter=',', header="Time [s], Fiducial X, Fiducial Y, IgnoreDataX, IgnoreDataY, WeightedX, WeightedY")

NetCSVFileName="{}MagneticGradientPredictionAlg.csv".format(OutputfileName)
AlgArray = np.stack((FinalData[:EndPoint,0], MagnetX,MagnetY,AlgGradX,AlgGradY,GradVelocityX,GradVelocityY))
np.savetxt(NetCSVFileName, AlgArray,delimiter=',', header="Time [s], Median X, Median Y, IgnoreDataX, IgnoreDataY, WeightedX, WeightedY")

# Updated Data Subtracted 
FiducialsUpdateX = MotionX-np.repeat(np.reshape(FlowVelocityX,(-1,1)), np.size(Motion2,1),1)
FiducialsUpdateY = MotionY-np.repeat(np.reshape(FlowVelocityY,(-1,1)), np.size(Motion2,1),1)
FiducialsTotalMotion = np.sqrt(FiducialsUpdateX**2+FiducialsUpdateY**2)

SwimmerUpdateX = np.reshape(MotionM[:,0::2],(-1,1))-np.reshape(FlowVelocityX,(-1,1))
SwimmersUpdateY = np.reshape(MotionM[:,1::2],(-1,1))-np.reshape(FlowVelocityX,(-1,1))

color = []
n = np.size(MotionX,1)
for i in range(n):
    value = np.random.choice(range(0, 100), size=3)
    saturation = np.max(value)-np.min(value)
    satnorm=.5/saturation
    value = value*satnorm
    norm = np.max(value)
    value = value/norm*.75
    #print(value)
    color.append(tuple(value))
    
#Flow Predicted By the Fiducials
FNetX = np.repeat(np.reshape(AlgFlowX,(-1,1)),np.size(MotionX,1),1)
FNetY = np.repeat(np.reshape(AlgFlowY,(-1,1)),np.size(MotionX,1),1)

# Flow Predicted By the Magnetic Spheres
MNetX = np.repeat(np.reshape(AlgGradX,(-1,1)),np.size(MotionMX,1),1)
MNetY = np.repeat(np.reshape(AlgGradY,(-1,1)),np.size(MotionMX,1),1)

SNetX = MotionMX
SNetY = MotionMY

#Switch the above to the third algorithm if your having a very noisy video


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
plt.plot(FinalData[:EndPoint,0],MotionX)
plt.plot(FinalData[:EndPoint,0].T,MotionMX,'k')
#plt.ylim([-15,35])
plt.xlabel('Time [ms]')
plt.ylabel('Non-Magnetic Relative Distance in X [pixels]')


plt.subplot(3, 2, 5)
plt.plot(FinalData[:EndPoint,0],MotionMagX)
plt.plot(FinalData[:EndPoint,0].T,MotionMX,'k')
#plt.ylim([-15,35])
plt.xlabel('Time [ms]')
plt.ylabel('Magnetic Relative Distance in X [pixels]')



plt.subplot(3, 2, 2)
# plt.plot(FinalData[:,0],1000*FlowVelocity)
# plt.ylim([-10,35])
plt.plot(FinalData[:EndPoint,0],MotionY)
plt.plot(FinalData[:EndPoint,0].T,(MotionMY), 'k')
plt.xlabel('Time [ms]')
plt.ylabel('Non-Magnetic Relative Distance in Y [pixels]')


plt.subplot(3, 2, 6)
# plt.plot(FinalData[:,0],1000*FlowVelocity)
# plt.ylim([-10,35])
plt.plot(FinalData[:EndPoint,0],MotionMagY)
plt.plot(FinalData[:EndPoint,0].T,(MotionMY), 'k')
plt.xlabel('Time [ms]')
plt.ylabel('Magnetic Relative Distance in Y [pixels]')
convert =2.31673


plt.rcParams.update({'font.size': 10})
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(3.25,6.5))
plt.subplot(5,1,1)
plt.plot(FieldData[:,0],FieldData[:,1]*180/3.14159,color='k',linewidth=0.5)
plt.xlabel('Time [s]')
plt.ylabel('Magnetic Field \n Orientation [$^\circ$]')
plt.xlim([0,80])
plt.ylim([-100,2000])



plt.subplot(5,1,3)
Regs = np.mean(np.sqrt(MotionY[:,1:]**2+MotionX[:,1:]**2),1)
RegsX = np.mean(MotionX[:,1:],1)
RegsY = np.mean(MotionY[:,1:],1)
Regs2 = np.sqrt(AlgFlowY**2+AlgFlowX**2)
Dir = np.arctan2(RegsY,RegsX)
RegStd = np.std(np.sqrt(MotionY[:,1:]**2+MotionX[:,1:]**2),1)
#slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(FinalData[:EndPoint,0], Regs)
#print(r_value)
plt.xlim([0,80])
plt.ylim([0,60])

#ax2=axes[1].twinx()
axes[2].plot(FinalData[:EndPoint,0],Regs2/convert,color = 'b', linewidth=0.5)
axes[2].fill_between(np.linspace(0,97.3607,EndPoint), (Regs2-RegStd).flatten()/convert, (Regs2+RegStd).flatten()/convert, color = 'b', alpha=.25)
axes[2].set_xlabel('Time [s]')
axes[2].set_ylabel('$\Delta$ [$\mu$m]',color = [83/250,19/250,211/250])
axes[2].tick_params(axis="y", labelcolor=[83/250,19/250,211/250])
#plt.ylim([0,75])

# ax2.plot(FinalData[:EndPoint,0]/1000, Dir*180/3.14159, color = 'k')
# ax2.set_ylabel('$Direction$ [$^\circ$]',color ='k')
# ax2.tick_params(axis="y", labelcolor="k")

plt.subplot(5,1,4)
colorG=np.ones((2,3))*[0,1,0]
colorG[1,:]=colorG[1,:]*.5
Mags = np.sqrt(MotionMagY**2+MotionMagX**2)
Regs2 = np.sqrt(AlgFlowY**2+AlgFlowX**2)

Diff = np.sqrt((MotionMagY.flatten()-AlgFlowY.flatten())**2+(MotionMagX.flatten()-AlgFlowX.flatten())**2)
Dir = np.arctan2(MotionMagY-np.roll(MotionMagY,10),MotionMagX-np.roll(MotionMagX,10))*180/3.14159
#Dir[Dir<-90]=Dir[Dir<-90]+360
#ax4=axes[2].twinx()
axes[3].plot(FinalData[:EndPoint,0],Diff/convert,color = [0.94,0.03,0.03], linewidth=0.5)
axes[2].plot(FinalData[:EndPoint,0],Mags[:EndPoint]/convert,color = [0.94,0.03,0.03], linewidth=0.5)
#axes[1].fill_between(np.linspace(0,87683.2814/1000,EndPoint), (Regs-RegStd).flatten()/convert, (Regs+RegStd).flatten()/convert, color = [83/250,19/250,211/250], alpha=.75)
axes[3].set_xlabel('Time [s]')
axes[3].set_ylabel('$\Delta$ [$\mu$m]',color = [240/250,8/250,0/250])
axes[3].tick_params(axis="y", labelcolor=[240/250,8/250,0/250])
#axes[2].set_ylim(-35,10)
#plt.ylim([-15,5])
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(FinalData[:EndPoint,0], Mags.flatten())
print(r_value)
plt.xlim([0,80])
plt.ylim([0,60])

# ax4.plot(FinalData[:EndPoint,0]/1000, Dir, color = 'k')
# ax4.set_ylabel('$Direction$ [$^\circ$]',color ='k')
# ax4.tick_params(axis="y", labelcolor="k")
plt.subplot(5,1,5)
Ms = np.sqrt(MotionMY**2+MotionMX**2)
Dir = np.arctan2(MotionMY-np.roll(MotionMY,10),MotionMX-np.roll(MotionMX,10))*180/3.14159
#Dir[Dir<-90]=Dir[Dir<-90]+360
#ax6=axes[3].twinx()
Diff2 = np.sqrt((MotionMY.flatten()-MotionMagY.flatten())**2+(MotionMX.flatten()-MotionMagX.flatten())**2)
plt.ylim([0,60])

axes[4].plot(FinalData[:707,0],Diff2[:707]/convert,color = [.443,0,.467], linewidth=0.5)
axes[4].plot(FinalData[707:EndPoint,0].flatten(),Diff2[707:EndPoint]/convert,color = [206.25/255,28.75/255,208.75/255] ,linewidth=0.5)
axes[2].plot(FinalData[:707,0],Ms[:707]/convert,color = [.443,0,.467], linewidth=0.5)
axes[2].plot(FinalData[707:EndPoint,0].flatten(),Ms[707:EndPoint]/convert,color = [206.25/255,28.75/255,208.75/255] ,linewidth=0.5)
#axes[1].fill_between(np.linspace(0,87683.2814/1000,EndPoint), (Regs-RegStd).flatten()/convert, (Regs+RegStd).flatten()/convert, color = [83/250,19/250,211/250], alpha=.75)
axes[4].set_xlabel('Time [s]')
axes[4].set_ylabel('$\Delta$ [$\mu$m]',color = [.443,0,.467])
axes[4].tick_params(axis="y", labelcolor=[.443,0,.467])
#axes[3].set_ylim(-35,10)
plt.xlim([0,80])
#plt.ylim([0,75])

plt.subplot(5,1,2)
plt.plot(SwimmerOrientation[:707,0],SwimmerOrientation[:707,1]*180/3.14159,color = [.443,0,.467],linewidth=0.5)
plt.plot(SwimmerOrientation[707:,0],SwimmerOrientation[707:,1]*180/3.14159,color = [206.25/255,28.75/255,208.75/255],linewidth=0.5)
plt.xlabel('Time [s]')
axes[1].tick_params(axis="y", labelcolor=[.443,0,.467])
axes[1].set_ylabel('Swimmer\n Orientation [$^\circ$]',color = [.443,0,.467])
plt.xlim([0,80])
plt.ylim([-100,2000])

# ax6.plot(FinalData[:EndPoint,0]/1000, Dir, color = 'k')
# ax6.set_ylabel('$Direction$ [$^\circ$]',color ='k')
# ax6.tick_params(axis="y", labelcolor="k")


plt.rcParams.update({'font.size': 10})
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(3.25,3.25))
plt.subplot(3,1,1)
plt.plot(FieldData[:EndPoint,0]/1000,FieldData[:EndPoint,1])
plt.xlabel('Time [s]')
plt.ylabel('Magnetic Field [deg]')

plt.subplot(3,1,2)
colorG=np.ones((2,3))*[0,1,0]
colorG[1,:]=colorG[1,:]*.5
Regs = np.mean(np.sqrt(MotionY[:,5:]**2+MotionX[:,5:]**2),1)
RegsX = np.mean(MotionX[:,5:],1)
RegsY = np.mean(MotionY[:,5:],1)
RegStd = np.std(np.sqrt(MotionY[:,5:]**2+MotionX[:,5:]**2),1)
Mags = np.sqrt(MotionMagY**2+MotionMagX**2)
for ii in range(1):
    plt.plot(FinalData[:EndPoint,0]/1000,Mags[:,ii]/convert,color=colorG[ii,:])
plt.plot(FinalData[:EndPoint,0]/1000,Regs/convert,'r')
#plt.plot(FinalData[:EndPoint,0]/1000,np.sqrt(MotionMY**2+MotionMX**2)/convert,'r')
plt.fill_between(np.linspace(0,87683.2814/1000,EndPoint), (Regs-RegStd).flatten()/convert, (Regs+RegStd).flatten()/convert, color=[1,.25,.25], alpha=.5)

plt.xlabel('Time [ms]')
plt.ylabel('$\Delta$ [$\mu$m]')

plt.subplot(3,1,3)
plt.scatter(MotionMagX[:,0]/convert,MotionMagY[:,0]/convert,2,color=colorG[0,:])
#plt.scatter(MotionMagX[:,1]/convert,MotionMagY[:,1]/convert,2,color=colorG[1,:])
#plt.scatter(MotionMX[:]/convert,MotionMY[:]/convert,2,color='b')
plt.scatter(RegsX/convert,RegsY/convert,2,color='r')
plt.xlabel('$\Delta$ X [$\mu$ m]')
plt.ylabel('$\Delta$ Y [$\mu$ m]')
plt.axis('equal')

plt.figure(figsize=(3.25,3.25))
plt.subplot(2,1,1)
plt.plot(FieldData[:EndPoint,0]/1000,FieldData[:EndPoint,1])
plt.xlabel('Time [ms]')
plt.ylabel('Magnetic Field [deg]')
# Make a video whose frame moves instead of the particle
# This will be easiest to test if we make the particle move

# Load the vide
videoPath = "TAEPull_MS19_constDirRotSine_trim.mp4"

#Create the Output Name
MovVideoOutputName = "{}MovingFrame.mp4".format(OutputfileName)
# The divisor is for skipping frames (so as to not make too large a video)
divisor = 4


# Create a video capture object to read videos
cap = cv2.VideoCapture(videoPath)
TotalFrames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
framesPerSecond=cap.get(cv2.CAP_PROP_FPS)
framerate=1/framesPerSecond
fourcc = cv2.VideoWriter_fourcc(*'MP4V')

# This just copies the video
success, frame = cap.read()
OutputIMG=frame

if not success:
  print('Failed to read video')
  sys.exit(1)
# 

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
  if hold%1==0:
      print('PercentBar',hold/TotalFrames*divisor)

 
  # quit on ESC button
  if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
    break
  hold=hold+1
# This portion of the code creates the new video    


LastxCenter=FinalData[0,1::2]
LastyCenter=FinalData[0,2::2]
XCenter = np.zeros((np.size(FramesToVideo,3),np.size(LastxCenter)),dtype=int)
YCenter = np.zeros((np.size(FramesToVideo,3),np.size(LastxCenter)),dtype=int)
ArrowMultiplier = 5

# This creates the New video
# If you want to do this with the magnetic motion change it to M Net instead of FNet
Zeros=np.zeros_like(MNetX)
XNetX = Zeros
XNetY = Zeros

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

StartX=0
StartY=0

video=cv2.VideoWriter(MovVideoOutputName,fourcc,framesPerSecond/divisor,(int(np.size(FramesToVideo,1)+abs(MaxMoveX)),int(abs(MaxMoveY)+np.size(FramesToVideo,0))))
for j in range(int(np.size(XNetX,0)/divisor)-1):
    BlankArray = np.zeros((int(np.size(FramesToVideo,0)+abs(MaxMoveY)),int(abs(MaxMoveX)+np.size(FramesToVideo,1)),3),int)
    
    ZeroAxis = int(StartY + XNetY[int(j*divisor),0])
    OneAxis = int(StartX - XNetX[int(j*divisor),0])
    BlankArray[ZeroAxis:ZeroAxis+np.size(FramesToVideo,0),OneAxis:OneAxis+np.size(FramesToVideo,1)] = FramesToVideo[:,:,:,j]
    ArrowFrame=BlankArray
    frameint=int((j-2)*divisor)
    for i in range(frameint):
        for k in range(np.size(MotionX,1)):
            cv2.circle(ArrowFrame,(int(FinalData[i,1+2*k]),np.size(ArrowFrame,0)-int(FinalData[i,2+2*k])),2,(255,0,0),2)
        cv2.circle(ArrowFrame,(int(FinalDataMag[i,1]),np.size(ArrowFrame,0)-int(FinalDataMag[i,2])),2,(0,0,255),2)
        cv2.circle(ArrowFrame,(int(FinalDataM[i,1]),np.size(ArrowFrame,0)-int(FinalDataM[i,2])),2,(255,0,255),2)
    video.write(np.uint8(BlankArray))
    if j%1==0:
      print('PercentBar2',j/TotalFrames*divisor)
      #print(np.shape(BlankArray))

video.release()
cv2.destroyAllWindows()
XNetX = MNetX
XNetY = MNetY

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

OutputfileName = "Vid10ResultsVideo"
MovVideoOutputName = "{}MovingFrame.mp4".format(OutputfileName)
video=cv2.VideoWriter(MovVideoOutputName,fourcc,framesPerSecond/divisor,(int(np.size(FramesToVideo,1)+abs(MaxMoveX)+35),int(abs(MaxMoveY)+35+np.size(FramesToVideo,0))))

for j in range(int(np.size(XNetX,0)/divisor)+1):
    
    BlankArray = np.zeros((int(np.size(FramesToVideo,0)+abs(MaxMoveY)+35),int(abs(MaxMoveX)+35+np.size(FramesToVideo,1)),3),int)
    #print(np.shape(BlankArray))
    ZeroAxis = int(StartY + XNetY[int(j*divisor),0])
    OneAxis = int(StartX - XNetX[int(j*divisor),0])
    ArrowFrame = np.ascontiguousarray(FramesToVideo[:,:,:,j],dtype=np.uint8)
    i=int((j-2)*divisor)
    for k in range(np.size(MotionX,1)):
        cv2.circle(ArrowFrame,(int(FinalData[0,1+2*k]+FNetX[i,0]),np.size(ArrowFrame,0)-(int(FinalData[0,2+2*k]+FNetY[i,0]))),15,(255,0,0),2)
    cv2.circle(ArrowFrame,(int(FinalDataMag[0,1]+MotionMagX[i,0]),np.size(ArrowFrame,0)-(int(FinalDataMag[0,2]+MotionMagY[i,0]))),15,(0,0,255),2)
    cv2.circle(ArrowFrame,(int(FinalDataM[0,1]+MotionMagX[i,0]),np.size(ArrowFrame,0)-(int(FinalDataM[0,2]+MotionMagY[i,0]))),15,(255,0,255),2)

    BlankArray[ZeroAxis:ZeroAxis+np.size(FramesToVideo,0),OneAxis:OneAxis+np.size(FramesToVideo,1)] =ArrowFrame
    video.write(np.uint8(BlankArray))
    if j%10==0:
      print('PercentBar2',j/TotalFrames*divisor)
      print(np.shape(BlankArray))

video.release()
cv2.destroyAllWindows()
Multiplier=.5


vals = np.linspace(-3.14159,3.14159,180)
r=150
circleX = r*np.sin(vals)
circleY = r*np.cos(vals)



# plt.subplot(2,3,1) 
# vertical_img = np.flip(OutputIMG,0)
# plt.imshow(vertical_img, origin= 'lower')
# #plt.imshow(OutputIMG)
# for ii in range (np.size(MotionX,1)):
#     #plt.scatter(FinalData[15,1+ii*2],FinalData[15,2+2*ii], 100, color = [128/255,0,128/255])
#     plt.scatter(FinalData[1,1+ii*2],FinalData[1,2+2*ii], 15, color = 'r', alpha = 0.75)
    
# plt.plot(circleX+FinalDataM[15,1],circleY+FinalDataM[15,2],[0.25,0.25,0.25], alpha = .75)
# plt.ylim([300,900])
# plt.axis('off')

#for ii in range(3):
    #plt.arrow(FinalData[1,1+ii*2],FinalData[1,2+2*ii], Multiplier*MotionX[EndPoint-1,ii], Multiplier*MotionY[EndPoint-1,ii], width = 6, color = [204/255,85/255,0])




reds =np.linspace(0,.75,np.size(MotionX,1))
Darkness = np.linspace(.5,1,EndPoint)
# plt.subplot(2,3,3) 
# vertical_img = np.flip(FramesToVideo[:,:,:,200],0)
# plt.imshow(vertical_img, origin= 'lower')
# #plt.imshow(OutputIMG)
# for ii in range (np.size(MotionX,1)):
#     plt.scatter(FinalData[1,1+ii*2]+FNetX[1000,ii],FinalData[1,2+2*ii]+FNetY[1000,ii], 15, color = 'r', alpha = 0.75)
#     #plt.arrow(FinalData[15,1+ii*2],FinalData[15,2+ii*2],MotionX[1500,ii], MotionY[1500,ii], width=4, color = 'r',alpha =0.5)
#     plt.plot([FinalData[1000,1+ii*2],FinalData[1,1+ii*2]+FNetX[1000,ii]],[FinalData[1000,2+2*ii],FinalData[1,2+ii*2]+FNetY[1000,ii]], 50,  color = [128/255,0,128/255],alpha =0.5)
    
# plt.ylim([300,900])
# plt.plot(circleX+FinalDataM[1000,1],circleY+FinalDataM[1000,2],[0.25,0.25,0.25], alpha = .5)
# plt.axis('off')

plt.rcParams.update({'font.size': 10})
fig, axs = plt.subplots(1,3)
fig.set_figwidth(3.5)
fig.tight_layout()

# vertical_img = np.flip(FramesToVideo[:,:,:,350],0)
# plt.imshow(vertical_img, origin= 'lower')
#plt.imshow(OutputIMG)
ii=0
colors=np.ones((EndPoint,3))*[1,0,0]
colors[HoldGood[:,0]==0,:]=[128/255,0.5,128/255]
convert =2.31673
#colors = (colors.T*Darkness).T
Darkness2=np.repeat(np.reshape(Darkness,[-1,1]),3,1)
axs[0].scatter(MotionX[:,ii]/convert,MotionY[:,ii]/convert, 1, color = [128/255,0,128/255], alpha = 0.25)
  
axs[0].scatter(FNetX[:EndPoint,ii]/convert,FNetY[:EndPoint,ii]/convert, 1, color = colors, alpha = 0.25)
  #plt.arrow(FinalData[15,1+ii*2],FinalData[15,2+ii*2],MotionX[1750,ii], MotionY[1750,ii], width=4, color = 'r',alpha =0.5)
    #plt.arrow(FinalData[15,1+ii*2],FinalData[15,2+ii*2],FNetX[1750,ii], FNetY[1750,ii], width=4, color = 'b',alpha =0.5,linestyle='--')
    #plt.plot([FinalData[2000,1+ii*2],FinalData[1,1+ii*2]+FNetX[2000,ii]],[FinalData[2000,2+2*ii],FinalData[1,2+ii*2]+FNetY[2000,ii]], 50,  color = [128/255,0,128/255],alpha =0.5)
    
#plt.plot(circleX+FinalDataM[1750,1],circleY+FinalDataM[1750,2],[0.25,0.25,0.25], alpha = .5)
axs[0].set(xlim=(-15,200), ylim=(-100,15))
axs[0].set_aspect('equal', 'box')
axs[0].set_ylabel('$\Delta$ y [$\mu$m]')
axs[0].set_xlabel('$\Delta$ x [$\mu$m]')


plt.subplot(1,3,2) 
ii=4
colors=np.ones((EndPoint,3))*[1,0,0]
colors[HoldGood[:,4]==0,:]=[128/255,0.5,128/255]
axs[1].scatter(MotionX[:,ii]/convert,MotionY[:,ii]/convert, 1, color = [128/255,0,128/255], alpha = 0.25)
  
axs[1].scatter(FNetX[:EndPoint,ii]/convert,FNetY[:EndPoint,ii]/convert, 1, color = colors, alpha = 0.25)
  #plt.arrow(FinalData[15,1+ii*2],FinalData[15,2+ii*2],MotionX[1750,ii], MotionY[1750,ii], width=4, color = 'r',alpha =0.5)
    #plt.arrow(FinalData[15,1+ii*2],FinalData[15,2+ii*2],FNetX[1750,ii], FNetY[1750,ii], width=4, color = 'b',alpha =0.5,linestyle='--')
    #plt.plot([FinalData[2000,1+ii*2],FinalData[1,1+ii*2]+FNetX[2000,ii]],[FinalData[2000,2+2*ii],FinalData[1,2+ii*2]+FNetY[2000,ii]], 50,  color = [128/255,0,128/255],alpha =0.5)
    
#plt.plot(circleX+FinalDataM[1750,1],circleY+FinalDataM[1750,2],[0.25,0.25,0.25], alpha = .5)
axs[1].set(xlim=(-15,200), ylim=(-100,15))
axs[1].set_aspect('equal', 'box')
#axs[1].set_ylabel('$\Delta$ y [$\mu$m]')
axs[1].set_xlabel('$\Delta$ x [$\mu$m]')


plt.subplot(1,3,3) 
ii=6
colors=np.ones((EndPoint,3))*[1,0,0]
colors[HoldGood[:,6]==0,:]=[128/255,0.5,128/255]
axs[2].scatter(MotionX[:,ii]/convert,MotionY[:,ii]/convert, 1, color = [128/255,0,128/255], alpha = 0.25)
  
axs[2].scatter(FNetX[:EndPoint,ii]/convert,FNetY[:EndPoint,ii]/convert, 1, color = colors, alpha = 0.25)
  #plt.arrow(FinalData[15,1+ii*2],FinalData[15,2+ii*2],MotionX[1750,ii], MotionY[1750,ii], width=4, color = 'r',alpha =0.5)
    #plt.arrow(FinalData[15,1+ii*2],FinalData[15,2+ii*2],FNetX[1750,ii], FNetY[1750,ii], width=4, color = 'b',alpha =0.5,linestyle='--')
    #plt.plot([FinalData[2000,1+ii*2],FinalData[1,1+ii*2]+FNetX[2000,ii]],[FinalData[2000,2+2*ii],FinalData[1,2+ii*2]+FNetY[2000,ii]], 50,  color = [128/255,0,128/255],alpha =0.5)
    
#plt.plot(circleX+FinalDataM[1750,1],circleY+FinalDataM[1750,2],[0.25,0.25,0.25], alpha = .5)
axs[2].set(xlim=(-15,200), ylim=(-100,15))
axs[2].set_aspect('equal', 'box')
#axs[2].set_ylabel('$\Delta$ y [$\mu$m]')
axs[2].set_xlabel('$\Delta$ x [$\mu$m]')

#plt.arrow(FinalData[15,1+ii*2],FinalData[15,2+ii*2],MotionX[1750,ii], MotionY[1750,ii], width=4, color = 'r',alpha =0.5)
#plt.arrow(FinalData[15,1+ii*2],FinalData[15,2+ii*2],FNetX[1750,ii], FNetY[1750,ii], width=4, color = 'b',alpha =0.5,linestyle='--')
#plt.plot([FinalData[2000,1+ii*2],FinalData[1,1+ii*2]+FNetX[2000,ii]],[FinalData[2000,2+2*ii],FinalData[1,2+ii*2]+FNetY[2000,ii]], 50,  color = [128/255,0,128/255],alpha =0.5)



# plt.subplot(3,3,4) 
# for ii in range (1):
#     plt.scatter(FinalData[1,1+ii*2]+FNetX[:EndPoint,ii],FinalData[1,2+2*ii]+FNetY[:EndPoint,ii], 15, color = 'r', alpha = 0.25)
#     plt.scatter(FinalData[:EndPoint,1+ii*2],FinalData[:EndPoint,2+2*ii], 15, color = [128/255,0,128/255], alpha = 0.25)
#     #plt.arrow(FinalData[15,1+ii*2],FinalData[15,2+ii*2],MotionX[1750,ii], MotionY[1750,ii], width=4, color = 'r',alpha =0.5)
#     #plt.arrow(FinalData[15,1+ii*2],FinalData[15,2+ii*2],FNetX[1750,ii], FNetY[1750,ii], width=4, color = 'b',alpha =0.5,linestyle='--')
#     #plt.plot([FinalData[2000,1+ii*2],FinalData[1,1+ii*2]+FNetX[2000,ii]],[FinalData[2000,2+2*ii],FinalData[1,2+ii*2]+FNetY[2000,ii]], 50,  color = [128/255,0,128/255],alpha =0.5)
    
# plt.plot(circleX+FinalDataM[1750,1],circleY+FinalDataM[1750,2],[0.25,0.25,0.25], alpha = .5)
# plt.axis('off')
# plt.axis("equal")
plt.figure()

plt.figure()
vertical_img = np.flip(FramesToVideo[:,:,:,350],0)
plt.imshow(vertical_img, origin= 'lower')
#plt.imshow(OutputIMG)
for ii in range (np.size(MotionX,1)):
    plt.plot([FinalData[1750,1+ii*2],FinalData[15,1+ii*2]+FNetX[1750,ii]-FNetX[15,ii]],[FinalData[1750,2+2*ii],FinalData[15,2+2*ii]+FNetY[1750,ii]-FNetY[15,ii]], 150,  color = 'b', alpha =.75)
    plt.scatter(FinalData[15,1+ii*2]+FNetX[1750,ii]-FNetX[15,ii],FinalData[15,2+2*ii]+FNetY[1750,ii]-FNetY[15,ii], 45, color = [128/255,0,128/255])
    plt.scatter(FinalData[1750,1+ii*2],FinalData[1750,2+2*ii], 45, color = [1,0, 0], alpha = 0.25)
    #plt.arrow(FinalData[15,1+ii*2],FinalData[15,2+ii*2],MotionX[1750,ii], MotionY[1750,ii], width=4, color = 'r',alpha =0.5)
    #plt.arrow(FinalData[1750,1+ii*2],FinalData[1750,2+ii*2],FNetX[1750,ii]-MotionX[1750,ii], FNetY[1750,ii]-MotionY[1750,ii], width=4, color = [1,0,0],alpha =0.75)
    
    
plt.plot(circleX+FinalDataM[1750,1],circleY+FinalDataM[1750,2],color=[0.5,0.5,0.5])
plt.axis('off')
plt.ylim([250,850])

# plt.subplot(3,1,4)
# Multiplier=1
# z = np.average(Motion2[XMinSwim:XMaxSwim,:],0)
# for ii in range (np.size(MotionX,1)):
#     colorVal = [1,0,0]
#     plt.scatter(FinalData[:EndPoint,0],np.arctan2(MotionX[:EndPoint,ii], MotionY[:EndPoint,ii])*180/3.14159, 1, color = colorVal)

# for ii in range(3):
#     plt.scatter(FinalData[:EndPoint,0],np.arctan2(MotionX[:EndPoint,ii], MotionY[:EndPoint,ii])*180/3.14159, 1, color =[0.5, .5, .5])
# plt.scatter(FinalData[:EndPoint,0],np.arctan2(FNetX[:EndPoint,ii], FNetY[:EndPoint,ii])*180/3.14159, 1, color = [128/255,0,128/255])    
# plt.axis('on')

# plt.ylim([100,150])
# plt.xlabel("time [s]")
# plt.ylabel ("Direction [deg]")

# plt.subplot(2,3,(5))
# for ii in range (np.size(MotionX,1)):
#     plt.scatter(FinalData[:EndPoint,0],Motion2[:,ii],1, color='r')

# for ii in range (3):
#     plt.scatter(FinalData[:EndPoint,0],Motion2[:,ii],1, color=[0.5,0.5,0.5])
# FNet = np.sqrt(FNetX**2+FNetY**2)
# plt.scatter(FinalData[:EndPoint,0],FNet[:,ii],1, color=[128/255,0,128/255])

# plt.xlabel("Time [s]")
# plt.ylabel("Motion [pixels]")
plt.rcParams.update({'font.size': 10})
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(3.25,1.75))
ii = 0
Error = np.sqrt((MotionX[:,ii]/convert-FNetX[:EndPoint,ii]/convert)**2+(MotionY[:,ii]/convert-FNetY[:EndPoint,ii]/convert)**2)
TotalDist = np.sqrt((MotionX[:,ii]/convert)**2+(MotionY[:,ii]/convert)**2)
TotalDist[TotalDist<10]=10
PercentError = Error/TotalDist
ax2=axes[0].twinx()
axes[0].scatter(FieldData[:EndPoint,0]/1000,Error, 1, color =  [83/250,19/250,211/250])
axes[0].set_ylabel("Error from Estimate [$\mu$m]", color =[83/250,19/250,211/250])
axes[0].set_xlabel("Time [s]")
axes[0].tick_params(axis="y", labelcolor=[83/250,19/250,211/250])
ax2.scatter(FieldData[:EndPoint,0]/1000,DistTot[:EndPoint,ii]/convert, 1, color = 'k')
ax2.set_ylabel('$Distance from Swimmer$ [$\mu$m]',color ='k')
ax2.tick_params(axis="y", labelcolor="k")
ax2.axhline(y=150, color=[0.5,0.5,0.5], linestyle='--')
ax2.set_ylim([0,270])
axes[0].set_ylim([0,50])

ii = 4
Error = np.sqrt((MotionX[:,ii]/convert-FNetX[:EndPoint,ii]/convert)**2+(MotionY[:,ii]/convert-FNetY[:EndPoint,ii]/convert)**2)
TotalDist = np.sqrt((MotionX[:,ii]/convert)**2+(MotionY[:,ii]/convert)**2)
TotalDist[TotalDist<10]=10
PercentError = Error/TotalDist
ax4=axes[1].twinx()
axes[1].scatter(FieldData[:EndPoint,0]/1000,Error, 1, color =  [83/250,19/250,211/250])
axes[1].set_ylabel("Error from Estimate [$\mu$m]", color =[83/250,19/250,211/250])
axes[1].set_xlabel("Time [s]")
ax4.set_ylim([0,270])
ax4.scatter(FieldData[:EndPoint,0]/1000,DistTot[:EndPoint,ii]/convert, 1, color = 'k')
ax4.set_ylabel('$Distance from Swimmer$ [$\mu$m]',color ='k')
ax4.tick_params(axis="y", labelcolor="k")
ax4.axhline(y=150, color=[0.5,0.5,0.5], linestyle='--')
axes[1].set_ylim([0,50])

plt.figure()
vertical_img = np.flip(FramesToVideo[:,:,:,350],0)
plt.imshow(vertical_img, origin= 'lower')
#plt.imshow(OutputIMG)
for ii in range (np.size(MotionX,1)):
    #plt.plot([FinalData[1750,1+ii*2],FinalData[15,1+ii*2]+FNetX[1750,ii]-FNetX[15,ii]],[FinalData[1750,2+2*ii],FinalData[15,2+2*ii]+FNetY[1750,ii]-FNetY[15,ii]], 150,  color = 'b', alpha =.75)
    #plt.scatter(FinalData[15,1+ii*2]+FNetX[1750,ii]-FNetX[15,ii],FinalData[15,2+2*ii]+FNetY[1750,ii]-FNetY[15,ii], 45, color = [128/255,0,128/255])
    #plt.scatter(FinalData[1750,1+ii*2],FinalData[1750,2+2*ii], 45, color = [1,0, 0], alpha = 0.25)
    #plt.arrow(FinalData[15,1+ii*2],FinalData[15,2+ii*2],MotionX[1750,ii], MotionY[1750,ii], width=4, color = 'r',alpha =0.5)
    plt.arrow(FinalData[1750,1+ii*2],FinalData[1750,2+ii*2],FNetX[1750,ii]-MotionX[1750,ii], FNetY[1750,ii]-MotionY[1750,ii], width=4, color = [83/250,19/250,211/250],alpha =0.75, hatch='X',length_includes_head=True)
plt.plot(circleX+FinalDataM[1750,1],circleY+FinalDataM[1750,2],color=[0.5,0.5,0.5])    
    
#plt.plot(circleX+FinalDataM[1750,1],circleY+FinalDataM[1750,2],color=[0.5,0.5,0.5])
plt.axis('off')

  #plt.arrow(FinalData[15,1+ii*2],FinalData[15,2+ii*2],MotionX[1750,ii], MotionY[1750,ii], width=4, color = 'r',alpha =0.5)
    #plt.arrow(FinalData[15,1+ii*2],FinalData[15,2+ii*2],FNetX[1750,ii], FNetY[1750,ii], width=4, color = 'b',alpha =0.5,linestyle='--')
    #plt.plot([FinalData[2000,1+ii*2],FinalData[1,1+ii*2]+FNetX[2000,ii]],[FinalData[2000,2+2*ii],FinalData[1,2+ii*2]+FNetY[2000,ii]], 50,  color = [128/255,0,128/255],alpha =0.5)
    
#plt.plot(circleX+FinalDataM[1750,1],circleY+FinalDataM[1750,2],[0.25,0.25,0.25], alpha = .5)
axs[0].set(xlim=(-15,200), ylim=(-100,15))
axs[0].set_aspect('equal', 'box')
axs[0].set_ylabel('$\Delta$ y [$\mu$m]')
axs[0].set_xlabel('$\Delta$ x [$\mu$m]')


plt.subplot(1,3,2) 
ii=4
colors=np.ones((EndPoint,3))*[1,0,0]
colors[HoldGood[:,4]==0,:]=[128/255,0.5,128/255]
axs[1].scatter(MotionX[:,ii]/convert,MotionY[:,ii]/convert, 1, color = [128/255,0,128/255], alpha = 0.25)
  
axs[1].scatter(FNetX[:EndPoint,ii]/convert,FNetY[:EndPoint,ii]/convert, 1, color = colors, alpha = 0.25)
  #plt.arrow(FinalData[15,1+ii*2],FinalData[15,2+ii*2],MotionX[1750,ii], MotionY[1750,ii], width=4, color = 'r',alpha =0.5)
    #plt.arrow(FinalData[15,1+ii*2],FinalData[15,2+ii*2],FNetX[1750,ii], FNetY[1750,ii], width=4, color = 'b',alpha =0.5,linestyle='--')
    #plt.plot([FinalData[2000,1+ii*2],FinalData[1,1+ii*2]+FNetX[2000,ii]],[FinalData[2000,2+2*ii],FinalData[1,2+ii*2]+FNetY[2000,ii]], 50,  color = [128/255,0,128/255],alpha =0.5)
    
#plt.plot(circleX+FinalDataM[1750,1],circleY+FinalDataM[1750,2],[0.25,0.25,0.25], alpha = .5)
axs[1].set(xlim=(-15,200), ylim=(-100,15))
axs[1].set_aspect('equal', 'box')
#axs[1].set_ylabel('$\Delta$ y [$\mu$m]')
axs[1].set_xlabel('$\Delta$ x [$\mu$m]')


plt.rcParams.update({'font.size': 10})
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(3.25,3.25))
plt.subplot(3,1,1)
plt.plot(FieldData[:EndPoint,0]/1000,FieldData[:EndPoint,1])
plt.xlabel('Time [ms]')
plt.ylabel('Magnetic Field [deg]')

plt.subplot(3,1,2)
colorG=np.ones((2,3))*[0,1,0]
colorG[1,:]=colorG[1,:]*.5
Regs = np.mean(np.sqrt(MotionY[:,:]**2+MotionX[:,:]**2),1)
RegsX = np.mean(MotionX[:,:],1)
RegsY = np.mean(MotionY[:,:],1)
RegStd = np.std(np.sqrt(MotionY[:,:]**2+MotionX[:,:]**2),1)
Mags = np.sqrt(MotionMagY**2+MotionMagX**2)
for ii in range(1):
    plt.plot(FinalData[:EndPoint,0]/1000,Mags[:,ii]/convert,color=colorG[ii,:])
plt.plot(FinalData[:EndPoint,0]/1000,Regs/convert,'r')
plt.plot(FinalData[:EndPoint,0]/1000,np.sqrt(MotionMY**2+MotionMX**2)/convert,'b')
plt.fill_between(np.linspace(0,97.36/1000,EndPoint), (Regs-RegStd).flatten()/convert, (Regs+RegStd).flatten()/convert, color=[1,.25,.25], alpha=.5)
plt.xlabel('Time [ms]')
plt.ylabel('$\Delta$ [$\mu$m]')


plt.subplot(3,1,3)
vertical_img = np.flip(FramesToVideo[:,:,:,1],0)
plt.imshow(vertical_img, origin= 'lower')
plt.arrow(FinalData[2,1],FinalData[2,2],MotionX[500,0],MotionY[500,0],color='r')
plt.arrow(FinalData[500,1],FinalData[500,2],MotionX[EndPoint-1,0]-MotionX[500,0],MotionY[EndPoint-1,0]-MotionY[500,0],color='r')
plt.arrow(FinalDataM[2,1],FinalDataM[2,2],MotionMX[500,0],MotionMY[500,0],color='b')
plt.arrow(FinalDataM[500,1],FinalDataM[500,2],MotionMX[EndPoint-1,0]-MotionMX[500,0],MotionMY[EndPoint-1,0]-MotionMY[500,0],color='b')
plt.arrow(FinalDataMag[2,1],FinalDataMag[2,2],MotionMagX[500,0],MotionMagY[500,0],color='g')
plt.arrow(FinalDataMag[500,1],FinalDataMag[500,2],MotionMagX[EndPoint-1,0]-MotionMagX[500,0],MotionMagY[EndPoint-1,0]-MotionMagY[500,0],color='g')

reds =np.linspace(0.25,1,np.size(MotionX,1))
fig=plt.figure()
fig.set_figwidth(3.5)
fig.tight_layout()
PercentErrorAll=np.zeros((EndPoint,np.size(MotionX,1)))
for ii in range (np.size(MotionX,1)):
    colors=np.ones((EndPoint,3))*[reds[ii],0,0]
    colors[HoldGood[:,ii]==0,:]=[128/255,0.5,128/255]
    Error = np.sqrt((FNetX[:EndPoint,ii]-MotionX[:EndPoint,ii])**2+(FNetY[:EndPoint,ii]-MotionY[:EndPoint,ii])**2)
    TotalMotion = np.sqrt(MotionX[:EndPoint,ii]**2+MotionY[:EndPoint,ii]**2)
    TotalMotion[TotalMotion<0.5]=0.5
    PercentErrorAll[:,ii]=Error/TotalMotion*100

 
GoodError = np.zeros((EndPoint,1)) 
Std = np.zeros((EndPoint,1)) 
for jj in range(EndPoint):
    GoodError[jj] = np.mean(PercentErrorAll[jj,HoldGood[jj,:]==1])
    Std[jj] = np.std(PercentErrorAll[jj,HoldGood[jj,:]==1])
    plt.scatter(FinalData[jj,0]/1000,GoodError[jj],.25,color='r')
Std[Std<1]=1
plt.fill_between(np.linspace(0,87683.2814/1000,EndPoint), (GoodError-Std).flatten(), (GoodError+Std).flatten(), color=[1,.25,.25], alpha=.5)

for jj in range(EndPoint):
    BadError = np.mean(PercentErrorAll[jj,HoldGood[jj,:]==0])
    
    plt.scatter(FinalData[jj,0]/1000,BadError,.25,color=[0.5, 0.5, 0.5])
Std[Std<1]=1
plt.axvline(x = 73397.5676/1000, color = 'k', label = 'axvline - full height')

plt.ylim([0,100])

#plt.scatter(FinalData[:EndPoint,0],0,2, color=[128/255,0,128/255])

plt.xlabel("Time [s]")
plt.ylabel("Percent Error [%]")

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7,5))
plt.subplot(2,2,1)
Ms = np.sqrt(MotionMY**2+MotionMX**2)
Mags = np.sqrt(MotionMagY**2+MotionMagX**2)
Dir = np.arctan2(MotionMY-np.roll(MotionMY,10),MotionMX-np.roll(MotionMX,10))*180/3.14159
#Dir[Dir<-90]=Dir[Dir<-90]+360
#ax6=axes[3].twinx()
Diff2 = np.sqrt((MotionMY.flatten()-MotionMagY.flatten())**2+(MotionMX.flatten()-MotionMagX.flatten())**2)
plt.ylim([0,60])

plt.plot(FinalData[:707,0],Diff2[:707]/convert,color = [.443,0,.467], linewidth=0.5)
plt.plot(FinalData[707:EndPoint,0].flatten(),Diff2[707:EndPoint]/convert,color = [206.25/255,28.75/255,208.75/255] ,linewidth=0.5)

slope, intercept, r, p, se =scipy.stats.linregress(FinalData[707:EndPoint,0].flatten(), (Diff2[707:]/convert).flatten())
slope2, intercept2, r2, p2, se2 =scipy.stats.linregress(MotionMX[707:EndPoint,0].flatten(), (MotionMY[707:]).flatten())
slope3, intercept3, r3, p3, se3 =scipy.stats.linregress(MotionMagX[707:EndPoint,0].flatten(), (MotionMagY[707:]).flatten())
slope4, intercept4, r4, p4, se4 =scipy.stats.linregress(MotionMX[:707].flatten(), (MotionMY[:707]).flatten())
slope5, intercept5, r5, p5, se5 =scipy.stats.linregress(MotionMagX[:707].flatten(), (MotionMagY[:707]).flatten())

#axes[1].fill_between(np.linspace(0,87683.2814/1000,EndPoint), (Regs-RegStd).flatten()/convert, (Regs+RegStd).flatten()/convert, color = [83/250,19/250,211/250], alpha=.75)
plt.xlabel('Time [s]')
plt.ylabel('$\Delta_{swim}$ [$\mu$m]',color = [.443,0,.467])
plt.tick_params(axis="y", labelcolor=[.443,0,.467])
plt.xlim([0,80])
plt.rcParams.update({'font.size': 10})

plt.subplot(2,2,3)

plt.plot(FieldData[:,0],FieldData[:,1]*180/3.14159,color='k',linewidth=0.5)
plt.xlabel('Time [s]')
plt.ylabel('Magnetic Field \n Orientation [$^\circ$]')
plt.xlim([0,80])

plt.subplot(2,2,(2,4))
plt.scatter(MotionX[:707,1],MotionY[:707,1],.5,color='b')
plt.scatter(MotionX[707:,1],MotionY[707:,1],.5,color='b')
plt.scatter(MotionMagX[:707,0],MotionMagY[:707,0],.5,color='r')
plt.scatter(MotionMagX[707:,0],MotionMagY[707:,0],.5,color='r')
plt.scatter(MotionMX[:707,0],MotionMY[:707,0],.5,color=[.443,0,.467])
plt.scatter(MotionMX[707:,0],MotionMY[707:,0],.5,color=[206.25/255,28.75/255,208.75/255])
plt.ylabel("$\Delta$ y")
plt.xlabel("$\Delta$ x")

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(7,6.5))
plt.subplot(1,4,1)
vertical_img = np.flip(FramesToVideo[:,:,:,1],0)
plt.imshow(vertical_img, origin= 'lower')
plt.ylim([250,1000])
plt.axis('off')
plt.scatter(FinalData[1,1::2],FinalData[1,2::2], 50, color = 'b', alpha = 0.25)
plt.scatter(FinalDataMag[1,1::2],FinalDataMag[1,2::2], 50, color = 'r', alpha = 0.25)
plt.scatter(FinalDataM[1,1::2],FinalDataM[1,2::2], 50, color = [165/255,23/255,167/255], alpha = 0.25)

plt.subplot(1,4,2)
vertical_img = np.flip(FramesToVideo[:,:,:,139],0)
plt.imshow(vertical_img, origin= 'lower')
plt.ylim([250,1000])
plt.axis('off')
plt.scatter(FinalData[1,1::2]+FNetX[694,:],FinalData[1,2::2]+FNetY[694,:], 50, color = 'b', alpha = 0.25)
plt.scatter(FinalDataMag[1,1::2]+MotionMagX[694,:],FinalDataMag[1,2::2]+MotionMagY[694,:], 50, color = 'r', alpha = 0.25)
plt.scatter(FinalDataM[1,1::2]+MotionMagX[694,:],FinalDataM[1,2::2]+MotionMagY[694,:], 50, color = [165/255,23/255,167/255], alpha = 0.25)
#plt.scatter(FinalDataMag[1,1::2],FinalDataMag[1,2::2], 50, color = 'r', alpha = 0.25)
#plt.scatter(FinalDataM[1,1::2],FinalDataM[1,2::2], 50, color = [165/255,23/255,167/255], alpha = 0.25)


plt.subplot(1,4,3)
vertical_img = np.flip(FramesToVideo[:,:,:,286],0)
plt.imshow(vertical_img, origin= 'lower')
plt.ylim([250,1000])
plt.axis('off')
plt.scatter(FinalData[1,1::2]+FNetX[1429,:],FinalData[1,2::2]+FNetY[1429,:], 50, color = 'b', alpha = 0.25)
plt.scatter(FinalDataMag[1,1::2]+MotionMagX[1429,:],FinalDataMag[1,2::2]+MotionMagY[1429,:], 50, color = 'r', alpha = 0.25)
plt.scatter(FinalDataM[1,1::2]+MotionMagX[1429,:],FinalDataM[1,2::2]+MotionMagY[1429,:], 50, color = [165/255,23/255,167/255], alpha = 0.25)



plt.subplot(1,4,4)
vertical_img = np.flip(FramesToVideo[:,:,:,429],0)
plt.imshow(vertical_img, origin= 'lower')
plt.scatter(FinalData[1,1::2]+FNetX[2145,:],FinalData[1,2::2]+FNetY[2145,:], 50, color = 'b', alpha = 0.25)
plt.scatter(FinalDataMag[1,1::2]+MotionMagX[2145,:],FinalDataMag[1,2::2]+MotionMagY[2145,:], 50, color = 'r', alpha = 0.25)
plt.scatter(FinalDataM[1,1::2]+MotionMagX[2145,:],FinalDataM[1,2::2]+MotionMagY[2145,:], 50, color = [165/255,23/255,167/255], alpha = 0.25)
plt.ylim([259,1000])
plt.axis('off')

axes[0].set_frame_on(1)


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(3,4.5))
plt.subplot(3,1,2)
plt.scatter(MotionX[:707,1],MotionY[:707,1],.5,color='b')
plt.scatter(MotionX[707:,1],MotionY[707:,1],.5,color='b')
plt.scatter(MotionMagX[:707,0],MotionMagY[:707,0],.5,color='r')
plt.scatter(MotionMagX[707:,0],MotionMagY[707:,0],.5,color='r')
plt.scatter(MotionMX[:707,0],MotionMY[:707,0],.5,color=[.443,0,.467])
plt.scatter(MotionMX[707:,0],MotionMY[707:,0],.5,color=[206.25/255,28.75/255,208.75/255])
plt.ylabel("$\Delta$ y")
plt.xlabel("$\Delta$ x")


plt.subplot(3,1,3)
Ms = np.sqrt(MotionMY**2+MotionMX**2)
Mags = np.sqrt(MotionMagY**2+MotionMagX**2)
Dir = np.arctan2(MotionMY-np.roll(MotionMY,10),MotionMX-np.roll(MotionMX,10))*180/3.14159
#Dir[Dir<-90]=Dir[Dir<-90]+360
#ax6=axes[3].twinx()
plt.plot(FinalData[:707,0],Diff2[:707]/convert,color = [.443,0,.467], linewidth=0.5)
plt.plot(FinalData[707:EndPoint,0].flatten(),Diff2[707:EndPoint]/convert,color = [206.25/255,28.75/255,208.75/255] ,linewidth=0.5)
#axes[1].fill_between(np.linspace(0,87683.2814/1000,EndPoint), (Regs-RegStd).flatten()/convert, (Regs+RegStd).flatten()/convert, color = [83/250,19/250,211/250], alpha=.75)
plt.xlabel('Time [s]')
plt.ylabel('$\Delta$ [$\mu$m]',color = [.443,0,.467])
plt.tick_params(axis="y", labelcolor=[.443,0,.467])

plt.subplot(3,1,1)
vertical_img = np.flip(FramesToVideo[:,:,:,429],0)
plt.imshow(vertical_img, origin= 'lower')
plt.scatter(FinalData[1,1::2]+FNetX[2145,:],FinalData[1,2::2]+FNetY[2145,:], 50, color = 'b', alpha = 0.25)
plt.scatter(FinalDataMag[1,1::2]+MotionMagX[2145,:],FinalDataMag[1,2::2]+MotionMagY[2145,:], 50, color = 'r', alpha = 0.25)
plt.scatter(FinalDataM[1,1::2]+MotionMagX[2145,:],FinalDataM[1,2::2]+MotionMagY[2145,:], 50, color = [165/255,23/255,167/255], alpha = 0.25)
plt.ylim([259,1000])
plt.axis('off')




