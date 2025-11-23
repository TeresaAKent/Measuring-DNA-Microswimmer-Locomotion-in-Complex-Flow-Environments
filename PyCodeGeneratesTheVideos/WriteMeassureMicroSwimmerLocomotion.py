# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 18:17:48 2025

@author: tkent
"""

import numpy as np
import pandas as pd
from ReadOnlyMeassureMicroSwimmerLocomotion import *
import scipy
from scipy import stats

## This section is the change section
# Give a Unifying Name to the Output Files
# CHANGE THIS or your previous results will be overwritten
OutputfileName = "Demonstration"

## Load Data
# First column of all should be time
# Second column onwards should be data

# Load the Data
## Sometimes the array loads but does not remove the header
## And sometimes TEMA's first data points jump, so we only consider data after the Header Row
#This does not need to be changed ev
HeaderRows = 15

## Load Fiducials 
Locations = "TasrMS2_7HzSine_DULR_Fiducials.xlsx"
FinalData = np.array(pd.read_excel(Locations))
FinalData = FinalData[HeaderRows:,:]

##Load Swimmer Data
Locations = "TasrMS2_7HzSine_DULR_Position.xlsx"
FinalDataM = np.array(pd.read_excel(Locations))
FinalDataM = FinalDataM[HeaderRows:,:]

##Load Magnetic Fiducial Data
Locations = "TasrMS2_7HzSine_DULR_magFiducials.xlsx"
FinalDataMag = np.array(pd.read_excel(Locations))
FinalDataMag = FinalDataMag[HeaderRows:,:]

##Load the data about the Magnetic Field
Locations1 = "DirectionField.xlsx"
FieldData1 = np.array(pd.read_excel(Locations1))
SwimmerOrientation = FieldData1[HeaderRows:,:]

Locations = "TAEPull_MS10_Angle.xlsx"
FieldData = np.array(pd.read_excel(Locations))
FieldData = FieldData[HeaderRows:,:]

# You need to change this value for every file...
# This is the data row where the osscilating field is turned on
# There is no affect on data just a color change indicator
OscillatingFieldStart = 800

# Load the video
videoPath = "TasrMS2_7HzSine_DULR.mp4"


#-----------------------------------------------------------------
## This section is the maybe change section
# This is how many pixels are in each mm (new microscope settings will change this)
convert =2.31673
# This is how many pixels is considered too close to the swimmer for reliable data
ExclusionRadius = 300



# What data point do you want to use as the end of the test
# I chose -10 which is near the end but not quite the last row
pltVal = -10

# This is which method you want to estimate the average motion of the fiducial types
#Options are Mean, ExclusionRemoval, or SpacialGradient
CompAnalMethod = 'ExclusionRemoval'

# The skipFrames is for skipping frames (so as to not make too large a video)
skipFrames = 5

# The circles are overlayed onto the video to indicate...
# where you expect the object to be based on other fiducials motion
# 1 overlays the circles, 0 does not
OverlayCircle=1

# There is one more variable you may want to change that is embeded in the function code
# If you command F XStabilization, you can change which variable you are stabilizing the video to
#-----------------------------------------------------------------
## This section runs functions and should not be changed

# close all previous plot windows
plt.close('all')

# Analyze The Data
IntraAnal, Spacial, AnalCompNoMag, AnalCompMag, SafeFiducials = AnalyzeRawData(FinalData, FinalDataM, FinalDataMag, FieldData, HeaderRows, ExclusionRadius)

# If you want to preserve the analyzed data for later uncomment
# np.save("IntraAnal", IntraAnal)
# np.save("Spacial", Spacial)
# np.save("AnalCompNoMag", AnalCompNoMag)
# np.save("AnalCompMag", AnalCompMag)
# the load command is np.load()

# Visualize the Data

# Figure Setup
# Set the colors for the number of fiducials to keep the color consistent in later figures 
n = np.size(IntraAnal["MotionX"],1)
Mn = np.size(IntraAnal["MotionMagX"],1)
Sn = np.size(IntraAnal["MotionMX"],1)
color = get_cmap_colors('hsv', n)

#Strictly Speaking this function does no Analysis, just modifies data for easier plotting
NetResults=FigArrayPrep(CompAnalMethod, AnalCompNoMag, AnalCompMag, n, Sn)
# This function quantifies the preformance of the chosen Method
PercentageRMSE = CalculateNonMagFidAccuracy(IntraAnal, NetResults)
    
# Plot the total motion of the swimmers (black) and non-magnetic fiducials
PlotTotalMotion(IntraAnal,FieldData)

# Plot the relative motion of the swimmers (black) and non-magnetic fiducials
PlotRelativeToAverageMotion(IntraAnal,NetResults,FieldData)

# Plot showing the total distance traveled
PlotStartEndVsExpectedEndNonMagFid(pltVal,n,color,FinalData, NetResults, FinalDataM)

# Two figures showing the total error rate and how the distance to a simmer affects it
LookAtPercentErrorNonMagFid(n,color,PercentageRMSE,convert, IntraAnal['time'], ExclusionRadius, Spacial)

# Summary data about relative velocity of motion
PlotSummaryMotionWithShading(FieldData,SwimmerOrientation,IntraAnal,OscillatingFieldStart,NetResults,convert, n, Mn, SafeFiducials)

# Summary data of swimmer motion relative to the magnetic fiducail
PlotSwimmerEvaluation(OscillatingFieldStart,FieldData, IntraAnal, convert, n, Mn, NetResults, SafeFiducials)

## The Videos
#Create the Output Name
MovVideoOutputName = "{}MovingFrame.mp4".format(OutputfileName)
BubbleVideoOutputName = "{}BubbleTrail2.mp4".format(OutputfileName)

# The stabilization decides what drives how the frame moves
# MNet Stabilizes to the Magnetic Fiducial
# FNet Stabilizes to the Non-Magnetic Fiducial
# A zero array will have all particles move
XStabilization = NetResults['MNetX']
YStabilization = NetResults['MNetY']

#MakeStabilizeVideo(videoPath,MovVideoOutputName,skipFrames, FinalData, FinalDataMag, FinalDataM, IntraAnal, XStabilization, YStabilization,n, Mn, Sn, OverlayCircle, HeaderRows)
MakeBubblesVideo(videoPath,BubbleVideoOutputName,skipFrames, FinalData, FinalDataMag, FinalDataM, IntraAnal, XStabilization, YStabilization,n, Mn, Sn, OverlayCircle, HeaderRows)