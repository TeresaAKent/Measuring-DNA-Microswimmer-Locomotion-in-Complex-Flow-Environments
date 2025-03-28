# Measuring-DNA-Microswimmer-Locomotion-in-Complex-Flow-Environments

Authors: Taryn Imamura, Teresa A Kent, Rebecca Taylor and Sarah Bergbreiter

Abstract: Microswimmers are sub-millimeter swimming robots that show potential as a platform for controllable locomotion in applications including targeted cargo delivery and minimally invasive surgery. To be viable for these target applications, microswimmers will eventually need to be able to navigate in environments with dynamic fluid flows and forces. Experimental studies with microswimmers towards this goal
are currently rare because of the difficulty isolating intentional microswimmer locomotion from environment-induced motion. In this work, we present a method for measuring microswimmer locomotion within a complex flow environment using fiducial microspheres. By tracking the particle motion of ferromagnetic and non-magnetic polystyrene fiducial microspheres, we capture the effect of fluid flow and magnetic field gradients on
microswimmer trajectories. We then determine the field-driven translation of these microswimmers relative to fluid flow and demonstrate the effectiveness of this method by illustrating the motion of multiple microswimmers through different flows.

Paper URL: https://arxiv-org.cmu.idm.oclc.org/abs/2412.15152

Please contact Taryn Imamura at tri@andrew.cmu.edu for questions about the paper or the original copies of the videos.

# Features
This code was developed to preform the analysis discussed in the above described paper. This code does the following:
* Estimates the motion caused by fluid flow (non-magnetic fiducials) and the combined fulid flow and magnetic field gradient of the magnetic coils (magnetic fiducials) by considering fiducials of a same type.
* Considers the spacial relationship between the micro-swimmer and the fiducials to quantify the affect the swimmer is having on the relative motion of the fiducials.
* Quantifies the self generated motion of the microswimmer during osscilatory swimming due to the magnetic field.
* Creates an output video which removes (stabilizes) external forces on the micro swimmer so the swimmer motion in the video is the self-generated motion of the swimmer only.
* Creates a video with predicted locations imposed on the video to visualize the accuracy of our (or new) methadologies used to explain fiducial and swimmer motion.

# Code Available
## Understanding the Code Logic
The Jupyter notebook contains a more descriptive version of the algorithms, designed for implementation on new data. Running throught the Juypter notebook we have mark down comments on each of the figures generated. The notebook is demonstrated on trial 10 data but can be run with any of the data sets in the GeneratingThePaperFigures folder. 

## Visualizing the analysis in video form
The code in the PyCodeGeneratesTheVideos folder contains the same functions as the Jupyter notebook, but is divided in a way that eases new analysis. There is also an additional function which creates the output videos seen for this paper.

The write file is broken into three sections. 
1. Change for new data
2. Change for new experimental set up or if you want a different analysis
3. Run the functions

The readme file contains all the same functions as the juypter notebook. 

## Generating the Figures from the Paper
In the GeneratingThePaperFigures Folder, you will find the actual code we used to convert the tracking data output from the TEMA tracking system for the objects (microswimmer posistion, magnetic fiducial, and non-magnetic fiducial, microswimmer's orientation and the magnetic field's direction) into the figures in the paper and some others. It is disorganized and not well commented, they are uploaded for documention, the authors highly recommend you use one of the two prior codes for new data files.

# Code Installation
Clone this repository:
```
git clone https://github.com/TeresaAKent/Measuring-DNA-Microswimmer-Locomotion-in-Complex-Flow-Environments.git
```
Install the required packages (if not already installed)
1. [Matplotlib](https://matplotlib.org/stable/install/index.html)
2. [Numpy](https://numpy.org/install/)
3. [pandas](https://pandas.pydata.org/docs/getting_started/install.html)
4. [scipy](https://scipy.org/install/)

These packages only need to be installed if you are generating videos:

5. [PIL](https://pypi.org/project/pillow/)
6. [Open CV](https://pypi.org/project/opencv-python/)

## Input
In the same folder as the code you wish to run there needs to be the following files in excel format
1. Excel files which gives the tracked posistions of the objects in pixels. The first column should be time, then data columns of the form x0, y0.....xn, yn. Even if you don't have objects of that type you need to create an excel file with a zero array.
  a. Non-Magnetic Fiducials
  b. Magnetic Fiducials
  c. Micro Swimmer
2. Excel file for the orientation of the micro swimmers. The format should be the first column time then the orientation of the swimmers in the columns.
3. Excel file for the magnetic field. The first column should be time then the magnetic field in the second column.
4. Video (not required for the juypter notebook). Should be the same mp4 file as the data generated from the tracking.



# Contributors

Teresa Kent- Responsible for the code and analysis.

Taryn Imamura - Collected the experimental data during trials she deisgned for the microswimmer.

Sarah Bergbreiter - Guided the research

Rebecca Taylor - Guided the research

## Want to Contribute
Please check the data folder which contains all the files (except videos) used for this paper and make sure your changes do not affect the running of any of the three trials in the figure generation folder (the number of non-magnetic fiducials, magnetic fiducials and swimmers varies by test and could be zero). Then feel free to commit to a branch with notes or email TeresaAKent@gmail.com to have your contribution pushed to main.


