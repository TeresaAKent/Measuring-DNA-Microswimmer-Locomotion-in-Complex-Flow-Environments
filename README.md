# Measuring-DNA-Microswimmer-Locomotion-in-Complex-Flow-Environments

Authors: Taryn Imamura, Teresa A Kent, Rebecca Taylor and Sarah Bergbreiter

Abstract: Microswimmers are sub-millimeter swimming robots that show potential as a platform for controllable locomotion in applications including targeted cargo delivery and minimally invasive surgery. To be viable for these target applications, microswimmers will eventually need to be able to navigate in environments with dynamic fluid flows and forces. Experimental studies with microswimmers towards this goal
are currently rare because of the difficulty isolating intentional microswimmer locomotion from environment-induced motion. In this work, we present a method for measuring microswimmer locomotion within a complex flow environment using fiducial microspheres. By tracking the particle motion of ferromagnetic and non-magnetic polystyrene fiducial microspheres, we capture the effect of fluid flow and magnetic field gradients on
microswimmer trajectories. We then determine the field-driven translation of these microswimmers relative to fluid flow and demonstrate the effectiveness of this method by illustrating the motion of multiple microswimmers through different flows.

Paper URL: https://arxiv-org.cmu.idm.oclc.org/abs/2412.15152

Please contact Taryn Imamura at tri@andrew.cmu.edu for questions about the paper or the original copies of the videos.

# Code Instruction
Clone this repository:
[git clone https://github.com/TeresaAKent/Measuring-DNA-Microswimmer-Locomotion-in-Complex-Flow-Environments.git](https://github.com/TeresaAKent/Measuring-DNA-Microswimmer-Locomotion-in-Complex-Flow-Environments.git)


## Generating the Figures from the Paper
In the GeneratingThePaperFigures Folder, you will find the tracking data output from the TEMA tracking system for the microswimmer, magnetic fiducial, and non-magnetic fiducial, as well as data for the microswimmer's orientation and the magnetic field's direction. Running the code will reproduce the figures in the paper and some others.

## Analyzing New Data
The Jupyter notebook contains a more descriptive version of the algorithms, designed for implementation on new data. The notebook is demonstrated on trial 10 data but can be run with any of the data sets in the GeneratingThePaperFigures folder. 

## Visualizing the analysis in video form
A copy of the code from the Jupyter notebook, but in a basic Python file, is in this file. 
