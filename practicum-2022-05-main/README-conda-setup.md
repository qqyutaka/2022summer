
# Installing dependencies using Anaconda
To install under an Anaconda environment, you can either use the `environment.yml` file to install
all the dependencies at once, or you can manually install them using the `conda` comand.

## Using the `environment.yml` file from the GitHub repository:
1.	Clone the repository to your machine.
2.	Open an Anaconda prompt and navigate to the root folder of the repository (which should contain
	the `environment.yml` file).
3.	Run the following command to create a new conda virtual environment and install all the packages 
	needed to run the provided code:
	```
	conda env create --name audiot -f environment.yml
	```
4.	If you're going to run anything within the newly created virtual environment, then activate it 
	first:
	```
	conda activate audiot
	```

## Manual environment setup using conda (and pip):
Run all the following commands in an Anaconda prompt (or any shell that has been configured to work
with Anaconda).  These commands give the created environment the name `audiot` (you can change this
if desired).
1.	Create and activate a virtual environment:
	```
	conda create -n audiot python=3.8
	conda activate audiot
	```
2.	Install pysoundfile dependencies, then install pysoundfile via pip (as per directions on its
	site---note that pysoundfile seems to not work if installed via conda):
	```
	conda install cffi numpy
	pip install pysoundfile
	```
3.	Install other packages for audio processing and feature computation:
	```
	conda install scikit-learn pyaudio matplotlib pandas librosa
	```
4.	Install packages I like to have for functionality in VSCode (code linting, formatting, and
	refactoring):
	```
	conda install pylint black rope
	```
