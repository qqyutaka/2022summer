# Installing dependencies using vanilla python, pip, and venv

These instructions have not been tested as much as the ones using the Anaconda distribution, but
have been found to work on a Windows computer.  If you find that any modifications are needed to get
them to work (for Windows or other platforms), let us know so we can update these instructions.

1.  Download and install Python 3.9 (3.9.9 was the current version as of this writing).  Although
	3.10 is available, I was unable to get librosa to install successfully under 3.10.  I had the
	installer add Python to my system path.
2.  Open a command prompt and update your system pip, setuptools, and wheel packages (this will
	install wheel if not already present):
    ```
    python -m pip install --upgrade pip setuptools wheel
    ```
3.  If you don't have a folder to store virtual environments in, create one.  This is usually a
	`.venv` folder inside your user's home folder.
4.	Create a new virtual environment using the appropriate path to your `.venv` folder (this command
	gives the virtual environment the name `audiot`, but you can choose another name if desired):
	```
	python -m venv path/to/.venv/audiot
	```
5.	Activate the virtual environment:
	-	If using Linux: 
		```
		path/to/.venv/audiot/Scripts/activate
		```
	-	If using Windows PowerShell: 
		```
		path/to/.venv/audiot/Scripts/Activate.ps1
		```
	-	If using Windows cmd: 
		```
		path/to/.venv/audiot/Scripts/activate.bat
		```
	The name of the activated environment should now show up at the beginning of your command prompt.
6.	Update pip/setuptools/wheel inside the virtual environment (if desired):
	```
	python -m pip install --upgrade pip setuptools wheel
	```
7.	Install packages:
	```
	python -m pip install numpy scipy pysoundfile scikit-learn matplotlib pandas pylint black rope librosa
	```
