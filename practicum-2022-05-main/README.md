# AudioT OMSA Practicum Repository
Contents:
-	[People you can ask for help on Slack](#people-you-can-ask-for-help-on-slack)
-	[Rules for using the repository](#rules-for-using-the-repository)
-	[Coding guidelines](#coding-guidelines)
-	[Python environment setup](#python-environment-setup)
	-	[Installing dependencies using Anaconda](#installing-dependencies-using-anaconda)
	-	[Installing dependencies using Pip](#installing-dependencies-using-pip)
	-	[Installing the repository in "editable" mode](#installing-the-repository-in-"editable"-mode)
-	[Running tests](#running-tests)

# People you can ask for help on Slack
* Guy Germain
* Kevin Perdomo
* Brandon Carroll

# Rules for using the repository
To keep things organized and prevent headaches, please follow the following rules.  If you're
confused at what a rule means (e.g. you don't know what branch is), please check with one of the
people listed above.  If you feel like you have a good reason for going against one of the rules,
also check with one of the people above first.

1. **Do not commit to the `main` branch of the repository.**  That branch is intended to hold code
	we provide that will be useful to everyone.  If you write code you think would be helpful to
	everyone else, check with us on getting it commited into the main branch.

2. **Create your own branch (with your name or the name of your team) to work in.**  In addition to
	using your own branch, we also recommend putting your files inside a subfolder with your name (or
	team name).  This will help keep things more organized and less likely to conflict if you need
	to merge with other people's branches at some point.
	
3. **Do not commit audio or other binary data / files to the repository, especially large files.
	Use the SharePoint drive or S3 buckets instead.**  Git and other version control systems are not
	designed to handle binary files well, so they can quickly bloat the size of the repository.
	Once they're in there, they cannot be removed without risky and complicated version history
	editing (deleting them and committing again won't remove them from the version history).  Note
	that there are a couple audio files in the `test_data` folder that are there for convenience for
	testing and for example code---please do not add any more.
	
4. **Use cross-platform file and folder names (AKA don't use characters that Windows doesn't allow
	in filenames).**   If you're using Linux or a Mac, please avoid using characters like
	`:*?"<>|` in filenames because then no one running Windows will be able to check out your
	branch.

5. **Please give useful commit messages that at least say something about what you changed.**
	Generally, the first line of your commit message should be a very short description of the
	commit (preferably <100 characters).  It should be followed by a blank line, and then additional
	details about the commit as appropriate.
	
# Coding guidelines
We won't try to dictate how you write your code, but please try to make it readable by other people
(e.g. use meaningful variable names, include comments when appropriate, etc).  If you want
recommendations, here are a few from Brandon:

* I generally try to follow 
	[Google's Python Style Guide](https://google.github.io/styleguide/pyguide.html) when coding in
	Python---particularly it's recommendations for how to format docstrings.  I especially try to
	make sure that code that other people will be using has good docstrings.
	
* I use [VSCode](https://code.visualstudio.com/) as my IDE.  It works well for visually debugging 
	Python code and has nice plugins for a variety of things (e.g. if you like Vim's modal editing
	interface, etc).
	
* I like using the [Black](https://black.readthedocs.io/en/stable/) code formatter to format my 
	Python code for me (it also integrates well with VSCode).

# Python environment setup
These setup instructions cover setting up a virtual environment that contains all the dependencies 
needed to use the AudioT code provided in this repository.  You may need to install other
packages/dependencies in addition to these to support your own code.  Note that these instructions 
give the virtual environment the name `audiot`, but you can change this if you'd prefer to give it a
different name.

Since there are multiple different distributions of Python and ways of seting up virtual 
environments, we have separated the instructions for installing dependencies into separate files
(linked below) for each of the supported methods.  If you run into problems and/or find other ways
to set up the environment, let us know so that we can correct and/or expand these instructions.  

Links to environment setup instructions (pick one):
-	[Vanilla Python with pip and venv (last updated for Python 3.9)](README-venv-setup.md)
-	[Anaconda (last updated for Python 3.8)](README-conda-setup.md)


## Installing the repository in "editable" mode
After installing dependencies, it's also a good idea to install the repository you'll be working on
in "editable" (or "development") mode within your virtual environment.  This will help you avoid
issues with Python being able to find both your files and the files we provide in the repository.
Below, we give an explanation of why this is necessary followed by instructions on how to do it.

### Background / explanation
When Python packages are installed, all their source code (or compiled code) files are copied into a
`site-packages` folder associated with the Python interpreter.  This `site-packages` folder is 
automatically included in the Python path, so all the installed packages can be imported into any
Python module without any issues with Python being able to find them.

When developing their own code with multiple Python files / modules, people often run into problems
with Python not being able to find and import their other modules if they're not in the same folder
as the file that is being executed.  Often, people resort to hacking the Python path (adding the 
location of their other modules to `sys.path`) to get around this, but this is generally not a great
solution.  It breaks easily (e.g. when files are moved around) and creates ugly code.

To avoid having to hack the Python path, it would be nice to just install all your modules as a 
Python package so that they can easily be imported and used anywhere (just like numpy or any other
packages you have installed).  This also has the benefit that your package will then behave the same
for you as it would for anyone else that installed it into their environment.  However, there's one
major problem with this approach for code that is under active development.  Since the package 
installation process makes a copy of your code in the `site-packages` folder, any changes you make
to the local code will not be reflected in the installed version until you re-run the installation
process to update the code in the `site-packages` folder to the current version.  Most people don't
want to have to re-install a package every time they make a minor change to their code and want to
test it.

To solve the above problem, Python allows you to install packages in "editable" (or "development") 
mode.  When a package is installed in editable mode, its files are not copied into the 
`site-packages` folder.  Instead, a link is placed in the `site-packages` folder that points to the 
original source files in your development directory so that your current source code is what gets
used as the "installed version".  This makes it so that you can import all your modules the same as
if they had been installed normally, but any changes you make to your code will also be immediately
reflected in the installed version without having to re-install.

### Instructions for installing a package in editable mode
1.	Open a command prompt.
2.	Navigate to the root folder of the package (should be the folder containing a `setup.py` file).
3.	Activate the virtual environment that you would like to install the package in (each virtual
	environment contains its own separate Python interpreter and `site-packages` folder).
4.	Run the following command, where the dot refers to the current directory (alternatively, you 
	could run the command from another folder and supply the correct path to the package folder in
	place of the dot):
	```
	pip install -e .
	```

Note that the above pip command works (and is the recommended method) for editable installs in both
Anaconda environments as well as vanilla venv/virtualenv environments.

