# Coding Environment
This document contains the instructions you need to follow in order to run the exercise notebooks. You are free to use the coding IDE or platform of your choice. The two main suggestions are coding locally and using Google Colab.

## Google Colab

Google provides free access to its *Colab* coding platform. Code usually runs slower than it would on your PC, but no installation is needed to start using it. You can access Colab [here](https://colab.research.google.com). Log in with your Google account and upload the notebooks you would like to work on.

Many of our notebooks pull datasets automatically from webservers, but others require you to manually upload a dataset. For those cases, upload your data to Google Drive and add at the top of your notebook:

```
# Connect to Google Colab 
from google.colab import drive  

# This will prompt for authorization to access your Google Drive from Colab. 
drive.mount('/content/drive', force_remount=True)  

# After mounting, you can navigate to a specific folder using the usual UNIX cd command. 
# Replace 'your_folder_path' with the actual path of your folder inside Google Drive. 
folder_path = '/content/drive/MyDrive/cnn_application/'  # Example path  

%cd "$folder_path"
```

## Coding locally 
Coding on your own PC is faster and allows you to use the coding IDE of your choice (e.g. VS Code). We have prepared a Conda environment file that should take care of the necessary package dependencies for this course. To set that up, follow the steps below:

### Installing Anaconda
We are going to use Anaconda to create an environment for DSAIE, so the first step is to install Anaconda.

[Here](https://www.anaconda.com/products/distribution), you can find the installer for your OS.

### Creating an environment from a .yml file

The second step is to create an environment in Anaconda containing all the required packages. For this purpose, we have created an environment file {download}`dsaie.yml<./environment/dsaie.yml>`, which you have to download.
This file tells Anaconda which versions of which python libraries to install.

You can create your DSAIE environment by opening your Anaconda prompt or terminal, and entering
```
conda env create -f dsaie.yml
```
After waiting for a bit to let Anaconda figure out all the dependencies, installation should follow.
You can check that installation was successful by running
```
conda env list
```
and verifying that `dsaie` is in the list.

### Activating the DSAIE environment
Now, all that's left is to activate the environment before using it, which is done by running
```
conda activate dsaie
```
Note that by default, no environment is activated, so if you restart your PC, you need to activate the environment again. 

Deactivating the DSAIE environment is done by running
```
conda deactivate
```

### Installing additional libraries
In case there are some additional libraries that you would like to install, make sure to do this installation via Anaconda! Packages can be installed by running
```
conda activate dsaie
conda install <package name>
```
If you don't know the name of your package (this is not necessarily the same name as the one in the import statement in your notebooks/python scripts), you can find it by searching the package you want in the [Anaconda repo](https://anaconda.org/anaconda/repo).

