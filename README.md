# SynthSeqs: data-driven design of context-specific regulatory elements

This project revolves around a fully data-driven system for designing synthetic regulatory elements.
We apply a large map of 3.5M+ annotated DNase I Hypersensitive Sites (DHSs, Meuleman et al., Nature 2020) as the basis for a training set to 
teach a Generative Adversarial Network (GAN) how regulatory elements are encoded in the human genome sequence.
This approach results in highly variable pools of synthetic sequences, yet never seen before in Nature.
We further apply a supervised adaptation strategy to tune these sequences towards pre-defined cellular contexts of interest.


### How to set up the SynthSeqs environment using `conda`

The general setup procedure is as follows:

#### Create a new python3.8 `SynthSeqs` virtualenv

We first create a fresh virtual environment dedicated to synthetic sequences development.  
This virtual environment will hold all of the dependencies needed to run the various components of the project.  
```
~$ conda create -n SynthSeqs python=3.8
~$ conda activate SynthSeqs
```
The above command creates a new environment named `SynthSeqs` that uses python3.8, and subsequently activates it.
After running these commands, you should see the name of your virtualenv at the left of your terminal prompt in parentheses like so:
```
(SynthSeqs) username:~$ ...
```
This means that the `SynthSeqs` virtualenv is active.  To deactivate the virtualenv, simply run:
```
~$ deactivate
```

#### Install required python packages in virtualenv

A running list of python package dependencies lives in `etc/requirements.txt`.  
Make sure you have your SynthSeqs environment active, and run the following:
```
~$ pip install -r etc/requirements.txt
```
This should install all of the necessary python packages in your virtualenv.

#### Running modules

The SynthSeqs code base is organized around several Python modules:
- `make_data` generates and preprocesses the numpy DHS sequence datasets,
- `generator` trains a GAN on a large set of DHS sequences,
- `classifier` trains a classifier network on more component-specific and high-confidence DHSs (can also perform hyperparam sweep),
- `optimize` handles the sequence tuning process.

Each module can be run with:
```
~$ python3 -m MODULE ARGS
```
Most modules don't actually require arguments.  
The modules are currently not perfectly set up to be controlled completely by the command line, so most of the tweaking is done in the `__main__.py` of each module.

To run the modules on the GPU, run:
```
sbatch submit/submit_{MODULE_NAME}.slurm
```
for a given module.  
`make_data` is the only module that does not have the option to run on the GPU.


<!--
This is preliminary documentation and development, and a few things that would help clean everything up would be:
- add a test suite to easily test each module, as well as quickly test that dev install worked.
- add a `dev-install.sh` script or a makefile to condense all of these steps.
- add more comprehensive CLI for each module.
- add better error messages for the modules that fail due to their dependency on other modules being run first (e.g., `optimize` relies on having data from `make_data` and trained models from `generator` and `classifier`).
-->


