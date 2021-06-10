# SynthSeqs: data-driven design of context-specific regulatory elements

This project revolves around a fully data-driven system for designing synthetic regulatory elements.
We apply a large map of 3.5M+ annotated DNase I Hypersensitive Sites (DHSs, Meuleman et al., Nature 2020) as the basis for a training set to 
teach a Generative Adversarial Network (GAN) how regulatory elements are encoded in the human genome sequence.
This approach results in highly variable pools of synthetic sequences, yet never seen before in nature.
We further apply a supervised adaptation strategy to tune these sequences towards pre-defined cellular contexts of interest.


### How to set up the development environment using `virtualenvwrapper`

From a recent overhaul of much of the code in this project, the steps required to run each module are now different and (I'm hoping) more simple.  The general setup procedure is as follows:
- Load the python3.6 module on the Altius server,
- Install `virtualenvwrapper` *locally* using pip,
- Create a new `synth-seqs` virtualenv,
- Install the required python packages in the `synth-seqs` virtualenv using pip,
- Run individual modules by executing `python3 -m MODULE ARGS...` from the command line, where `MODULE` is one of `make_data`, `generator`, `classifier`, or `optimize`. Or run `sbatch submit/submit_MODULE.slurm` when using GPUs.

#### Load the python3.6 module

This project requires python3.6 or higher.  On the Altius server, we can get python3.6 by running: 
```
~$ module load python/3.6.4
```
You can also install python3.6+ locally if that is easier.

#### Install `virtualenvwrapper` locally

[virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) is an extension of `virtualenv` that is extremely convenient for easily handling multiple virtual environments for different python projects.

On the Altius server, run:
```
~$ python3 -m pip install --user virtualenvwrapper
```
to install `virtualenvwrapper` and all of its dependencies locally without needing sudo privileges.  `virtualenvwrapper` will be installed in `/home/{USER}/.local/bin`.
Make sure to execute `source /home/{USER}/.local/bin/virtualenvwrapper.sh` and add this command to your shell login script too (e.g. ~/.bash_profile).

#### Create a new python3.6 `synth-seqs` virtualenv

Now we need to create a fresh virtual environment dedicated to synthetic sequences development.  This virtual environment will hold all of the dependencies needed to run the various components of the project.  We can create this virtualenv using the command `mkvirtualenv`.  Make sure to specify which python you would like to use for your virtualenv with the `-p` flag - I've been using the python3.6 in the `python/3.6.4` module.
```
~$ mkvirtualenv -p /net/module/sw/python/3.6.4/bin/python3.6 synth-seqs
```
The above command creates a new virtualenv named `synth-seqs` that uses python3.6 located at `/net/module/sw/python/3.6.4/bin/python3.6`.  After running this command, you should see the name of your virtualenv at the left of your terminal prompt in parentheses like so:
```
(synth-seqs) pbromley:~$ ...
```
This means that the `synth-seqs` virtualenv is active.  To deactivate the virtualenv, simply run:
```
~$ deactivate
```
To show a list of all of your virtualenvs, run:
```
~$ workon
```
To activate an existing virtualenv, run:
```
~$ workon NAME_OF_VENV
```

#### Install required python packages in virtualenv

First, confirm that your `python3` command uses your virtualenv's python3 by running:
```
~$ which python3
```
You should see the following output:
```
~/.virtualenvs/{NAME_OF_VENV}/bin/python3
```

A running list of python package dependencies lives in `etc/requirements.txt`.  (None of these dependencies are pinned which is probably bad).  Make sure you have your synth seqs virtualenv active, and run the following:
```
~$ python3 -m pip install -r etc/requirements.txt
```
This should install all of the necessary python packages in your virtualenv.

#### Running modules

You can test that everything went smoothly by trying to run the modules.  There are four modules:
- `make_data` generates and preprocesses the numpy DHS sequence datasets,
- `generator` trains a GAN on the full DHS universe,
- `classifier` trains a classifier network on the more "pure" DHSs (can also perform hyperparam sweep,
- `optimize` handles the sequence tuning process.

Each module can be run with:
```
~$ python3 -m MODULE ARGS
```
Most modules don't actually require arguments.  As of now (2020/06/16), the modules are not perfectly set up to be controlled completely by the command line, so most of the tweaking is done in the `__main__.py` of each module.

To run the modules on the GPU, run:
```
sbatch submit/submit_{MODULE_NAME}.sh
```
for a given module.  `make_data` is the only module that does not have the option to run on the GPU.

This is preliminary documentation and development, and a few things that would help clean everything up would be:
- add a test suite to easily test each module, as well as quickly test that dev install worked.
- add a `dev-install.sh` script or a makefile to condense all of these steps.
- add more comprehensive CLI for each module.
- add better error messages for the modules that fail due to their dependency on other modules being run first (e.g., `optimize` relies on having data from `make_data` and trained models from `generator` and `classifier`).


