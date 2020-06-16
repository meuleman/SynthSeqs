### How to set up development environment using `virtualenvwrapper` - PB20200616

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



### Experiences with getting this code to work - Fall 2019 (outdated)
## Wouter Meuleman -- WM20191006

From the current environment at Altius, which was mostly Tensorflow based, I had to install pyTorch first.
The version recommended by Peter (0.4.1) by default works with CUDA 9.0.x, and unfortunately we only have/had
NVIDIA drivers installed that are compatible with CUDA up to and including 8.0.
I could not immediately figure out how to force pip3 to install a pyTorch variant that worked with CUDA 8.0.
However, conda did seem to have that possibility, but asked for root privileges when trying to install system-wide.

Luckily conda explicitly supports virtual environments, so I created a new virtual environment:
```
conda create -n SynthSeqs pip python=3.5.3
```

Then, activated this new environment:
```
source activate SynthSeqs
```

And then installed pyTorch in that environment, while explicitly requiring CUDA 8.0 hooks:
```
conda --debug install pytorch=0.4.1 cuda80 -c pytorch
```

I then added the `source activate SynthSeqs` line to the SLURM submission script, right before the python3 call.

This did not immediately solve things, I had to install a few more Python packages:
```
conda --debug install six
conda --debug install matplotlib
conda --debug install cycler
```

However, `matplotlib` did not work correctly, as it required the dateutil package.
Installing this via conda, would result in downgrading Python to 2.7, which I wasn't going to do.
Since at this point generating figures is not critical, I commented out all `matplotlib` references from the code.

Lastly, I had to remove `transpose()` from line 63 in `optimize.py`, and 
had to explicitly convert the output file name path to a string.

It now all seems to work -- impressive job Peter!


