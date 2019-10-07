### Experiences with getting this code to work
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

######################################################################################################################################

