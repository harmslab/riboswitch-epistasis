# riboswitch-epistasis
Calculations and analysis files for "Disentangling contact and ensemble
epistasis in a riboswitch" by Wonderlick, Widom and Harms. 

## Data files

+ Our experimental data is in *2AP_corrected.csv*
+ All of our ML fitting results and MCMC samples are in the *all-samples* 
  directory.

## Models

The binding models are implemented in a set of python scripts:
+ 4.5 (four_state_five_param.py)
+ 4.4 (four_state_five_param.py)
+ 3.4 (three_state_four_param.py)
+ 3.3 (three_state_three_param.py)
+ 2.3 (two_state_three_param.py)
+ apparent binding constants, not shown in manuscript (kapp_one_param.py)

## Setting up environment

To set up the environment for reproducing and extending the analyses in the 
paper, create a python environment (>=3.8) with standard scientific
computing libraries: (matplotlib scipy tqdm numpy pandas jupyter-lab). We 
recommend conda. 

The only non-standard dependency is the "likelihood" library. This can be 
installed using the following commands:

```
git clone git@github.com:harmslab/likelihood.git
cd likelihood
python setup.py install
```

## Reproducing fits

All ML and MCMC fits can be reproduced by running: 

```
bash run-all-fits.sh
```

This could take a long time. (For the manuscript, we ran the MCMC samples 
in parallel on a computing cluster). You might want to run each fit with its
own script. 

## Reproducing figures and tables

The analyses shown in the figures and tables can be reproduced using two 
jupyter notebooks:

+ Fig2B-S2-S3_TableS1-S2.ipynb
+ Fig-3-4-5-6.ipynb

Notes are in each notebook. 

