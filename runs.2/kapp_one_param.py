"""
Linked binding of adenine and magnesium (single sites) to an RNA molecule.
"""
__author__ = "Michael J. Harms (harmsm@gmail.com)"
__date__ = "2020-02-14"

import numpy as np
import pandas as pd

def fx_A(logK_app=-3,some_df=None,At=50):
    '''
    Calculates fractional 2AP saturation (theta) given a logged apparent equilibrium
    constant estimate and a dataframe (some_df) with total RNA concentrations in nM 
    and total magnesium concentrations in mM.
    
    Uses single-site hill binding model for RNA binding to 2AP without assuming excess
    RNA or 2AP concentrations. At is always 50 nM.
    '''
    
    K_app = 10**logK_app
    x = some_df.Rna + At + (1 / K_app)
    theta = ( x - np.sqrt((x**2) - 4*some_df.Rna*At) ) / (2*At)

    return theta 

