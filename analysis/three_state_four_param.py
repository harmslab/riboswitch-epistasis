"""
Linked binding of adenine and magnesium (single sites) to an RNA molecule.
"""
__author__ = "Michael J. Harms (harmsm@gmail.com)"
__date__ = "2020-02-14"

import numpy as np
import pandas as pd
from scipy import optimize


def _calc_all_conc(K_dock,K_2AP,K_mg,n_mg,Rt,At,Mt,A):
    """
    Calculate the concentration of all  species given the equilibrium
    constants, total RNA concentration, total adenine concentration, total
    magnesium concentration, and free adenine concentration.
    
    Daria Wonderlick, 2021/04/14
    OneNote S2021, 3state_K_mg
    Solved for M by eliminating M^n 
    Use M to find E
    Use equation for E from Rt-At equation 
    
    Uses K_mg equilibrium constant
    Adenine regression using percentages
    Best_A once calculated R,M,A meet convergence requirements 
    """
    
    ## Solve for E having guessed A
    
    M = Mt - n_mg*(At-A) - n_mg*(At-A)/(K_2AP*A)
    
    if M < 0:
        raise ValueError

    
    # Calcualte M given that we now have A, E
    
    # From Ra: model used in earlier 3 state analyses
    # Variables: Rt,At,K_dock,M,A,n
    E = (Rt - At + A) / (1 + K_dock*((K_mg*M)**n_mg))

    if E < 0:
        raise ValueError
        
    # Calculate all other species
    
    D = K_dock*((K_mg*M)**n_mg)*E 
    DA = K_2AP*A*D

    return E, D, DA, M


def _A_residual(params,K_dock,K_2AP,K_mg,n_mg,Rt,At,Mt):
    """
    A residual function for finding the free value of adenine.  The
    value of free adenine is the fit parameter.  The difference between
    the calculated and total rna, adenine, and magnesium species' concentrations
    is the residual.
    """

    # Calculate the species concentration given our guess for [C]...do you mean [A]?
    A = params[0]
    E, D, DA, M = _calc_all_conc(K_dock,K_2AP,K_mg,n_mg,Rt,At,Mt,A)

    # calculate the total concentrations in calcium and protein
    Rt_calc = E + D + DA
    At_calc = A + DA
    Mt_calc = M + (n_mg*D) + (n_mg*DA)

    """
    r_r = Rt_calc - Rt
    r_a = At_calc - At
    r_m = Mt_calc - Mt

    """
    if Rt == 0:
        r_r = 0
    else:
        r_r = (Rt_calc - Rt)/Rt

    if At == 0:
        r_a = 0
    else:
        r_a = (At_calc - At)/At

    if Mt == 0:
        r_m = 0
    else:
        r_m = (Mt_calc - Mt)/Mt
    

    # Return difference between actual and total
    return np.array([r_r,r_a,r_m])

def _species_conc(K_dock,K_2AP,K_mg,n_mg,
                  Rt,At,Mt,
                  convergence_cutoff=0.01,
                  guess_resolution=0.1,
                  verbose=True):
    """
    Get species of L, H, HM, HA, HMA, A, and M given the equilibrium
    constants and the total protein and calcium concentrations. The equilibrium
    constants and concentrations must be floats.

    Ka: adenine binding constant
    K_mg: magnesium binding constant
    Ks: apo to active constant
    Rt: total RNA concentration
    At: total adenine concentration
    Mt: total magnesium concentration

    convergence_cutoff: percent difference between actual and calculated totals
    guess_resolution: if the first guess doesn't succeed, try another guess that
                      is guess-resolution away
    verbose: if True, record all all residuals and spit out if regression fails.

    returns: L, H, HM, HA, HMA, A, M
    """
    
    # If zero adenine total, we already know A
    if At == 0:
        A = 0.0
    else:

        best_residual = None
        best_A = None
        best_sum_residual = None

        guesses = np.arange(0,1+guess_resolution,guess_resolution)

        successful = False
        residuals = []
        for g in guesses:
            
            guess = At*g

            # Find value for C between 0 and Ct that finds species concentrations
            # that sum to Ct and Pt.
            try:
                fit = optimize.least_squares(_A_residual,
                                             [guess],
                                             bounds=(0,At),
                                             args=np.array((K_dock,K_2AP,K_mg,n_mg,Rt,At,Mt)),
                                             tr_solver="exact",
                                             method="dogbox")
            except ValueError:
                continue

            # If we did not converge in the fit, try next guess
            if fit.success:
                A = fit.x[0]
                
            else:
                continue
                
            # get residuals
            residual = _A_residual([A],K_dock,K_2AP,K_mg,n_mg,Rt,At,Mt)

            # Record residuals if this is verbose
            if verbose:
                residuals.append((g,A,residual))
                if best_sum_residual is None or np.sum(np.abs(residual)) < best_sum_residual:
                    best_residual = np.copy(residual)
                    best_sum_residual = np.sum(residual)
                    best_A = A

            # Have we converged?  If not, change the guess
            if np.abs(residual[0]) > convergence_cutoff or \
               np.abs(best_residual[1]) > convergence_cutoff or \
               np.abs(best_residual[2]) > convergence_cutoff:
                continue

            # If we get here, we've converged
            successful = True
            break

        if not successful:

            err = "\nCould not find solution within convergence cutoff.\n"
            if not verbose:
                err += "Run again, setting verbose = True to get information\n"
                err += "about the regression that is failing.\n"
            else:
                err += "K_dock: {}\n".format(K_dock)
                err += "K_2AP: {}\n".format(K_2AP)
                err += "K_mg: {}\n".format(K_mg)
                err += "n_mg: {}\n".format(n_mg)
                err += "Rt: {}\n".format(Rt)
                err += "At: {}\n".format(At)
                err += "Mt: {}\n".format(Mt)
                err += "\nAll residuals:\n\n"
                for r in residuals:
                    err += "{}\n".format(r)

                err += "\nBest residual:\n\n"
                err += "residual: {}\n".format(best_residual)
                err += "free [A]: {}".format(best_A)

            raise RuntimeError(err)

    # Calculate concentration of each species given the equilibrium constants,
    # RNA, and free adenine conc
    E, D, DA, M = _calc_all_conc(K_dock,K_2AP,K_mg,n_mg,Rt,At,Mt,A)

    return E, D, DA, A, M


def species_conc(logK_dock,logK_2AP,logK_mg,n_mg,
                 Rt,At,Mt,
                 convergence_cutoff=0.01,
                 guess_resolution=0.1,
                 verbose=True):
    """
    Get species of L, H, HM, HA, HMA, A, and M given the equilibrium
    constants and the total protein and calcium concentrations.  This is the
    core, publically implementation of the model.  The equilibrium constants and
    concentrations may be floats or arrays, where all arrays have the same length.

    Ka: adenine binding constant
    K_mg: magnesium binding constant
    Ks: apo to active constant
    Rt: total RNA concentration
    At: total adenine concentration
    Mt: total magnesium concentration

    convergence_cutoff: percent difference between actual and calculated totals
    guess_resolution: if the first guess doesn't succeed, try another guess that
                      is guess-resolution away
    verbose: if True, record all all residuals and spit out if regression fails.

    returns: L, H, HM, HA, HMA, A, M
    """

    mismatch_error = False
    values = [logK_dock,logK_2AP,logK_mg,n_mg,Rt,At,Mt]
    is_array = [type(v) is pd.Series or type(v) is np.ndarray for v in values]

    # We have a series somewhere.  Make sure all series have the same length
    # and record this as series_length
    if sum(is_array) > 0:
        lengths = set([len(v) for i, v in enumerate(values) if is_array[i]])
        if len(lengths) != 1:
            mismatch_error = True
        else:
            series_length = list(lengths)[0]

    # No series.  series_length is 1.
    else:
        series_length = 1

    if mismatch_error:
        err = "Ka, K_mg, Ks, Rt, At, Mt must either be float values or\n"
        err += "arrays.  Any arrays that are present must have the same length.\n"
        raise ValueError(err)

    # Go through values and either cast the series/array as an array or
    # repeat single values series_length times as arrays
    for i in range(len(values)):
        if is_array[i]:
            values[i] = np.array(values[i])
        else:
            values[i] = np.array([values[i] for _ in range(series_length)])

    # Pull values out of array
    logK_dock, logK_2AP, logK_mg, n_mg, Rt, At, Mt = values
    
    E, D, DA, A, M = [], [], [], [], []
    for i in range(len(Rt)):

        concs = _species_conc(10.0**(logK_dock[i]),
                              10.0**(logK_2AP[i]),
                              10.0**(logK_mg[i]),
                              n_mg[i],
                              Rt[i],At[i],Mt[i],
                              convergence_cutoff=convergence_cutoff,
                              guess_resolution=guess_resolution,
                              verbose=verbose)
        
        E.append(concs[0])
        D.append(concs[1])
        DA.append(concs[2])
        A.append(concs[3])
        M.append(concs[4])

    
    E = np.array(E)
    D = np.array(D)
    DA = np.array(DA)
    A = np.array(A)
    M = np.array(M)

    return E, D, DA, A, M

def fx_A(logK_dock=-1,logK_2AP=-3,logK_mg=-6,n_mg=1,some_df=None,At=50):

    E, D, DA, A, M = species_conc(logK_dock,logK_2AP,logK_mg,n_mg,
                                  some_df.Rna,At,some_df.Mg*1e6)

    return (At - A)/At 

