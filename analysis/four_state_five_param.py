"""
Linked binding of adenine and magnesium to an RNA molecule.
"""
__author__ = "Michael J. Harms (harmsm@gmail.com)"
__date__ = "2020-02-14"

import numpy as np
import pandas as pd
from scipy import optimize


def _calc_all_conc(K_link,K_dock,K_2AP,K_mg,n_mg,Rt,At,Mt,A):
    """
    Calculate the concentration of all species given the equilibrium
    constants (K_link,K_dock,K_2AP,K_mg,n_mg), total RNA concentration (Rt), 
    total 2AP concentration (At), total magnesium concentration (Mt), 
    and free 2AP concentration (A). 
    """
    
    # Define convenience variables
    S = 1 + K_2AP*A
    T = 1 + K_2AP*K_link*A
    
    # Solve for M having guessed A
    M = Mt + ((n_mg*T)/(T-S))*( Rt*(S-1) - S*At + S*A )
    
    
    if M < 0:
        raise ValueError
    
    # Calculate E given that we now have A, M
    E = Rt / (1 + K_2AP*A +(K_dock*((K_mg*M)**n_mg)*(1+K_2AP*K_link*A)) )
    
    if E < 0:
        raise ValueError

    # Calculate all other species
    D = K_dock*((K_mg*M)**n_mg)*E
    EA = K_2AP*A*E
    DA = D*K_2AP*K_link*A

    return E,D,EA,DA,M


def _A_residual(params,K_link,K_dock,K_2AP,K_mg,n_mg,Rt,At,Mt):
    """
    Residual function for finding the free value of 2AP (A).  
    
    Fit parameter: The value of free 2AP. 
    Residual: Percent difference between calculated and total RNA, 2AP, 
    and magnesium species' concentrations.
    """

    # Calculate the species concentration given our guess for [A]
    A = params[0]
    E, D, EA, DA, M = _calc_all_conc(K_link,K_dock,K_2AP,K_mg,n_mg,Rt,At,Mt,A)

    # Calculate the total concentrations in RNA, 2AP, and magnesium
    Rt_calc = E + D + EA + DA
    At_calc = A + EA + DA
    Mt_calc = M + (n_mg*D) + (n_mg*DA)

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

def _species_conc(K_link,K_dock,K_2AP,K_mg,n_mg,
                  Rt,At,Mt,
                  convergence_cutoff=0.01,
                  guess_resolution=0.1,
                  verbose=True):
    """
    Get species concentrations given the equilibrium
    constants and the total RNA (Rt), 2AP (At), and Mg2+ (Mt) concentrations. 
    The equilibrium constants and concentrations must be floats.
    
    K_link: constant linking magnesium binding and 2AP binding
    K_dock: E -> D equilibrium constant
    K_2AP: affinity of 2AP for the D conformation
    K_mg: relative affinity of D and E conformations for Mg2+ ions
    n_mg: difference in number of bound Mg2+ ions for D and E conformations
    Rt: total RNA concentration
    At: total 2AP concentration
    Mt: total magnesium concentration

    convergence_cutoff: percent difference between actual and calculated totals
    guess_resolution: if the first guess doesn't succeed, try another guess that
                      is guess-resolution away
    verbose: if True, record all all residuals and spit out if regression fails.

    returns: E, D, EA, DA, A, M
    """

    # If zero 2AP total, we already know A
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

            # Find value for A between 0 and At that finds species concentrations
            # that sum to Rt, At, and Mt.
            try:
                fit = optimize.least_squares(_A_residual,
                                             [guess],
                                             bounds=(0,At),
                                             args=np.array((K_link,K_dock,K_2AP,K_mg,n_mg,Rt,At,Mt)),
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
            residual = _A_residual([A],K_link,K_dock,K_2AP,K_mg,n_mg,Rt,At,Mt)

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
                err += "K_link: {}\n".format(K_link)
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
    # total RNA, 2AP, Mg2+, and free 2AP concentrations
    E, D, EA, DA, M = _calc_all_conc(K_link,K_dock,K_2AP,K_mg,n_mg,Rt,At,Mt,A)

    return E, D, EA, DA, A, M


def species_conc(logK_link,logK_dock,logK_2AP,logK_mg,n_mg,
                 Rt,At,Mt,
                 convergence_cutoff=0.01,
                 guess_resolution=0.1,
                 verbose=True):
    """
    Get species of E, D, EA, DA, A, and M given logged equilibrium
    constants (except n_mg) and the total RNA (Rt), 2AP (At), and Mg2+ (Mt) concentrations.  
    This is the core, publically implementation of the model.  
    The equilibrium constants and concentrations may be floats or arrays, 
    where all arrays have the same length.

    logK_link: logged constant linking magnesium binding and 2AP binding
    logK_dock: logged E -> D equilibrium constant
    logK_2AP: logged affinity of 2AP for the D conformation
    logK_mg: logged relative affinity of D and E conformations for Mg2+ ions
    n_mg: difference in number of bound Mg2+ ions for D and E conformations
    Rt: total RNA concentration
    At: total 2AP concentration
    Mt: total magnesium concentration

    convergence_cutoff: percent difference between actual and calculated totals
    guess_resolution: if the first guess doesn't succeed, try another guess that
                      is guess-resolution away
    verbose: if True, record all all residuals and spit out if regression fails.

    returns: E, D, EA, DA, A, M
    """

    mismatch_error = False
    values = [logK_link,logK_dock,logK_2AP,logK_mg,n_mg,Rt,At,Mt]
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
        err = "logK_link, logK_dock, logK_2AP, logK_mg, n_mg, Rt, At, Mt must either be float values or\n"
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
    logK_link, logK_dock, logK_2AP, logK_mg, n_mg, Rt, At, Mt = values

    E, D, EA, DA, A, M = [], [], [], [], [], []
    for i in range(len(Rt)):

        concs = _species_conc(10.0**(logK_link[i]),
                              10.0**(logK_dock[i]),
                              10.0**(logK_2AP[i]),
                              10.0**(logK_mg[i]),
                              n_mg[i],
                              Rt[i],At[i],Mt[i],
                              convergence_cutoff=convergence_cutoff,
                              guess_resolution=guess_resolution,
                              verbose=verbose)
        
        E.append(concs[0])
        D.append(concs[1])
        EA.append(concs[2])
        DA.append(concs[3])
        A.append(concs[4])
        M.append(concs[5])

    
    E = np.array(E)
    D = np.array(D)
    EA = np.array(EA)
    DA = np.array(DA)
    A = np.array(A)
    M = np.array(M)

    return E, D, EA, DA, A, M

def fx_A(logK_link=-6,logK_dock=-3,logK_2AP=-3,logK_mg=-6,n_mg=0,some_df=None,At=50):
    '''
    Calculates fractional 2AP saturation given logged equilibrium constant estimates 
    (except for n_mg) and a dataframe (some_df) with total RNA concentrations in nM 
    and total magnesium concentrations in mM.
    
    Calls species_conc to get free adenine concentration. At is always 50 nM.
    '''
   
    E, D, EA, DA, A, M = species_conc(logK_link,logK_dock,logK_2AP,logK_mg, n_mg,
                                      some_df.Rna,At,some_df.Mg*1e6)
    return (At - A)/At 
