__usage__ = "run.py  genotype fit_type (one of ml bayesian) model (one of 2.3 3.3 3.4 4.4 4.5)"

from matplotlib import pyplot as plt

import two_state_three_param
import three_state_four_param
import three_state_three_param
import four_state_five_param

import pandas as pd
import numpy as np

import likelihood

import pickle, sys


def run(g,fit_type,model):

    if model == "2.3":
        fx_A = two_state_three_param.fx_A
    elif model == "3.3":
        fx_A = three_state_three_param.fx_A
    elif model == "3.4":
        fx_A = three_state_four_param.fx_A
    elif model == "4.4":
        fx_A = four_state_five_param.fx_A
    elif model == "4.5":
        fx_A = four_state_five_param.fx_A
    else:
        err = f"model '{model}' not recognized\n"
        raise ValueError(err)

    df = pd.read_csv("2AP_corrected.csv")

    cmap_pos = np.linspace(0.25,1,4)
    colors = [plt.cm.Greens(x) for x in cmap_pos]

    g_df = df[df.Geno == g]

    # Construct likliehood model wrapper for fitting
    lm = likelihood.ModelWrapper(fx_A)
    lm.some_df = g_df
    lm.At = 50
    num_fit_param = len(lm.fit_parameters)

    # fix the dock parameter to zero for 4.4
    if model == "4.4":
        lm.logK_dock = 0
        lm.logK_dock.fixed = True
        num_fit_param = num_fit_param - 1

    # Create fitter object
    if fit_type == "bayesian":
        f = likelihood.BayesianFitter(num_walkers=100,num_steps=1500)
    else:
        f = likelihood.MLFitter()
    
    # constrain "n_mg"
    bounds = [[-np.inf for _ in range(num_fit_param)],
              [ np.inf for _ in range(num_fit_param)]]

    bounds[0][-1] = -5
    bounds[1][-1] = 5

    # Do fit
    f.fit(lm,y_obs=g_df.FS_mean,y_stdev=g_df.FS_std_cutoff,bounds=bounds)

    # Get fit results
    fit_dict = dict(zip(f.names,f.estimate))        
    fig, ax = plt.subplots(1,1,figsize=(5,5))
   
    # Plot fit results 
    r = np.arange(np.min(df.Rna),np.max(df.Rna),50)
    for i, m in enumerate(np.unique(g_df.Mg)):
        mg = [m for _ in range(len(r))]
    
        g_m_df = g_df[g_df.Mg == m]

        ax.plot(g_m_df.Rna,g_m_df.FS_mean,"o",color=colors[i])
    
        x = g_m_df.Rna
        y = g_m_df.FS_mean
        yerr = g_m_df.FS_std_cutoff
        plt.errorbar(x,y,yerr,color="gray",fmt="none",ls="none")
    
        plot_df = pd.DataFrame({"Rna":r,"Mg":mg})
    
        ax.plot(plot_df.Rna,fx_A(**fit_dict,some_df=plot_df),"-",color=colors[i],lw=2,label=m)

        # If bayesian, plot indiv samples with low alpha
        if fit_type == "bayesian":
    
            # Plot if samples made
            if f.samples.shape != (0,):
                for j in np.arange(0,f.samples.shape[0],f.samples.shape[0]//50):
                    tmp_fit_dict = dict(zip(f.names,f.samples[j,:]))
                    try:
                        ax.plot(plot_df.Rna,fx_A(**tmp_fit_dict,some_df=plot_df),"-",color=colors[i],alpha=0.1,label='_nolegend_')
                    except RuntimeError:
                        continue
        
    ax.set_title(g)
    ax.set_xscale("log")
    ax.set_xlim((100,10000))
    ax.set_ylim((0,1))
    ax.legend()
    
    # save outputs

    fig.savefig(f"{model}/{fit_type}/{g}_fit.pdf")
    plt.show()

    f.fit_to_df.to_csv(f"{model}/{fit_type}/{g}_results.csv")

    if f.samples.shape != (0,):
        fig = f.corner_plot()
        fig.savefig(f"{model}/{fit_type}/{g}_corner.pdf")
        plt.show()

        f.write_samples(f"{model}/{fit_type}/{g}_samples.pickle")

    h = open(f"{model}/{fit_type}/{g}_likelihood.txt","w")
    h.write("lnL,N\n")
    h.write(f"{f.ln_like(f.estimate)},{f.num_params}\n")
    h.close()
        
def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    try:
        genotype = argv[0]

        fit_type = argv[1]
        if fit_type not in ["ml","bayesian"]:
            raise IndexError

        model = argv[2] 
       
    except IndexError:
        err = f"incorrect arguments. usage:\n\n{__usage__}\n\n"
        raise ValueError(err)

    run(genotype,fit_type,model)

if __name__ == "__main__":
    main()
