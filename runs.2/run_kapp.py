__usage__ = "run.py  genotype fit_type (one of ml bayesian) model (kapp)"

import matplotlib
from matplotlib import pyplot as plt

import kapp_one_param

import pandas as pd
import numpy as np

import likelihood

import pickle, sys


def run(g,fit_type,model):

    if model == "kapp":
        fx_A = kapp_one_param.fx_A
    else:
        err = f"model '{model}' not recognized\n"
        raise ValueError(err)
       
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    colors = [plt.cm.Greens(x) for x in np.linspace(0.25,1,4)]
    
    df = pd.read_csv("runs/2AP_corrected.csv")
    r = np.arange(np.min(df.Rna),np.max(df.Rna),50)
    g_df = df[df.Geno == g]
    fit_dict = {}
    fit_results = []
    
    for i, mg_mM in enumerate(np.unique(g_df.Mg)):
        g_m_df = g_df[g_df.Mg == mg_mM]

        # Construct likelihood model wrapper for fitting
        lm = likelihood.ModelWrapper(fx_A)
        lm.some_df = g_m_df
        lm.At = 50
        num_fit_param = len(lm.fit_parameters)

        # Create fitter object
        if fit_type == "bayesian":
            f = likelihood.BayesianFitter(num_walkers=100,num_steps=1500)
        else:
            f = likelihood.MLFitter()

        bounds = [[-np.inf for _ in range(num_fit_param)],
                  [ np.inf for _ in range(num_fit_param)]]

        # Do fit
        f.fit(lm,y_obs=g_m_df.FS_mean,y_stdev=g_m_df.FS_std_cutoff,bounds=bounds)
        
        fit_dict[f.names[0]] = f.estimate
        f.fit_to_df.insert(1,'Mg',mg_mM)
        fit_results.append(f.fit_to_df)

        ax.plot(g_m_df.Rna,g_m_df.FS_mean,"o",color=colors[i])
    
        x = g_m_df.Rna
        y = g_m_df.FS_mean
        yerr = g_m_df.FS_std_cutoff
        plt.errorbar(x,y,yerr,color="gray",fmt="none",ls="none")
        
        mg = [mg_mM for _ in range(len(r))]
        plot_df = pd.DataFrame({"Rna":r,"Mg":mg})
    
        ax.plot(plot_df.Rna,fx_A(**fit_dict,some_df=plot_df),"-",color=colors[i],lw=2,label=mg_mM)

        # If bayesian, plot indiv samples with low alpha
        if fit_type == "bayesian":
    
            # Plot if samples made
            if f.samples.shape != (0,):
                for j in np.arange(0,f.samples.shape[0],f.samples.shape[0]//50):
                    tmp_fit_dict = {f.names[0]:f.samples[j,:]}
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
    
    fit_result_df = pd.concat(fit_results)
    fit_result_df.to_csv(f"{model}/{fit_type}/{g}_results.csv")

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