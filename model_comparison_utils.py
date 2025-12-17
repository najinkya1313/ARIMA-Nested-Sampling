from ARIMA_ns import loglikelihood,prior_parameters,ARIMA_Nested_Sampler
from ARIMA import ARIMA_fast
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.special import logsumexp

def ARIMA_model_comparison(data,max_p,max_q,d,num_live,num_delete,seed,mu_mean=0,mu_scale=1,prior_scale=1,normalize=True,inner_steps_factor=6,file_name=None,prior_bounds={}):
    """Evaluates model log posterior probabilities on a grid of ARMA models.
    Arguments:
    data : time series data to be analyzed
    max_p : max AR order of the grid
    max_q : max MA order of the grid
    d : differencing order d of the ARIMA Model
    num_live : number of live points to be used.
    num_delete : number of points to delete at each iteration
    seed : random seed
    mu_mean : prior distribution mean of the long term mean of data
    mu_scale : prior distribution scale of the long term mean of data
    prior_scale : prior distribution scale for the AR and MA coefficients (phi and theta)
    normalize : Return model log posterior probabilities; setting this to False returns log evidences.
    file_name : save the results of the run to a .txt file
    """
    evidences = []
    evidence_err = []
    order_done = []
    orders = [(p,d,q) for p in range(max_p+1) for q in range(max_q+1)]
    orders.remove((0,d,0))
    seeds = [seed]*len(orders)
    for order,seed in zip(orders,seeds):
        model = ARIMA_Nested_Sampler(data,order,mu_mean,mu_scale,num_live,num_delete,seed,prior_bounds=prior_bounds,inner_steps_factor=inner_steps_factor,prior_scale=prior_scale)
        evidence = evidences.append(model.log_evidence)
        evidence_err.append(model.log_evidence_err)
        order_done.append(order)
        evidence_arr = np.array(evidences)
        index = np.where(evidence_arr==max(evidence_arr))[0][0]
        print("----------------------x-------------------x---------------------x------")
        print(f"Evidence for {order} : {model.log_evidence} ; Error : {model.log_evidence_err}")
        print(f"Highest Evidence so far : {max(evidences)} for order : {order_done[index]}")
        print("----------------------x-------------------x----------------------x-----")
        
        # Save result to file after each run
        if file_name:
         with open(file_name, "a") as f:
            f.write(f"Order={order}, Seed={seed}, Evidence={model.log_evidence},Error={model.log_evidence_err}\n")
    
    evidences = np.array(evidences)
    evidence_err = np.array(evidence_err)
    normalization = logsumexp(evidences)
    log_posteriors = evidences-normalization
    if normalize:
        return log_posteriors,evidence_err
    else:
        return evidences,evidence_err

def load_evidence_file(file_name,normalization=True):
    """ Function to open and read the model log posterior probabilities from the .txt file of ARIMA_model_comparison.
    """
    evidences=[]
    errs = []

    with open(file_name, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines
            try:
                parts = line.split(",")
                # Evidence part is like " Evidence=-123.456"
               
                evidence_str = [p for p in parts if "Evidence" in p][0]
                error_str = [p for p in parts if "Error=" in p][0]

                evidence_val = float(evidence_str.split("=")[1])
                error = float(error_str.split("=")[1])

                evidences.append(evidence_val)
                errs.append(error)
            except Exception as e:
                print(f"Skipping line due to parse error: {line}\nError: {e}")
    if normalization:
     normalization = logsumexp(evidences)
     logP = evidences-normalization
     model_posteriors = logP,errs
     return model_posteriors
    else:
     return evidences,errs


def plot_evidence_heatmap(data, max_order, contrast=0, highlight_max=True,annotate=True,invert=False,axes=None,figure=None, **kwargs):
    """Plots a heatmap of the model log posterior probabilities and their associated errors on a grid of ARMA models.
    Arguments:
    data : the list of model log posterior probabilities and their associated error passed as a tuple (model_log_P,error)
    max_order : max order of the ARMA grid
    highlight_max : highlight the ARMA model with highest posterior probability (or evidence)
    annotate : annotate the grid with values of the model log posterior probabilities and their errors
    invert : invert the colorbar (to be used for plotting AIC/BIC heatmaps).
    """
    logP, err = data
    p_values = np.arange(max_order + 1)
    q_values = np.arange(max_order + 1)

    P, Q = np.meshgrid(p_values, q_values, indexing="ij")
    Z_flat, Z_err_flat = np.array(logP), np.array(err)

    heatmap_data = np.full((max_order + 1, max_order + 1), np.nan)
    heatmap_err = np.full((max_order + 1, max_order + 1), np.nan)
    for p, q, z, zerr in zip(P.flatten()[1:], Q.flatten()[1:], Z_flat, Z_err_flat):
        heatmap_data[q, p] = z
        heatmap_err[q, p] = zerr

    vmin, vmax = min(Z_flat) + contrast, max(Z_flat)

    # --- consistent figure size (double column) ---
    fig_width_pt, inches_per_pt, golden_mean = 508.0, 1.0 / 72.27, 0.6
    fig_width, fig_height = fig_width_pt * inches_per_pt, fig_width_pt * inches_per_pt * golden_mean
    width = kwargs.get('fig_width',fig_width)
    height = kwargs.get('fig_height',fig_height)
    fig, ax = plt.subplots(figsize=(width, height))
    if axes:
     ax = axes
    if figure:
     fig = figure
    if invert:

     im = ax.imshow(heatmap_data, origin="lower", cmap="inferno_r", vmin=vmin, vmax=vmax)
    else:
     im = ax.imshow(heatmap_data, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
        
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\log{P_i}$", fontsize=8)

    ax.set_xlabel("AR (p)", fontsize=9)
    ax.set_ylabel("MA (q)", fontsize=9)
    ax.set_xticks(np.arange(max_order+1))
    ax.set_yticks(np.arange(max_order+1))
    ax.set_xticklabels(np.arange(max_order + 1))
    ax.set_yticklabels(np.arange(max_order + 1))
    ax.tick_params(axis="both", labelsize=7)

    # Highlight maximum
    if highlight_max:
        j, i = np.unravel_index(np.nanargmax(heatmap_data), heatmap_data.shape)
        ax.scatter(i, j, s=40, facecolors='none', edgecolors='cyan', linewidths=1)

    # Annotate ALL tiles with values ± errors
    if annotate:
     for i in range(max_order + 1):
        for j in range(max_order + 1):
            if not np.isnan(heatmap_data[j, i]):
                ax.text(
                    i, j, f"{heatmap_data[j, i]:.1f}\n±{heatmap_err[j, i]:.1f}",
                    ha="center", va="center", color="black", fontsize=6,linespacing=1.2
                )

    fig.tight_layout(pad=0.5)
    return fig

