from ARIMA_ns import loglikelihood,prior_parameters,ARIMA_Nested_Sampler
from ARIMA import ARIMA_fast
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def ARIMA_model_comparison(data,orders,num_live,num_delete,seeds,mu_mean=0,mu_scale=1,prior_scale=1,file_name=None,prior_bounds={}):
    evidences = []
    evidence_err = []
    order_done = []
    for order,seed in zip(orders,seeds):
        model = ARIMA_Nested_Sampler(data,order,mu_mean,mu_scale,num_live,num_delete,seed,prior_bounds=prior_bounds,prior_scale=prior_scale)
        evidences.append(model.log_evidence)
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
            f.write(f"Order={order}, Seed={seed}, Evidence={model.log_evidence}, Error={model.log_evidence_err}\n")
       
    return evidences,evidence_err

def load_evidence_file(file_name):
    evidences = []
    evidence_errs = []

    with open(file_name, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines
            try:
                parts = line.split(",")
                # Evidence part is like " Evidence=-123.456"
                evidence_str = [p for p in parts if "Evidence=" in p][0]
                error_str = [p for p in parts if "Error=" in p][0]

                evidence = float(evidence_str.split("=")[1])
                error = float(error_str.split("=")[1])

                evidences.append(evidence)
                evidence_errs.append(error)
            except Exception as e:
                print(f"Skipping line due to parse error: {line}\nError: {e}")
    mixed_evidences = evidences,evidence_errs
    return mixed_evidences

def plot_evidence_heatmap(mixed_evidences,max_order,contrast=0,title=None,invert=False):
 evidences,evidence_err = mixed_evidences
 p_values = np.arange(0, max_order+1)
 q_values = np.arange(0, max_order+1)

 # Flattened meshgrid for p, q pairs
 P, Q = np.meshgrid(p_values, q_values, indexing='ij')
 P_flat = P.flatten()
 Q_flat = Q.flatten()

 Z_flat = np.array(evidences)

 Z_err_flat = np.array(evidence_err)


 # Initialize with NaN so missing (like (0,0)) stay blank
 heatmap_data = np.full((max_order+1, max_order+1), np.nan)
 heatmap_err  = np.full((max_order+1, max_order+1), np.nan)

 # Fill skipping (0,0) since evidences exclude it
 for p, q, z, zerr in zip(P_flat[1:], Q_flat[1:], Z_flat, Z_err_flat):
    heatmap_data[q, p] = z
    heatmap_err[q, p] = zerr

 vmin = min(Z_flat) + contrast
 vmax = max(Z_flat)

 plt.figure(figsize=(17,17))
 if invert==False:
  plt.imshow(heatmap_data, origin='lower', cmap='plasma',vmin=vmin,vmax=vmax)
 else:
  plt.imshow(heatmap_data, origin='lower', cmap='plasma_r',vmin=vmin,vmax=vmax)
 plt.colorbar(label='Log Evidence')
 plt.xticks(np.arange(0,max_order+1))
 plt.yticks(np.arange(0,max_order+1))
 plt.xlabel('AR(p)', fontsize=15)
 plt.ylabel('MA(q)', fontsize=15)

# Annotate with text values (value ± error)
 for i in range(max_order+1):
    for j in range(max_order+1):
        if not np.isnan(heatmap_data[j, i]):
            plt.text(i, j, f"{heatmap_data[j, i]:.2f}±{heatmap_err[j, i]:.2f}", 
                     ha='center', va='center', color='black', fontsize=10)
 if title:
     plt.title(title,fontsize=20)
 else:
     plt.title(r'Log Evidence Heatmap',fontsize=20)
 
 plt.show()
