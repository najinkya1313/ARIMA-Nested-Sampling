


def ARIMA_model_comparison(data,orders,prior_type,num_live,num_delete,seeds,scale=1,mu_mean=0,mu_scale=1,file_name=None,prior_bounds={}):
    evidences = []
    evidence_err = []
    order_done = []
    for order,seed in zip(orders,seeds):
        model = ARIMA_Nested_Sampler(data,order,prior_type,scale,mu_mean,mu_scale,num_live,num_delete,seed,prior_bounds={})
        evidences.append(model.log_evidence)
        evidence_err.append(model.log_evidence_err)
        order_done.append(order)
        evidence_arr = np.array(evidences)
        index = np.where(evidence_arr==max(evidence_arr))[0][0]
        print(index)
        print("----------------------x-------------------x---------------------x------")
        print(f"Evidence : {model.log_evidence} ; Error : {model.log_evidence_err}")
        print(f"Highest Evidence so far : {max(evidences)} for order : {order_done[index]}")
        print("----------------------x-------------------x----------------------x-----")
  
        # Save result to file after each run
        if file_name:
         with open(file_name, "a") as f:
            f.write(f"Order={order}, Seed={seed}, Evidence={model.log_evidence}, Error={model.log_evidence_err}\n")
       
    return evidences,evidence_err
