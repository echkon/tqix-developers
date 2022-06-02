def fit(objective_function,optimizer,init_params,return_loss_hist=None,loss_break=None):
    if return_loss_hist:
        opt_params,loss,_,loss_hist = optimizer.optimize(len(init_params), objective_function, initial_point=init_params,return_loss_hist=return_loss_hist,loss_break=loss_break)
        return opt_params,loss,loss_hist
    opt_params,loss,_ = optimizer.optimize(len(init_params), objective_function, initial_point=init_params,return_loss_hist=return_loss_hist,loss_break=loss_break)
    return opt_params,loss