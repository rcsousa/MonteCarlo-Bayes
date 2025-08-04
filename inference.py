from parameters import parameters

def update_prior(param_name, successes, trials):
    prior = parameters[param_name]
    alpha_prior = prior["alpha"]
    beta_prior = prior["beta"]

    new_alpha = alpha_prior + successes
    new_beta = beta_prior + (trials - successes)

    return {
        "new_alpha": new_alpha,
        "new_beta": new_beta
    }
