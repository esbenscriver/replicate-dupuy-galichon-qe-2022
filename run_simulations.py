import jax
from jax import numpy as jnp

import pandas as pd

from tabulate import tabulate

from matching import MatchingModel, Data, ModelParameters

# Increase precision to 64 bit
jax.config.update("jax_enable_x64", True)

include_transfer_constant = False
include_scale_parameters = True

# Set dimensions
X, Y, N, M = 50, 30, 5, 3

covariates_X = jax.random.normal(jax.random.PRNGKey(211), (X, Y, N))
covariates_Y = jax.random.normal(jax.random.PRNGKey(212), (X, Y, M))
marginal_distribution_X = jnp.ones((X, 1)) / X
marginal_distribution_Y = jnp.ones((1, Y)) / Y

assert jnp.isclose(
    jnp.sum(marginal_distribution_X), jnp.sum(marginal_distribution_Y)
)

# Simulate parameters
beta_X = jax.random.uniform(jax.random.PRNGKey(311), (N,))
beta_Y = -jax.random.uniform(jax.random.PRNGKey(312), (M,))

if include_scale_parameters is True:
    sigma_X = jnp.asarray([2.00])
    sigma_Y = jnp.asarray([1.50])
else:
    sigma_X = jnp.asarray([1.0])
    sigma_Y = jnp.asarray([1.0])

transfer_constant = jnp.asarray([1.0]   )

# set mean and variance of measurement errors
mu, sigma = transfer_constant, 0.001

# Simulate data
errors = mu + jnp.sqrt(sigma) * jax.random.normal(jax.random.PRNGKey(111), (X, Y))

model = MatchingModel(
    covariates_X=covariates_X,
    covariates_Y=covariates_Y,

    marginal_distribution_X = marginal_distribution_X,
    marginal_distribution_Y = marginal_distribution_Y,

    continuous_distributed_attributes = False,
    include_transfer_constant = include_transfer_constant,
    include_scale_parameters = include_scale_parameters,
)
parameter_names_X = [f"beta_X {n}" for n in range(N)]
parameter_names_Y = [f"beta_Y {m}" for m in range(M)]

parameter_names = parameter_names_X + parameter_names_Y

if model.include_transfer_constant is True:
    parameter_names += ["salary constant"]
if model.include_scale_parameters is True:
    parameter_names += ["scale (X)", "scale (Y)"]

mp_true = ModelParameters(
    beta_X=beta_X,
    beta_Y=beta_Y,
    sigma_X=sigma_X,
    sigma_Y=sigma_Y,
    transfer_constant=transfer_constant,
)
parameter_values = model.class2vec(mp_true, transform=False)
parameter_values_transformed = model.class2vec(mp_true, transform=True)

utility_X, utility_Y = model.Utilities_of_agents(mp_true)
transfer = model.solve(utility_X=utility_X, utility_Y=utility_Y, mp=mp_true, verbose=False)
observed_treansfer = transfer + errors

data = Data(
    transfers=observed_treansfer, 
    matches=jnp.ones_like(observed_treansfer, dtype=float)
)

guess = jnp.zeros_like(parameter_values)

estimates = model.fit(guess, data, maxiter=100, verbose=False)
mp_estim = model.extract_model_parameters(estimates, transform=True)
estimates_transformed = model.class2vec(mp_estim, transform=False)

# Print estimated parameters
print(
    tabulate(
        list(zip(parameter_names, parameter_values, guess, estimates, estimates_transformed)),
        headers=["True parameter values", "Parameter guess", "Parameter estimates", "Transformed estimates"],
        tablefmt="grid",
    )
)
print(f"number of observations: {3*observed_treansfer.size}\n")

print(f"{model.neg_log_likelihood(parameter_values_transformed, data) = }")
print(f"{model.neg_log_likelihood(estimates, data) = }\n")

df_estimates = pd.DataFrame(
    {
        "name": parameter_names,
        "estimates": estimates,
        "transformed_estimates": estimates_transformed,
    }
)

variance, mean = model.compute_moments(estimates, data)
moment_estimates = jnp.asarray([mean, variance])
moment_names = ['mean','variance']

# Print estimated mean and variance of measurement errors
print(
    tabulate(
        list(zip(moment_names, moment_estimates)),
        headers=["Parameter estimates"],
        tablefmt="grid",
    )
)