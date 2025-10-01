import jax
from jax import numpy as jnp

import pandas as pd

from tabulate import tabulate

from matching import MatchingModel, Data

import sys

# Increase precision to 64 bit
jax.config.update("jax_enable_x64", True)

# Set dimensions
X, Y, N, M = 50, 30, 5, 3

# set mean and variance of measurement errors
mu, sigma = 0.0, 0.1

# Simulate data
errors = mu + sigma * jax.random.normal(jax.random.PRNGKey(111), (X, Y))

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

sigma_X = jnp.asarray(2.0)
sigma_Y = jnp.asarray(1.0)
transfer_constant = jnp.asarray(0.0)

parameters = jnp.concatenate(
    [beta_X, beta_Y, jnp.asarray([jnp.log(sigma_X), jnp.log(sigma_Y)])],
    axis=0,
)

model = MatchingModel(
    covariates_X=covariates_X,
    covariates_Y=covariates_Y,

    marginal_distribution_X = marginal_distribution_X,
    marginal_distribution_Y = marginal_distribution_Y,

    include_scale_parameters=True,
    replication=False,
    centered_variance=False,
)

mp = model.extract_model_parameters(parameters)
utility_X, utility_Y = model.Utilities_of_agents(mp)
transfer = model.solve(utility_X=utility_X, utility_Y=utility_Y, mp=mp, verbose=False)
observed_treansfer = transfer + errors

covariate_names_X = [f"beta_X {n}" for n in range(N)]
covariate_names_Y = [f"beta_Y {m}" for m in range(M)]
covariate_names = covariate_names_X + covariate_names_Y

parameter_names = covariate_names.copy()
# if model.centered_variance is False:
#     parameter_names += ["Salary constant"]
parameter_names += ["scale (X)", "scale (Y)"]

data = Data(transfers=observed_treansfer, matches=jnp.ones_like(observed_treansfer, dtype=float))

guess = jnp.zeros_like(parameters)

neg_log_lik = model.neg_log_likelihood(guess, data)
print(f"{model.replication = }: number of observations: {3*observed_treansfer.size}")
print(f"{neg_log_lik = }")

estimates = model.fit(guess, data, maxiter=100, verbose=True)
mp = model.extract_model_parameters(estimates)

estimates_transformed = jnp.concatenate(
    [mp.beta_X, mp.beta_Y, jnp.asarray([mp.sigma_X, mp.sigma_Y])],
    axis=0,
)

# Print estimated parameters
print(
    tabulate(
        list(zip(parameter_names, parameters, guess, estimates, estimates_transformed)),
        headers=["True parameter values", "Parameter guess", "Parameter estimates", "Transformed estimates"],
        tablefmt="grid",
    )
)

df_estimates = pd.DataFrame(
    {
        "name": parameter_names,
        "estimates": estimates,
        "transformed_estimates": estimates_transformed,
    }
)

mean, variance = model.compute_moments(estimates, data)
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