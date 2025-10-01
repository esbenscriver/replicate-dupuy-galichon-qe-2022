import jax
from jax import numpy as jnp

import pandas as pd

from matching import MatchingModel, Data, ModelParameters

# Increase precision to 64 bit
jax.config.update("jax_enable_x64", True)

include_transfer_constant = True
include_scale_parameters = True

# Set dimensions
X, Y, N, M = 50, 30, 5, 3

covariates_X = jax.random.normal(jax.random.PRNGKey(211), (X, Y, N))
covariates_Y = jax.random.normal(jax.random.PRNGKey(212), (X, Y, M))
marginal_distribution_X = jnp.ones((X, 1)) / X
marginal_distribution_Y = jnp.ones((1, Y)) / Y

assert jnp.isclose(jnp.sum(marginal_distribution_X), jnp.sum(marginal_distribution_Y))

# Simulate parameters
beta_X = jax.random.uniform(jax.random.PRNGKey(311), (N,))
beta_Y = -jax.random.uniform(jax.random.PRNGKey(312), (M,))

if include_scale_parameters is True:
    sigma_X = jnp.asarray([2.00])
    sigma_Y = jnp.asarray([1.50])
else:
    sigma_X = jnp.asarray([1.0])
    sigma_Y = jnp.asarray([1.0])

transfer_constant = jnp.asarray([1.0])

# set mean and variance of measurement errors
mu, sigma = transfer_constant, jnp.asarray([0.001])

# Simulate data
errors = mu + jnp.sqrt(sigma) * jax.random.normal(jax.random.PRNGKey(111), (X, Y))

model = MatchingModel(
    covariates_X=covariates_X,
    covariates_Y=covariates_Y,
    marginal_distribution_X=marginal_distribution_X,
    marginal_distribution_Y=marginal_distribution_Y,
    continuous_distributed_attributes=False,
    include_transfer_constant=include_transfer_constant,
    include_scale_parameters=include_scale_parameters,
)
parameter_names_X = [f"beta_X {n}" for n in range(N)]
parameter_names_Y = [f"beta_Y {m}" for m in range(M)]

parameter_names = parameter_names_X + parameter_names_Y

if model.include_transfer_constant is True:
    parameter_names += ["salary constant"]
if model.include_scale_parameters is True:
    parameter_names += ["scale parameter (workers)", "scale parameter (firms)"]

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
transfer = model.solve(
    utility_X=utility_X, utility_Y=utility_Y, mp=mp_true, verbose=False
)
observed_treansfer = transfer + errors

data = Data(
    transfers=observed_treansfer, matches=jnp.ones_like(observed_treansfer, dtype=float)
)

guess = jnp.zeros_like(parameter_values)

estimates = model.fit(guess, data, maxiter=1, verbose=False)
mp_estim = model.extract_model_parameters(estimates, transform=True)
estimates_transformed = model.class2vec(mp_estim, transform=False)

print("=" * 80)
print("Parameter Estimates")
print("=" * 80)
df_estimates = (
        pd.DataFrame(
        {
            "name": parameter_names,
            "true parameters": parameter_values_transformed,
            "estimated parameters": estimates_transformed,
        }
    )
    .round(3)
    .set_index("name")
)
print(df_estimates)
print("=" * 80)
print(f"Number of estimated parameters: {len(df_estimates)}")
print(f"number of observations: {observed_treansfer.size}\n")

print(f"{model.neg_log_likelihood(parameter_values_transformed, data) = }")
print(f"{model.neg_log_likelihood(estimates, data) = }\n")

variance, mean = model.compute_moments(estimates, data)

print("=" * 80)
print("Estimated moments")
print("=" * 80)
df_moments = (
    pd.DataFrame(
        {
            "name": ["mean", "variance"],
            "true parameters": jnp.concatenate([mu, sigma], axis=0),
            "estimated parameters": jnp.asarray([mean, variance]),
        }
    )
    .set_index("name")
    .round(3)
)
print(df_moments)
print("=" * 80)
