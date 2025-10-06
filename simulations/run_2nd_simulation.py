import jax
from jax import numpy as jnp

import pandas as pd

from matching import MatchingModel, Data, ModelParameters

# Increase precision to 64 bit
jax.config.update("jax_enable_x64", True)

include_transfer_constant = False
include_scale_parameters = True

# Set dimensions
Z, N, M = 1000, 5, 3

print("\n" + "=" * 80)
print("Simulation Settings")
print("=" * 80)
print(f"Number of observations: {Z}")
print(f"Number of worker covariates: {N}")
print(f"Number of firm covariates: {M}")
print(f"Include transfer constant: {include_transfer_constant}")
print(f"Include scale parameters: {include_scale_parameters}")
print("=" * 80)

covariates_X = jax.random.normal(jax.random.PRNGKey(111), (Z, Z, N))
covariates_Y = jax.random.normal(jax.random.PRNGKey(112), (Z, Z, M))

marginal_distribution_X = jnp.ones((Z, 1)) / Z
marginal_distribution_Y = jnp.ones((1, Z)) / Z

assert jnp.isclose(jnp.sum(marginal_distribution_X), jnp.sum(marginal_distribution_Y))

# Simulate parameters
beta_X = -jax.random.uniform(jax.random.PRNGKey(211), (N,))
beta_Y = jax.random.uniform(jax.random.PRNGKey(212), (M,))

if include_scale_parameters is True:
    sigma_X = jnp.asarray([2.00])
    sigma_Y = jnp.asarray([1.50])
else:
    sigma_X = jnp.asarray([1.0])
    sigma_Y = jnp.asarray([1.0])

transfer_constant = jnp.asarray([0.0])

# set mean and variance of measurement errors
mu, sigma = transfer_constant, jnp.asarray([0.01])

# Simulate data
errors = mu + jnp.sqrt(sigma) * jax.random.normal(jax.random.PRNGKey(311), (Z,))

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
parameter_values = model.class2vec(mp_true)

# Solve model given true parameters
utility_X, utility_Y = model.Utilities_of_agents(mp_true)
transfer = model.solve(
    utility_X=utility_X, utility_Y=utility_Y, mp=mp_true, verbose=True
)
# print(f"utility_X:\n{utility_X}")
# print(f"utility_Y:\n{utility_Y}")
# print(f"transfer:\n{transfer}")
# print(f"payoff_X:\n{model.Payoff_X(transfer, utility_X, sigma_X)}")
# print(f"payoff_Y:\n{model.Payoff_Y(transfer, utility_Y, sigma_Y)}")

# Simulate observed data
observed_treansfer = jnp.diag(transfer) + errors
# print(f"observed_treansfer:\n{observed_treansfer}")

data = Data(
    transfers=observed_treansfer, matches=jnp.ones_like(observed_treansfer, dtype=float)
)

guess = jnp.zeros(len(parameter_names_X + parameter_names_Y))
if model.include_transfer_constant is True:
    guess = jnp.concatenate([guess, jnp.array([0.0])], axis=0)

if model.include_scale_parameters is True:
    guess = jnp.concatenate([guess, jnp.array([1.0, 1.0])], axis=0)

print(f"\nlogL(guess)={-model.neg_log_likelihood(guess, data)}\n")

estimates = model.fit(guess, data, maxiter=100)
variance, mean = model.compute_moments(estimates, data)

mp_estim = model.extract_model_parameters(estimates)

# # Solve model given true parameters
# utility_X, utility_Y = model.Utilities_of_agents(mp_estim)
# estimated_transfer = model.solve(
#     utility_X=utility_X, utility_Y=utility_Y, mp=mp_true, verbose=True
# )
# print(f"estimated transfer:\n{estimated_transfer}")

print("\n" + "=" * 80)
print("Parameter Estimates")
print("=" * 80)
df_estimates = pd.DataFrame(
    {
        "name": parameter_names,
        "true parameters": parameter_values,
        "estimated parameters": estimates,
    }
).round(3).set_index("name").rename_axis(None)

print(df_estimates)
print("=" * 80)
print(f"Number of estimated parameters: {len(estimates)}")
print(f"number of observations: {observed_treansfer.size}\n")

print(f"\nlogL(parameter_values)={-model.neg_log_likelihood(parameter_values, data)}")
print(f"logL(estimates)={-model.neg_log_likelihood(estimates, data)}\n")

print("=" * 80)
print("Estimated Moments of Measurement Errors")
print("=" * 80)
df_moments = pd.DataFrame(
    {
        "name": ["mean", "variance"],
        "true parameters": jnp.concatenate([mu, sigma], axis=0),
        "estimated parameters": jnp.asarray([mean, variance]),
    }
).round(3).set_index("name").rename_axis(None)

print(df_moments)
print("=" * 80)
model.R_squared(estimates, data)
