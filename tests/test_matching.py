# import JAX
import jax
import jax.numpy as jnp

# import solver for one-to-one matching model
from module.matching import MatchingModel, Data

import pytest

# Increase precision to 64 bit
jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "include_scale_parameters, include_transfer_constant, log_transform_scale",
    [
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (True, False, False),
    ],
)
def test_mle(include_scale_parameters, include_transfer_constant, log_transform_scale):
    simulate_marginal_distributions = True

    # Set dimensions
    X, Y, N, M = 9, 5, 3, 2

    covariates_X = jax.random.uniform(jax.random.PRNGKey(111), (X, Y, N))
    covariates_Y = jax.random.uniform(jax.random.PRNGKey(112), (X, Y, M))

    if simulate_marginal_distributions is True:
        m_X = jax.random.uniform(jax.random.PRNGKey(211), (X, 1))
        m_Y = jax.random.uniform(jax.random.PRNGKey(212), (1, Y))
        marginal_distribution_X = m_X / jnp.sum(m_X)
        marginal_distribution_Y = m_Y / jnp.sum(m_Y)
    else:
        marginal_distribution_X = jnp.ones((X, 1)) / X
        marginal_distribution_Y = jnp.ones((1, Y)) / Y

    assert jnp.isclose(jnp.sum(marginal_distribution_X), jnp.sum(marginal_distribution_Y))

    # Simulate parameters
    beta_X = -jax.random.uniform(jax.random.PRNGKey(311), (N,))
    beta_Y = jax.random.uniform(jax.random.PRNGKey(312), (M,))

    parameter_names_X = [f"beta_X {n}" for n in range(N)]
    parameter_names_Y = [f"beta_Y {m}" for m in range(M)]

    sigma_X = jnp.asarray([2.00])
    sigma_Y = jnp.asarray([1.50])

    if include_scale_parameters and log_transform_scale:
        sigma_X = jnp.log(sigma_X)
        sigma_Y = jnp.log(sigma_Y)

    transfer_constant = jnp.asarray([1.0])

    # set mean and variance of measurement errors
    mu, sigma = transfer_constant, jnp.asarray([0.0])

    # Simulate data
    errors = mu + jnp.sqrt(sigma) * jax.random.normal(jax.random.PRNGKey(411), (X, Y))

    parameter_values = jnp.concatenate([beta_X, beta_Y], axis=0)

    parameter_names = parameter_names_X + parameter_names_Y

    if include_transfer_constant:
        parameter_values = jnp.concatenate([parameter_values, transfer_constant], axis=0)
        parameter_names += ["salary constant"]

    if include_scale_parameters:
        parameter_values = jnp.concatenate([parameter_values, sigma_X, sigma_Y], axis=0)
        parameter_names += ["scale parameter (workers)", "scale parameter (firms)"]

    model = MatchingModel(
        covariates_X=covariates_X,
        covariates_Y=covariates_Y,
        marginal_distribution_X=marginal_distribution_X,
        marginal_distribution_Y=marginal_distribution_Y,
        continuous_distributed_attributes=False,
        include_transfer_constant=include_transfer_constant,
        include_scale_parameters=include_scale_parameters,
        log_transform_scale=log_transform_scale,
    )
    mp_true = model.extract_model_parameters(parameter_values)

    # Solve model given true parameters
    utility_X, utility_Y = model.Utilities_of_agents(mp_true)

    transfer = model.solve(
        utility_X=utility_X, utility_Y=utility_Y, mp=mp_true, verbose=True
    )

    # Simulate observed data
    observed_matches_X = jnp.exp(model.log_Demand_X(
        transfer=transfer, utility_X=utility_X, sigma_X=mp_true.sigma_X
    ))
    observed_matches_Y = jnp.exp(model.log_Demand_Y(
        transfer=transfer, utility_Y=utility_Y, sigma_Y=mp_true.sigma_Y
    ))
    assert jnp.allclose(observed_matches_X, observed_matches_Y)
    observed_treansfer = transfer + errors

    data = Data(transfers=observed_treansfer, matches=observed_matches_X)

    variance = model.compute_moments(parameter_values, data)[0]

    assert jnp.isclose(variance, 0.0), (
        f"{mu = }, {sigma = }: The variance of the measurement error is different from zero: {variance = }"
    )
