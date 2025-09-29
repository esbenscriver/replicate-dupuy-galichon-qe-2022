"""
JAX implementation of fixed-point iteration algoritm to solve and one-to-one matching model with transferable utility

Reference:
Esben Scriver Andersen, Note on solving one-to-one matching models with linear transferable utility, 2025 (https://arxiv.org/pdf/2409.05518)
"""

import jax
import jax.numpy as jnp
from jax import Array

from jax.scipy.special import logsumexp

# import simple_pytree (used to store variables)
from simple_pytree import Pytree, dataclass

# import solvers
from jaxopt import FixedPointIteration, AndersonAcceleration, LBFGS
from squarem_jaxopt import SquaremAcceleration

SolverTypes = (
    type[SquaremAcceleration] | type[AndersonAcceleration] | type[FixedPointIteration]
)


@dataclass
class ModelParameters(Pytree, mutable=False):
    """Model parameters to be estimated

    Attributes:
        beta_X (Array): linear utility parameters of agent of type X
        beta_Y (Array): linear utility parameters of agent of type Y
        sigma_X (Array): scale parameter for EV type-I shock for agents of type X
        sigma_Y (Array): scale parameter for EV type-I shock for agents of type Y
        wage_scale (Array): scale parameter for wages
    """

    beta_X: Array
    beta_Y: Array
    sigma_X: Array
    sigma_Y: Array
    wage_scale: Array


@dataclass
class MatchingModel(Pytree, mutable=False):
    """Matching model

    Attributes:
        covariates_X (Array): covariates of utility function of agents of type X
        covariates_Y (Array): covariates of utility function of agents of type Y
    """

    covariates_X: Array
    covariates_Y: Array

    def log_ChoiceProbabilities(self, v: Array, axis: int) -> Array:
        """Compute the logit choice probabilities for inside and outside options

        Args:
            v (Array): choice-specific payoffs

        Returns:
        log_p (Array):
            logarithm of choice probabilities of inside options.
        """
        return v - logsumexp(v, axis=axis, keepdims=True)

    def Utility(self, covariates: Array, parameters: Array) -> Array:
        """Computes match-specific utilities

        Args:
            covariates (Array): covariates of utility function
            parameters (Array): parameters of utility function

        Returns:
            utility (Array): match-specific utilities
        """
        return jnp.einsum("ijk, k -> ij", covariates, parameters)

    def log_ChoiceProbabilities_X(
        self, transfer: Array, utility_X: Array, sigma_X: Array
    ) -> Array:
        """Computes agents of type X's demand for agents of type Y

        Args:
            transfer (Array): match-specific transfers
            utility_X (Array): match-specific utilities

        Returns:
            demand (Array): demand for inside options
        """
        v_X = jax.lax.add(utility_X, transfer) / sigma_X
        return self.log_ChoiceProbabilities(v_X, axis=1)

    def log_ChoiceProbabilities_Y(
        self, transfer: Array, utility_Y: Array, sigma_Y: Array
    ) -> Array:
        """Computes agents of type Y's demand for agents of type X

        Args:
            transfer (Array): match-specific transfers
            utility_Y (Array): match-specific utilities

        Returns:
            demand (Array): demand for inside options
        """
        v_Y = jax.lax.sub(utility_Y, transfer) / sigma_Y
        return self.log_ChoiceProbabilities(v_Y, axis=0)

    def UpdateTransfers(
        self,
        t_initial: Array,
        utility_X: Array,
        utility_Y: Array,
        mp: ModelParameters,
    ) -> Array:
        """Updates fixed point equation for transfers

        Args:
            t_initial (Array): initial transfers
            utility_X (Array): utility of agents of type X
            utility_Y (Array): utility of agents of type Y
            mp (ModelParameters): model parameters

        Returns:
            t_updated (Array): updated transfers
        """
        # Calculate demand for both sides of the market
        log_demand_X = self.log_ChoiceProbabilities_X(
            t_initial, utility_X, mp.sigma_X
        )  # type X's demand for type Y
        log_demand_Y = self.log_ChoiceProbabilities_Y(
            t_initial, utility_Y, mp.sigma_Y
        )  # type Y's demand for type X

        # Update transfer
        t_updated = t_initial + 1 / 2 * (log_demand_Y - log_demand_X)
        return t_updated

    def solve(
        self,
        utility_X: Array,
        utility_Y: Array,
        mp: ModelParameters,
        fixed_point_solver: SolverTypes = SquaremAcceleration,
        tol: float = 1e-10,
        maxiter: int = 1_000,
        verbose: bool = False,
    ) -> Array:
        """Solve for equilibrium transfer

        Args:
            utility_X (Array): utilities of agents of type X
            utility_Y (Array): utilities of agents of type Y
            fixed_point_solver (SolverTypes): solver used for solving fixed point equation (FixedPointIteration, AndersonAcceleration, SquaremAcceleration)
            tol (float): stopping tolerance for step length of fixed-point iterations, x_{i+1} - x_{i}
            maxiter (int): maximum number of iterations
            verbose (bool): whether to print information on every iteration or not.

        Returns:
            transfers (Array): equilibrium transfers
        """
        # Initial guess for equilibrium transfers
        transfer_init = jnp.zeros(
            (self.covariates_Y.shape[0], self.covariates_X.shape[1])
        )

        # Find equilibrium transfers
        result = fixed_point_solver(
            self.UpdateTransfers,
            maxiter=maxiter,
            tol=tol,
            verbose=verbose,
        ).run(transfer_init, utility_X, utility_Y, mp)
        return result.params

    def extract_model_parameters(self, params: Array) -> ModelParameters:
        """Extract and store model parameters

        Args:
            params (Array): parameters of agents' utility functions

        Returns:
        mp (ModelParameters):
            model parameters
        """
        number_of_parameters_X = self.covariates_X.shape[-1]
        number_of_parameters_Y = self.covariates_Y.shape[-1]

        parameters_X = params[:number_of_parameters_X]
        parameters_Y = params[
            number_of_parameters_X : number_of_parameters_X + number_of_parameters_Y
        ]

        return ModelParameters(
            beta_X=parameters_X,
            beta_Y=parameters_Y,
            wage_scale=params[-3],
            sigma_X=jnp.exp(params[-2]),
            sigma_Y=jnp.exp(params[-1]),
        )

    def Utilities_of_agents(self, mp: ModelParameters) -> tuple[Array, Array]:
        """Compute match-specific utilities for agents of type X and Y

        Args:
            mp (ModelParameters): model parameters

        Returns:
        utility_X (Array):
            utilities for agents of type X
        utility_Y (Array):
            utilities for agents of type Y
        """

        utility_X = self.Utility(self.covariates_X, mp.beta_X)
        utility_Y = self.Utility(self.covariates_Y, mp.beta_Y)
        return utility_X, utility_Y

    def variance_of_measurement_error(
        self, transfer: Array, observed_transfer: Array, transfer_constant: Array
    ) -> Array:
        """Estimates the variance of the normal distributed measurement error of the transfers

        Args:
            transfer (Array): model consistent trasfers
            observed_transfer (Array): observed transfers
            transfer_constant (Array): transfer constant

        Returns:
            sigma^2 (Array): variance of the measurement error
        """
        transfer_diag = jnp.diag(transfer) + transfer_constant
        return jnp.mean((transfer_diag - observed_transfer) ** 2)

    def neg_log_likelihood(self, params: Array, observed_transfer: Array) -> Array:
        """Computes the negative log-likelihood function

        Args:
            params (Array): parameters of agents' utility functions
            observed_transfer (Array): observed transfers and numbers of matched and unmatched agents

        Returns:
            neg_log_lik (Array): negative log-likelihood value
        """
        mp = self.extract_model_parameters(params)
        utility_X, utility_Y = self.Utilities_of_agents(mp)

        transfer = self.solve(utility_X, utility_Y, mp)

        number_of_observations = 2 * observed_transfer.size

        log_lik_transfer = -jnp.log(
            self.variance_of_measurement_error(transfer, observed_transfer, mp.wage_scale)
        ) * (observed_transfer.size / 2)
        log_lik_matched_X = jnp.nansum(
            jnp.diag(self.log_ChoiceProbabilities_X(transfer, utility_X, mp.sigma_X))
        )
        log_lik_matched_Y = jnp.nansum(
            jnp.diag(self.log_ChoiceProbabilities_Y(transfer, utility_Y, mp.sigma_Y))
        )

        neg_log_lik = (
            -(log_lik_transfer + log_lik_matched_X + log_lik_matched_Y)
            / number_of_observations
        )

        return neg_log_lik

    def fit(
        self,
        guess: Array,
        observed_transfer: Array,
        tol: float = 1e-8,
        maxiter: int = 1_000,
        verbose: bool | int = True,
    ) -> Array:
        """Estimate parameters of matching model by maximum likelihood (minimize the negative log-likelihood function)

        Args:
            guess (Array): initial parameter guess
            observed_transfer (Array): observed transfers and numbers of matched and unmatched agents
            tol (float): tolerance of the stopping criterion
            maxiter (int): maximum number of proximal gradient descent iterations
            verbose (bool): if set to True or 1 prints the information at each step of the solver, if set to 2, print also the information of the linesearch

        Returns:
            params (Array): parameter estimates
        """

        result = LBFGS(
            fun=self.neg_log_likelihood,
            tol=tol,
            maxiter=maxiter,
            verbose=verbose,
        ).run(guess, observed_transfer)
        return result.params
