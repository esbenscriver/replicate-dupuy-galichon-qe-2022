"""
JAX implementation of fixed-point iteration algoritm to solve and one-to-one matching model with transferable utility

Reference:
Esben Scriver Andersen, Note on solving one-to-one matching models with linear transferable utility, 2025 (https://arxiv.org/pdf/2409.05518)
"""

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import logsumexp
from jax.scipy.optimize import minimize, OptimizeResults

# import simple_pytree (used to store variables)
from simple_pytree import Pytree, dataclass, static_field

# import solvers
from jaxopt import FixedPointIteration, AndersonAcceleration
from squarem_jaxopt import SquaremAcceleration

SolverTypes = (
    type[SquaremAcceleration] | type[AndersonAcceleration] | type[FixedPointIteration]
)


@dataclass
class Data(Pytree, mutable=False):
    """Observed data

    Attributes:
        transfers (Array): observed transfers
        matches (Array): observed matches
    """

    transfers: Array
    matches: Array


@dataclass
class ModelParameters(Pytree, mutable=False):
    """Model parameters to be estimated

    Attributes:
        beta_X (Array): linear utility parameters of agent of type X
        beta_Y (Array): linear utility parameters of agent of type Y
        sigma_X (Array): scale parameter for EV type-I shock for agents of type X
        sigma_Y (Array): scale parameter for EV type-I shock for agents of type Y
        transfer_constant (Array): Additive constant for transfers
    """

    beta_X: Array
    beta_Y: Array
    sigma_X: Array
    sigma_Y: Array
    transfer_constant: Array


@dataclass
class MatchingModel(Pytree, mutable=False):
    """Matching model

    Attributes:
        covariates_X (Array): covariates of utility function of agents of type X
        covariates_Y (Array): covariates of utility function of agents of type Y
        marginal_distribution_X (Array): marginal distribution of agents of type X
        marginal_distribution_Y (Array): marginal distribution of agents of type Y
        continuous_distributed_attributes (bool): replicate empirical model of Dupuy and Galichon (2022) if True
        include_scale_parameters (bool): include scale parameters if True
        include_salary_constant (bool): use centered variance of measurement errors if True
    """

    covariates_X: Array
    covariates_Y: Array

    marginal_distribution_X: Array
    marginal_distribution_Y: Array

    continuous_distributed_attributes: bool = static_field(default=True)
    include_transfer_constant: bool = static_field(default=True)
    include_scale_parameters: bool = static_field(default=True)
    reference: int = static_field(default=0)
    log_transform_scale: bool = static_field(default=True)

    def ChoiceProbabilities(self, v: Array, axis: int) -> Array:
        """Compute the logit choice probabilities for inside and outside options

        Args:
            v (Array): choice-specific payoffs
            axis (int): axis that describes the choice set

        Returns:
        P_inside (Array):
            choice probabilities of inside options.
        P_outside (Array):
            choice probabilities of outside option.
        """
        v_max = jnp.max(v, axis=axis, keepdims=True)

        # exponentiated centered payoffs of inside options
        nominator = jnp.exp(v - v_max)

        # denominator of choice probabilities
        denominator = jnp.sum(nominator, axis=axis, keepdims=True)
        return nominator / denominator
    
    def log_ChoiceProbabilities(self, v: Array, axis: int) -> Array:
        """ compute the log of the logit choice probabililities using logsumexp from scipy.special
        
        Args:
            v (Array): choice-specific payoffs
            axis (int): axis that describes the choice set
            
        Returns:
            log_p (Array): log of choice probabilities.
        """

        return v - logsumexp(v, axis=axis, keepdims=True)

    def Utility(self, covariates: Array, parameters: Array) -> Array:
        """Computes match-specific utilities

        Args:
            covariates (Array): covariates of utility function
            parameters (Array): parameters of utility function

        Returns:
            demand (Array): match-specific utilities
        """
        return jnp.einsum("ijk, k -> ij", covariates, parameters)

    def Payoff_X(self, transfer: Array, utility_X: Array, sigma_X: Array) -> Array:
        """Computes match-specific payoffs for agents of type X

        Args:
            transfer (Array): match-specific transfers
            utility_X (Array): match-specific utilities for agents of type X
            sigma_X (Array): scale parameter for EV type-I shock for agents of type X

        Returns:

            payoff_X (Array): match-specific payoffs for agents of type X
        """
        return jax.lax.add(utility_X, transfer) / sigma_X
    
    def Payoff_Y(self, transfer: Array, utility_Y: Array, sigma_Y: Array) -> Array:
        """Computes match-specific payoffs for agents of type Y

        Args:
            transfer (Array): match-specific transfers
            utility_Y (Array): match-specific utilities for agents of type Y
            sigma_Y (Array): scale parameter for EV type-I shock for agents of type Y

        Returns:
            payoff_Y (Array): match-specific payoffs for agents of type Y
        """
        return jax.lax.sub(utility_Y, transfer) / sigma_Y

    def ChoiceProbabilities_X(
        self, transfer: Array, utility_X: Array, sigma_X: Array
    ) -> Array:
        """Computes choice probabilities of agents of type X

        Args:
            transfer (Array): match-specific transfers
            utility_X (Array): match-specific utilities for agents of type X

        Returns:
            ChoiceProbabilities (Array): match-specific choice probabilities for agents of type X
        """
        v_X = self.Payoff_X(transfer, utility_X, sigma_X)
        return self.ChoiceProbabilities(v_X, axis=1)
    
    def log_ChoiceProbabilities_X(
        self, transfer: Array, utility_X: Array, sigma_X: Array
    ) -> Array:
        """Compute the log of the logit choice probabililities using logsumexp from scipy.special
        
        Args:
            transfer (Array): match-specific transfers
            utility_X (Array): match-specific utilities for agents of type X
            sigma_X (Array): scale parameter for EV type-I shock for agents of type X

        Returns:
            log_p (Array): log of choice probabilities.
        """
        v_X = self.Payoff_X(transfer, utility_X, sigma_X)
        return self.log_ChoiceProbabilities(v_X, axis=1)
    
    def ChoiceProbabilities_Y(
        self, transfer: Array, utility_Y: Array, sigma_Y: Array
    ) -> Array:
        """Computes choice probabilities of agents of type Y

        Args:
            transfer (Array): match-specific transfers
            utility_Y (Array): match-specific utilities for agents of type Y

        Returns:
            ChoiceProbabilities (Array): match-specific choice probabilities for agents of type Y
        """
        v_Y = self.Payoff_Y(transfer, utility_Y, sigma_Y)
        return self.ChoiceProbabilities(v_Y, axis=0)
    
    def log_ChoiceProbabilities_Y(
        self, transfer: Array, utility_Y: Array, sigma_Y: Array
    ) -> Array:
        """Compute the log of the logit choice probabililities using logsumexp from scipy.special
        
        Args:
            transfer (Array): match-specific transfers
            utility_Y (Array): match-specific utilities for agents of type Y
            sigma_Y (Array): scale parameter for EV type-I shock for agents of type Y

        Returns:
            log_p (Array): log of choice probabilities.
        """
        v_Y = self.Payoff_Y(transfer, utility_Y, sigma_Y)
        return self.log_ChoiceProbabilities(v_Y, axis=0)

    def log_Demand_X(self, transfer: Array, utility_X: Array, sigma_X) -> Array:
        """Computes agents of type X's demand for agents of type Y

        Args:
            transfer (Array): match-specific transfers
            utility_X (Array): match-specific utilities

        Returns:
            demand (Array): demand for inside options
        """
        return jnp.log(self.marginal_distribution_X) + self.log_ChoiceProbabilities_X(
            transfer, utility_X, sigma_X
        )

    def log_Demand_Y(self, transfer: Array, utility_Y: Array, sigma_Y) -> Array:
        """Computes agents of type Y's demand for agents of type X

        Args:
            transfer (Array): match-specific transfers
            utility_Y (Array): match-specific utilities

        Returns:
            demand (Array): demand for inside options
        """
        return jnp.log(self.marginal_distribution_Y) + self.log_ChoiceProbabilities_Y(
            transfer, utility_Y, sigma_Y
        )

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

        Returns:
            t_updated (Array): updated transfers
        """
        # Calculate demand for both sides of the market
        log_demand_X = self.log_Demand_X(
            t_initial, utility_X, mp.sigma_X
        )
        log_demand_Y = self.log_Demand_Y(
            t_initial, utility_Y, mp.sigma_Y
        )

        adjust_step = (mp.sigma_X * mp.sigma_Y) / (mp.sigma_X + mp.sigma_Y)

        # Update transfer
        t_updated = t_initial + adjust_step * (log_demand_Y - log_demand_X)
        return t_updated - t_updated[self.reference, self.reference] # normalize transfers

    def solve(
        self,
        utility_X: Array,
        utility_Y: Array,
        mp: ModelParameters,
        fixed_point_solver: SolverTypes = SquaremAcceleration,
        tol: float = 1e-10,
        maxiter: int = 1_0,
        verbose: bool = False,
    ) -> Array:
        """Solve for equilibrium transfer

        Args:
            utility_X (Array): utilities of agents of type X
            utility_Y (Array): utilities of agents of type Y
            fixed_point_solver (SolverTypes): solver used for solving fixed-point equation (FixedPointIteration, AndersonAcceleration, SquaremAcceleration)
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
        # print(result.state)
        return result.params

    def extract_model_parameters(self, params: Array) -> ModelParameters:
        """Extract and store model parameters

        Args:
            params (Array): parameters of agents' utility functions

        Returns:
        mp (ModelParameters):
            model parameters
        """
        number_of_covariates_X = self.covariates_X.shape[-1]
        number_of_covariates_Y = self.covariates_Y.shape[-1]

        beta_X = params[:number_of_covariates_X]
        beta_Y = params[
            number_of_covariates_X : number_of_covariates_X + number_of_covariates_Y
        ]

        # Use static conditions since they're model attributes, not traced values
        if self.include_transfer_constant and self.include_scale_parameters:
            transfer_constant = params[-3:-2]
        elif self.include_transfer_constant and not self.include_scale_parameters:
            transfer_constant = params[-1:]
        else:
            transfer_constant = jnp.asarray([0.0])

        if self.include_scale_parameters and self.log_transform_scale:
            sigma_X = jnp.exp(params[-2:-1])
            sigma_Y = jnp.exp(params[-1:])
        elif self.include_scale_parameters and not self.log_transform_scale:
            sigma_X = params[-2:-1]
            sigma_Y = params[-1:]
        else:
            sigma_X = jnp.asarray([1.0])
            sigma_Y = jnp.asarray([1.0])

        return ModelParameters(
            beta_X=beta_X,
            beta_Y=beta_Y,
            transfer_constant=transfer_constant,
            sigma_X=sigma_X,
            sigma_Y=sigma_Y,
        )

    def transform_parameters(self, params_raw: Array) -> Array:
        """Transform model parameters

        Args:
            params_raw (Array): raw model parameters

        Returns:
        params (Array):
            transformed model parameters
        """
        mp = self.extract_model_parameters(params_raw)
        params = jnp.concatenate([mp.beta_X, mp.beta_Y], axis=0)

        if self.include_transfer_constant:
            params = jnp.concatenate([params, mp.transfer_constant], axis=0)

        if self.include_scale_parameters:
            params = jnp.concatenate([params, mp.sigma_X, mp.sigma_Y], axis=0)
        return params

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

    def moments_of_measurement_error(
        self, model_transfer: Array, observed_transfer: Array
    ) -> tuple[Array, Array, Array]:
        """Estimates the mean and variance of the normal distributed measurement error of the transfers

        Args:
            model_transfer (Array): model consistent trasfers
            observed_transfer (Array): observed transfers

        Returns:
            mu (Array): 
        mean of the measurement error
            sigma^2 (Array): 
        variance of the measurement error
            R_squared (Array): 
        R-squared of measurement error
        """
        error = observed_transfer - model_transfer

        mu = jnp.mean(error)        

        if self.include_transfer_constant:
            sigma_sq = jnp.mean(error**2)
        else:
            sigma_sq =  jnp.mean((error - mu) ** 2)

        SST = jnp.var(observed_transfer)
        R_squared = 1 - sigma_sq / SST
        return sigma_sq, mu, R_squared

    def compute_moments(self, params: Array, data: Data) -> tuple[Array, Array, Array]:
        """Computes the mean and variance of the measurement errors

        Args:
            params (Array): parameters of agents' utility functions
            data (Data): observed data

        Returns:
        mu (Array):
            mean of measurement error
        sigma^2 (Array):
            variance of measurement error
        """
        mp = self.extract_model_parameters(params)
        utility_X, utility_Y = self.Utilities_of_agents(mp)

        transfer = self.solve(utility_X, utility_Y, mp) + mp.transfer_constant

        if self.continuous_distributed_attributes:
            model_transfer = jnp.diag(transfer)
        else:
            model_transfer = transfer

        return self.moments_of_measurement_error(model_transfer, data.transfers)
        
    def neg_log_likelihood(self, params: Array, data: Data) -> Array:
        """Computes the negative log-likelihood function

        Args:
            params (Array): parameters of agents' utility functions
            data (Data): observed data

        Returns:
            neg_log_lik (Array): negative log-likelihood value
        """
        mp = self.extract_model_parameters(params)
        utility_X, utility_Y = self.Utilities_of_agents(mp)

        transfer = self.solve(utility_X, utility_Y, mp)

        if self.continuous_distributed_attributes:
            model_transfer = jnp.diag(transfer) + mp.transfer_constant
            log_pX = jnp.diag(self.log_ChoiceProbabilities_X(transfer, utility_X, mp.sigma_X))
            log_pY = jnp.diag(self.log_ChoiceProbabilities_Y(transfer, utility_Y, mp.sigma_Y))
        else:
            model_transfer = transfer + mp.transfer_constant
            log_pX = self.log_ChoiceProbabilities_X(transfer, utility_X, mp.sigma_X)
            log_pY = self.log_ChoiceProbabilities_Y(transfer, utility_Y, mp.sigma_Y)

        number_of_observations = data.transfers.size + 2 * data.matches.sum()

        variance_of_error = self.moments_of_measurement_error(model_transfer, data.transfers)[0]

        log_lik_transfers= -jnp.log(variance_of_error) * (data.transfers.size / 2)
        log_lik_matches = jnp.sum(data.matches * (log_pX + log_pY))
        
        neg_log_lik = -(log_lik_transfers + log_lik_matches) / number_of_observations

        return neg_log_lik

    def fit(
        self,
        guess: Array,
        data: Data,
        tol: float = 1e-6,
        maxiter: int = 1_000,
        verbose: bool | int = True,
    ) -> Array:
        """Estimate parameters of matching model by maximum likelihood (minimize the negative log-likelihood function)

        Args:
            guess (Array): initial parameter guess
            data (Data): observed data
            tol (float): tolerance of the stopping criterion
            maxiter (int): maximum number of proximal gradient descent iterations
            verbose (bool): if set to True or 1 prints the information at each step of the solver, if set to 2, print also the information of the linesearch

        Returns:
            params (Array): parameter estimates
        """
        assert jnp.isclose(
            jnp.sum(self.marginal_distribution_X), jnp.sum(self.marginal_distribution_Y)
        )

        result = minimize(
            lambda x: self.neg_log_likelihood(x, data),
            guess,
            method="BFGS",
            tol=tol,
            options={"maxiter": maxiter},
        )
        print(
            f"\niterations: {result.nit}, status: {result.status}, final gradient norm: {jnp.linalg.norm(result.jac):.6f}"
        )
        print(f"\nGradients:\n {result.jac}\n")
        return result.x

        # result = BFGS(
        #     fun=self.neg_log_likelihood,
        #     tol=tol,
        #     maxiter=maxiter,
        #     verbose=verbose,
        #     jit=False,
        # ).run(guess, data)

        # print(f"\niterations: {result.state.iter_num}, final gradient norm: {jnp.linalg.norm(result.state.grad)}")
        # print(f"\nGradients:\n {result.state.grad}\n")
        # return result.params
