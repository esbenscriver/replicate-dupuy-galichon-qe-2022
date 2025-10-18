[![Test](https://github.com/esbenscriver/replicate-dupuy-galichon-qe-2022/actions/workflows/test.yml/badge.svg)](https://github.com/esbenscriver/replicate-dupuy-galichon-qe-2022/actions/workflows/test.yml)
# Description
This project aims to replicate the empirical results of [Dupuy and Galichon (2022)](https://doi.org/10.3982/QE928), who estimate the value of job amenities and labor productivity in the United States for 2017. Their estimation relies on maximum likelihood estimation of a one-to-one matching model with transferable utility, in which the demand and supply of labor are described by a logit specification.

Dupuy and Galichon have made their Matlab code and dataset publicly available. However, we were unable to execute their code without making modifications. Therefore, we implemented their estimation procedure in Python, and our implementation is publicly available in this repository that also includes the dataset.

From our estimation results, we conclude that the parameter estimates reported by Dupuy and Galichon cannot be fully reproduced. Based on their own parameter estimates and the average earnings in their sample, they report a value of statistical life of $6.3 million (in 2017 US dollars). However, as their final dataset does not include information on earnings — only imputed hourly wages — we are unable to verify this calculation.

## Replication of descriptive statistics
The dataset consists of a cross-sectional sample of 3,454 employed individuals from 2017. Below, we successfully replicate Table 1 from Dupuy and Galichon, which presents descriptive statistics.

{{output/summary_stats.md}}

The variable risk measures the average number of fatal injuries per 100,000 by the occupation and industry the individual is employed.

## Replication of maximum likelihood estimates
For estimation, we apply the same transformations to the data as Dupuy and Galichon, who use the logarithm of hourly wages and standardize the variables years of schooling, years of experience, and risk of fatal injuries.

The table below compares our estimates to Dupuy and Galichon. As shown we are not able to fully recover their estimates.

From the table, we observe that our estimates for job attributes (rows 1–3) closely resemble those of Dupuy and Galichon. However, the estimated worker productivity terms (rows 4–14) do not match their estimates. Additionally, our estimated scale parameter of the firms’ taste shocks differ substantially from their reported values.

{{output/estimated_parameters_standardize_True.md}}

During the maximization of the log-likelihood function, our optimizer terminates prematurely because it fails to converge to the specified tolerance level of $10^{-6}$. Therefore, we experimented with several different starting values — including the reported estimates of Dupuy and Galichon — and retained the estimates that achieved the highest likelihood. The table below shows the our achieved log-likelihood value and R-squared. It should be emphasized that the reported results for Dupuy and Galichon are the implied values derived from their rounded parameter estimates.  

{{output/objective_standardize_True.md}}

Recall that Dupuy and Galichon assume that the measurment errors, $\varepsilon_{i}$, are iid normal distributed, $N(m,s^2)$, with mean zero, $m=0$. However, the inclusion of a salary constant is similar to allow the measurement errors to have a non-zero mean. Consequently, the salary constant can be concentrated out of the likelihood function in the same manner as the mean or variance of the measurement errors,

$$
    \hat{\varepsilon}_{i}(\Theta) = w_{i} - \hat{w}_{i}(\Theta), 
$$
$$
    \hat{m}(\Theta) = \tfrac{1}{N} \sum_{i=1}^N \hat{\varepsilon}_{i}(\Theta),
$$
$$
    \hat{s}^2(\Theta) = \tfrac{1}{N} \sum_{i=1}^N \left(\hat{\varepsilon}_{i}(\Theta) - \hat{m}(\Theta)\right)^2.
$$

The table below reports the implied mean and variance of the wage measurement errors given the parameter estimates, where the mean of the measurement can be interpreted as a salary constant. Note that the mean differs from the salary constant reported by Dupuy and Galichon. This is because we use a different normalization of the wage distribution then they impose. We need to impose a normalization, as the wage distribution is otherwise not uniquely determined.

{{output/estimated_moments_standardize_True.md}}


