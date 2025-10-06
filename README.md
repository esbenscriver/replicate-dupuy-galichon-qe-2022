[![Test](https://github.com/esbenscriver/replicate-dupuy-galichon-qe-2022/actions/workflows/test.yml/badge.svg)](https://github.com/esbenscriver/replicate-dupuy-galichon-qe-2022/actions/workflows/test.yml)
# Description
This project aims to replicate the empirical results of [Dupuy and Galichon (2022)](https://doi.org/10.3982/QE928), who estimate the value of job amenities and labor productivity in the United States for 2017. Their estimation relies on maximum likelihood estimation of a one-to-one matching model with transferable utility, in which the demand and supply of labor are described by a logit specification.

Dupuy and Galichon have made their Matlab code and dataset publicly available. However, we were unable to execute their code without making modifications. Therefore, we implemented their estimation procedure in Python, and our implementation is publicly available in this repository that also includes the dataset.

From our estimation results, we conclude that the parameter estimates reported by Dupuy and Galichon cannot be fully reproduced. Based on their own parameter estimates and the average earnings in their sample, they report a value of statistical life of $6.3 million (in 2017 US dollars). However, as their final dataset does not include information on earnings — only imputed hourly wages — we are unable to verify this calculation.

## Replication of descriptive statistics
The dataset consists of a cross-sectional sample of 3,454 employed individuals from 2017. Below, we successfully replicate Table 1 from Dupuy and Galichon, which presents descriptive statistics.

|                     |   Mean |   Std |   Min |    Max |
|:--------------------|-------:|------:|------:|-------:|
| Workers             |        |       |       |        |
| Years of schooling  |  13.35 |  2.24 |  1.00 |  21.00 |
| Years of experience |  20.67 | 12.97 |  0.00 |  51.00 |
| Female              |   0.52 |  0.50 |  0.00 |   1.00 |
| Married             |   0.50 |  0.50 |  0.00 |   1.00 |
| White               |   0.63 |  0.48 |  0.00 |   1.00 |
| Black               |   0.12 |  0.32 |  0.00 |   1.00 |
| Asian               |   0.06 |  0.24 |  0.00 |   1.00 |
| Wage (hourly)       |  17.95 |  9.02 |  3.75 |  70.00 |
| Firms               |        |       |       |        |
| Risk (per 100,000)  |   3.44 | 13.05 |  0.00 | 345.70 |
| Public              |   0.12 |  0.33 |  0.00 |   1.00 |

The variable risk measures the average number of fatal injuries per 100,000 by the occupation and industry the individual is employed.

## Replication of maximum likelihood estimates
For estimation, we apply the same transformations to the data as Dupuy and Galichon, who use the logarithm of hourly wages and standardize the variables years of schooling, years of experience, and risk.

The table below compares our estimates to Dupuy and Galichon. As shown we are not able to fully recover their estimates.

From the table, we observe that our estimates for job attributes (rows 1–3) closely resemble those of Dupuy and Galichon. However, the estimated worker productivity terms (rows 4–14) do not match their estimates. Additionally, our estimated salary constant and the scale parameter of the firms’ taste shocks differ substantially from their reported values.

|                               |   Dupuy and Galichon (2022) |   Our estimates |
|:------------------------------|----------------------------:|----------------:|
| Risk                          |                      -0.023 |          -0.023 |
| Public                        |                      -0.062 |          -0.061 |
| Public x Years of schooling   |                       0.081 |           0.082 |
| Years of schooling            |                       0.057 |           0.054 |
| Years of experience           |                       0.084 |           0.071 |
| Female                        |                      -0.404 |          -0.403 |
| Married                       |                       0.050 |           0.052 |
| White                         |                       0.046 |           0.048 |
| Black                         |                      -0.108 |          -0.105 |
| Asian                         |                      -0.069 |           0.074 |
| Years of experience (squared) |                      -0.051 |          -0.051 |
| Risk x Years of schooling     |                      -0.059 |          -0.069 |
| Risk x Years of experience    |                       0.074 |           0.078 |
| Risk x Females                |                      -2.388 |          -2.388 |
| Public x Years of schooling   |                       0.838 |           0.863 |
| Public x Years of experience  |                       0.096 |           0.212 |
| Public x Females              |                       0.548 |           0.527 |
| Salary constant               |                       2.981 |           2.606 |
| Scale parameter (workers)     |                       0.046 |           0.047 |
| Scale parameter (firms)       |                       2.233 |           2.223 |

During the maximization of the log-likelihood function, our optimizer terminated prematurely as it failed to converge with a tolerance level of $1e-6$. However, our obtained average log-likelihood value of $-5.076$ is much larger than the average log-likelihood value of $-5.191$ implied by the estimates of Dupuy and Galichon. It should be emphasized that in this later exercise we rely on their rounded parameter estimates.

Recall that the measurment errors, $\varepsilon_{i}$, are assumed to be iid normal distributed, $N(m,s^2)$, with mean zero, $m=0$. The table below reports the implied mean and variance of the wage measurement errors given the parameter estimates. We estimate the variance of the measurement errors to $0.140$ which corresponds to a $R^2$ of $0.233$.

|                 |   Dupuy and Galichon (2022) |   Our estimates |
|:----------------|----------------------------:|----------------:|
| mean, $m$       |                      -0.369 |          -0.000 |
| variance, $s^2$ |                       0.276 |           0.140 |

Observe that, if the parameter vector $\hat{\Theta}$ maximizes the likelihood function, the inclusion of a salary constant implies that the mean of the measurement error is zero. Hence, including a salary constant is equivalent to a zero mean for the measurement error. Consequently, the salary constant can be concentrated out of the likelihood function in the same manner as the variance of the measurement error,

$$
    \hat{\varepsilon}_{i}(\Theta) = w_{i} - \hat{w}_{i}(\Theta), 
$$
$$
    \hat{m}(\Theta) = \tfrac{1}{N} \sum_{i=1}^N \hat{\varepsilon}_{i}(\Theta),
$$
$$
    \hat{s}^2(\Theta) = \tfrac{1}{N} \sum_{i=1}^N \left(\hat{\varepsilon}_{i}(\Theta) - \hat{m}(\Theta)\right)^2.
$$

In turn, we would have one less parameter to optimize the log-likelihood function with respect to.

