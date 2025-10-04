
# Description
This project aims at replicating the empirical results of [Dupuy and Galichon (2022)](https://doi.org/10.3982/QE928) who estimates the value of a statistical life using compensating wage differentials for the risk of fatal injury on the job. Using US data for 2017, they estimate the value of statistical life to be $6.3 million ($2017). Their estimate relies on maximum likelihood estimation of a one-to-one matching model with transferable utility, where the demand and supply of labor is given by the logit formula. The estimation procedure simultaneously fits both the matching patterns and the wage curve.

## Replication of descriptive statistics
Below, we have succesfully replicated Table 1 of Dupuy and Galichon (2022) that show some descriptive statistics for the analyzed data set.

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

## Replication of maximum likelihood estimates
The table below compares our estimates to Dupuy and Galichon (2022).

|                               |   Dupuy and Galichon (2022) |   Andersen (2025) |
|:------------------------------|----------------------------:|------------------:|
| Risk                          |                      -0.023 |            -0.023 |
| Public                        |                      -0.062 |            -0.061 |
| Public x Years of schooling   |                       0.081 |             0.080 |
| Years of schooling            |                       0.057 |             0.062 |
| Years of experience           |                       0.084 |             0.070 |
| Female                        |                      -0.404 |            -0.368 |
| Married                       |                       0.050 |             0.053 |
| White                         |                       0.046 |             0.046 |
| Black                         |                      -0.108 |            -0.106 |
| Asian                         |                      -0.069 |             0.070 |
| Years of experience (squared) |                      -0.051 |            -0.051 |
| Risk x Years of schooling     |                      -0.059 |            -0.065 |
| Risk x Years of experience    |                       0.074 |             0.072 |
| Risk x Females                |                      -2.388 |            -2.137 |
| Public x Years of schooling   |                       0.838 |             0.798 |
| Public x Years of experience  |                       0.096 |             0.221 |
| Public x Females              |                       0.548 |             0.402 |
| Salary constant               |                       2.981 |             2.608 |
| Scale parameter (workers)     |                       0.046 |             0.047 |
| Scale parameter (firms)       |                       2.233 |             1.977 |

The table below reports the implied mean and variance of the wage measurement errors given the parameter estimates.

|          |   Dupuy and Galichon (2022) |   Andersen (2025) |
|:---------|----------------------------:|------------------:|
| mean     |                      -0.369 |             0.000 |
| variance |                       0.276 |             0.140 |

Notice, that if the estimated parameters maximize the likelihood function, including a salary constant effectively imposes that the mean of the measurement errors is zero.

Thus, including a salary constant is equivalent to allowing the measurement errors to have a non-zero mean.
Consequently, the salary constant can be concentrated out of the likelihood function in the same way as the variance of the measurement error,

$$
    \hat{\varepsilon}_{i}(\Theta) = w_{i} - \hat{w}_{i}(\Theta), 
$$
$$
    \hat{\mu}(\Theta) = \tfrac{1}{N} \sum_{i=1}^N \hat{\varepsilon}_{i}(\Theta),
$$
$$
    \hat{\sigma}^2(\Theta) = \tfrac{1}{N} \sum_{i=1}^N \left(\hat{\varepsilon}_{i}(\Theta) - \hat{\mu}(\Theta)\right)^2.
$$

In turn, we have one less parameter to optimize the log-likelihood function with respect to.

