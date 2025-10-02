
# Description
This project aims at replicating the empirical results of [Dupuy and Galichon (2022)](https://doi.org/10.3982/QE928) who estimates the value of a statistical life using compensating wage differentials for the risk of fatal injury on the job. Using US data for 2017, they estimate the value of statistical life to be $6.3 million ($2017). Their estimate relies on maximum likelihood estimation of a one-to-one matching model with transferable utility, where the demand and supply of labor is given by the logit formula. The estimation procedure simultaneously fits both the matching patterns and the wage curve.

## Replication of descriptive statistics
Below, we have succesfully replicated the descriptive statistics of Dupuy and Galichon (2022).

|                     |   mean |   std |   min |    max |
|:--------------------|-------:|------:|------:|-------:|
| Wage (hourly)       |  17.95 |  9.02 |  3.75 |  70.00 |
| Years of schooling  |  13.35 |  2.24 |  1.00 |  21.00 |
| Years of experience |  20.67 | 12.97 |  0.00 |  51.00 |
| Female              |   0.52 |  0.50 |  0.00 |   1.00 |
| Married             |   0.50 |  0.50 |  0.00 |   1.00 |
| White               |   0.63 |  0.48 |  0.00 |   1.00 |
| Black               |   0.12 |  0.32 |  0.00 |   1.00 |
| Asian               |   0.06 |  0.24 |  0.00 |   1.00 |
| Risk (per 100,000)  |   3.44 | 13.05 |  0.00 | 345.70 |
| Public              |   0.12 |  0.33 |  0.00 |   1.00 |

## Replication of maximum likelihood estimates
The table below compares our estimates to Dupuy and Galichon (2022).

|                               |   Dupuy and Galichon (2022) |   Andersen (2025) |
|:------------------------------|----------------------------:|------------------:|
| Years of schooling            |                       0.057 |            -0.037 |
| Years of experience           |                       0.084 |             0.457 |
| Female                        |                      -0.404 |             0.986 |
| Married                       |                       0.050 |             0.118 |
| White                         |                       0.046 |             0.048 |
| Black                         |                      -0.108 |            -0.228 |
| Asian                         |                      -0.069 |             0.040 |
| Years of experience (squared) |                      -0.051 |             0.272 |
| Risk x Years of schooling     |                      -0.059 |             0.125 |
| Public x Years of schooling   |                       0.074 |             0.594 |
| Risk x Years of experience    |                      -2.388 |            -0.062 |
| Public x Years of experience  |                       0.838 |             0.015 |
| Risk x Females                |                       0.096 |             1.014 |
| Public x Females              |                       0.548 |             0.007 |
| Risk                          |                      -0.023 |            -0.003 |
| Public                        |                      -0.062 |             0.113 |
| Public x Years of schooling   |                       0.081 |             0.780 |
| Salary constant               |                       2.981 |             2.653 |
| Scale parameter (workers)     |                       0.046 |             1.000 |
| Scale parameter (firms)       |                       2.233 |             1.000 |

