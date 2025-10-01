
# Description
This project aims at replicating the empirical results of [Dupuy and Galichon (2022)](https://doi.org/10.3982/QE928) who estimates the value of a statistical life using compensating wage differentials for the risk of fatal injury on the job. Using US data for 2017, they estimate the value of statistical life to be $6.3 million ($2017). Their estimate relies on maximum likelihood estimation of a one-to-one matching model with transferable utility, where the demand and supply of labor is given by the logit formula. The estimation procedure simultaneously fits both the matching patterns and the wage curve.

## Estimates
|                               |   Dupuy-Galichon (2022) |   estimates |
|:------------------------------|------------------------:|------------:|
| Years of schooling            |                   0.057 |       0.953 |
| Years of experience           |                   0.084 |       0.993 |
| Female                        |                  -0.404 |       0.999 |
| Married                       |                   0.05  |       1.022 |
| White                         |                   0.046 |       1.02  |
| Black                         |                  -0.108 |       0.959 |
| Asian                         |                  -0.069 |       0.961 |
| Years of experience (squared) |                  -0.051 |       0.971 |
| Risk x Years of schooling     |                  -0.059 |       1.007 |
| Public x Years of schooling   |                   0.074 |       1.003 |
| Risk x Years of experience    |                  -2.388 |       1.006 |
| Public x Years of experience  |                   0.838 |       0.97  |
| Risk x Females                |                   0.096 |       1.001 |
| Public x Females              |                   0.548 |       0.971 |
| Risk                          |                  -0.023 |       0.995 |
| Public                        |                  -0.062 |       1.014 |
| Public x Years of schooling   |                   0.081 |       1.002 |
| salary constant               |                   2.981 |       0.064 |
| scale (X)                     |                   0.046 |       1     |
| scale (Y)                     |                   2.233 |       1     |

## Descriptives
|                     |   mean |   std |   min |   max |
|:--------------------|-------:|------:|------:|------:|
| Wage (hourly)       |  17.95 |  9.02 |  3.75 |  70   |
| Years of schooling  |  13.35 |  2.24 |  1    |  21   |
| Years of experience |  20.67 | 12.97 |  0    |  51   |
| Female              |   0.52 |  0.5  |  0    |   1   |
| Married             |   0.5  |  0.5  |  0    |   1   |
| White               |   0.63 |  0.48 |  0    |   1   |
| Black               |   0.12 |  0.32 |  0    |   1   |
| Asian               |   0.06 |  0.24 |  0    |   1   |
| Risk (per 100,000)  |   3.44 | 13.05 |  0    | 345.7 |
| Public              |   0.12 |  0.33 |  0    |   1   |

