# Seminar
## Potential models
### Random effects model

```
y[i, t] = alpha[i] + beta[i] * x[i, t] + gamma[i] * w[t] + epsilon[i, t]
alpha[i] = alpha_0 + eta_alpha[i]
beta[i] = beta_0 + eta_beta[i]
gamma[i] = gamma_0 + eta_gamma[i]
alpha_0 ~ N(0, tau_alpha)
beta_0 ~ N(0, tau_beta)
gamma_0 ~ N(0, tau_gamma)
eta_alpha[i] ~ N(0, tau_eta_alpha)
eta_beta[i] ~ N(0, tau_eta_beta)
eta_gamma[i] ~ N(0, tau_eta_gamma)
tau_alpha ~ Gamma(1, 1)
tau_beta ~ Gamma(1, 1)
tau_gamma ~ Gamma(1, 1)
tau_eta_alpha ~ Gamma(1, 1)
tau_eta_beta ~ Gamma(1, 1)
tau_eta_gamma ~ Gamma(1, 1)

epsilon[i, t] ~ N(0, tau_epsilon)
tau_epsilon ~ Gamma(1, 1)
```

Where `y` is inflation, `x` are country-specific time-varying regressors, and `w` is a global time-varying regressor.

### Distance-based model
```
y[i,t] = alpha[i] + beta[i] * x[i, t] + gamma[i] * w[t] + epsilon[i, t]
... (fewer constraints on beta than above)
beta[i] - beta[j] ~ N(0, dist(i, j) * sigma_beta) 
# incorporate prior that betas are similar for closeby countries
# only for countries i and j that are main trade partners
```
distance based on 1. trades 2. geographical similarity within continent

## Techniques to incorporate
### Check convergence of MCMC
- autocorrelation 
- trace plot
- Gelman-Rubin statistic
- effective sample size

Important:
- use sufficient burn-in

### Obtain acceptance rate of around 40% for MCMC

### Brier score to evaluate forecast accuracy

### DIC comparison of models

## Plots to make
- `alpha[i]` on world map
- joint distribution of `alpha[i]` and `beta[i]`
- actual inflation forecasts and 95% confidence intervals (and what fraction falls within)


## Useful links
- https://florianwilhelm.info/2020/10/bayesian_hierarchical_modelling_at_scale/
- https://www.pymc.io/welcome.html
PyMC might be slow, in that case we can try PyJAGS
- https://github.com/michaelnowotny/pyjags