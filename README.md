# Seminar
## Potential models
### Fixed effects model

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


Where `y` is inflation, `x` are country-specific regressors, and `w` is a global regressor.