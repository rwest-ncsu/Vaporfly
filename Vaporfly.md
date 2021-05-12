Vaporfly
================
Robert West
5/11/2021

# Introduction

Since their introduction to the market in 2016, Nike’s `Vaporfly` shoes
have set the marathon community ablaze. The shoes are supposedly
designed with the intent of optimizing every step that a runner takes.
Huge accomplishments in marathon racing have been made during this time
as well, including Nike’s project of breaking the daunting 2-hour
barrier. The release of `Vaporflys` along with the decrease in marathon
times has led many to criticize the use of the shoes, suggesting that
there is a substantial mechanical advantage to those who wear them. Some
have gone as far to call the use of the shoes “Mechanical Doping” and
call for bans on them in official races. The goal of this analysis is
not to argue for or against their use, but simply to determine if the
use of Nike Vaporflys has any effect on the time of marathon runners and
if that effect is changed across gender, individual runner, and marathon
course.

# Packages

``` r
library(rjags)#MCMC
library(coda)#MCMC
library(ggplot2)#Data Visualization
library(tidyr)#Data tidying
library(dplyr)#Data wrangling
library(varhandle)
library(knitr)#Display tables
library(readr)
```

# Data Import

``` r
men = read_csv("men_sampled_shoe.csv", col_names=T)%>%
  select(name_age, match_name, marathon, year, time_minutes, vaporfly)%>%
  mutate(sex=0)
women = read_csv("women_sampled_shoe.csv", col_names=T)%>%
  select(name_age, match_name, marathon, year, time_minutes, vaporfly)%>%
  mutate(sex=1)
```

# Data Cleaning

``` r
final = full_join(men, women) %>%
  mutate(
    name_age = trimws(name_age, which="right"),
    age = ifelse(check.numeric(substr(name_age, nchar(name_age)-2, nchar(name_age)-1)), 
                 as.numeric(substr(name_age, nchar(name_age)-2, nchar(name_age)-1)),
                 0)
  )%>%
  filter(!is.na(vaporfly))

#Transform the marathon variable to a dummy variable for JAGS
final = cbind(final, to.dummy(v=final$marathon, prefix = "marathon"))
#MAY NEED TO TAKE OUT LAKEFRONT MARATHON
```

# Visualizations

![](Vaporfly_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->![](Vaporfly_files/figure-gfm/unnamed-chunk-4-2.png)<!-- -->![](Vaporfly_files/figure-gfm/unnamed-chunk-4-3.png)<!-- -->![](Vaporfly_files/figure-gfm/unnamed-chunk-4-4.png)<!-- -->![](Vaporfly_files/figure-gfm/unnamed-chunk-4-5.png)<!-- -->![](Vaporfly_files/figure-gfm/unnamed-chunk-4-6.png)<!-- -->![](Vaporfly_files/figure-gfm/unnamed-chunk-4-7.png)<!-- -->

# Methods

To determine the effect of Vaporflys, I utilized a mixture of Bayesian
Linear Regression, Bayesian Mixed-Random Effects, and Bayesian
Multi-level Interaction Models to attempt to explain the overall effect
that the shoes have on marathon times when applied to different genders,
runners, and courses. Upon first inspection, the distribution of time
was bi-modal (figure 1).

![](Vaporfly_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

The 2 peaks of this distribution were explained by the differences in
male and female performances. The distributions of time conditioned on
sex were both bell-shaped and skewed to the right, which suggests that
sex should play a role in the marathon time of an individual. While not
surprising, this added the need to include sex as a predictor in each
model to avoid fitting every model for men and women.

![](Vaporfly_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

For every model, instead of treating the raw time as the response, I
treat log(time) as the response as the time variable was a little too
right-skewed to meet the Normality condition of my models. Taking the
log allowed for more reasonable Gaussian assumptions (Figure 2, 3).
Using a consistent Log-Normal likelihood allowed for more precise model
comparison after fitting as
well.

![](Vaporfly_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->![](Vaporfly_files/figure-gfm/unnamed-chunk-7-2.png)<!-- -->

# Creating analysis variables

``` r
y = final$time_minutes
logy = log(y)
X = cbind(1, final)
n = length(logy)
```

# Models Under Consideration

## Model 1

  
![
log(Y\_i) \\sim N(\\mu\_i, \\sigma^2)\\\\
\\mu\_i = B\_0 + B\_1V\_i\\\\
V\_i \\in {0,1}: Vaporfly\_i\\\\
B\_0, B\_1 \\sim N(0, (\\sqrt{10})^2)\\\\
\\sigma^2 \\sim InvGamma(0.1, 1)
](https://latex.codecogs.com/png.latex?%0Alog%28Y_i%29%20%5Csim%20N%28%5Cmu_i%2C%20%5Csigma%5E2%29%5C%5C%0A%5Cmu_i%20%3D%20B_0%20%2B%20B_1V_i%5C%5C%0AV_i%20%5Cin%20%7B0%2C1%7D%3A%20Vaporfly_i%5C%5C%0AB_0%2C%20B_1%20%5Csim%20N%280%2C%20%28%5Csqrt%7B10%7D%29%5E2%29%5C%5C%0A%5Csigma%5E2%20%5Csim%20InvGamma%280.1%2C%201%29%0A
"
log(Y_i) \\sim N(\\mu_i, \\sigma^2)\\\\
\\mu_i = B_0 + B_1V_i\\\\
V_i \\in {0,1}: Vaporfly_i\\\\
B_0, B_1 \\sim N(0, (\\sqrt{10})^2)\\\\
\\sigma^2 \\sim InvGamma(0.1, 1)
")  

This model suggests that the only contributing factor to a runner’s time
is the binary `vaporfly` variable:

``` r
X_mod = X%>%select(1, vaporfly)%>%
  mutate(vaporfly = as.numeric(vaporfly))
data = list(X_mod=X_mod, n=n, logy=logy)

model_string = textConnection("model{
    #Likelihood 
    for(i in 1:n){
      logy[i] ~ dnorm(mu[i], tau)
      mu[i] = B1*X_mod[i,1] + B2*X_mod[i,2]
    }
    
    #Priors
    B1 ~ dnorm(0, 1/10)
    B2 ~ dnorm(0, 1/10)
    tau ~ dgamma(0.1, 1)
    sigma = 1/sqrt(tau)
}")

#Initialize parameters and construct model
inits = list(B1=0, B2=0, tau=1)
model = jags.model(model_string, data=data, inits=inits, n.chains=2, quiet=T)

#Thin the burn-in samples
update(model, 1000, progress.bar="none")

params = c("B1","B2","sigma")
samples = coda.samples(model, variable.names = params, n.iter=2000, progress.bar="none")

#Compute DIC
dic_1 = dic.samples(model, n.iter=2000, progress.bar="none")
```

    ## 
    ## Iterations = 1001:3000
    ## Thinning interval = 1 
    ## Number of chains = 2 
    ## Sample size per chain = 2000 
    ## 
    ## 1. Empirical mean and standard deviation for each variable,
    ##    plus standard error of the mean:
    ## 
    ##           Mean       SD  Naive SE Time-series SE
    ## B1     5.00594 0.002465 3.898e-05      4.294e-05
    ## B2    -0.03146 0.006990 1.105e-04      1.282e-04
    ## sigma  0.09383 0.001657 2.619e-05      2.619e-05
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##           2.5%      25%      50%      75%    97.5%
    ## B1     5.00106  5.00427  5.00598  5.00758  5.01073
    ## B2    -0.04523 -0.03631 -0.03144 -0.02674 -0.01746
    ## sigma  0.09070  0.09266  0.09384  0.09493  0.09720

![](Vaporfly_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

    ## Potential scale reduction factors:
    ## 
    ##       Point est. Upper C.I.
    ## B1             1          1
    ## B2             1          1
    ## sigma          1          1
    ## 
    ## Multivariate psrf
    ## 
    ## 1

By popular convergence diagnostics (Gelman Statistic and Trace Plots),
this model has converged to the parameter estimates given. Now, we fit
more sophisticated models to attempt to characterize the relationship
more thoroughly.

## Model 2

  
![
log(Y\_i) \\sim N(\\mu\_i, \\sigma^2)\\\\
\\mu\_i = B\_0+B\_1V\_i+B\_2S\_i\\\\
V\_i \\in 0,1: Vaporfly\\\\
S\_i \\in 0,1: Sex\\\\
B\_0, B\_1, B\_2 \\sim N(0, (\\sqrt{10})^2)\\\\
\\sigma^2 \\sim InvGamma(0.1, 1)\\\\
](https://latex.codecogs.com/png.latex?%0Alog%28Y_i%29%20%5Csim%20N%28%5Cmu_i%2C%20%5Csigma%5E2%29%5C%5C%0A%5Cmu_i%20%3D%20B_0%2BB_1V_i%2BB_2S_i%5C%5C%0AV_i%20%5Cin%200%2C1%3A%20Vaporfly%5C%5C%0AS_i%20%5Cin%200%2C1%3A%20Sex%5C%5C%0AB_0%2C%20B_1%2C%20B_2%20%5Csim%20N%280%2C%20%28%5Csqrt%7B10%7D%29%5E2%29%5C%5C%0A%5Csigma%5E2%20%5Csim%20InvGamma%280.1%2C%201%29%5C%5C%0A
"
log(Y_i) \\sim N(\\mu_i, \\sigma^2)\\\\
\\mu_i = B_0+B_1V_i+B_2S_i\\\\
V_i \\in 0,1: Vaporfly\\\\
S_i \\in 0,1: Sex\\\\
B_0, B_1, B_2 \\sim N(0, (\\sqrt{10})^2)\\\\
\\sigma^2 \\sim InvGamma(0.1, 1)\\\\
")  

``` r
X_mod = X%>%select(1, vaporfly, sex)%>%
  mutate(vaporfly = as.numeric(vaporfly))
p = ncol(X_mod)
data = list(X_mod = X_mod, logy=logy, n=n, p=p)

model_string = textConnection("model{
    #Likelihood
    for(i in 1:n){
      logy[i] ~ dnorm(mu[i], tau)
      mu[i] = inprod(X_mod[i,], B[])
    }
    
    #Priors
    for(j in 1:p){
      B[j] ~ dnorm(0, 1/10)
    }
    
    tau ~ dgamma(0.1, 0.1)
    sigma = 1/sqrt(tau)
}")

inits = list(B=rep(0,p), tau=1)
model = jags.model(model_string, data=data, inits=inits, n.chains=2, quiet=T)

update(model, 10000, progress.bar="none")

params = c("B","sigma")
samples = coda.samples(model, variable.names = params, n.iter=3000, progress.bar="none")

#Convergence criterion
summary(samples)
```

    ## 
    ## Iterations = 10001:13000
    ## Thinning interval = 1 
    ## Number of chains = 2 
    ## Sample size per chain = 3000 
    ## 
    ## 1. Empirical mean and standard deviation for each variable,
    ##    plus standard error of the mean:
    ## 
    ##           Mean        SD  Naive SE Time-series SE
    ## B[1]   4.93658 0.0018649 2.408e-05      4.142e-05
    ## B[2]  -0.01897 0.0039110 5.049e-05      5.965e-05
    ## B[3]   0.14103 0.0026250 3.389e-05      5.711e-05
    ## sigma  0.05231 0.0009192 1.187e-05      1.187e-05
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##           2.5%      25%      50%     75%    97.5%
    ## B[1]   4.93296  4.93533  4.93655  4.9379  4.94027
    ## B[2]  -0.02676 -0.02151 -0.01896 -0.0164 -0.01135
    ## B[3]   0.13584  0.13926  0.14104  0.1428  0.14615
    ## sigma  0.05058  0.05168  0.05228  0.0529  0.05419

``` r
gelman.diag(samples)
```

    ## Potential scale reduction factors:
    ## 
    ##       Point est. Upper C.I.
    ## B[1]           1          1
    ## B[2]           1          1
    ## B[3]           1          1
    ## sigma          1          1
    ## 
    ## Multivariate psrf
    ## 
    ## 1

``` r
#Compute DIC
dic_2 = dic.samples(model, n.iter=3000, progress.bar="none")
```

## Model 3

  
![
log(Y\_i) \\sim N(\\mu\_i, \\sigma^2)\\\\
\\mu\_i = \\beta\_0+\\beta1V\_i+\\beta\_2S\_i+\\beta\_3S\_i\*V\_i\\\\
V\_i \\in 0,1: Vaporfly\\\\
S\_i \\in 0,1: Sex\\\\
\\beta\_i \\sim N(0, 10^2)\\\\
\\sigma^2 \\sim InvGamma(0.1, 1)\\\\
](https://latex.codecogs.com/png.latex?%0Alog%28Y_i%29%20%5Csim%20N%28%5Cmu_i%2C%20%5Csigma%5E2%29%5C%5C%0A%5Cmu_i%20%3D%20%5Cbeta_0%2B%5Cbeta1V_i%2B%5Cbeta_2S_i%2B%5Cbeta_3S_i%2AV_i%5C%5C%0AV_i%20%5Cin%200%2C1%3A%20Vaporfly%5C%5C%0AS_i%20%5Cin%200%2C1%3A%20Sex%5C%5C%0A%5Cbeta_i%20%5Csim%20N%280%2C%2010%5E2%29%5C%5C%0A%5Csigma%5E2%20%5Csim%20InvGamma%280.1%2C%201%29%5C%5C%0A
"
log(Y_i) \\sim N(\\mu_i, \\sigma^2)\\\\
\\mu_i = \\beta_0+\\beta1V_i+\\beta_2S_i+\\beta_3S_i*V_i\\\\
V_i \\in 0,1: Vaporfly\\\\
S_i \\in 0,1: Sex\\\\
\\beta_i \\sim N(0, 10^2)\\\\
\\sigma^2 \\sim InvGamma(0.1, 1)\\\\
")  

``` r
X_mod = X%>%select(1, vaporfly, sex)%>%
  mutate(vaporfly = as.numeric(vaporfly))
p = ncol(X_mod)

data = list(logy=logy, X_mod=X_mod, p=p, n=n)

model_string = textConnection("model{
    #Likelihood
    for(i in 1:n){
      logy[i] ~ dnorm(mu[i], tau)
      mu[i] = inprod(X_mod[i,], B[]) + C*X_mod[i,2]*X_mod[i,3]
    }

    #Priors
    for(j in 1:p){
      B[j] ~ dnorm(0, 0.01)
    }
    
    C ~ dnorm(0, 0.01)
    tau ~ dgamma(0.1, 1)
    sigma = 1/sqrt(tau)
}")

inits = list(B=rep(0, p), C=0, tau=1)
model = jags.model(model_string, data=data, inits=inits, n.chains=2, quiet=T)
update(model, 1000, progress.bar="none")

params = c("B", "C", "sigma")
samples = coda.samples(model, n.iter=2000, progress.bar="none", variable.names = params)

summary(samples)
```

    ## 
    ## Iterations = 1001:3000
    ## Thinning interval = 1 
    ## Number of chains = 2 
    ## Sample size per chain = 2000 
    ## 
    ## 1. Empirical mean and standard deviation for each variable,
    ##    plus standard error of the mean:
    ## 
    ##             Mean       SD  Naive SE Time-series SE
    ## B[1]   4.9364731 0.002278 3.602e-05      6.905e-05
    ## B[2]  -0.0188373 0.006081 9.615e-05      1.677e-04
    ## B[3]   0.1411324 0.003269 5.168e-05      9.786e-05
    ## C     -0.0001531 0.009563 1.512e-04      2.602e-04
    ## sigma  0.0620780 0.001094 1.731e-05      1.729e-05
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##           2.5%       25%        50%       75%    97.5%
    ## B[1]   4.93208  4.934966  4.9364814  4.938034  4.94081
    ## B[2]  -0.03103 -0.022948 -0.0187429 -0.014728 -0.00702
    ## B[3]   0.13472  0.138960  0.1411193  0.143309  0.14764
    ## C     -0.01849 -0.006806 -0.0001725  0.006257  0.01910
    ## sigma  0.06006  0.061324  0.0620256  0.062839  0.06423

``` r
gelman.diag(samples)
```

    ## Potential scale reduction factors:
    ## 
    ##       Point est. Upper C.I.
    ## B[1]           1       1.00
    ## B[2]           1       1.00
    ## B[3]           1       1.01
    ## C              1       1.00
    ## sigma          1       1.02
    ## 
    ## Multivariate psrf
    ## 
    ## 1

``` r
dic_3 = dic.samples(model, n.iter=2000, progress.bar="none")
```

## Model 4

  
![
log(Y\_i) \\sim N(\\mu\_i, \\sigma^2)\\\\
\\mu\_i = \\beta\_0+\\beta1V\_i+\\beta\_2S\_i+\\beta\_3S\_i\*V\_i +
\\alpha\_i\\\\
V\_i \\in 0,1: Vaporfly\\\\
S\_i \\in 0,1: Sex\\\\
\\alpha\_i: effect \\space of \\space maraton \\space of \\space
runner\_i
\\beta\_i \\sim N(0, 10^2)\\\\
\\alpha\_i \\sim N(0, 10^2)
\\sigma^2 \\sim InvGamma(0.1, 1)\\\\
](https://latex.codecogs.com/png.latex?%0Alog%28Y_i%29%20%5Csim%20N%28%5Cmu_i%2C%20%5Csigma%5E2%29%5C%5C%0A%5Cmu_i%20%3D%20%5Cbeta_0%2B%5Cbeta1V_i%2B%5Cbeta_2S_i%2B%5Cbeta_3S_i%2AV_i%20%2B%20%5Calpha_i%5C%5C%0AV_i%20%5Cin%200%2C1%3A%20Vaporfly%5C%5C%0AS_i%20%5Cin%200%2C1%3A%20Sex%5C%5C%0A%5Calpha_i%3A%20effect%20%5Cspace%20of%20%5Cspace%20maraton%20%5Cspace%20of%20%5Cspace%20runner_i%0A%5Cbeta_i%20%5Csim%20N%280%2C%2010%5E2%29%5C%5C%0A%5Calpha_i%20%5Csim%20N%280%2C%2010%5E2%29%0A%5Csigma%5E2%20%5Csim%20InvGamma%280.1%2C%201%29%5C%5C%0A
"
log(Y_i) \\sim N(\\mu_i, \\sigma^2)\\\\
\\mu_i = \\beta_0+\\beta1V_i+\\beta_2S_i+\\beta_3S_i*V_i + \\alpha_i\\\\
V_i \\in 0,1: Vaporfly\\\\
S_i \\in 0,1: Sex\\\\
\\alpha_i: effect \\space of \\space maraton \\space of \\space runner_i
\\beta_i \\sim N(0, 10^2)\\\\
\\alpha_i \\sim N(0, 10^2)
\\sigma^2 \\sim InvGamma(0.1, 1)\\\\
")  

``` r
X_mod = X%>%select(1, vaporfly, sex, marathon)%>%
  mutate(vaporfly = as.numeric(vaporfly),
         marathon = as.numeric(factor(marathon)))
n_mar = length(unique(X_mod$marathon))

data = list(logy=logy, n_mar=n_mar, X_mod=X_mod, n=n)

model_string = textConnection("model{
  #Likelihood
  for(i in 1:n){
    logy[i] ~ dnorm(mu[i], tau)
    mu[i] = B0 + B1*X_mod[i,2] +B2*X_mod[i,3] + alpha[X_mod[i,4]]
  }
  
  #Random effects
  for(j in 1:n_mar){
    alpha[j] ~ dnorm(0, 10)
  }
  
  B0 ~ dnorm(0, 10)
  B1 ~ dnorm(0, 10)
  B2 ~ dnorm(0, 10)
  tau ~ dgamma(0.1, 1)
  sigma = 1/sqrt(tau)
}")

inits_random = list(B0=0, B1=0, B2=0, alpha=rep(0, n_mar), tau=1)
model = jags.model(model_string, data=data, inits = inits_random, n.chains = 2, quiet=T)
update(model, 10000, progress.bar="none")

params = c("B0", "B1", "B2", "alpha", "sigma")
samples = coda.samples(model, variable.names = params, n.iter=50000, progress.bar="none")

gelman.diag(samples)
```

    ## Potential scale reduction factors:
    ## 
    ##           Point est. Upper C.I.
    ## B0              1.22       1.75
    ## B1              1.00       1.00
    ## B2              1.00       1.00
    ## alpha[1]        1.22       1.74
    ## alpha[2]        1.22       1.74
    ## alpha[3]        1.22       1.74
    ## alpha[4]        1.21       1.71
    ## alpha[5]        1.20       1.69
    ## alpha[6]        1.22       1.74
    ## alpha[7]        1.22       1.74
    ## alpha[8]        1.22       1.74
    ## alpha[9]        1.22       1.74
    ## alpha[10]       1.12       1.43
    ## alpha[11]       1.19       1.66
    ## alpha[12]       1.22       1.74
    ## alpha[13]       1.22       1.74
    ## alpha[14]       1.22       1.74
    ## alpha[15]       1.22       1.73
    ## alpha[16]       1.19       1.64
    ## alpha[17]       1.21       1.71
    ## alpha[18]       1.22       1.74
    ## alpha[19]       1.22       1.74
    ## alpha[20]       1.21       1.72
    ## alpha[21]       1.21       1.71
    ## alpha[22]       1.15       1.54
    ## alpha[23]       1.22       1.74
    ## sigma           1.00       1.00
    ## 
    ## Multivariate psrf
    ## 
    ## 1.1

``` r
#Compute DIC
dic_4 = dic.samples(model, n.iter=50000, progress.bar="none")
```

This model suggests that there is no significant difference between
marathon courses since the effective sample sizes are so small even
after 50,000 iterations of MCMC. From this, I conclude that the marathon
effects are constant and are “encoded” in the constant intercept
![\\beta\_0](https://latex.codecogs.com/png.latex?%5Cbeta_0 "\\beta_0").
For further justification:

``` r
alphas = apply(samples[[1]][ , 4:26],
      MARGIN = 2,
      FUN = function(x){
        c(mean(x), sd(x))
      })

alphas %>%
  t()%>%
  as.data.frame()%>%
  rename(mean=V1, sd=V2)%>%
  ggplot(mapping=aes(x=1:23, y=mean))+
  geom_point(aes(color="Mean"))+
  geom_point(aes(y=sd, color="sd"))+
  labs(color="Measurement", x=expression(alpha), 
       y="Value", main="Mean and Standard Deviation of Alpha effects")
```

![](Vaporfly_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

Clearly, all of the
![\\alpha](https://latex.codecogs.com/png.latex?%5Calpha "\\alpha")
effects behave similarly and produced relatively similar estimates and
Posterior Distributions. Because of this and the low effective sample
size of the ![\\beta\_0](https://latex.codecogs.com/png.latex?%5Cbeta_0
"\\beta_0") in this model, I propose the removal of the Marathon effect
term in the model.

## Model 5
