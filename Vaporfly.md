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
    ## B1     5.00599 0.002480 3.921e-05      4.367e-05
    ## B2    -0.03138 0.006998 1.107e-04      1.255e-04
    ## sigma  0.09389 0.001670 2.640e-05      2.668e-05
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##           2.5%      25%      50%      75%    97.5%
    ## B1     5.00118  5.00428  5.00599  5.00770  5.01067
    ## B2    -0.04529 -0.03609 -0.03133 -0.02650 -0.01819
    ## sigma  0.09072  0.09273  0.09386  0.09495  0.09720

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
    ## B[1]   4.93649 0.0019319 2.494e-05      4.399e-05
    ## B[2]  -0.01880 0.0039272 5.070e-05      5.618e-05
    ## B[3]   0.14114 0.0026634 3.438e-05      5.812e-05
    ## sigma  0.05234 0.0009279 1.198e-05      1.215e-05
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##           2.5%      25%      50%      75%    97.5%
    ## B[1]   4.93267  4.93519  4.93651  4.93778  4.94025
    ## B[2]  -0.02645 -0.02145 -0.01880 -0.01617 -0.01099
    ## B[3]   0.13602  0.13929  0.14111  0.14299  0.14637
    ## sigma  0.05055  0.05171  0.05234  0.05295  0.05418

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
    ## B[1]   4.9366374 0.002346 3.709e-05      7.766e-05
    ## B[2]  -0.0191371 0.006079 9.611e-05      1.700e-04
    ## B[3]   0.1409369 0.003382 5.347e-05      1.023e-04
    ## C      0.0003999 0.009417 1.489e-04      2.649e-04
    ## sigma  0.0620621 0.001087 1.718e-05      1.718e-05
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##           2.5%       25%        50%      75%     97.5%
    ## B[1]   4.93192  4.935073  4.9366195  4.93825  4.941145
    ## B[2]  -0.03152 -0.023179 -0.0190998 -0.01526 -0.007166
    ## B[3]   0.13430  0.138717  0.1409391  0.14318  0.147436
    ## C     -0.01865 -0.005757  0.0005601  0.00678  0.018334
    ## sigma  0.05992  0.061340  0.0620667  0.06277  0.064242

``` r
gelman.diag(samples)
```

    ## Potential scale reduction factors:
    ## 
    ##       Point est. Upper C.I.
    ## B[1]           1       1.00
    ## B[2]           1       1.01
    ## B[3]           1       1.00
    ## C              1       1.01
    ## sigma          1       1.00
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
course\\\\
\\beta\_i \\sim N(0, 10^2)\\\\
\\alpha\_i \\sim N(0, 10^2)
\\sigma^2 \\sim InvGamma(0.1, 1)\\\\
](https://latex.codecogs.com/png.latex?%0Alog%28Y_i%29%20%5Csim%20N%28%5Cmu_i%2C%20%5Csigma%5E2%29%5C%5C%0A%5Cmu_i%20%3D%20%5Cbeta_0%2B%5Cbeta1V_i%2B%5Cbeta_2S_i%2B%5Cbeta_3S_i%2AV_i%20%2B%20%5Calpha_i%5C%5C%0AV_i%20%5Cin%200%2C1%3A%20Vaporfly%5C%5C%0AS_i%20%5Cin%200%2C1%3A%20Sex%5C%5C%0A%5Calpha_i%3A%20effect%20%5Cspace%20of%20%5Cspace%20maraton%20%5Cspace%20of%20%5Cspace%20course%5C%5C%0A%5Cbeta_i%20%5Csim%20N%280%2C%2010%5E2%29%5C%5C%0A%5Calpha_i%20%5Csim%20N%280%2C%2010%5E2%29%0A%5Csigma%5E2%20%5Csim%20InvGamma%280.1%2C%201%29%5C%5C%0A
"
log(Y_i) \\sim N(\\mu_i, \\sigma^2)\\\\
\\mu_i = \\beta_0+\\beta1V_i+\\beta_2S_i+\\beta_3S_i*V_i + \\alpha_i\\\\
V_i \\in 0,1: Vaporfly\\\\
S_i \\in 0,1: Sex\\\\
\\alpha_i: effect \\space of \\space maraton \\space of \\space course\\\\
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
    ## B0              1.07       1.28
    ## B1              1.00       1.00
    ## B2              1.00       1.00
    ## alpha[1]        1.07       1.28
    ## alpha[2]        1.07       1.27
    ## alpha[3]        1.07       1.28
    ## alpha[4]        1.06       1.26
    ## alpha[5]        1.06       1.25
    ## alpha[6]        1.07       1.27
    ## alpha[7]        1.07       1.28
    ## alpha[8]        1.07       1.27
    ## alpha[9]        1.07       1.27
    ## alpha[10]       1.03       1.13
    ## alpha[11]       1.06       1.24
    ## alpha[12]       1.07       1.27
    ## alpha[13]       1.07       1.28
    ## alpha[14]       1.07       1.27
    ## alpha[15]       1.07       1.27
    ## alpha[16]       1.05       1.23
    ## alpha[17]       1.07       1.26
    ## alpha[18]       1.07       1.27
    ## alpha[19]       1.07       1.27
    ## alpha[20]       1.07       1.27
    ## alpha[21]       1.07       1.26
    ## alpha[22]       1.04       1.19
    ## alpha[23]       1.07       1.27
    ## sigma           1.00       1.00
    ## 
    ## Multivariate psrf
    ## 
    ## 1.04

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
term in the model. However, we have yet to consider the interaction that
the use of Vaporflys may have with age and marathon course. Good
modeling practice says to include all lower order terms, so I will keep
the marathon effect.

## Model 5

  
![
log(Y\_i) \\sim N(\\mu\_i, \\sigma^2)\\\\
\\mu\_i =
B\_0+B\_1V\_i+B\_2S\_i+M\_i+R\_i+C\_1M\_iR\_i+C\_2R\_iV\_i+C\_3S\_iV\_i+C\_4M\_iV\_i\\\\
V\_i : Indicator \\space of \\space Vaporflys\\\\
S\_i : Sex\\\\
M\_i: Effect \\space of \\space Marathon \\space of \\space Y\_i\\\\
R\_i: Effect \\space of \\space Runner \\space of \\space Y\_i\\\\
B\_0, B\_1, B\_2 \\sim N(0, (1/10)^2)\\\\
M\_i \\sim N(0, (1/10)^2)\\\\
R\_i \\sim N(0, (1/10)^2)\\\\
C\_1, C\_2, C\_3, C\_4 \\sim N(0, (1/10)^2)\\\\
\\sigma^2 \\sim InvGamma(0.1, 1)
](https://latex.codecogs.com/png.latex?%0Alog%28Y_i%29%20%5Csim%20N%28%5Cmu_i%2C%20%5Csigma%5E2%29%5C%5C%0A%5Cmu_i%20%3D%20B_0%2BB_1V_i%2BB_2S_i%2BM_i%2BR_i%2BC_1M_iR_i%2BC_2R_iV_i%2BC_3S_iV_i%2BC_4M_iV_i%5C%5C%0AV_i%20%3A%20Indicator%20%5Cspace%20of%20%5Cspace%20Vaporflys%5C%5C%0AS_i%20%3A%20Sex%5C%5C%0AM_i%3A%20Effect%20%5Cspace%20of%20%5Cspace%20Marathon%20%5Cspace%20of%20%5Cspace%20Y_i%5C%5C%0AR_i%3A%20Effect%20%5Cspace%20of%20%5Cspace%20Runner%20%5Cspace%20of%20%5Cspace%20Y_i%5C%5C%0AB_0%2C%20B_1%2C%20B_2%20%5Csim%20N%280%2C%20%281%2F10%29%5E2%29%5C%5C%0AM_i%20%5Csim%20N%280%2C%20%281%2F10%29%5E2%29%5C%5C%0AR_i%20%5Csim%20N%280%2C%20%281%2F10%29%5E2%29%5C%5C%0AC_1%2C%20C_2%2C%20C_3%2C%20C_4%20%5Csim%20N%280%2C%20%281%2F10%29%5E2%29%5C%5C%0A%5Csigma%5E2%20%5Csim%20InvGamma%280.1%2C%201%29%0A
"
log(Y_i) \\sim N(\\mu_i, \\sigma^2)\\\\
\\mu_i = B_0+B_1V_i+B_2S_i+M_i+R_i+C_1M_iR_i+C_2R_iV_i+C_3S_iV_i+C_4M_iV_i\\\\
V_i : Indicator \\space of \\space Vaporflys\\\\
S_i : Sex\\\\
M_i: Effect \\space of \\space Marathon \\space of \\space Y_i\\\\
R_i: Effect \\space of \\space Runner \\space of \\space Y_i\\\\
B_0, B_1, B_2 \\sim N(0, (1/10)^2)\\\\
M_i \\sim N(0, (1/10)^2)\\\\
R_i \\sim N(0, (1/10)^2)\\\\
C_1, C_2, C_3, C_4 \\sim N(0, (1/10)^2)\\\\
\\sigma^2 \\sim InvGamma(0.1, 1)
")  

``` r
X_mod = X%>%select(1, vaporfly, sex, marathon, match_name)%>%
  mutate(marathon = as.numeric(factor(marathon)),
         vaporfly = as.numeric(vaporfly),
         runner = as.numeric(factor(match_name)))%>%
  select(-match_name)

n_run = max(X_mod$runner)
n_mar = max(X_mod$marathon)

data = list(logy=logy, X_mod=X_mod, n_run=n_run, n_mar=n_mar, n=n)

model_string = textConnection("model{
    #likelihood
    for(i in 1:n){
      logy[i] ~ dnorm(mu[i], tau)
      mu[i] = B0 + B1*X_mod[i,2] + B2*X_mod[i,3] + 
      M[X_mod[i,4]] + R[X_mod[i,5]] + 
      C1*M[X_mod[i,4]]*R[X_mod[i,5]] + C2*R[X_mod[i,5]]*X_mod[i,2] + 
      C3*X_mod[i,3]*X_mod[i,2] + C4*X_mod[i,2]*M[X_mod[i,4]]
    }
    
    #Random marathon effect
    for(j in 1:n_mar){
      M[j] ~ dnorm(0, 1)
    }
    
    #Random runner effect
    for(k in 1:n_run){
      R[k] ~ dnorm(0, 1)
    }
    
    #Priors
    B0~dnorm(0, 1)
    B1~dnorm(0, 1)
    B2~dnorm(0, 1)
    C1~dnorm(0, 1)
    C2~dnorm(0, 1)
    C3~dnorm(0, 1)
    C4~dnorm(0, 1)
    tau~dgamma(0.1, 1)
    sigma = 1/sqrt(tau)
}")

inits = list(B0=0, B1=0, B2=0, C1=0, C2=0, C3=0,C4=0, M=rep(0, n_mar), R=rep(0, n_run), tau=1)
model = jags.model(model_string, data=data, inits = inits, n.chains = 2, quiet=T)
update(model, 10000, progress.bar="none")

params = c("B0", "B1", "B2","C1", "C2", "C3", "C4", "sigma", "M", "R")
samples = coda.samples(model, variable.names = params, n.iter=10000, progress.bar="none")

dic_5 = dic.samples(model, n.iter=1000, progress.bar="none")
```
