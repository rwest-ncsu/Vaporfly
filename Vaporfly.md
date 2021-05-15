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
samples_1 = coda.samples(model, variable.names = params, n.iter=2000, progress.bar="none")

#Compute DIC
dic_1 = dic.samples(model, n.iter=2000, progress.bar="none")
```

``` r
summary(samples_1)
plot(samples_1)
gelman.diag(samples_1)
```

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
B\_0, B\_1, B\_2 \\sim N(0, 10)\\\\
\\sigma^2 \\sim InvGamma(0.1, 1)\\\\
](https://latex.codecogs.com/png.latex?%0Alog%28Y_i%29%20%5Csim%20N%28%5Cmu_i%2C%20%5Csigma%5E2%29%5C%5C%0A%5Cmu_i%20%3D%20B_0%2BB_1V_i%2BB_2S_i%5C%5C%0AV_i%20%5Cin%200%2C1%3A%20Vaporfly%5C%5C%0AS_i%20%5Cin%200%2C1%3A%20Sex%5C%5C%0AB_0%2C%20B_1%2C%20B_2%20%5Csim%20N%280%2C%2010%29%5C%5C%0A%5Csigma%5E2%20%5Csim%20InvGamma%280.1%2C%201%29%5C%5C%0A
"
log(Y_i) \\sim N(\\mu_i, \\sigma^2)\\\\
\\mu_i = B_0+B_1V_i+B_2S_i\\\\
V_i \\in 0,1: Vaporfly\\\\
S_i \\in 0,1: Sex\\\\
B_0, B_1, B_2 \\sim N(0, 10)\\\\
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
samples_2 = coda.samples(model, variable.names = params, n.iter=3000, progress.bar="none")

#Compute DIC
dic_2 = dic.samples(model, n.iter=3000, progress.bar="none")
```

``` r
#Convergence criterion
summary(samples_2)
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
    ## B[1]   4.93654 0.0019145 2.472e-05      4.225e-05
    ## B[2]  -0.01887 0.0038798 5.009e-05      5.963e-05
    ## B[3]   0.14104 0.0026420 3.411e-05      5.580e-05
    ## sigma  0.05233 0.0009127 1.178e-05      1.199e-05
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##           2.5%      25%      50%      75%    97.5%
    ## B[1]   4.93285  4.93525  4.93654  4.93784  4.94032
    ## B[2]  -0.02657 -0.02142 -0.01883 -0.01626 -0.01136
    ## B[3]   0.13592  0.13926  0.14107  0.14282  0.14622
    ## sigma  0.05057  0.05171  0.05233  0.05293  0.05416

``` r
gelman.diag(samples_2)
```

    ## Potential scale reduction factors:
    ## 
    ##       Point est. Upper C.I.
    ## B[1]           1       1.00
    ## B[2]           1       1.00
    ## B[3]           1       1.00
    ## sigma          1       1.01
    ## 
    ## Multivariate psrf
    ## 
    ## 1

## Model 3

  
![
log(Y\_i) \\sim N(\\mu\_i, \\sigma^2)\\\\
\\mu\_i = \\beta\_0+\\beta1V\_i+C\_1S\_i+\\beta\_3S\_i\*V\_i\\\\
V\_i \\in 0,1: Vaporfly\\\\
S\_i \\in 0,1: Sex\\\\
\\beta\_i \\sim N(0, 10)\\\\
C\_1 \\sim N(0, 10)\\\\
\\sigma^2 \\sim InvGamma(0.1, 1)\\\\
](https://latex.codecogs.com/png.latex?%0Alog%28Y_i%29%20%5Csim%20N%28%5Cmu_i%2C%20%5Csigma%5E2%29%5C%5C%0A%5Cmu_i%20%3D%20%5Cbeta_0%2B%5Cbeta1V_i%2BC_1S_i%2B%5Cbeta_3S_i%2AV_i%5C%5C%0AV_i%20%5Cin%200%2C1%3A%20Vaporfly%5C%5C%0AS_i%20%5Cin%200%2C1%3A%20Sex%5C%5C%0A%5Cbeta_i%20%5Csim%20N%280%2C%2010%29%5C%5C%0AC_1%20%5Csim%20N%280%2C%2010%29%5C%5C%0A%5Csigma%5E2%20%5Csim%20InvGamma%280.1%2C%201%29%5C%5C%0A
"
log(Y_i) \\sim N(\\mu_i, \\sigma^2)\\\\
\\mu_i = \\beta_0+\\beta1V_i+C_1S_i+\\beta_3S_i*V_i\\\\
V_i \\in 0,1: Vaporfly\\\\
S_i \\in 0,1: Sex\\\\
\\beta_i \\sim N(0, 10)\\\\
C_1 \\sim N(0, 10)\\\\
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
      B[j] ~ dnorm(0, 0.1)
    }
    
    C ~ dnorm(0, 0.1)
    tau ~ dgamma(0.1, 1)
    sigma = 1/sqrt(tau)
}")

inits = list(B=rep(0, p), C=0, tau=1)
model = jags.model(model_string, data=data, inits=inits, n.chains=2, quiet=T)
update(model, 1000, progress.bar="none")

params = c("B", "C", "sigma")
samples_3 = coda.samples(model, n.iter=2000, progress.bar="none", variable.names = params)

dic_3 = dic.samples(model, n.iter=2000, progress.bar="none")
```

``` r
summary(samples_3)
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
    ## B[1]   4.937e+00 0.002348 3.712e-05      7.293e-05
    ## B[2]  -1.876e-02 0.006128 9.689e-05      1.849e-04
    ## B[3]   1.411e-01 0.003335 5.273e-05      1.072e-04
    ## C     -8.883e-05 0.009612 1.520e-04      2.810e-04
    ## sigma  6.210e-02 0.001094 1.730e-05      1.704e-05
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##           2.5%       25%        50%       75%     97.5%
    ## B[1]   4.93201  4.934894  4.937e+00  4.938147  4.941121
    ## B[2]  -0.03078 -0.022925 -1.880e-02 -0.014625 -0.006707
    ## B[3]   0.13472  0.138828  1.411e-01  0.143433  0.147691
    ## C     -0.01904 -0.006553  4.386e-05  0.006446  0.018489
    ## sigma  0.06000  0.061365  6.207e-02  0.062780  0.064270

``` r
gelman.diag(samples_3)
```

    ## Potential scale reduction factors:
    ## 
    ##       Point est. Upper C.I.
    ## B[1]           1       1.01
    ## B[2]           1       1.00
    ## B[3]           1       1.00
    ## C              1       1.00
    ## sigma          1       1.00
    ## 
    ## Multivariate psrf
    ## 
    ## 1

## Model 4

  
![
log(Y\_i) \\sim N(\\mu\_i, \\sigma^2)\\\\
\\mu\_i = \\beta\_0+\\beta1V\_i+\\beta\_2S\_i+\\beta\_3S\_i\*V\_i +
\\alpha\_i\\\\
V\_i \\in 0,1: Vaporfly\\\\
S\_i \\in 0,1: Sex\\\\
\\alpha\_i: marathon \\space \\space effect\\\\
\\beta\_i \\sim N(0, 10)\\\\
\\alpha\_i \\sim N(0, 10)\\\\
\\sigma^2 \\sim InvGamma(0.1, 1)\\\\
](https://latex.codecogs.com/png.latex?%0Alog%28Y_i%29%20%5Csim%20N%28%5Cmu_i%2C%20%5Csigma%5E2%29%5C%5C%0A%5Cmu_i%20%3D%20%5Cbeta_0%2B%5Cbeta1V_i%2B%5Cbeta_2S_i%2B%5Cbeta_3S_i%2AV_i%20%2B%20%5Calpha_i%5C%5C%0AV_i%20%5Cin%200%2C1%3A%20Vaporfly%5C%5C%0AS_i%20%5Cin%200%2C1%3A%20Sex%5C%5C%0A%5Calpha_i%3A%20marathon%20%5Cspace%20%5Cspace%20effect%5C%5C%0A%5Cbeta_i%20%5Csim%20N%280%2C%2010%29%5C%5C%0A%5Calpha_i%20%5Csim%20N%280%2C%2010%29%5C%5C%0A%5Csigma%5E2%20%5Csim%20InvGamma%280.1%2C%201%29%5C%5C%0A
"
log(Y_i) \\sim N(\\mu_i, \\sigma^2)\\\\
\\mu_i = \\beta_0+\\beta1V_i+\\beta_2S_i+\\beta_3S_i*V_i + \\alpha_i\\\\
V_i \\in 0,1: Vaporfly\\\\
S_i \\in 0,1: Sex\\\\
\\alpha_i: marathon \\space \\space effect\\\\
\\beta_i \\sim N(0, 10)\\\\
\\alpha_i \\sim N(0, 10)\\\\
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
  
  B0 ~ dnorm(0, 0.1)
  B1 ~ dnorm(0, 0.1)
  B2 ~ dnorm(0, 0.1)
  tau ~ dgamma(0.1, 1)
  sigma = 1/sqrt(tau)
}")

inits_random = list(B0=0, B1=0, B2=0, alpha=rep(0, n_mar), tau=1)
model = jags.model(model_string, data=data, inits = inits_random, n.chains = 2, quiet=T)
update(model, 10000, progress.bar="none")

params = c("B0", "B1", "B2", "alpha", "sigma")
samples_4 = coda.samples(model, variable.names = params, n.iter=50000, progress.bar="none")

#Compute DIC
dic_4 = dic.samples(model, n.iter=50000, progress.bar="none")
```

``` r
gelman.diag(samples_4)
```

    ## Potential scale reduction factors:
    ## 
    ##           Point est. Upper C.I.
    ## B0              1.10       1.35
    ## B1              1.00       1.00
    ## B2              1.00       1.00
    ## alpha[1]        1.10       1.36
    ## alpha[2]        1.09       1.35
    ## alpha[3]        1.09       1.35
    ## alpha[4]        1.09       1.34
    ## alpha[5]        1.08       1.31
    ## alpha[6]        1.09       1.35
    ## alpha[7]        1.09       1.35
    ## alpha[8]        1.09       1.34
    ## alpha[9]        1.09       1.35
    ## alpha[10]       1.04       1.15
    ## alpha[11]       1.07       1.28
    ## alpha[12]       1.09       1.34
    ## alpha[13]       1.09       1.35
    ## alpha[14]       1.09       1.35
    ## alpha[15]       1.09       1.34
    ## alpha[16]       1.07       1.28
    ## alpha[17]       1.09       1.33
    ## alpha[18]       1.09       1.35
    ## alpha[19]       1.09       1.34
    ## alpha[20]       1.09       1.33
    ## alpha[21]       1.09       1.33
    ## alpha[22]       1.05       1.21
    ## alpha[23]       1.09       1.35
    ## sigma           1.00       1.00
    ## 
    ## Multivariate psrf
    ## 
    ## 1.05

This model suggests that there is no significant difference between
marathon courses since the effective sample sizes are so small even
after 50,000 iterations of MCMC. From this, I conclude that the marathon
effects are constant and are “encoded” in the constant intercept
![\\beta\_0](https://latex.codecogs.com/png.latex?%5Cbeta_0 "\\beta_0").
For further justification:

``` r
alphas = apply(samples_4[[1]][ , 4:26],
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

![](Vaporfly_files/figure-gfm/unnamed-chunk-17-1.png)<!-- -->

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
M\_i: Marathon \\space \\space effect\\\\
R\_i: Runner \\space \\space effect\\\\
B\_0, B\_1, B\_2 \\sim N(0, 10)\\\\
M\_i \\sim N(0, 10)\\\\
R\_i \\sim N(0, 10)\\\\
C\_1, C\_2, C\_3, C\_4 \\sim N(0, 10)\\\\
\\sigma^2 \\sim InvGamma(0.1, 1)
](https://latex.codecogs.com/png.latex?%0Alog%28Y_i%29%20%5Csim%20N%28%5Cmu_i%2C%20%5Csigma%5E2%29%5C%5C%0A%5Cmu_i%20%3D%20B_0%2BB_1V_i%2BB_2S_i%2BM_i%2BR_i%2BC_1M_iR_i%2BC_2R_iV_i%2BC_3S_iV_i%2BC_4M_iV_i%5C%5C%0AV_i%20%3A%20Indicator%20%5Cspace%20of%20%5Cspace%20Vaporflys%5C%5C%0AS_i%20%3A%20Sex%5C%5C%0AM_i%3A%20Marathon%20%5Cspace%20%5Cspace%20effect%5C%5C%0AR_i%3A%20Runner%20%5Cspace%20%5Cspace%20effect%5C%5C%0AB_0%2C%20B_1%2C%20B_2%20%5Csim%20N%280%2C%2010%29%5C%5C%0AM_i%20%5Csim%20N%280%2C%2010%29%5C%5C%0AR_i%20%5Csim%20N%280%2C%2010%29%5C%5C%0AC_1%2C%20C_2%2C%20C_3%2C%20C_4%20%5Csim%20N%280%2C%2010%29%5C%5C%0A%5Csigma%5E2%20%5Csim%20InvGamma%280.1%2C%201%29%0A
"
log(Y_i) \\sim N(\\mu_i, \\sigma^2)\\\\
\\mu_i = B_0+B_1V_i+B_2S_i+M_i+R_i+C_1M_iR_i+C_2R_iV_i+C_3S_iV_i+C_4M_iV_i\\\\
V_i : Indicator \\space of \\space Vaporflys\\\\
S_i : Sex\\\\
M_i: Marathon \\space \\space effect\\\\
R_i: Runner \\space \\space effect\\\\
B_0, B_1, B_2 \\sim N(0, 10)\\\\
M_i \\sim N(0, 10)\\\\
R_i \\sim N(0, 10)\\\\
C_1, C_2, C_3, C_4 \\sim N(0, 10)\\\\
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
      M[j] ~ dnorm(0, .1)
    }
    
    #Random runner effect
    for(k in 1:n_run){
      R[k] ~ dnorm(0, .1)
    }
    
    #Priors
    B0~dnorm(0, .1)
    B1~dnorm(0, .1)
    B2~dnorm(0, .1)
    C1~dnorm(0, .1)
    C2~dnorm(0, .1)
    C3~dnorm(0, .1)
    C4~dnorm(0, .1)
    tau~dgamma(0.1, 1)
    sigma = 1/sqrt(tau)
}")

inits = list(B0=2.3, B1=-0.2, B2=0, C1=-0.37, C2=0, C3=0,C4=0.05, M=rep(0, n_mar), R=rep(0, n_run), tau=1)
model = jags.model(model_string, data=data, inits = inits, n.chains = 2, quiet=T)
update(model, 10000, progress.bar="none")

params = c("B0", "B1", "B2","C1", "C2", "C3", "C4", "sigma", "M", "R")
samples_5 = coda.samples(model, variable.names = params, n.iter=10000, progress.bar="none")

dic_5 = dic.samples(model, n.iter=1000, progress.bar="none")
```

This model suffers from over-parameterization. From the MCMC
diagnostics, it appears that any term involving marathon effects ![M\_i,
C\_1, C\_4](https://latex.codecogs.com/png.latex?M_i%2C%20C_1%2C%20C_4
"M_i, C_1, C_4") failed to converge. To show this, Below are the trace
plots of those parameters. I would move forward by removing these terms
in the model. Their inclusion leads to a deceiving DIC value since their
posteriors are not approximately Normal even after thousands of MCMC
iterations.

``` r
plot(samples_5[[1]][, c(4, 8, 9:11)])
```

![](Vaporfly_files/figure-gfm/unnamed-chunk-19-1.png)<!-- -->![](Vaporfly_files/figure-gfm/unnamed-chunk-19-2.png)<!-- -->

Proposing the following adjustment to model 5 for the most compicated
model fit. I will keep the notation of ![C\_2,
C\_3](https://latex.codecogs.com/png.latex?C_2%2C%20C_3 "C_2, C_3") to
denote that this is a subset of a model previously attempted. I will
also tighen the prior variances.

  
![
log(Y\_i) \\sim N(\\mu\_i, \\sigma^2)\\\\
\\mu\_i = B\_0+B\_1V\_i+B\_2S\_i+R\_i+C\_2R\_iV\_i+C\_3S\_iV\_i\\\\
V\_i : Indicator \\space \\space of \\space \\space Vaporflys\\\\
S\_i : Sex\\\\
R\_i: Runner \\space \\space effect\\\\
B\_0, B\_1, B\_2 \\sim N(0, 1)\\\\
R\_i \\sim N(0, 1)\\\\
C\_2, C\_3\\sim N(0, 1)\\\\
\\sigma^2 \\sim InvGamma(0.1, 1)
](https://latex.codecogs.com/png.latex?%0Alog%28Y_i%29%20%5Csim%20N%28%5Cmu_i%2C%20%5Csigma%5E2%29%5C%5C%0A%5Cmu_i%20%3D%20B_0%2BB_1V_i%2BB_2S_i%2BR_i%2BC_2R_iV_i%2BC_3S_iV_i%5C%5C%0AV_i%20%3A%20Indicator%20%5Cspace%20%5Cspace%20of%20%5Cspace%20%5Cspace%20Vaporflys%5C%5C%0AS_i%20%3A%20Sex%5C%5C%0AR_i%3A%20Runner%20%5Cspace%20%5Cspace%20effect%5C%5C%0AB_0%2C%20B_1%2C%20B_2%20%5Csim%20N%280%2C%201%29%5C%5C%0AR_i%20%5Csim%20N%280%2C%201%29%5C%5C%0AC_2%2C%20C_3%5Csim%20N%280%2C%201%29%5C%5C%0A%5Csigma%5E2%20%5Csim%20InvGamma%280.1%2C%201%29%0A
"
log(Y_i) \\sim N(\\mu_i, \\sigma^2)\\\\
\\mu_i = B_0+B_1V_i+B_2S_i+R_i+C_2R_iV_i+C_3S_iV_i\\\\
V_i : Indicator \\space \\space of \\space \\space Vaporflys\\\\
S_i : Sex\\\\
R_i: Runner \\space \\space effect\\\\
B_0, B_1, B_2 \\sim N(0, 1)\\\\
R_i \\sim N(0, 1)\\\\
C_2, C_3\\sim N(0, 1)\\\\
\\sigma^2 \\sim InvGamma(0.1, 1)
")  

``` r
X_mod = X%>%select(1, vaporfly, sex, match_name)%>%
  mutate(vaporfly = as.numeric(vaporfly),
         runner = as.numeric(factor(match_name)))%>%
  select(-match_name)

n_run = max(X_mod$runner)

data = list(logy=logy, X_mod=X_mod, n_run=n_run, n=n)

model_string = textConnection("model{
    #likelihood
    for(i in 1:n){
      logy[i] ~ dnorm(mu[i], tau)
      mu[i] = B0 + B1*X_mod[i,2] + B2*X_mod[i,3] + 
      R[X_mod[i,4]] + 
      C2*R[X_mod[i,4]]*X_mod[i,2] + C3*X_mod[i,3]*X_mod[i,2]
    }
    
    #Random runner effect
    for(k in 1:n_run){
      R[k] ~ dnorm(0, 1)
    }
    
    #Priors
    B0~dnorm(0, 1)
    B1~dnorm(0, 1)
    B2~dnorm(0, 1)
    C2~dnorm(0, 1)
    C3~dnorm(0, 1)
    tau~dgamma(0.1, 1)
    sigma = 1/sqrt(tau)
}")

inits = list(B0=5, B1=0, B2=0.2, C2=0, C3=0, R=rep(0, n_run), tau=1)
model = jags.model(model_string, data=data, inits = inits, n.chains = 2, quiet=T)
update(model, 200000, progress.bar="none")

params = c("B0", "B1", "B2", "C2", "C3", "sigma", "R")
samples_5 = coda.samples(model, variable.names = params, n.iter=200000, progress.bar="none")

dic_5Star = dic.samples(model, n.iter=10000, progress.bar="none")
```

## Model Comparisons

``` r
dic_1
```

    ## Mean deviance:  -3292 
    ## penalty 2.987 
    ## Penalized deviance: -3289

``` r
dic_2
```

    ## Mean deviance:  -5030 
    ## penalty 4.106 
    ## Penalized deviance: -5025

``` r
dic_3
```

    ## Mean deviance:  -4922 
    ## penalty 5.134 
    ## Penalized deviance: -4917

``` r
dic_4
```

    ## Mean deviance:  -5146 
    ## penalty 26 
    ## Penalized deviance: -5120

``` r
dic_5Star
```

    ## Mean deviance:  -5292 
    ## penalty 583.1 
    ## Penalized deviance: -4709
