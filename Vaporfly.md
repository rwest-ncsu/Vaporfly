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

``` r
ggplot(data=final, mapping=aes(x=time_minutes))+
  labs(x="Time in Minutes", title = "FIGURE 1: Histogram of Time(min) for all runners")+
  geom_histogram(bins=75, fill="red", col="black")
```

![](Vaporfly_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
ggplot(data=final%>%filter(sex==F), mapping=aes(x=log(time_minutes)))+
  labs(title="FIGURE 2: Distribution of Men's log time", x="Men's Log minutes")+
  geom_histogram(bins=85, fill="blue", col="black")
```

![](Vaporfly_files/figure-gfm/unnamed-chunk-4-2.png)<!-- -->

``` r
ggplot(data=final%>%filter(sex==T), mapping=aes(x=log(time_minutes)))+
  labs(title="FIGURE 3: Distribution of Women's log time", x="Women's Log minutes")+
  geom_histogram(bins=85, fill="pink", col="black")
```

![](Vaporfly_files/figure-gfm/unnamed-chunk-4-3.png)<!-- -->

``` r
ggplot(data=final, mapping=aes(x=log(time_minutes)), fill=as.factor(sex))+
  labs(title="FIGURE 2: Histogram of log(time)", x="log(Time)")+
  geom_histogram(bins=75, col="black")
```

![](Vaporfly_files/figure-gfm/unnamed-chunk-4-4.png)<!-- -->

``` r
ggplot(data=final, mapping=aes(x=time_minutes, fill=vaporfly))+
  geom_histogram(bins = 60)
```

![](Vaporfly_files/figure-gfm/unnamed-chunk-4-5.png)<!-- -->

``` r
ggplot(data=final%>%filter(sex==0), mapping=aes(x=time_minutes, fill=vaporfly))+
  geom_histogram(bins = 60)
```

![](Vaporfly_files/figure-gfm/unnamed-chunk-4-6.png)<!-- -->

``` r
ggplot(data=final%>%filter(sex==1), mapping=aes(x=time_minutes, fill=vaporfly))+
  geom_histogram(bins = 60)
```

![](Vaporfly_files/figure-gfm/unnamed-chunk-4-7.png)<!-- -->

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
    ## B1     5.00600 0.002444 3.864e-05      4.358e-05
    ## B2    -0.03149 0.006954 1.100e-04      1.215e-04
    ## sigma  0.09381 0.001622 2.564e-05      2.519e-05
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##           2.5%      25%      50%      75%    97.5%
    ## B1     5.00119  5.00438  5.00607  5.00765  5.01071
    ## B2    -0.04515 -0.03615 -0.03136 -0.02692 -0.01788
    ## sigma  0.09072  0.09267  0.09382  0.09492  0.09714

![](Vaporfly_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

    ## Potential scale reduction factors:
    ## 
    ##       Point est. Upper C.I.
    ## B1             1       1.00
    ## B2             1       1.01
    ## sigma          1       1.00
    ## 
    ## Multivariate psrf
    ## 
    ## 1

By popular convergence diagnostics (Gelman Statistic and Trace Plots),
this model has converged to the parameter estimates given. Now, we fit
more sophisticated models to attempt to characterize the relationship
more thoroughly.

## Model 2

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
    ##           Mean       SD  Naive SE Time-series SE
    ## B[1]   4.93654 0.001915 2.472e-05      4.424e-05
    ## B[2]  -0.01896 0.003890 5.023e-05      5.871e-05
    ## B[3]   0.14106 0.002630 3.395e-05      5.711e-05
    ## sigma  0.05234 0.000921 1.189e-05      1.218e-05
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##           2.5%      25%      50%      75%    97.5%
    ## B[1]   4.93273  4.93526  4.93659  4.93785  4.94023
    ## B[2]  -0.02635 -0.02166 -0.01901 -0.01638 -0.01131
    ## B[3]   0.13588  0.13932  0.14100  0.14281  0.14631
    ## sigma  0.05056  0.05171  0.05233  0.05294  0.05416

``` r
gelman.diag(samples)
```

    ## Potential scale reduction factors:
    ## 
    ##       Point est. Upper C.I.
    ## B[1]           1       1.01
    ## B[2]           1       1.01
    ## B[3]           1       1.00
    ## sigma          1       1.00
    ## 
    ## Multivariate psrf
    ## 
    ## 1

``` r
#Compute DIC
dic_2 = dic.samples(model, n.iter=3000, progress.bar="none")
```

## Model 3

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
    ## B[1]   4.9364441 0.002310 3.653e-05      6.784e-05
    ## B[2]  -0.0185867 0.006200 9.803e-05      1.781e-04
    ## B[3]   0.1412012 0.003273 5.175e-05      9.708e-05
    ## C     -0.0004753 0.009594 1.517e-04      2.753e-04
    ## sigma  0.0620439 0.001097 1.734e-05      1.725e-05
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##           2.5%       25%        50%       75%     97.5%
    ## B[1]   4.93200  4.934857  4.9364615  4.938039  4.941062
    ## B[2]  -0.03072 -0.022773 -0.0185376 -0.014460 -0.006519
    ## B[3]   0.13465  0.139043  0.1412339  0.143399  0.147464
    ## C     -0.01869 -0.007046 -0.0006164  0.005926  0.018690
    ## sigma  0.05991  0.061277  0.0620481  0.062796  0.064157

``` r
gelman.diag(samples)
```

    ## Potential scale reduction factors:
    ## 
    ##       Point est. Upper C.I.
    ## B[1]           1          1
    ## B[2]           1          1
    ## B[3]           1          1
    ## C              1          1
    ## sigma          1          1
    ## 
    ## Multivariate psrf
    ## 
    ## 1

``` r
dic_3 = dic.samples(model, n.iter=2000, progress.bar="none")
```

## Model 4

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
    ## B0              2.26       4.49
    ## B1              1.00       1.00
    ## B2              1.00       1.00
    ## alpha[1]        2.26       4.48
    ## alpha[2]        2.26       4.48
    ## alpha[3]        2.26       4.48
    ## alpha[4]        2.21       4.36
    ## alpha[5]        2.18       4.27
    ## alpha[6]        2.26       4.48
    ## alpha[7]        2.25       4.47
    ## alpha[8]        2.24       4.44
    ## alpha[9]        2.25       4.46
    ## alpha[10]       1.74       3.09
    ## alpha[11]       2.11       4.08
    ## alpha[12]       2.24       4.44
    ## alpha[13]       2.26       4.48
    ## alpha[14]       2.25       4.46
    ## alpha[15]       2.24       4.44
    ## alpha[16]       2.10       4.06
    ## alpha[17]       2.21       4.36
    ## alpha[18]       2.25       4.46
    ## alpha[19]       2.25       4.45
    ## alpha[20]       2.22       4.37
    ## alpha[21]       2.22       4.37
    ## alpha[22]       1.93       3.60
    ## alpha[23]       2.25       4.46
    ## sigma           1.00       1.00
    ## 
    ## Multivariate psrf
    ## 
    ## 1.57

``` r
#Compute DIC
dic_4 = dic.samples(model, n.iter=50000, progress.bar="none")
```

This model suggests that there is no significant difference between
marathon courses since the effective sample sizes are so small even
after 50,000 iterations of MCMC. From this, I conclude that the marathon
effects are constant and are “encoded” in the constant intercept
\(\beta_0\). For further justification:

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

Clearly, all of the \(\alpha\) effects behave similarly and produced
relatively similar estimates and Posterior Distributions. Because of
this and the low effective sample size of the \(\beta_0\) in this model,
I propose the removal of the Marathon effect term in the model.

## Model 5
