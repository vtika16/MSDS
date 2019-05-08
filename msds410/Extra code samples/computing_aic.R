# Chad R Bhatti
# 10.25.2017
# computing_aic.R


library(MASS)

# Boston Housing Data Set
str(Boston)


# Let's fit a lm() model so we have a model to work with for the example;
model.1 <- lm(medv ~ rm + age, data=Boston);
summary(model.1)


# Use the AIC() function;
AIC(model.1)

> AIC(model.1)
[1] 3306.142


# Here is the general form of AIC.  When we think of AIC, this is the formula
# that we should think of for general use in all GLM and time series models.
# We have a regression specific formula in our book, but that is only relevant
# in linear regression.
# aic = -2*logLik +2*p where p is the number of model parameters

# Compute the AIC value by hand;
my.aic <- -2*logLik(model.1)+2*length(model.1$coef)

> my.aic
'log Lik.' 3304.142 (df=4)	<- Here (df=4) is a hint that R is counting
					model parameters differently than we are.

# Whoa!  The values do not match!


# Let's compute the obvious choices and see if we can get a match.
-2*logLik(model.1)+2*3

> -2*logLik(model.1)+2*3
'log Lik.' 3304.142 (df=4)	

-2*logLik(model.1)+2*4

> -2*logLik(model.1)+2*4
'log Lik.' 3306.142 (df=4)



# Looks like the AIC() function uses p=4.  The AIC() function calls the logLik()
# function, which is saying 4 parameters, not 3 parameters.
# How many model parameters do we have?  I would say 3.  R is saying 4.  Why?
# See the help page for logLik().  For the lm() function R is counting the estimate
# of the error variance as a model parameter.
# (This tricky question is asked in your study questions on linear regression.)
# When the estimation is Maximum Likelihood Estimation, then sigma^2 is considered
# a model parameter that has to be jointly estimated with the model coefficients.
# When the estimation is Least Squares the estimate of sigma^2 is an after-the-fact
# computation, and not a true parameter estimate.

help(logLik)


# Compute AIC using the regression formula;
# aic = n*log(RSS/n) + 2*p

aic.reg <- dim(Boston)[1]*log( sum(model.1$resid^2)/dim(Boston)[1] ) + 2*3

> aic.reg
[1] 1868.176



# This is an advanced distinction above what we require from Predict 410;
# What should we know?

# (1) Different software will compute AIC and BIC differently!  Terrible pain in your
# bottom, but it will always be this way.  These differences tend to be related to the
# inclusion or exclusion of constants in the likelihood function.  Constants do not
# affect the optimization so they are frequently dropped by programmers when they 
# build the software. 

# (2) The values of AIC and BIC do not matter.  It is their relative value related
# to alternate models that is important! However, all of the AIC or BIC values 
# need to be computed using the same software!










