# Chad R Bhatti
# 06.24.2017
# anscombe.R

# Are we familiar with Anscombe's quartet?

# The paper is in the Predict 410 Course Reserves.
# Anscombe, Francis J. (1973) Graphs in statistical analysis. American Statistician, 27, 17–21. 
# Please read this paper before running this code on your own.



# R has some famous data sets preloaded into Base R;
help(anscombe)

# Just type anscombe and it will appear.  No need to load anything.
anscombe


# Note that this is four pairs of data (X1,Y1), (X2,Y2), (X3,Y3), and (X4,Y4)
# For this example we will attach the dataframe.
# In general I do not recommend that you attach the dataframe.
# Attaching the dataframe will allow us to use x1 instead of anscombe$x1.

# Attach dataframe
attach(anscombe)


# Each of these four small data sets is designed to have the same correlation between
# the x and y components.

cor(x1,y1)
cor(x2,y2)
cor(x3,y3)
cor(x4,y4)

# They also have the same variance in the x component.
var(x1)
var(x2)
var(x3)
var(x4)

# They have almost the same variance in the y component.
# This is an old paper so they have the same variance to two decimal places which is what he
# wanted for hand computations.

var(y1)
var(y2)
var(y3)
var(y4)



# Since these pairs have the same correlations and the same variances to two decimal places, 
# if we performed the computations by hand they will also have the same beta coefficients
# when fitting a simple linear regression model.  Do we know why?


# Now let's fit these four models in R.

lm.1 <- lm(y1~x1)
summary(lm.1)

lm.2 <- lm(y2~x2)
summary(lm.2)

lm.3 <- lm(y3~x3)
summary(lm.3)

lm.4 <- lm(y4~x4)
summary(lm.4)


# All of these models are essentially the same.  They all have the same parameter estimate (beta
# coefficients).  They all have the same R-Squared values, F-statistics, and residual standard
# errors.  However, do all of these models fit equally well?


# Maybe numerical summaries can hide model problems, especially in small data sets.
# Let's plot these four data sets into a four panel plot;
par(mfrow=c(2,2))
plot(x1,y1, main='Ascombe #1')
plot(x2,y2, main='Ascombe #2')
plot(x3,y3, main='Ascombe #3')
plot(x4,y4, main='Ascombe #4')



# Do we still think that all of these regression models are equally as good?
# Why or why not?


# Maybe instead of just looking at statistical metrics, we should also look at some
# prediction metrics.  Here all of these prediction metrics are in-sample, but still
# very useful.  Let's look at the Mean Square Error and the Mean Absolute Error.

# Mean Square Error
mse.1 <- mean(lm.1$residuals^2)
mse.2 <- mean(lm.2$residuals^2)
mse.3 <- mean(lm.3$residuals^2)
mse.4 <- mean(lm.4$residuals^2)


> mse.1
[1] 1.251154
> mse.2
[1] 1.25239
> mse.3
[1] 1.250563
> mse.4
[1] 1.249317



# Mean Absolute Error
mae.1 <- mean(abs(lm.1$residuals))
mae.2 <- mean(abs(lm.2$residuals))
mae.3 <- mean(abs(lm.3$residuals))
mae.4 <- mean(abs(lm.4$residuals))


> mae.1
[1] 0.837405
> mae.2
[1] 0.9679339
> mae.3
[1] 0.7159669
> mae.4
[1] 0.9027273


# From the plot we should think that Anscombe #1 looks like the most traditional view
# of a linear regression data set.  
# BUT, Ascombe #3 is the 'nicest' linear regression data set.
# Anscombe #3 is a perfectly straight line with no noise and a single outlier.  
# From the MAE we see that Acombe #3 is fit the best by it's linear regression model.


# Let's plot the four data sets again, but this time overlay the fitted regression models
# using the R function abline()
par(mfrow=c(2,2))
plot(x1,y1, main='Ascombe #1')
abline(lm.1$coef[1],lm.1$coef[2],lwd=2,col='red')
plot(x2,y2, main='Ascombe #2')
abline(lm.2$coef[1],lm.1$coef[2],lwd=2,col='red')
plot(x3,y3, main='Ascombe #3')
abline(lm.3$coef[1],lm.1$coef[2],lwd=2,col='red')
plot(x4,y4, main='Ascombe #4')
abline(lm.4$coef[1],lm.1$coef[2],lwd=2,col='red')







##############################################################################################
# Comments:
##############################################################################################
(1) Traditional statistical modeling has always relied heavily on the use of statistical 
graphics.  We should note that traditional statistical modeling has was developed to validate
the GoF of statistical model assumptions on small data sets fitted with the use of statistical
inference in mind.

(2) As data sets get larger we cannot use statistical graphics in the same manner as we would on
small data sets.  We will have some statistcal graphics that will still be useful, and some 
that will not.  

(3) With large data sets we need to produce summary statistics to evaluate our models.  These
summary statistics will not validate our models that well for statistical inference, BUT they
work very well for predictive modeling.  We typically produce both the MSE and the MAE for all
modeling efforts.  Later in the course we will also note that we produce both of these metrics
on the in-sample data and the out-of-sample data.

(4) Your Practice - Perform a traditional residual analysis on the quartet models and see if you
can ascertain some models to fit better than some other models.  Look at residual versus predictor
plots and Cook's distance.







