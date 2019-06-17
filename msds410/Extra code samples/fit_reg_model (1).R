# Chad R Bhatti
# 12.31.2016
# fit_reg_model.R


# Read (or reload) the .RData object as an R data frame
sample.df <- readRDS('C:\\Users\\Chad R Bhatti\\Dropbox\\Northwestern_MSPA\\Predict_410_R\\Data\\ames_sample.Rdata');

# Check it
str(sample.df)

# Technically we should perform our EDA on the training data set
train.df <- subset(sample.df,train==1);


# Fit a linear regression model with R
model.1 <- lm(SalePrice ~ TotalSqftCalc, data=train.df)
 
# Display model summary
summary(model.1)

# List out components of lm object
names(model.1)

# Access a component of lm object
model.1$coefficients

# Access residuals to compute Mean Square Error (MSE) and Mean Absolute Error (MAE)
mse.1 <- mean(model.1$residuals^2)
mae.1 <- mean(abs(model.1$residuals))


# BASE R diagnostic plot for lm object
plot(model.1)
# Not too useful for writing a report

# Panel the plots
par(mfrow = c(2, 2), oma = c(0, 0, 2, 0))
plot(model.1)

# Are these the diagnostic plots that we want?
# What diagnostic plots are of primary interest to us?
# What plots are most useful for validating model assumptions (for inference)?

# Hint:  We need to check the assumptions of normality and homoscedasticity;
# (1) QQ Plot
# (2) Scatterplot of residuals versus predictor


# Use the Base R functon qqplot() to assess the normality of the residuals
qqnorm(model.1$residuals)
qqline(model.1$residuals)

# Make a scatterplot
plot(train.df$TotalSqftCalc,model.1$residuals)
title('Residual vs Predictor')
# Do we detect any issues here?


# Note that these plots are sufficient to assess the GOF of a regression model in
# Predict 410.





########################################################################################
# Diagnostic plots using Weisberg's car package
########################################################################################
# First we need to install the car package
# See Section 1.2.4 p.31 of CAR (Companion to Applied Regression)

# Install package and all other needed packages
install.packages('car', dependencies=TRUE)

# Load library into your active R session
library(car)

# Use the function qqPlot() in the car package to assess the Studentized residuals
qqPlot(model.1)

# Note that this is not the typical QQ plot.  Also note that the Studentized residuals
# have a different distribution than the standard residuals.


# Cook's Distance Plot
influenceIndexPlot(model.1,vars=c('Cook','hat'))

# Note that the x index is not correct.  They are the row labels from the original
# data frame.

rownames(train.df) <- seq(1,length(model.1$residuals),1)
model.1 <- lm(SalePrice ~ TotalSqftCalc, data=train.df)
influenceIndexPlot(model.1,vars=c('Cook','hat'))

# If we want the labels correct, then we have to go back to the original data frame
# and fix the row names, then refit the model, and then call the plot.






































