# Chad R Bhatti
# 07.10.2017
# log_linear_models.R


# Read (or reload) the .RData object as an R data frame
sample.df <- readRDS('C:\\Users\\Chad R Bhatti\\Dropbox\\Northwestern_MSPA\\Predict_410_R\\Data\\ames_sample.Rdata');

# Check it
str(sample.df)

# Table GarageCars
table(sample.df$GarageCars)

# Here is the table that you should see;
> table(sample.df$GarageCars)

  0   1   2   3   4 
 15 266 669 182   3 


# Let's create a family of indicator variables;
# We will take the baseline category to be 0;
# What does this mean?  Why am I creating three indicator variables?
# Should I be creating an indicator variable for 0?
sample.df$garage1 <- ifelse(sample.df$GarageCars==1,1,0);
sample.df$garage2 <- ifelse(sample.df$GarageCars==2,1,0);
sample.df$garage3 <- ifelse(sample.df$GarageCars>=3,1,0);


# Check the indicator assignment against the original table results;
table(sample.df$garage1)
table(sample.df$garage2)
table(sample.df$garage3)



# Fit a linear regression model with R
model.1 <- lm(SalePrice ~ TotalSqftCalc + TotalBathCalc + garage2 + garage3, data=sample.df)
 
# Display model summary
summary(model.1)


# Fit a second model;
model.2 <- lm(log(SalePrice) ~ TotalSqftCalc + TotalBathCalc + garage2 + garage3, data=sample.df)
 
# Display model summary
summary(model.2)

# Note that all statistical metrics like R-Squared, MSE, and MAE are computed on the 
# scale of the response variable.  Here we have two models on two different scales?
# How do we compute MSE and MAE on the same scale?
# We want to compute the MSE and MAE on the original scale SalePrice;


mse.1 <- mean(model.1$residuals^2);
mae.1 <- mean(abs(model.1$residuals));

mse.2 <- mean((sample.df$SalePrice-exp(model.2$fitted.values))^2);
mae.2 <- mean(abs(sample.df$SalePrice-exp(model.2$fitted.values)));

mse.1/mse.2
mae.1/mae.2


> mse.1
[1] 1252933135
> mae.1
[1] 25554.32
> mse.2
[1] 1109542574
> mae.2
[1] 23546.68
> mse.1/mse.2
[1] 1.129234
> mae.1/mae.2
[1] 1.085262






























