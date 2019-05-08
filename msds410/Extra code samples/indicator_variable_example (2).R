# Chad R Bhatti
# 07.10.2017
# indicator_variable_example.R


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
model.1 <- lm(SalePrice ~ TotalSqftCalc + garage1 + garage2 + garage3, data=sample.df)
 
# Display model summary
summary(model.1)


# Fit a second model;
model.2 <- lm(SalePrice ~ TotalSqftCalc + garage2 + garage3, data=sample.df)
 
# Display model summary
summary(model.2)



































