# Chad R Bhatti
# 11.16.2017
# stepAIC_example.R


# Load the MASS library
library(MASS)

# The MASS library contains a famous data set called the Boston Housing data set;
# We do not need to load or require this data set explicitly;
# Once we load the MASS library then 'Boston' is an object in the active workspace;
help(Boston)

head(Boston)
str(Boston)



#######################################################################
# Example 1: Backwards Variable Selection
#######################################################################

# Use the Full Model as the initial model;
# Note that this notational shortcut uses all variables on the data 
# frame as predictor variables;

initial.lm <- lm(medv ~.,data=Boston)
summary(initial.lm)



# Note that chas, zn, and rad are all treated as continuous variables.
# Check the str().  See that no variables are defined as factors.
# When we include chas, zn, and rad as numeric in a regression we are then
# treating them as continuous variables. Do we want that?

> table(Boston$chas)

  0   1 
471  35 


> table(Boston$zn)

   0 12.5 17.5   18   20   21   22   25   28   30   33   34   35   40   45 52.5 
 372   10    1    1   21    4   10   10    3    6    4    3    3    7    6    3 
  55   60   70   75   80 82.5   85   90   95  100 
   3    4    3    3   15    2    2    5    4    1 


> table(Boston$rad)

  1   2   3   4   5   6   7   8  24 
 20  24  38 110 115  26  17  24 132 

# chas - okay since it is a binary variable and will estimate to be an intercept
# adjustment

# zn - maybe okay based on plots.  No apriori reason for it to be okay.
plot(Boston$zn,Boston$medv)
boxplot(Boston$medv ~ Boston$zn)
 

# rad - probably not a good candidate to be a continuous variable
plot(Boston$rad,Boston$medv)
boxplot(Boston$medv ~ Boston$rad)


# Define a new data frame;
my.Boston <- Boston;
my.Boston$rad <- as.factor(Boston$rad);
str(my.Boston)


# With factor rad;
initial.lm <- lm(medv ~.,data=my.Boston)
summary(initial.lm)

# Apply backward variable selection
backward.lm <- stepAIC(object=initial.lm, direction=c('backward'));
summary(backward.lm)


# With numeric rad;
initial.lm2 <- lm(medv ~.,data=Boston)
summary(initial.lm2)

backward.lm2 <- stepAIC(object=initial.lm2, direction=c('backward'));
summary(backward.lm2)


# Recall: variable selection with AIC is to select the model with the
# smallest AIC value!  For backward variable selection R will compute the 
# AIC value for the initial model,
# and then the AIC value for every 1-delta model, i.e. the initial model
# MINUS a single predictor that is available in the pool.  It will continue
# to DELETE variables until the null model has the smallest AIC value.



#######################################################################
# Example 2: Forward Variable Selection 
#######################################################################
# Going back to the original data set to focus on the programming
# and interpretation of variable selection in R.
#######################################################################

# Use the Intercept Model as the initial model;
lower.lm <- lm(medv ~ 1,data=Boston)
summary(lower.lm)

# What is the Intercept Model?
# Hint:
mean(Boston$medv)


# Define upper model;
upper.lm <- lm(medv ~.,data=Boston)
summary(upper.lm)


# Apply forward variable selection
forward.lm <- stepAIC(object=lower.lm, direction=c('forward'),
	scope=list(upper=formula(upper.lm), lower=formula(lower.lm))
	);

summary(forward.lm)


# Recall: variable selection with AIC is to select the model with the
# smallest AIC value!  For forward variable selection R will compute the 
# AIC value for the initial model,
# and then the AIC value for every 1-delta model, i.e. the initial model
# PLUS a single predictor that is available in the pool.  It will continue
# to ADD variables until the null model has the smallest AIC value.




#######################################################################
# Example 3: Stepwise Variable 
#######################################################################

initial.lm <- lm(medv ~ lstat,data=Boston)
summary(initial.lm)


# Apply stepwise variable selection
stepwise.lm <- stepAIC(object=initial.lm, direction=c('both'),
	scope=list(upper=formula(upper.lm), lower=formula(lower.lm))
	);

summary(stepwise.lm)


# Stepwise variable selection is the combination of both forward variable
# selection and backwards variable selection.  Look at the 1-delta models.
# They all have a + or - in front of them.  Those are the AIC values when 
# you add a variable from the pool to the initial model or subtract a 
# variable from the existing model.



















