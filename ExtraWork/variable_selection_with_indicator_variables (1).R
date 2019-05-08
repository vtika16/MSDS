# Chad R Bhatti
# 11.19.2017
# variable_selection_with_indicator_variables.R


# Load the MASS library
library(MASS)

# The MASS library contains a famous data set called the Boston Housing data set;
# We do not need to load or require this data set explicitly;
# Once we load the MASS library then 'Boston' is an object in the active workspace;
help(Boston)

head(Boston)


################################################################################
# Example 1: Backwards Variable Selection with rad as a continuous variable
################################################################################

# Use the Full Model as the initial model;
# Note that this notational shortcut uses all variables on the data 
# frame as predictor variables;

initial.lm <- lm(medv ~.,data=Boston)
summary(initial.lm)



# Note that chas, zn, and rad are all treated as continuous variables in
# this approach.  Do we want that?

> table(Boston$rad)

  1   2   3   4   5   6   7   8  24 
 20  24  38 110 115  26  17  24 132 

table(Boston$rad)/sum(table(Boston$rad))

> table(Boston$rad)/sum(table(Boston$rad))

         1          2          3          4          5          6          7 
0.03952569 0.04743083 0.07509881 0.21739130 0.22727273 0.05138340 0.03359684 
         8         24 
0.04743083 0.26086957 




# rad - probably not a good candidate to be a continuous variable
plot(Boston$rad,Boston$medv)
boxplot(Boston$medv ~ Boston$rad)

# Maybe some of these values should be grouped together?  Maybe 4 and 6?  Maybe others?
# How could we make that decision?  Could we use variable selection to make that decision?



initial.lm <- lm(medv ~.,data=Boston)
summary(initial.lm)

backward.lm <- stepAIC(object=initial.lm, direction=c('backward'));
summary(backward.lm)


# Note that when rad is treated as numeric, the whole variable is either selected
# or not selected.  Maybe some groups are different and the model would benefit
# from including intercept adjustments for those groups.





################################################################################
# Example 2: Backwards Variable Selection with rad as a factor
################################################################################

# Use the Full Model as the initial model;
initial.lm <- lm(medv ~ crim+zn+indus+chas+nox+rm+age+dis ++
	+ as.factor(rad)+tax+ptratio+black+lstat,data=Boston)
summary(initial.lm)



# Apply backward variable selection
backward.lm <- stepAIC(object=initial.lm, direction=c('backward'));

summary(backward.lm)

# Note that the factor variable is treated as a single variable;
# Still how do we consider effects for some groups or groupings?
# ANSWER -> code the factor variables as a family of indicator variables




################################################################################
# Example 3: Backwards Variable Selection with rad as indicators
################################################################################

my.Boston <- Boston;

table(my.Boston$rad)
> table(my.Boston$rad)

  1   2   3   4   5   6   7   8  24 
 20  24  38 110 115  26  17  24 132 


# Take rad==1 as the base category;
# In general I recommend taking the smallest category as the baseline category.
# Here several are similar in size so we will take 1 instead of 7 as the baseline.



# Code the indicators for the other factor levels;
my.Boston$rad2 <- ifelse(my.Boston$rad==2,1,0);
my.Boston$rad3 <- ifelse(my.Boston$rad==3,1,0);
my.Boston$rad4 <- ifelse(my.Boston$rad==4,1,0);
my.Boston$rad5 <- ifelse(my.Boston$rad==5,1,0);
my.Boston$rad6 <- ifelse(my.Boston$rad==6,1,0);
my.Boston$rad7 <- ifelse(my.Boston$rad==7,1,0);
my.Boston$rad8 <- ifelse(my.Boston$rad==8,1,0);
my.Boston$rad24 <- ifelse(my.Boston$rad==24,1,0);



# Use the Full Model as the initial model;
initial.lm <- lm(medv ~ crim+zn+indus+chas+nox+rm+age+dis ++
	+tax+ptratio+black+lstat ++
	+rad2+rad3+rad4+rad5+rad6+rad7+rad8+rad24,
	data=my.Boston)
summary(initial.lm)



# Apply backward variable selection
backward.lm <- stepAIC(object=initial.lm, direction=c('backward'));
summary(backward.lm)


# Note that each indicator variable is treated separately;
# The backward variable selection kicked out rad2 and rad6;
# What does that mean?
# It means that rad=2 and rad=6 are not different from the baseline category
# rad=1.  Now our baseline category contains the values 1,2, and 6.

# Since estimation precision is affected by sample size, variable selection will
# typically lump some small categories together or with other larger categories
# while keeping the larger categories separate.  The larger categories using have 
# an 'estimatable intercept adjustment'.





















