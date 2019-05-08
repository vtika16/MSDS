# Chad R Bhatti
# 12.31.2016
# edaWithR.R

# How do we perform Exploratory Data Analysis in R?

# Read (or reload) the .RData object as an R data frame
sample.df <- readRDS('C:\\Users\\Chad R Bhatti\\Dropbox\\Northwestern_MSPA\\Predict_410_R\\Data\\ames_sample.Rdata');

# Check it
str(sample.df)

# Technically we should perform our EDA on the training data set
train.df <- subset(sample.df,train==1);



##################################################################################
# BASE R Scatterplot
##################################################################################

plot(train.df$TotalSqftCalc,train.df$SalePrice)


# Let's control the R plot to make it pretty
plot(train.df$TotalSqftCalc,train.df$SalePrice/1000,xlab='Total SQFT',ylab='Sale Price (000)',
	main='SQFT and Sale Price')

# In low dimensions scatterplots can be useful for visualizing the relationship
# between the response and a predictor variable


##################################################################################
# BASE R Box Plot
##################################################################################

boxplot(train.df$SalePrice/1000 ~ train.df$Neighborhood)
# Notice that R does not display all of the neighborhood names
# Need to use the graphical parameters
help(par)
# We want the parameter las

boxplot(train.df$SalePrice/1000 ~ train.df$Neighborhood, las=2)
title('Sale Price By Neighborhood')


# Boxplots are the tool of choice for visualizing factor variables



##################################################################################
# Treating a discrete variable as continuous 
##################################################################################

plot(train.df$TotalBathCalc,train.df$SalePrice/1000)


# Sometime we will choose to treat a discrete variable as a continuous variable.
# In these cases the discrete variable needs to have 'enough' values, and a 'nice' 
# relationship with the response variable.

# What do we think here?  Do we have enough values?  Do we have a 'nice' relationship?




# Notes:

(1) See the book 'R For Everyone' for more plotting options and examples of ggplot,
if you are interested in using ggplot.

(2) Do you know how to plot out a loess scatterplot smoother?  See if you can figure it out.














 









