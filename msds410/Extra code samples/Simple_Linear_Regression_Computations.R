# RABE 5th Edition
# Problem 2.12
# p. 53


# In this script we will use the data from #2.12 to explore R functions and R programming;
# We will also focus on computing basic statistical quantities that are output from
# fitting linear regression models;

# Assign file paths for sample data location;
my.path <- 'C:\\Users\\Chad R Bhatti\\Dropbox\\Northwestern_MSPA\\MSDS_410_R\\RABE\\';
my.file <- paste(my.path,'newspaper_data.txt',sep='');

# Read in a tab delimited file with read.table() and sep='\t';
my.data <- read.table(my.file,header=TRUE,sep='\t');


# Part(a)
# Standard Base R scatterplot;
plot(my.data$Daily,my.data$Sunday)


# User manipulated Base R scatterplot;
plot(my.data$Daily,my.data$Sunday,xlab='Daily Circulation',ylab='Sunday Circulation',
	xlim=c(0,700),ylim=c(0,1200),main='Sunday Circulation')

# Scatter plot suggests a linear relationship between daily circulation and Sunday 
# circulation;


# Part(b)
# Fit a regression model in R;
sunday.lm <- lm(Sunday ~ Daily, data=my.data);

# Display a summary of the regression model;
summary(sunday.lm)

# Display the retained components of the regression model;
names(sunday.lm)

# Extract the coefficients of a regression model;
sunday.lm$coef

# Extract the fitted values (Y-hat) of the regression model;
plot(my.data$Sunday,sunday.lm$fitted.values,xlab='Actual',ylab='Predicted',
xlim=c(0,2000),ylim=c(0,2000))
abline(a=0,b=1)

# Extract the model residuals;
mean(sunday.lm$residuals)

# Compare this 5 number summary of the residuals to the one from the regression summary;
summary(sunday.lm$residuals)

# Construct a histogram of the residuals;
hist(sunday.lm$residuals,main='Model Residuals',xlab='')

# Compute sigma.hat 
# p. 37 EQ(2.23)
sigma2.hat <- sum(sunday.lm$residuals^2)/sunday.lm$df.residual;

# Compute SE(beta.0)
mean.x <- mean(my.data$Daily);
sse.x <- sum((my.data$Daily-mean.daily)^2);
SE.beta0 <- sqrt(sigma2.hat)*sqrt(1/dim(my.data)[1] + (mean.x^2)/sse.x);

# Compute SE(beta.1)
SE.beta1 <- sqrt(sigma2.hat)/sqrt(sse.x);

# Compute t-statistics for null hypothesis of 0;
t.0 <- (sunday.lm$coef[1] -0)/SE.beta0;
t.1 <- (sunday.lm$coef[2] -0)/SE.beta1;

# Compute p-values for t-statistics;
p.0 <- 2*pt(-abs(t.0),df=sunday.lm$df.residual);
p.1 <- 2*pt(-abs(t.1),df=sunday.lm$df.residual);




















