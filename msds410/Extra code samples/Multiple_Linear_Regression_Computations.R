# Multiple Linear Regression Computations
# Formula not in RABE
# Need a book like Sheather - pdf available from library


# Assign file paths for sample data location;
my.path <- 'C:\\Users\\Chad R Bhatti\\Dropbox\\Northwestern_MSPA\\MSDS_410_R\\RABE\\';
my.file <- paste(my.path,'supervisor_performance.txt',sep='');

# Read in a tab delimited file with read.table() and sep='\t';
my.data <- read.table(my.file,header=TRUE,sep='\t');

# List out variables types;
str(my.data)

# See header of data frame;
head(my.data)


# Fit the full model
model.1 <- lm(Y ~ X1+X2+X3+X4+X5+X6,data=my.data)
summary(model.1)

# Compare this solution to the solution on p. 70 RABE;

##################################################################################
# Note that there is an R shortcut for including all of the variables in a
# data frame in a linear regression model.

model.2 <- lm(Y ~ .,data=my.data)
summary(model.2)
##################################################################################



##################################################################################
# How do we get these estimates?
##################################################################################
# Note: In R matrix multiplication is denoted by %*%;
# One way is to directly solve the normal equations;
# Y = X%*%B
# t(X)%*%Y = t(X)%*%X%*%B - Normal Equations

one <- rep(1,dim(my.data)[1]);
X <- as.matrix(cbind(one,my.data[,-1]));
Y <- as.matrix(my.data[,1]);
XX <- t(X)%*%X;
XY <- t(X)%*%Y;
beta.hat <- solve(a=XX, b=XY)


# Frequently this solution is written as as if the inv(XX) is guaranteed to exist;
# This inverse is not guaranteed to exist;
# beta.hat <- inv(t(X)%*%X)%*%X%*%Y;

XX.inv <- solve(a=XX,b=diag(7));

# Check the inverse matrix;
XX.inv%*%XX

# Note that numerical computing does not return nice answers;
# Let's round the matrix to make it look nice;
round(XX.inv%*%XX,1)

# Looks like our inverse matrix is correct;

beta.hat2 <- XX.inv%*%XY;


# Numerical implementations should be solving the normal equations directly;
# In linear regression this is typically this is performed using a QR factorization.




##################################################################################
# How do we get these standard errors for the estimates?
##################################################################################

# First we need an estimate for sigma;
anova(model.1)

# sigma2 is the Residual Mean Sq 1149/23 = 49.96

# The standard errors of the beta coefficients are the diagonal elements of XX.inv
# multiplied by sigma2

se <- sqrt(49.96*diag(XX.inv))

# Compare to the fitted model
summary(model.1)


##################################################################################
# How do we get the t-statistics for the estimates?
##################################################################################

t <- beta.hat/se

# Compare to the fitted model;
summary(model.1)



##################################################################################
# How do we get the p-values?
##################################################################################
# Here we will use the R function sapply() to apply the pt() to the vector
# of t-stats all at one time.

p <- 2*sapply(-abs(t),FUN=pt,df=model.1$df.residual);





##################################################################################
# How do we get the fitted values using the hat matrix?
##################################################################################
# Compute the Hat matrix times Y;
# Remember the hat matrix puts the hats on Y;
HY <- X%*%XX.inv%*%XY

# Eyeball compare to the software computed fitted values;
model.1$fitted

# Quantitatively compare our fitted values to the software fitted values;
# We should expect to see zeros;
round(H-model.1$fitted,0.001)










anova(model.1)