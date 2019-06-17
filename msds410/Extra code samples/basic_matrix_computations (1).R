# basic_matrix_computations.R

# Basic matrix computations using R;
# Read the course reserve on matrix operations;
# Chapter 2 Matrix Algebra in Methods of Multivariate Analysis by Rencher and Christensen;


#########################################################################
# Define a vector.  Note that R doesn't distinguish the programming
# concepts of a vector (a mathematical object) and an array (a programming
# object).
#########################################################################

# R will define the vector as a row vector by default;
# In mathematics vectors are column vectors by default;

vector.1 <- c(2,3,5);

> vector.1
[1] 2 3 5



#########################################################################
# Multiple the vector by a scalar;
# A scalar is a number.  Here we will multiply the vector by 2;
#########################################################################

2*vector.1

> 2*vector.1
[1]  4  6 10



#########################################################################
# 'Multiply' the two vectors.  What is this?
#########################################################################
# Define a second vector;
vector.2 <- c(2,2,2);

> vector.2
[1] 2 2 2

vector.1*vector.2

> vector.1*vector.2
[1]  4  6 10

# This is NOT vector multiplication!  This is componentwise array 
# multiplication.



#########################################################################
# Matrix Multiply the two vectors.  What is this?
#########################################################################
# Matrix multiplication in R is %*%;
# Matrix multiply two vectors;

vector.1%*%vector.2

> vector.1%*%vector.2
     [,1]
[1,]   20

# This is called the 'inner product' or the 'scalar product'.


# How does matrix multiplication work for vectors?
# Multiply the components and then sum them up.

sum(vector.1*vector.2)

> sum(vector.1*vector.2)
[1] 20



#########################################################################
# Define a 2x2 matrix;
#########################################################################

a <- seq(1,4,1);
A <- matrix(data=a,nrow=2,ncol=2,byrow=TRUE);

>  matrix(data=a,nrow=2,ncol=2,byrow=TRUE);
     [,1] [,2]
[1,]    1    2
[2,]    3    4



#########################################################################
# Define the transpose of A;
#########################################################################
A.t <- t(A);

>  t(A)
     [,1] [,2]
[1,]    1    3
[2,]    2    4



#########################################################################
# Define the inverse of A;
#########################################################################
# Compute the inverse using solve();
A.inv <- solve(A);

> solve(A)
     [,1] [,2]
[1,] -2.0  1.0
[2,]  1.5 -0.5

# Show that the inverse is correct;
A%*%A.inv

> A%*%A.inv
     [,1]         [,2]
[1,]    1 1.110223e-16
[2,]    0 1.000000e+00


round(A%*%A.inv,1)

> round(A%*%A.inv,1)
     [,1] [,2]
[1,]    1    0
[2,]    0    1

# A matrix times its inverse will always return the identity matrix;
# The identity matrix is the matrix with 1 on the main diagonal and 0
# everywhere else;

# Only square matrices have inverses;
# Not every matrix, including not every square matrix, will have an inverse;



#########################################################################
# Conforming Matrices
#########################################################################
# In order to multiply to matrices together the dimensions need to be
# 'conforming'.

# [n,p] %*% [p,k] = [n,k]
# The inner dimensions of the two matrices need to be the same;

b <- seq(1,6,1);
B <- matrix(data=b,nrow=2,ncol=3,byrow=TRUE);

> B
     [,1] [,2] [,3]
[1,]    1    2    3
[2,]    4    5    6


# Multiply A and B;
A%*%B

> A%*%B
     [,1] [,2] [,3]
[1,]    9   12   15
[2,]   19   26   33


# Multiply B and A;
B%*%A

> B%*%A
Error in B %*% A : non-conformable arguments


# Multiplying matrices is not like multiplying numbers!
# Why did we get this error message from R?
# What does it mean when we see this error message?


# Multiple t(B) and A;
t(B)%*%A

> t(B)%*%A
     [,1] [,2]
[1,]   13   18
[2,]   17   24
[3,]   21   30

# Why are the dimensions of A%*%B different from t(B)%*%A?



#########################################################################
# Compute eigenvectors and eigenvalues for a matrix;
#########################################################################
# Use the function eigen();

a <- c(1,7,51,10);
A <- matrix(data=a,nrow=2,ncol=2,byrow=FALSE);
 
# Compute both eigenvalues and eigenvectors;
eigen.out <- eigen(A,only.values=FALSE);

> eigen.out
$values
[1]  24.92292 -13.92292

$vectors
           [,1]       [,2]
[1,] -0.9053451 -0.9597572
[2,] -0.4246765  0.2808311


# Note that the eigenvalue and eigenvector are a pair.
# The 'first' eigenvalue is the largest eigenvalue and it has the 
# corresponding eigenvector e.1, which is the first column on E.

lamba.1 <- eigen.out$values[1];
e.1 <- eigen.out$vectors[,1];

> eigen.out$values[1];
[1] 24.92292
> eigen.out$vectors[,1];
[1] -0.9053451 -0.4246765



#########################################################################
#########################################################################
#########################################################################












