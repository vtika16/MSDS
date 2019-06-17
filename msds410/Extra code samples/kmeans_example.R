# Chad R Bhatti
# 11.28.2017
# kmeans_example.R


# Load library
library(MASS)

# Print header
head(Boston)


# Use zn as the labels;
my.Boston <- Boston[,c(-2)];
head(my.Boston)



# Note that we will not center or scale the data;
# We want to focus on the kmeans() function;
########################################################################
# Fit k=5 cluster model;
########################################################################

k.a <- kmeans(x=my.Boston,centers=5)
names(k.a)

# Table the distribution of zn to the clusters;
table(k.a$cluster)

# Access the centers of the 5 clusters;
k.a$centers



########################################################################
# Fit another k=5 cluster model;
########################################################################

k.b <- kmeans(x=my.Boston,centers=5)

# What output is returned by kmeans()?
names(k.b)


# Table the distribution of zn to the clusters;
table(k.b$cluster)

# Access the centers of the 5 clusters;
k.b$centers



# NOTICE THAT WE HAVE DIFFERENT CLUSTER DISTRIBUTIONS!
# WE ALSO HAVE DIFFERENT CLUSTERS!
# WHY?  kmeans() has a random initialization.
# We will not get the same answer each time on the same data set.
# If you want to get a repeatable answer, then you will need to set the
# seed in the R random number generator.



########################################################################
# Fit two cluster models;
########################################################################

# Set the seed for the RNG;
set.seed(123)
k.A <- kmeans(x=my.Boston,centers=5)

# Need to set the seed for the RNG each time!
set.seed(123)
k.B <- kmeans(x=my.Boston,centers=5)


# Check the distributions of the two cluster models;
table(k.A$cluster)
table(k.B$cluster)


# Check that the cluster assignment matches;
# What does this compute?
mean(k.A$cluster==k.B$cluster)


# Figure it out?
# If not, consider this example;
mean(c(0,0,1))




########################################################################
# What output does kmeans() return?
########################################################################

# List out output;
names(k.A)

# Compare;
table(k.A$cluster)
k.A$size


# Access the centers of the 5 clusters;
k.A$centers

# Access the Sum of Squares measures;
k.A$withinss
k.A$betweenss
k.A$totss
k.A$tot.withinss


# Compute WithinSS as a measure of cluster tightness;
# Do this for cluster == 1;
center.1 <- k.A$centers[1,];
cluster.1 <- my.Boston[k.A$cluster==1,];
dim(cluster.1)

head(cluster.1)

# Use Euclidean distance of point from cluster center;
# Sum over all observations, which is equivalent to summing the entire vector;
WithinSS.1 <- sum((cluster.1-matrix(data=center.1,nrow=100,ncol=13,byrow=TRUE))^2)



# Table zn by cluster assignment;
table(Boston$zn,k.A$cluster)





