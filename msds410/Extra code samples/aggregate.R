# Chad R Bhatti
# 01.02.2017
# aggregate.R

# Read (or reload) the .RData object as an R data frame
sample.df <- readRDS('C:\\Users\\Chad R Bhatti\\Dropbox\\Northwestern_MSPA\\Predict_410_R\\Data\\ames_sample.Rdata');

# Check it
str(sample.df)

# Subset to our training data set
train.df <- subset(sample.df,train==1);

# Compute conditional summaries using the aggregate() function
help(aggregate)

# Compute the mean sale price by neighborhood
aggregate(train.df$SalePrice, by=list(Neighborhood=train.df$Neighborhood), FUN=mean)

# Want to change the column names?
avgSalePrice <- aggregate(train.df$SalePrice, by=list(Neighborhood=train.df$Neighborhood), FUN=mean)
colnames(avgSalePrice) <- c('Neighborhood','AvgSalePrice')


# Compute the min sale price by neighborhood
minSalePrice <- aggregate(train.df$SalePrice, by=list(Neighborhood=train.df$Neighborhood), FUN=min)
colnames(minSalePrice) <- c('Neighborhood','MinSalePrice')


# Compute the median sqft by neighborhood
medianSQFT <- aggregate(train.df$TotalSqftCalc, by=list(Neighborhood=train.df$Neighborhood), FUN=median)
colnames(medianSQFT) <- c('Neighborhood','MedianSQFT')














