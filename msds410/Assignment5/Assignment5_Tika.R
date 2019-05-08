library("tidyverse", lib.loc="~/R/win-library/3.5")
library("ggplot2", lib.loc="~/R/win-library/3.5")
library("stargazer", lib.loc="~/R/win-library/3.5")
library("PerformanceAnalytics", lib.loc="~/R/win-library/3.5")
library("corrplot", lib.loc="~/R/win-library/3.5")
library("MASS", lib.loc="~/R/win-library/3.5")

ames.df = read.csv("c:/Users/vtika/Desktop/MSDS/msds_410/ames_housing_data.csv", header = TRUE,stringsAsFactors = FALSE)

ames.df = data.frame(ames.df)

str(ames.df)
########################################
###determining our sample set of data###
########################################

##Narrowing down to single family homes and removing houses where GrllivArea is less than 
##800 and greater than 4000

ames.df$dropCondition <- ifelse(ames.df$BldgType!='1Fam','01: Not SFR',
  ifelse(ames.df$SaleCondition!='Normal','02: Non-Normal Sale',
  ifelse(ames.df$Street!='Pave','03: Street Not Paved',
  ifelse(ames.df$YearBuilt <1950,'04: Built Pre-1950',
  ifelse(ames.df$TotalBsmtSF <1,'05: No Basement',
  ifelse(ames.df$GrLivArea <800,'06: LT 800 SqFt',
  '99: Eligible Sample')
  )))))

##follow up with a table to look at the eligible population

table(ames.df$dropCondition)

##Making a matrix to export to create a pic for the report

waterfall <- table(ames.df$dropCondition);

waterfall.matrix <- as.matrix(waterfall,4,2)


out.path <- "c:/Users/vtika/Desktop/R/msds_410/"
file.name <- 'summarstatistics.html';

stargazer(ames.df, type=c('html'),out=paste(out.path,file.name,sep=''),
          title=c('Table XX: Summary Statistics for Boston Housing'),
          align=TRUE, digits=2, digits.extra=2, initial.zero=TRUE, median=TRUE)


# Eliminate all observations that are not part of the eligible sample population;
elig.pop <- subset(ames.df,dropCondition=='99: Eligible Sample');


#Check that all remaining observations are eligible;
table(elig.pop$dropCondition)


##setting training/testing sets
set.seed(123)
elig.pop$u <- runif(n=dim(elig.pop)[1],min=0,max=1);

# Define these two variables for later use;
elig.pop$QualityIndex <- elig.pop$OverallQual*elig.pop$OverallCond;
elig.pop$TotalSqftCalc <- elig.pop$BsmtFinSF1+elig.pop$BsmtFinSF2+elig.pop$GrLivArea;
elig.pop$HouseAge <- elig.pop$YrSold - elig.pop$YearBuilt

# Create train/test split;
train.df <- subset(elig.pop, u<0.70);
test.df <- subset(elig.pop, u>=0.70);

# Checking the data split. The sum of the parts should equal the whole.
total_ames_count = dim(elig.pop)[1]
total_training = dim(train.df)[1]
total_testing = dim(test.df)[1]
testing_and_training = dim(train.df)[1]+dim(test.df)[1]

print(paste("Total count of eligible population is ", total_ames_count))
print(paste("Total count of eligible population is ", testing_and_training))

##remove unecessary fields and create a subset for further predcition

train.clean<- train.df %>%
         dplyr::select(SalePrice,SaleCondition,SaleType, HouseAge, TotalSqftCalc,
         QualityIndex,GrLivArea,FirstFlrSF,SecondFlrSF,
         TotalBsmtSF,GarageArea,YrSold,GarageYrBlt,GarageArea,
         Zoning,LotFrontage,LotArea,
         LotShape,Neighborhood,BldgType,HouseStyle,
         OverallCond,YearBuilt,YearRemodel)

train.clean <-train.df[,!(names(elig.pop) %in% drop.list)]

library(corrplot)
# Correlation only makes sense for numerical features. 
co <- cor(train.clean[sapply(train.clean, function(x) is.integer(x))], use = "complete.obs") 
corrplot(co)




