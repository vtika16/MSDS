# Chad R Bhatti
# 12.25.2016
# define_sample_population.R

# Read in csv file for Ames housing data;

# Note that back slash is an escape character in R so we use \\ when we want \;
path.name <- 'C:\\Users\\Chad R Bhatti\\Dropbox\\Northwestern_MSPA\\Predict_410_R\\Data\\';
file.name <- paste(path.name,'ames_housing_data.csv',sep='');

# Read in the csv file into an R data frame;
ames.df <- read.csv(file.name,header=TRUE,stringsAsFactors=FALSE);


# Create a waterfall of drop conditions;
# Work the data frame as a 'table' like you would in SAS or SQL;
ames.df$dropCondition <- ifelse(ames.df$BldgType!='1Fam','01: Not SFR',
	ifelse(ames.df$SaleCondition!='Normal','02: Non-Normal Sale',
	ifelse(ames.df$Street!='Pave','03: Street Not Paved',
	ifelse(ames.df$YearBuilt <1950,'04: Built Pre-1950',
	ifelse(ames.df$TotalBsmtSF <1,'05: No Basement',
	ifelse(ames.df$GrLivArea <800,'06: LT 800 SqFt',
	'99: Eligible Sample')
	)))));

# Save the table
waterfall <- table(ames.df$dropCondition);

# Format the table as a column matrix for presentation;
as.matrix(waterfall,7,1)


# Eliminate all observations that are not part of the eligible sample population;
eligible.population <- subset(ames.df,dropCondition=='99: Eligible Sample');

# Check that all remaining observations are eligible;
table(eligible.population$dropCondition);


##########################################################################################
# Create a list of interesting predictor variables
##########################################################################################

str(ames.df)

# Predictor variables that I like:
LotFrontage
LotArea
LotConfig
Neighborhood
HouseStyle
OverallQual
OverallCond
YearBuilt
YearRemodel
Exterior1
BsmtFinSF1
BsmtFinSF2
CentralAir
GrLivArea
BsmtFullBath
BsmtHalfBath
FullBath
HalfBath
BedroomAbvGr
TotRmsAbvGrd
Fireplaces
GarageCars
GarageArea
WoodDeckSF
OpenPorchSF
EnclosedPorch
ThreeSsnPorch
ScreenPorch
PoolArea
MoSold
YrSold
SaleCondition
SalePrice


# How do I restrict my data frame to just these columns that interest me?
# Make a vector of the names;

keep.vars <- c('SID','PID','LotFrontage','LotArea','LotConfig','Neighborhood',
'HouseStyle','OverallQual','OverallCond','YearBuilt','YearRemodel','Exterior1',
'BsmtFinSF1','BsmtFinSF2','CentralAir','GrLivArea','BsmtFullBath','BsmtHalfBath',
'FullBath','HalfBath','BedroomAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars',
'GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','ThreeSsnPorch',
'ScreenPorch','PoolArea','MoSold','YrSold','SaleCondition','SalePrice'
);


# Note that the R data frame is a (rectabgular) list object which means that it can be 
# accessed in two ways - as a matrix or as a list;
# Note that the keep.vars are the COLUMNS that we want to keep, not the rows;

skinny.df <- eligible.population[,keep.vars];

# Use the structure command to view the contents of the data frame;
str(skinny.df)




##########################################################################################
# Delete observations with missing values 
##########################################################################################
sample.df <- na.omit(skinny.df);

# Check the change in dimension;
dim(skinny.df)
dim(sample.df)

dim(skinny.df)-dim(sample.df)



##########################################################################################
# Define some discrete variables and indicator variables 
##########################################################################################

# Define total square footage
sample.df$TotalSqftCalc <- sample.df$BsmtFinSF1+sample.df$BsmtFinSF2+sample.df$GrLivArea;

# Define total bathrooms
sample.df$TotalBathCalc <- sample.df$BsmtFullBath + 0.5*sample.df$BsmtHalfBath ++
	+ sample.df$FullBath + 0.5*sample.df$HalfBath;


# Corner lot indicator
# ifelse(condition,valueTrue,valueFalse)
sample.df$CornerLotInd <- ifelse(sample.df$LotConfig=='Corner',1,0);

# Check how the indicator is assigned
table(sample.df$CornerLotInd,sample.df$LotConfig)

# Define two indicators for fire places
table(sample.df$Fireplaces)

# Intercept Adjustment for a single fireplace
sample.df$FireplaceInd1 <- ifelse((sample.df$Fireplaces>0)&(sample.df$Fireplaces<2),1,0);
table(sample.df$FireplaceInd1,sample.df$Fireplaces)

# Intercept Adjustment for 2 or more fireplaces
sample.df$FireplaceInd2 <- ifelse((sample.df$Fireplaces>1),1,0);
table(sample.df$FireplaceInd2,sample.df$Fireplaces)

# Additive Intercept Adjustment for a single fireplace
sample.df$FireplaceAdder1 <- ifelse((sample.df$Fireplaces>0),1,0);

# Additive Intercept Adjustment for 2 or more fireplaces
sample.df$FireplaceAdder2 <- ifelse((sample.df$Fireplaces>1),1,0);

table(sample.df$FireplaceAdder1,sample.df$Fireplaces)
table(sample.df$FireplaceAdder2,sample.df$Fireplaces)



# Central Air Indicator
sample.df$CentralAirInd <- ifelse(sample.df$CentralAir=='Y',1,0);
table(sample.df$CentralAirInd) 
# Looks like this is not useful since almost all homes have central air


# Exterior Siding Type
sample.df$BrickInd <- ifelse(sample.df$Exterior1=='BrkFace',1,0);
sample.df$VinylSidingInd <- ifelse(sample.df$Exterior1=='VinylSd',1,0);

# Pool Indicator
sample.df$PoolInd <- ifelse(sample.df$PoolArea>0,1,0);

# Wood Deck Indicator
sample.df$WoodDeckInd <- ifelse(sample.df$WoodDeckSF>0,1,0);

# Porch Indicator - Open Porch OR Screen Porch
sample.df$PorchInd <- ifelse((sample.df$OpenPorchSF>0)||(sample.df$ScreenPorch>0),1,0);

# Quality Index
sample.df$QualityIndex <- sample.df$OverallQual*sample.df$OverallCond;

table(sample.df$QualityIndex)

 
# Year Sold Indicators
sample.df$I2006 <- ifelse(sample.df$YrSold==2006,1,0);
sample.df$I2007 <- ifelse(sample.df$YrSold==2007,1,0);
sample.df$I2008 <- ifelse(sample.df$YrSold==2008,1,0);
sample.df$I2009 <- ifelse(sample.df$YrSold==2009,1,0);
sample.df$I2010 <- ifelse(sample.df$YrSold==2010,1,0);

table(sample.df$YrSold)
table(sample.df$I2006)
table(sample.df$I2007)
table(sample.df$I2008)
table(sample.df$I2009)
table(sample.df$I2010)

 
# List out sample.df
str(sample.df)



##########################################################################################
# Add a train/test flag to split the sample 
##########################################################################################
sample.df$u <- runif(n=dim(sample.df)[1],min=0,max=1);
sample.df$train <- ifelse(sample.df$u<0.70,1,0);

# Check the counts on the train/test split
table(sample.df$train)

# Check the train/test split as a percentage of whole
table(sample.df$train)/dim(sample.df)[1]



##########################################################################################
# Save data frame as an .RData data object 
##########################################################################################

# Save the R data frame as an .RData object
saveRDS(sample.df,file='C:\\Users\\Chad R Bhatti\\Dropbox\\Northwestern_MSPA\\Predict_410_R\\Data\\ames_sample.RData');

# Read (or reload) the .RData object as an R data frame
a <- readRDS('C:\\Users\\Chad R Bhatti\\Dropbox\\Northwestern_MSPA\\Predict_410_R\\Data\\ames_sample.Rdata');

# Check it
str(a)







##########################################################################################
# Notes:
##########################################################################################

(1) Handling and manipulating data in R requires knowledge of BASE R.  There are some R
packages that can help, like dplyer and tidyr, but still you need to undersstand BASE R.
Using BASE R has more to do with understanding R as a programming languge than using R
as a statistical software.

(2) Save your sample data set as an .RData object for future assignments.  We do not want
to have to rebuild our data set for every assignment.





















