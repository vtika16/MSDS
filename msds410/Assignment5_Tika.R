library(MASS)
library(car)

ames.df = read.csv("C:/Users/vtika/Downloads/assignment5/ames_housing_data.csv", header = TRUE,stringsAsFactors = TRUE)

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
    ifelse(ames.df$GrLivArea >4000,'07: GT 4000 SqFt',
    '99: Eligible Sample')
    ))))))

table(ames.df$dropCondition)

# Eliminate all observations that are not part of the eligible sample population;
elig.pop <- subset(ames.df,dropCondition=='99: Eligible Sample');

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
print(paste("Total count of training population is ", total_training))
print(paste("Total count of testing population is ", total_testing))



#First model: first tentative
drop.list <- c('SalePrice','GarageCond','SaleType', 'HouseAge', 'TotalSqftCalc','QualityIndex','FirstFlrSF','SecondFlrSF','TotalBsmtSF','GarageArea','GarageYrBlt','GarageArea','Zoning','FullBath','LotArea','LotShape','Neighborhood','Exterior1','HouseStyle','OverallCond','YearRemodel');

#Second model: Using those with higher rate in the upper model
drop.list <- c('SalePrice','LotArea','Neighborhood','HouseStyle1','OverallCond','Exterior1','TotalBsmtSF','FirstFlrSF','SecondFlrSF','FullBath','GarageYrBlt','GarageArea','QualityIndex','TotalSqftCalc','HouseAge','LotShape','YearRemodel')

#Third model: removing Neighborhood as it givesa VIF value equal to 20 and replacing OverallCond with HeatingQC
drop.list <- c('SalePrice','LotArea','HouseStyle','HeatingQC','Exterior1','TotalBsmtSF','FirstFlrSF','SecondFlrSF','FullBath','GarageYrBlt','GarageArea','QualityIndex','TotalSqftCalc','HouseAge','LotShape','YearRemodel')


train.clean <-train.df[,(names(elig.pop) %in% drop.list)];
#have to clean the NAs in the train clean to avoid the stepIAC function crashes
train.clean <- na.omit(train.clean)

# Define the upper model as the FULL model
upper.lm <- lm(SalePrice ~ .,data=train.clean);
summary(upper.lm)
# Define the lower model as the Intercept model
lower.lm <- lm(SalePrice ~ 1,data=train.clean);
# Need a SLR to initialize stepwise selection
sqft.lm <- lm(SalePrice ~ TotalSqftCalc,data=train.clean);
summary(sqft.lm)

# Note: There is only one function for classical model selection in R - stepAIC();
# stepAIC() is part of the MASS library.
# The MASS library comes with the BASE R distribution, but you still need to load it;
# Done at the beginning of this script
# Call stepAIC() for variable selection
forward.lm <- stepAIC(object=lower.lm,scope=list(upper=formula(upper.lm),lower=~1),direction=c('forward'));
summary(forward.lm)
forward.lm
backward.lm <- stepAIC(object=upper.lm,direction=c('backward'));
summary(backward.lm)

stepwise.lm <- stepAIC(object=sqft.lm,scope=list(upper=formula(upper.lm),lower=~1),direction=c('both'));
summary(stepwise.lm)

#here do not need to clean NAs
junk.lm <- lm(SalePrice ~ OverallQual + OverallCond + QualityIndex + GrLivArea + TotalSqftCalc, data=train.df)
summary(junk.lm)


sort(vif(forward.lm),decreasing=TRUE)
sort(vif(backward.lm),decreasing=TRUE)
sort(vif(stepwise.lm),decreasing=TRUE)
vif(forward.lm)
vif(backward.lm)
vif(stepwise.lm)
vif(junk.lm)

#R-sqare obtained from summary(model).

#AIC and BIC: the lower the better
AIC(forward.lm)
AIC(junk.lm)

#BIC
BIC(forward.lm)
BIC(junk.lm)

#mean squared error
mean((train.clean$SalePrice - predict(forward.lm))^2)
mean((train.df$SalePrice - predict(junk.lm))^2)

#mean absolute error
mean(abs(train.clean$SalePrice - predict(forward.lm)))
mean(abs(train.df$SalePrice - predict(junk.lm)))

#the model junk includes additional variables that were removed in the model
drop.list <- c(drop.list,"OverallQual","OverallCond","GrLivArea")

#remove not used columns following the models drop.llist
test.clean <-test.df[,(names(elig.pop) %in% drop.list)];
#have to clean the NAs in the test clean as well to avoid the predict function to complain
test.clean <- na.omit(test.clean)


#predict complains that it Exterior1 has a value not considered in the model. That is cos the train data selected did not contain that type 'ImStucc'. Therefore remove it.
length(test.clean[test.clean$Exterior1=='ImStucc',]$Exterior1)
#1 record to remove
test.clean<-test.clean[test.clean$Exterior1!='ImStucc',]

forward.test <- predict(forward.lm,newdata=test.clean);
junk.test <- predict(junk.lm,newdata=test.clean);

#mean squared error
mean((test.clean$SalePrice - forward.test)^2)
mean((test.clean$SalePrice - junk.test)^2)

#mean absolute error
mean(abs(test.clean$SalePrice - forward.test))
mean(abs(test.clean$SalePrice - junk.test))


#Training Data
# Abs Pct Error
forward.pct <- abs(forward.lm$residuals)/train.clean$SalePrice;
junk.pct <- abs(junk.lm$residuals)/train.df$SalePrice;
# Assign Prediction Grades;
forward.PredictionGrade <- ifelse(forward.pct<=0.10,'Grade 1: [0.0.10]', ifelse(forward.pct<=0.15,'Grade 2: (0.10,0.15]', ifelse(forward.pct<=0.25,'Grade 3: (0.15,0.25]','Grade 4: (0.25+]')))
forward.trainTable <- table(forward.PredictionGrade)
forward.trainTable/sum(forward.trainTable)

junk.PredictionGrade <- ifelse(junk.pct<=0.10,'Grade 1: [0.0.10]', ifelse(junk.pct<=0.15,'Grade 2: (0.10,0.15]', ifelse(junk.pct<=0.25,'Grade 3: (0.15,0.25]','Grade 4: (0.25+]')))
junk.trainTable <- table(junk.PredictionGrade)
junk.trainTable/sum(junk.trainTable)

##find mean of grade actuals vs grade predicted


# Test Data
# Abs Pct Error
forward.testPCT <- abs(test.clean$SalePrice-forward.test)/test.clean$SalePrice;
#Comment both lines as the model is the same for forward, backward and stepwise
#backward.testPCT <- abs(test.df$SalePrice-backward.test)/test.df$SalePrice;
#stepwise.testPCT <- abs(test.df$SalePrice-stepwise.test)/test.df$SalePrice;
junk.testPCT <- abs(test.clean$SalePrice-junk.test)/test.clean$SalePrice;

# Assign Prediction Grades;
forward.testPredictionGrade <- ifelse(forward.testPCT<=0.10,'Grade 1: [0.0.10]', ifelse(forward.testPCT<=0.15,'Grade 2: (0.10,0.15]', ifelse(forward.testPCT<=0.25,'Grade 3: (0.15,0.25]', 'Grade 4: (0.25+]')))
forward.testTable <-table(forward.testPredictionGrade)
forward.testTable/sum(forward.testTable)


junk.testPredictionGrade <- ifelse(junk.testPCT<=0.10,'Grade 1: [0.0.10]', ifelse(junk.testPCT<=0.15,'Grade 2: (0.10,0.15]', ifelse(junk.testPCT<=0.25,'Grade 3: (0.15,0.25]', 'Grade 4: (0.25+]')))
junk.testTable <-table(junk.testPredictionGrade)
junk.testTable/sum(junk.testTable)