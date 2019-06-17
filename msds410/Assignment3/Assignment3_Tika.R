library("tidyverse", lib.loc="~/R/win-library/3.5")
library("ggplot2", lib.loc="~/R/win-library/3.5")
library("stargazer", lib.loc="~/R/win-library/3.5")
library("PerformanceAnalytics", lib.loc="~/R/win-library/3.5")
library(corrplot, lib.loc="~/R/win-library/3.5")

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

##Select appopriate fields from eligible population
elig.pop <- elig.pop %>%
  select(SalePrice,SaleCondition,SaleType,
         GrLivArea,FirstFlrSF,SecondFlrSF,
         TotalBsmtSF,GarageArea,YrSold,
         MoSold,GarageType,GarageYrBlt,
         GarageArea,GarageCond,SubClass,
         Zoning,LotFrontage,LotArea,
         LotShape,Utilities,Neighborhood,
         BldgType,HouseStyle,OverallQual,
         OverallCond,YearBuilt,YearRemodel)


##create the elig.po as house dataset

house <- elig.pop
#View(house)

##### Question 2 #####

##perform EDA on two variables as potential predictor variables

##creating variables as well to determine a few things
library(corrplot)
# Correlation only makes sense for numerical features. 
co <- cor(house[sapply(house, function(x) is.integer(x))], use = "complete.obs") 
corrplot(co)

# looking at corrplot sale price seem to has highest correlation with GrLivArea and GarageArea

# We take these two variables to fit a regression model 

fit_1 <- lm(house$SalePrice ~ house$GrLivArea)
fit_2 <- lm(house$SalePrice ~ house$GarageArea)

summary(fit_1)
# Adjusted R-2 squared value of 0.639 
summary(fit_2)
# Adjusted R-2 squared value of 0.4348
# First model is a better fit

plot(house$GrLivArea, house$SalePrice, xlab="General Living Area", ylab = "Sale Price", main = "General Living Area to Sales Price")
abline(fit_1, col="red")


plot(house$GarageArea, house$SalePrice, xlab="Garage Area", ylab="Sale Price", main = "Garage Area to Sales Price")
abline(fit_2, col="red")

# Looking at residual plots for both fits 

plot(fit_1$residuals, ylab = "Residuals", main = "Residuals of General Living Area")
plot(fit_2$residuals, ylab= " Residuals", main = "Residuals of Garage Area")

par(mfrow=c(2,2))
plot(fit_1)

par(mfrow=c(2,2), main = "Residuals versus Fitted for Garage Area")
plot(fit_2)

# Both residual seem to have zero mean and about constant variance. 
# We have a closer look at the quantile plots 

par(mfrow=c(1,2))
qqnorm(fit_1$residuals)
qqline(fit_1$residuals)
qqnorm(fit_2$residuals)
qqline(fit_2$residuals)

# In the first fit the residuals lie more closely on the line so it is more closer to a normal distribution. 
# Coefficient estimates, p-values and t-values etc are listed in the summary tables 
summary(fit_1)
summary(fit_2)

##### ##### 

##### Question 3 ##### 

fit_m <- lm(house$SalePrice~house$GrLivArea+house$GarageArea)
summary(fit_m)
# The AdjustedR squared value has increased to 0.7215 which is greater than the individual models 
# Thus our model has improved. 
# But this may not always be the case 
summary(fit_m$residuals)

plot(fit_m$residuals, ylab = "Residuals", main = "Residuals of Multi-Linear Regression")

par(mfrow=c(1,1))
plot(fit_m$residuals, ylab="Residuals")
qqnorm(fit_m$residuals)
qqline(fit_m$residuals, col = "darkred")

# The residuals fit the normal more closely

##### ##### 

##### Question 4 ##### 

# Making new columns for residuals 
house$resid <- fit_m$residuals
table(house$Neighborhood)
boxplot(house$resid ~ house$Neighborhood, horizontal=T, main = "Boxplot of Residuals from MLR")

# Residual is observed value - value from fit 
# Thus is rediuals >> 0 means the property is underpriced by our model 
# If the residuals is << 0 means the property is overpriced by our model 

house$X <- house$SalePrice/house$GrLivArea
y <- house %>% 
  group_by(Neighborhood) %>%
  summarize(Y = mean(abs(resid)))
x <- house %>%
  group_by(Neighborhood) %>%
  summarize(x=mean(X))
plot(x$x,y$Y, xlab = "Price/Sqft", ylab = "MAE", main = "Scatterplot of MAE against Price/SqFt", col = "darkgreen")
# The MAE is higher for either low price/sqft or high price/sqft 
# For the average range it is almost the same. 

# Creating new categorical variable to account for change in price/sqft 
house$indi <- 0
house$indi[house$X < 100] <- 1
house$indi[house$X > 100 & house$X < 120] <- 2
house$indi[house$X > 120 & house$X < 140] <- 3
house$indi[house$X > 140 & house$X < 160] <- 4
house$indi[house$X > 160] <- 5
unique(house$indi)

house$indi <- as.factor(house$indi)

fit_new <- lm(house$SalePrice~house$GrLivArea+house$GarageArea+house$indi)
summary(fit_new)
# Adjusted R squared goes to 0.932! 

# Comparing MAEs 
mean(abs(fit_m$residuals))
mean(abs(fit_new$residuals)) # new model has lower MAE 

##### ##### 

##### Question 5 ##### 

# We chose any four continous and any 1 discrete variable
fit1 <- lm(house$SalePrice~house$LotFrontage+house$LotArea+house$YearBuilt+house$TotalBsmtSF+house$GrLivArea)
fit2 <- lm(log(house$SalePrice)~house$LotFrontage+house$LotArea+house$YearBuilt+house$TotalBsmtSF+house$GrLivArea)

par(mfrow=c(1,1))
plot(fit1$residuals, main = "Residuals of Sales Price MLR")
par(mfrow=c(1,1))
plot(fit2$residuals,main = "Residuals of Log(Sales Price) MLR")
# After the log transformation the variance in the error term reduces 
# This transformation might be useful when there is heteroskedasticity in the model 
# In our case it is not the case, but ususally it can help 


summary(fit1)
summary(fit2) # R squared value of second model is better 

par(mfrow=c(2,2))
hist(house$LotFrontage, col = "darkblue", main = "Histogram of Lot Frontage")
hist(house$LotArea, col = "darkgreen", main = "Histogram of Lot Area")
hist(house$GrLivArea, col = "darkred", main = "Histogram of Gen living Area")
hist(house$TotalBsmtSF, col = "pink", main = "Histogram of Total Bsmt SqFt")

# Lets Transform LotArea 

fit3 <- lm(log(house$SalePrice)~house$LotFrontage+log(house$LotArea)+house$YearBuilt+house$TotalBsmtSF+house$GrLivArea)
summary(fit3)
# R square value improved slightly!