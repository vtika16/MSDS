library("tidyverse", lib.loc="~/R/win-library/3.5")
library("ggplot2", lib.loc="~/R/win-library/3.5")


ames.df = read.csv("c:/Users/vtika/Desktop/R/msds_410/ames_housing_data.csv", header = TRUE,stringsAsFactors = FALSE)

ames.df = data.frame(ames.df)
train <- subset(ames.df, train=1)

##view dataset 
head(ames.df)

str(ames.df)

table(ames.df$Fence, useNA = c('always'))

##create waterfall drop conditions

ames.df$dropCondition <- ifelse(ames.df$BldgType!='1Fam','01: Not SFR',
  ifelse(ames.df$SaleCondition!='Normal','02: Non-Normal Sale',
  ifelse(ames.df$Street!='Pave','03: Street Not Paved',
  ifelse(ames.df$YearBuilt <1950,'04: Built Pre-1950',
  ifelse(ames.df$TotalBsmtSF <1,'05: No Basement',
  ifelse(ames.df$GrLivArea <800,'06: LT 800 SqFt',
  '99: Eligible Sample')
  )))));

table(ames.df$dropCondition)

#creating waterfall table
waterfall <- table(ames.df$dropCondition);

waterfall.matrix <- as.matrix(waterfall,7,1)


# Eliminate all observations that are not part of the eligible sample population;
elig.pop <- subset(ames.df,dropCondition=='99: Eligible Sample');

#Check that all remaining observations are eligible;
table(elig.pop$dropCondition);


elig.pop <- elig.pop %>%
  select(SalePrice,SaleCondition,SaleType,GrLivArea,FirstFlrSF,SecondFlrSF,TotalBsmtSF,GarageArea,YrSold,MoSold,GarageType,GarageYrBlt,GarageArea,GarageCond,SubClass,Zoning,LotFrontage,LotArea,LotShape,Utilities,Neighborhood,BldgType,HouseStyle,OverallQual,OverallCond,YearBuilt,YearRemodel)

##selecting sample dataset and variables for analysis. It was noted in the data dictionary
##that I would recommend removing any houses with more than 4000 square feet from the data set 
##(which eliminates these 5 unusual observations) before assigning it to students. I will
##Be removing those first

elig.pop <- elig.pop[which(elig.pop$GrLivArea<=4000),]


#I want to select the first variable total square footage and then total quality as my second

#My calculation for total square footage of the house is garage,first floor, second floor and total basement sq

elig.pop$totalhousesf <- elig.pop$FirstFlrSF + elig.pop$SecondFlrSF + elig.pop$TotalBsmtSF + elig.pop$GarageArea

elig.pop$totalQualInd <- elig.pop$OverallQual * elig.pop$OverallCond / 2

head(elig.pop$totalhousesf)

##analysis of sales price
par(mfrow = c(1,2))
hist(elig.pop$SalePrice, main = "Histogram of SalePrice", col = "darkblue", xlab = "Sales Price")

qqnorm(elig.pop$SalePrice, col = "Red", main = "Q-Q plot of Sales Price")
qqline(elig.pop$SalePrice, col = "darkgreen")

##lets take the log function of Sales price to evaluate the normality more

elig.pop$logSalePrice<-log10(elig.pop$SalePrice)

par(mfrow = c(1,2))
hist(elig.pop$logSalePrice, main = "Histogram of SalePrice", col = "darkblue", xlab = "Sales Price")

qqnorm(elig.pop$logSalePrice, col = "Red", main = "Q-Q plot of Sales Price")
qqline(elig.pop$logSalePrice, col = "darkgreen")

##It seems as though it is still slightly skewed to the left based on the histogram and QQ analysis



###################  regressions against each predictor seperately
##let's run our regression modelling for total square footage of the house vs sales price

fit <- lm(SalePrice ~ totalhousesf, data = elig.pop)

anova(fit)
summary(fit)

par(mfrow=c(2,2))
plot(fit)

ggplot(elig.pop, aes(x = totalhousesf, y = SalePrice))+geom_point()+geom_smooth(method="lm")+
  ggtitle("Sales Price versus total House SF")

##Next I will want to do the same analysis for the total quality index
fit2 <- lm(SalePrice ~ totalQualInd, data = elig.pop)

anova(fit2)
summary(fit2)

par(mfrow=c(2,2))
plot(fit2)

ggplot(elig.pop, aes(x = totalQualInd, y = SalePrice))+geom_point()+geom_smooth(method="lm")+
  ggtitle("Sales Price versus total quality index")

###################  regressions against both predictors at once
fit3 <- lm(SalePrice ~ totalQualInd + totalhousesf, data = elig.pop)

anova(fit3)
summary(fit3)

par(mfrow=c(2,2))
plot(fit3)



###################  regressions against each predictor seperately against log of Sales price
##let's run our regression modelling for total square footage of the house vs log sales price

fit4 <- lm(logSalePrice ~ totalhousesf, data = elig.pop)

anova(fit4)
summary(fit4)

par(mfrow=c(2,2))
plot(fit4)

ggplot(elig.pop, aes(x = totalhousesf, y = logSalePrice))+geom_point()+geom_smooth(method="lm")+
  ggtitle("Sales Price versus total House SF")

##Next I will want to do the same analysis for the total quality index
fit5 <- lm(logSalePrice ~ totalQualInd, data = elig.pop)

anova(fit5)
summary(fit5)

par(mfrow=c(2,2))
plot(fit5)

ggplot(elig.pop, aes(x = totalQualInd, y = logSalePrice))+geom_point()+geom_smooth(method="lm")+
  ggtitle("Sales Price versus total quality index")



###################  regressions against both predictors at once against log of sales price
fit6 <- lm(logSalePrice ~ totalQualInd + totalhousesf, data = elig.pop)

anova(fit6)
summary(fit6)

par(mfrow=c(2,2))
plot(fit6)
