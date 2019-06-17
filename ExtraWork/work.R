

####################################################################################

##investigating lot area 
str(ames.df$LotArea)


p1 <- ggplot(ames.df, aes(x=LotArea, y=SalePrice)) + geom_point() + xlab("Lot Shapes")


##investigating lot shape
table(ames.df$LotShape)

lot_bar <- ggplot(ames.df, aes(LotShape)) + geom_bar() + xlab("Lot Shapes")


##investigating sale condition
table(ames.df$SaleCondition)

scond <- ggplot(ames.df, aes(SaleCondition)) + geom_bar() + xlab("Lot Shapes")



##investigating sale type
table(ames.df$SaleType)

stype <- ggplot(ames.df, aes(SaleType)) + geom_bar() + xlab("Lot Shapes")


##investigating sales price data
hist(ames.df$SalePrice)
max(ames.df$SalePrice)
