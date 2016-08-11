
setwd("~/Documents/Analytics/Tata Case Study")

library(data.table)
library(lubridate)
library(date)
library(ggplot2)

train = fread("train.csv",drop=c("V1"))

train = train[,":="(HB = ifelse(grepl("HB",train$Type),1,0),
                    FB = ifelse(grepl("FB",train$Type),1,0),
                    Bifocal = ifelse(grepl("Bifocal",train$Type),1,0),
                    Progressive = ifelse(grepl("Progressive",train$Type),1,0),
                    SingleVision = ifelse(grepl("Single Vision",train$Type),1,0),
                    AGE_F = ifelse(AGE_F>100, NA, AGE_F))]


train = train[, ":="(Category = factor(Category),
                   Docdt = as.POSIXct( strptime(Docdt, "%Y-%m-%d")),
                   Type= factor(Type),
                   Sqr_Ft_Area = as.numeric(gsub(",","",Sqr_Ft_Area)),
                   Store_Type = factor(Store_Type),
                   City = factor(City),
                   Store_open_date = as.POSIXct( strptime(Store_open_date, "%d-%b-%y")),
                   AGE_F = AGE_F,
                   Gender_F = factor(tolower(Gender_F)),
                   Gender_Present = ifelse(is.na(Gender_F),0,1),
                   Marital_Status = factor(Marital_Status),
                   rpt_flag = rpt_flag)]

train$Gender_F = ifelse(train$Gender_F=="female",1,0)


train$Year = year(train$Docdt)
train$Month = month(train$Docdt)

test = fread("test.csv")

ggplot(data = train, aes(x = AGE_F))  + geom_bar()


#store profiles
stores = unique(train[,.(Area = Sqr_Ft_Area, City = City, Store_Type = Store_Type, Volume = sum(Qty), HB = mean(HB),
                FB = mean(FB), Bifocal = mean(Bifocal), Progressive = mean(Progressive), SingleVision = mean(SingleVision),
                MeanAge = mean(AGE_F, na.rm=TRUE), totalUniqueCustomers= length(unique(Cust_id)),
                EstabDate = Store_open_date, femalePerc = sum(Gender_F/sum(Gender_Present))  ),.(Store)])

topStores = train[, .(returnRatio = mean(rpt_flag)),.(Store,Year)]
setorder(topStores,-returnRatio)

topStores = merge(topStores, stores, by = c("Store"))
setorder(topStores,-returnRatio)

write.csv(topStores, "topStoresData.csv")

###############

storesYear = unique(train[,.(HB = mean(HB),FB = mean(FB), Bifocal = mean(Bifocal),
                          Progressive = mean(Progressive), SingleVision = mean(SingleVision),
                          uniqueCustomers =  length(unique(Cust_id)), returnRatio = mean(rpt_flag)), .(Store,Year)])
setkey(storesYear, "Store","Year")
write.csv(storesYear, "storesYear.csv")

###############

productData = train[,.(HB = sum(HB), FB = sum(FB), Progressive = sum(Progressive),
                       Bifocal = sum(Bifocal), SingleVision = sum(SingleVision) ),.(Month,Year)]
setkey(productData,"Year", "Month")
productData$Order = seq(1,36)

ggplot(data = productData) + geom_line( aes(x = Order, y  = HB)) + geom_line( aes(x = Order, y  = FB),colour = "green") +
  geom_line( aes(x = Order, y  = Progressive),colour = "blue") + geom_line( aes(x = Order, y  = SingleVision),colour = "red") +
  geom_line( aes(x = Order, y  = Bifocal),colour = "yellow") + scale_x_continuous(breaks=seq(0,36,2))

write.csv(productData,"productData.csv")


todayDate = as.POSIXct(strptime("01-5-2016","%d-%m-%Y"))
train[is.na(train$Store_open_date),]$Store_open_date = as.POSIXct("2010-07-17")
train$StoreAge = todayDate - train$Store_open_date

categoryData = train[,.(returnRatio = mean(rpt_flag), returnCount = sum(rpt_flag), totalSize = length(rpt_flag)),.(Category)]
ggplot(train, aes(Category)) + geom_bar()
barplot(categoryData$returnRatio,names.arg= categoryData$Category,xlab = "Category",ylab = "returnRatio")


monthData = categoryData = train[,.(returnRatio = mean(rpt_flag), returnCount = sum(rpt_flag), totalSize = length(rpt_flag)),.(Month)]
setkey(monthData,Month)
ggplot(train, aes(Month)) + geom_bar()
barplot(monthData$returnRatio,names.arg= monthData$Category,xlab = "Month of Purchase",ylab = "returnRatio")




library(caTools)
splt = sample.split(train$rpt_flag, SplitRatio = 0.7)
test = train[!splt,]
train = train[splt,]

##############
age = train$AGE_F
area = train$Sqr_Ft_Area
train$Sqr_Ft_Area=NULL
train$AGE_F=NULL
library(randomForest)
model =randomForest(rpt_flag ~HB+FB+Bifocal+Progressive + SingleVision + Year + StoreAge + 
                      Gender_F + Gender_Present + Month +Category + Store_Type+
                      City, data = train)
