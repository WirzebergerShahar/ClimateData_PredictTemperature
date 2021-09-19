library(caret)
library(magrittr)
library(data.table)
library(tidyselect)
library(dplyr)
library(imputeTS)
library(rlang)
library(xgboost)
library(dummies)
library(dplyr)
library(ggplot2)
library(JOUSBoost)
library(randomForest)

data1 <- fread(choose.files())
test <- fread(choose.files())



##### Data Summaries ####
summary(data)
#summary(test)


data[,lapply(.SD, class),]

NA_col <- data[,round(colSums(is.na(data))/nrow(data),7),]
NA_col[NA_col>0]


#### Complete Missing Data ####

#### gbrd ####
gbrd_NA <- data %>% group_by(hr)%>% summarize(NA_gbrd = sum(is.na(gbrd))/length(gbrd))
plot(gbrd_NA)
data$gbrd[is.na(data$gbrd)] <- 0

#### prcp ####
prcp_NA <- data1 %>% group_by(mo)%>% summarize(prcp_NA = sum(is.na(prcp))/length(prcp))
plot(prcp_NA)

#prcp- too much NA
data <- subset(x = data,select = -c(prcp))



#### zscore stp,smin/max- remove outliners ####
data$zstp<-(data$stp-mean(data$stp))/sd(data$stp)
data$zmin<-(data$smin-mean(data$smin))/sd(data$smin)
data$zmax<-(data$smax-mean(data$smax))/sd(data$smax)


data$smin[(data$zmin>=3 | data$zmin<=-3)]<-NA
data$smax[(data$zmax>=3 | data$zmax<=-3)]<-NA
data$stp[(data$zstp>=3 | data$zstp<=-3)]<-NA


#### remove rows with missing temp ####
data <- data[!is.na(data$temp),]

#### TS complete ####
stations <- unique(data$wsid)
NAcolNames <- names(NA_col[NA_col>0])

for(i in stations) {
  for(j in NAcolNames) {
       data[wsid==i,j]<-na_interpolation( data[wsid==i,j,with=F],option = 'spline')
  }  }
data[,round(colSums(is.na(data))/nrow(data),7),]

plotNA.distribution(x = data1[wsid==353 & yr==2010  &mo==2 & da==1,wdsp],y = data1[wsid==353 & yr==2010  &mo==2 & da==1,as.numeric(data1$date)] )
plotNA.distribution(y = data[wsid==353 & yr==2010  &mo==2& da==1 ,as.numeric(data1$date)],x = data[wsid==353 & yr==2010  &mo==2& da==1,wdsp])


#### Cor ####
numericCol <- names(data)[sapply(data,class)!="character"]
corrplot::corrplot(cor(data[,numericCol[5:22],with=F]))

tdata <- subset(data,select = -c(wsid,wsnm,inme,city,mdct,date))

RJdata <- tdata[tdata$prov=='RJ',]
RJdata <- subset(RJdata,select = -c(prov))

ESdata <- tdata[tdata$prov=='ES',]
ESdata <- subset(ESdata,select = -c(prov))

MGdata <- tdata[tdata$prov=='MG',]
MGdata <- subset(MGdata,select = -c(prov))

SPdata <- tdata[tdata$prov=='SP',]
SPdata <- subset(SPdata,select = -c(prov))


#### Xgboost ####
bstRJ <- xgb.cv(data = as.matrix(RJdata), label = RJdata$temp,nfold = 5, max.depth = 20, eta = 1, nrounds = 10,
                 nthread = 11, objective = "reg:squarederror") 

ggplot(bstRJ$evaluation_log) +
  geom_line(aes(iter, train_rmse_mean), color = "red") +
  geom_line(aes(iter, test_rmse_mean), color = "blue")+
  ggtitle('RJ model')

bstES <- xgb.cv(data = as.matrix(ESdata), label = ESdata$temp,nfold = 5, max.depth = 20, eta = 1, nrounds = 10,
              nthread = 11, objective = "reg:squarederror")

ggplot(bstES$evaluation_log) +
  geom_line(aes(iter, train_rmse_mean), color = "red") +
  geom_line(aes(iter, test_rmse_mean), color = "blue")+
  ggtitle('ES model')

bstMG <- xgb.cv(data = as.matrix(MGdata), label = MGdata$temp,nfold = 5, max.depth = 20, eta = 1, nrounds = 10,
                nthread = 11, objective = "reg:squarederror")

ggplot(bstMG$evaluation_log) +
  geom_line(aes(iter, train_rmse_mean), color = "red") +
  geom_line(aes(iter, test_rmse_mean), color = "blue")+
  ggtitle('MG model')

bstSP <- xgb.cv(data = as.matrix(SPdata), label = SPdata$temp,nfold = 5, max.depth = 20, eta = 1, nrounds = 10,
                nthread = 11, objective = "reg:squarederror")

ggplot(bstSP$evaluation_log) +
  geom_line(aes(iter, train_rmse_mean), color = "red") +
  geom_line(aes(iter, test_rmse_mean), color = "blue")+
  ggtitle('SP model')

#### Reg ####
MyTrainControl=trainControl(
  method = "repeatedcv",
  number=5,
  repeats=5
)

RJml <- train(temp~.,data=RJdata,method="lm",trControl =MyTrainControl)
mean(RJml$residuals^2)

ESml <- train(temp~.,data=ESdata,method="lm",trControl =MyTrainControl)
mean(ESml$residuals^2)

MGml <- train(temp~.,data=MGdata,method="lm",trControl =MyTrainControl)
mean(MGml$residuals^2)

SPml <- train(temp~.,data=SPdata,method="lm",trControl =MyTrainControl)
mean(SPml$residuals^2)

#### RF ####
ntrees <- seq(20,200,by = 10)
train_error <- numeric(length(ntrees))
test_error <- numeric(length(ntrees))

for (i in 1:length(ntrees)) {
  m_rf_i <- randomForest(temp~., data = RJdata, ntree=ntrees[i], mtry = 10)
  train_error[i] <- mean(predict(m_rf_i) != temp.train$temp)
  test_error[i] <- mean(predict(m_rf_i, temp.test) != temp.test$temp)
}

plot(train_error~ntrees, type = "b", ylim = c(0.04,0.07))
points(test_error~ntrees, col = 2,type = "b")

train_error <- numeric(length(ntrees))
test_error <- numeric(length(ntrees))

for (i in 1:length(ntrees)) {
  m_rf_i <- randomForest(temp~., data = ESdata, ntree=ntrees[i], mtry = 10)
  train_error[i] <- mean(predict(m_rf_i) != temp.train$temp)
  test_error[i] <- mean(predict(m_rf_i, temp.test) != temp.test$temp)
}

plot(train_error~ntrees, type = "b", ylim = c(0.04,0.07))
points(test_error~ntrees, col = 2,type = "b")

train_error <- numeric(length(ntrees))
test_error <- numeric(length(ntrees))

for (i in 1:length(ntrees)) {
  m_rf_i <- randomForest(temp~., data = MGdata, ntree=ntrees[i], mtry = 10)
  train_error[i] <- mean(predict(m_rf_i) != temp.train$temp)
  test_error[i] <- mean(predict(m_rf_i, temp.test) != temp.test$temp)
}

plot(train_error~ntrees, type = "b", ylim = c(0.04,0.07))
points(test_error~ntrees, col = 2,type = "b")

train_error <- numeric(length(ntrees))
test_error <- numeric(length(ntrees))

for (i in 1:length(ntrees)) {
  m_rf_i <- randomForest(temp~., data = SPdata, ntree=ntrees[i], mtry = 10)
  train_error[i] <- mean(predict(m_rf_i) != temp.train$temp)
  test_error[i] <- mean(predict(m_rf_i, temp.test) != temp.test$temp)
}

plot(train_error~ntrees, type = "b", ylim = c(0.04,0.07))
points(test_error~ntrees, col = 2,type = "b")


#### Adaboost ####
n <- sample.int(nrow(data1),nrow(data1)*.7)
train_set <- RJdata[n,]
val_set<- RJdata[-n,]
ada = adaboost(train_set, train_set$temp, tree_depth = 10,
               n_rounds = 100)

yhat_ada = predict(ada, val_set)

mean((RJdata$temp- yhat_ada)^2)


train_set <- ESdata[n,]
val_set<- ESdata[-n,]
ada = adaboost(train_set, train_set$temp, tree_depth = 10,
               n_rounds = 100)

yhat_ada = predict(ada, val_set)

mean((ESdata$temp- yhat_ada)^2)

train_set <- MGdata[n,]
val_set<- MGdata[-n,]
ada = adaboost(train_set, train_set$temp, tree_depth = 10,
               n_rounds = 100)

yhat_ada = predict(ada, val_set)

mean((MGdata$temp- yhat_ada)^2)


train_set <- SPdata[n,]
val_set<- SPdata[-n,]
ada = adaboost(train_set, train_set$temp, tree_depth = 10,
               n_rounds = 100)

yhat_ada = predict(ada, val_set)

mean((SPdata$temp- yhat_ada)^2)

