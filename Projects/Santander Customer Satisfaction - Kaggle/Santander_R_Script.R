# Clear the environment
rm(list=ls())

# Load the required libraries
library(data.table)
library(caret)
library(randomForest)

train <- fread("~/train.csv/train.csv",stringsAsFactors = FALSE,data.table=F)

y <- data.frame(train[,c(1,ncol(train))])

train <- train[,-c(1,ncol(train))]


prep <- preProcess(train,method=c("nzv","center","medianImpute"))
train <- predict(prep,train)
train <- na.roughfix(train)

rf_train <- train(train[,-1],y$TARGET,method="rf",ntrees=100,verbose=TRUE,
                   objective           = "binary:logistic",
                   eval_metric         = "auc") 

test <- fread("~/test.csv/test.csv",stringsAsFactors = FALSE,data.table=F)
