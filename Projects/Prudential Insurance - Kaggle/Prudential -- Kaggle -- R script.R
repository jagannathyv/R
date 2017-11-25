# All variables in the environment are cleared
rm(list = ls())

# Importing all packages necessary for our computations
library(caret)
library(randomForest)

# Reading the training files and the submission file
Prudential_train<-read.csv('Prudential.csv')
Prudential_submission<-read.csv('test.csv')

# Creating a stratified sample of the training data set
set.seed(123)
sample<-createDataPartition(Prudential_train$Response,p = 0.9, list = FALSE)
train<-Prudential_train[sample,]
test<-Prudential_train[-sample,]


# Dividing the most significant variable into bins.
train$Ins_Age <- cut(train$Ins_Age,breaks=8)
test$Ins_Age <- cut(test$Ins_Age,breaks=8)

# Transforming categorical variables into levels
feature.names <- names(train)[2:(ncol(train)-1)]

for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}


# Fitting a Random Forest model using caret

fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10)

rforest <- train(Response~.,train,
                     method='rf',
                     preProc=c('center','scale'),
                     trControl=fitControl,
                     na.action=na.roughfix)

knn_train <- train(Response~.,train,
                 method='knn',
                 preProc=c('center','scale'),
                 trControl=fitControl,
                 na.action=na.roughfix,
                 k=10)

# Predicting the values on the test data set
Prudential_test["Rf_Response"]<- predict(rforest,test)

# checking the misclassification rate
table(test$Rf_Response,test$Response)









