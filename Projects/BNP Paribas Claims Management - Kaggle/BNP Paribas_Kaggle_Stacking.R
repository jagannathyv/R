# ***BNP Paribas Insurance***
# ***Venkata Jagannath***

#**IN PROGRESS**

# Clears the environment
rm(list = ls())

# All required packages are loaded
library(caret)
library(xgboost)
library(randomForest)
library(nnet)
library(data.table)
library(caretEnsemble)
library(pls)

# The .csv files containing our data are read and stored in a data frame
Mainfile <- fread("C:/Users/Venkata Jagannath/Desktop/R-Scripts/BNP Paribas/train.csv/train.csv",stringsAsFactors = TRUE,data.table=FALSE)
# Testfile <- fread("C:/Users/Venkata Jagannath/Desktop/R-Scripts/BNP Paribas/test.csv/test.csv",stringsAsFactors = TRUE,data.table=FALSE)

sample <- createDataPartition(Mainfile$target,p=0.5,list=FALSE)
train <- Mainfile[sample,]
test <- Mainfile[-sample,]
rm(Mainfile)

# reg2 = pcr(target~.,scale=T,data=train)
# reg3 = pcr(target~.,scale=T,data=train)

# Separate the ID and target variables from train

y <- train[,c(1,2)]
train <- train[,-c(1,2)]
y$target <- factor(y$target)

# The ID field from our Test.csv file is stored in the Submission data frame
Submission <- data.frame(Testfile["ID"])

# Transforming categorical variables into levels
feature.names <- names(train)[1:(ncol(train))]

for (f in feature.names)
{
  if(class(train[[f]])=="factor")
  {
    levels <- unique(train[[f]])
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]] <- as.integer(factor(test[[f]], levels=levels))
  }
}

rm(feature.names,f,levels)

prep <- preProcess(train,method=c("knnImpute","nzv"),k=2)
train <- predict(prep,train)
Testfile <- predict(prep,Testfile[,-1])

y$target <- factor(y$target)
levels(y$target)[levels(y$target)==1] <- "Class_1"
levels(y$target)[levels(y$target)==0] <- "Class_0"


rf_tune <- tuneRF(x=train,y=y$target, mtryStart=1, ntreeTry=500, stepFactor=5, improve=0.05,
trace=TRUE, plot=TRUE, doBest=TRUE)

model_1 <- caretList(x=train,y=y$target,
                     trControl=trainControl(method = "cv",
                                            number = 3,verboseIter = TRUE,
                                            returnData = FALSE,returnResamp = "all",
                                            classProbs = TRUE,summaryFunction = twoClassSummary,
                                            allowParallel = TRUE,sampling="down",
                                            savePredictions="final"),
                     methodList=c("gbm"),
                     tuneGrid= expand.grid(n.trees=1000,
                                           interaction.depth=10,
                                           shrinkage=.1,
                                           n.minobsinnode = 10),
                     distribution="bernoulli")

stacked_model <- caretStack(model_1, method="rpart", tuneLength=2)


Submission["PredictedProb"] <- data.frame(predict(stacked_model,Testfile,type="prob"))

# The file is written to a new .csv file to be stored in our working directory.
write.csv(Submission,file="BNP_Submission.csv",row.names = FALSE)
head(Submission)


