# ***BNP Paribas Insurance***
# ***Venkata Jagannath***

# Clears the environment
rm(list = ls())

# All required packages are loaded
library(caret)
library(xgboost)
library(randomForest)
library(nnet)
library(data.table)

# The .csv files containing our data are read and stored in a data frame
Mainfile <- fread("C:/Users/Venkata Jagannath/Desktop/R-Scripts/BNP Paribas/train.csv/train.csv",stringsAsFactors = TRUE,data.table=FALSE)
Testfile <- fread("C:/Users/Venkata Jagannath/Desktop/R-Scripts/BNP Paribas/test.csv/test.csv",stringsAsFactors = TRUE,data.table=FALSE)

# Separate the 'target' and 'Id' variables

target <- Mainfile[,c(1,2)]
Mainfile <- Mainfile[,-c(1,2)]

# Pre-processing

dataprep <- preProcess(Mainfile,method=c("knnImpute"))
Main_file_transformed <- predict(dataprep,Mainfile)
Main_file <- cbind(target,Main_file_transformed)

# Data is partitioned in a 70-30 split
set.seed(123)
sample <- createDataPartition(Main_file$target,p=0.7,list = FALSE)
train <- Main_file[sample,]
test <- Main_file[-sample,]
rm(sample)


# An XGBoost model is built to our data with several tuning parameters

train$target <- factor(train$target)
levels(train$target)[levels(train$target)==1] <- "Class_1"
levels(train$target)[levels(train$target)==0] <- "Class_0"

# set up the cross-validated hyper-parameter search
xgb_grid = expand.grid(nrounds = 80,
                       eta = c(0.01, 0.001, 0.0001),
                       max_depth = c(6, 8, 10),
                       gamma = 1,
                       colsample_bytree=c(0.6),
                       min_child_weight=c(0.8))

# Specify the training control parameters
xgb_trcontrol = trainControl(method = "cv",
                             number = 3,verboseIter = TRUE,
                             returnData = FALSE,returnResamp = "all",            # save losses across all models
                             classProbs = TRUE,summaryFunction = twoClassSummary,# set to TRUE for AUC to be computed
                             allowParallel = TRUE,
                             sampling="down")

# Training the XGBoost model for each tuning parameter combination in the grid, using CV to evaluate

xgb_train = train(train[,-c(1,2)],train[,2],
                    trControl = xgb_trcontrol,
                    tuneGrid = xgb_grid,
                    method = "xgbTree",
                    metric="auc")


xgb_predict <- predict(xgb_train,test)

# A K-means model is built
attach(train)

levels(Test_file$v113) <- levels(train$v113)

train$target <- factor(train$target)

fit <- knn3(target~., train, k=20)

knn_predict <- predict(fit,Test_file,type="prob")

knn_predict <- knn_predict[,2]

knn_predict <- data.frame(knn_predict)



# Random Forest

rf_train <- train(target~.,train,
                  method='rf',
                  ntree=100,
                  maxnodes=1000,
                  tuneLength=5,
                  preProcess=c("center","scale"),
                  trControl=trainControl(sampling="down"))


rf_predict <- predict(rf_train,newdata=Test_file,type="prob")
rf_predict <- rf_predict[,2]
rf_predict <- data.frame(rf_predict)

# Neural Network in caret

my_grid <- expand.grid(.decay = 0.5, .size = 6)
Neural_train <- train(target ~ ., data = train,
                      method = "nnet", maxit = 1000, tuneGrid = my_grid, trace = F,preProcess=c("center","scale"),
                      trControl=trainControl(sampling="up"))


Neural_predict <- predict(Neural_train, newdata = Test_file,type="prob")
Neural_predict <- Neural_predict[,2]

# The class is converted to data frame
Test_file <- data.frame(Test_file)

# The model is fitted on our new data and prediction probabilities are computed
Submission_knn <- data.frame("PredictedProb" = knn_predict)

# The class is converted to data matrix
Test_file <- data.matrix(Test_file)

# The model is fitted on our new data and prediction probabilities are computed
Submission_xgb <- data.frame(PredictedProb = predict(xgb_train,Test_file,type="prob")[,2])

Test_file <- data.frame(Test_file)

# Assigning weights to the different predictions

Calc <- data.frame(((1*Submission_knn$knn_predict+2*rf_predict$rf_predict+0*Submission_xgb$PredictedProb+1*Neural_predict)/4))
colnames(Calc) <- c("PredictedProb")

Calc <- data.frame(((1*Submission_xgb$PredictedProb)/1))
colnames(Calc) <- c("PredictedProb")

# The ID field from our Test.csv file is stored in the Submission data frame
Submission <- data.frame(Test_file["ID"])

Submission["PredictedProb"] <- data.frame(Calc$PredictedProb)

# The file is written to a new .csv file to be stored in our working directory.
write.csv(Submission,file="BNP_Submission.csv",row.names = FALSE)
head(Submission)

