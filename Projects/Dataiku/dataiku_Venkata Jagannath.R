#Clear the environment
rm(list=ls())

# Load the required libraries
library(data.table)
library(caret)
library(ggplot2)

# Read the file to the environment
train <- fread("C:/Users/Venkata Jagannath/Desktop/dataiku/us_census_full/us_census_full/census_income_learn.csv",data.table = FALSE)

# ****************************************************************************************
# Explore the data
# ****************************************************************************************
# Age
summary(Age)

# Kernel density plot
qplot(Age, data=train, geom="density", fill=Sex, alpha=I(.5),
      main="Distribution of Age", xlab="Age",ylab="Density")  #No statistical difference

plot(train$Age,train$`Wage/hr`,ylab = "Wage/Hr", xlab = "Age",
     pch = 19, col = rgb(0, 0, 0, 0.05),main = "Age against Wage/Hr")

plot(train$Age,train$`weeks worked in year`,ylab = "Weeks worked in a year", xlab = "Age",
     pch = 19, col = rgb(0, 0, 0, 0.05),main = "Age against Weeks worked")

# Age division by occupation

qplot(Age, data=train, geom="density", fill=`Occupation code`, alpha=I(.5),
      main="Age against Occupation code", xlab="Age",ylab="Density")

# Education

summary(Education)


qplot(Age,data=train,geom = "auto",fill=Education,alpha=I(.5),binwidth = 1,main = "Age by Education")

qplot(train$`weeks worked in year`,fill=Education,alpha=I(.5),binwidth = 1,main = "Weeks worked by Education",data = train,xlab="Weeks worked/year")


# Profiles of people who earn greater than 50000
train_ <- train[train$Income=="50000+.",]

# Kernel density plot
qplot(Age, data=train_, geom="density", fill=Sex, alpha=I(.5),
      main="Distribution of Age", xlab="Age",ylab="Density")  #No statistical difference

plot(train_$Age,train_$`Wage/hr`,ylab = "Wage/Hr", xlab = "Age",
     pch = 19, col = rgb(0, 0, 0, 0.05),main = "Age against Wage/Hr")

plot(train_$Age,train_$`weeks worked in year`,ylab = "Weeks worked in a year", xlab = "Age",
     pch = 19, col = rgb(0, 0, 0, 0.05),main = "Age against Weeks worked")

# Age division by occupation

qplot(Age, data=train_, geom="density", fill=`Occupation code`, alpha=I(.5),
      main="Distribution of Age", xlab="Age",ylab="Density")
# *************************************************************************************
# *************************************************************************************

# Feature Engineering

 # First summarize every feature of the dataset and make changes if necessary
lapply(train,unique)


# Engineering feature to group advanced degrees

train["Edu_"]<- ifelse(train$Education=="Bachelors degree(BA AB BS)"|train$Education=="Masters degree(MA MS MEng MEd MSW MBA)"|
                         train$Education=="Prof school degree (MD DDS DVM LLB JD)"|train$Education=="Doctorate degree(PhD EdD)"|train$Education=="Associates degree-academic program",1,0)


# dummies <- dummyVars(Income~Sex+`Industry code`,data=train)
# train<- cbind(train,predict(dummies,train))


#Change columns types
changetype<-function(data)
{
  ix <- c("Class of worker","Industry recode","Occupation recode","Education","Edu enroll","Marital status","Industry code","Occupation code","Race","Hispanic?","Sex","Labour union?",
          "Unemployment reason?","Full/part time?","Tax filer status","Prev res region","Prev res state","detailed household and family stat","detailed household summary in household",
          "migration code-change in msa","migration code-change in reg","migration code-move within reg","live in this house 1 year ago",
          "migration prev res in sunbelt","family members under 18","country of birth father","country of birth mother","country of birth self","citizenship","own business or self employed",
          "fill inc questionnaire for veteran's admin","veterans benefits","year","Income")
  data[ix]<-lapply(data[ix],as.factor)
  
  # Transforming categorical variables into levels
  
  feature.names <- names(data)[1:(ncol(data))]
  
  for (f in feature.names)
  {
    if(class(data[[f]])=="factor")
    {
      levels <- unique(data[[f]])
      data[[f]] <- as.integer(factor(data[[f]], levels=levels))
    }
  }
  return(data)
}
train <- changetype(train)

# Missing values?
sum(is.na(train)) #874

# Median value imputation
preproc <- preProcess(train,method = c("medianImpute","zv")) 
train <- predict(preproc,train)


# Find correlated variable and remove them

trainCorr <- findCorrelation(cor(train), cutoff = .85)
train <- train[,-trainCorr]

# Outliers are removed
train <- train[train$`Capital gains`<50000,]

train <- train[train$`Wage/hr`<2500,]

train <- train[train$`Capital losses`<3000,]

train <- train[train$`Dividends from stocks`<75000,]


# Changing labels
train[train$Income==1,"Income"] <- "C1"
train[train$Income==2,"Income"] <- "C2"

# Data is partitioned in a 30-70 split
set.seed(123)
sample <- createDataPartition(train$Income,p=0.3,list = FALSE)
validation <- train[-sample,]
train <- train[sample,]

# Store target
y <- factor(train$Income)
train$Income <- NULL


# Feature selection - Recursive feature elimination

normalize <- preProcess(train)
x <- data.frame(predict(normalize,train))
subsets <- c(1:5,10,15,20,25,30)



ctrl <- rfeControl(functions = lrFuncs,
                   method = "repeatedcv",
                   repeats = 4,
                   verbose = TRUE)
set.seed(20)

lrrfe<- rfe(x,y,sizes = subsets,rfeControl = ctrl)
lrrfe

# Train an simple model

# lm_train <- glm(y~.,data=train,family = "binomial")
# lm_predict <- predict(lm_train,validation)
# lm_predict <- ifelse(lm_predict>0.50,"C2","C1")
# confusionMatrix(lm_predict,validation$Income)
# lmImp <- varImp(lm_train,scale=FALSE)
# plot(lmImp,main="Top 20 important variables (LR) ")


# Train a random forest model
set.seed(111)
rf_train <- train(y ~ .,train,method="rf",metric = "Kappa",
                  trControl = trainControl(method = "repeatedcv",number = 5,repeats=2,
                                           summaryFunction = twoClassSummary,classProbs = TRUE,
                                           search = "random",verbose=TRUE,allowParallel = TRUE))

rf_predict <- predict(rf_train,validation)
confusionMatrix(rf_predict,validation$Income)

# Variable Importance from Random Forest
rfImp <- varImp(rf_train,scale=FALSE)
plot(rfImp,top=20,main="Top 20 important variables (RF)")

# Train an XGBoost model using the adaptive resampling technique and cross validation techniques

fitCtrl <- trainControl(method = "repeatedcv",
                        number = 5,repeats = 2,
                        verboseIter = TRUE,returnResamp = "all",            # save losses across all models
                        allowParallel = TRUE,search = "random")


xgb_grid = expand.grid(nrounds = (10:30)*10,
                       eta = seq(0.05,0.1,by=0.02),
                       max_depth = c(6:10),
                       gamma = 1,
                       colsample_bytree=c(seq(0.1,1,by=0.15)),
                       min_child_weight=c(5:10))
set.seed(111)
xgb_train <- train(y~.,train,
                   trControl = fitCtrl,
                   method = "xgbTree",
                   importance = TRUE,
                   objective = "binary:logistic",
                   eval_metric = "logloss")

xgb_predict <- predict(xgb_train,validation)
confusionMatrix(xgb_predict,validation$Income)

# Predicting the test set
# ********************************

# Read the test set
test <- fread("C:/Users/Venkata Jagannath/Desktop/dataiku/us_census_full/us_census_full/census_income_test.csv",data.table = FALSE)

# Completing the feature engineering step
test["Edu_"]<- ifelse(test$Education=="Bachelors degree(BA AB BS)"|test$Education=="Masters degree(MA MS MEng MEd MSW MBA)"|
                         test$Education=="Prof school degree (MD DDS DVM LLB JD)"|test$Education=="Doctorate degree(PhD EdD)"|test$Education=="Associates degree-academic program",1,0)

# Changing data types
test <- changetype(test)

#Making predictions
rf_test <- predict(rf_train,test,type="prob")
xgb_test <- predict(xgb_train,test,type="prob")

pred <- data.frame((rf_test+xgb_test)/2)

pred["Prediction"] <- ifelse(pred[,1]>0.5,"-50000"," 50000+.")

write.csv(pred,"C:/Users/Venkata Jagannath/Desktop/dataiku/us_census_full/us_census_full/pred_test.csv")


