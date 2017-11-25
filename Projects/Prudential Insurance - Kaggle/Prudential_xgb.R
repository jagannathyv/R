#********************************************************
# Name of Project: Prudential Life Assessment - XGBoost model
#        VENKATA JAGANNATH 
#         IN-PROGRESS
#********************************************************

rm(list = ls())

## Loading required packages: xgboost,Metrics
require(xgboost)
require(Metrics)
require(caret)
require(randomForest)

Prudential_train <- read.csv('Prudential.csv')
submission <- read.csv('test.csv')

new_targets<-paste("Prudential_train$Response_",(1:8),"<-0",sep="")
for(item in new_targets)
{
  eval(parse(text=item))
}
for(x in seq(1,8,by=1))
{
  for(y in 1:nrow(Prudential_train))
  {
    if(Prudential_train$Response[y]==x)
    {
      temp<-paste("Prudential_train$Response_",x,"[",y,"]","<-1",sep="")
      eval(parse(text=temp))
    }
  }
}
remove(item,new_targets,x,y,temp)


sample<-createDataPartition(Prudential_train$Response,p=0.7,list = FALSE)
train<-Prudential_train[sample,]
test<-Prudential_train[-sample,]

# # Dividing the most significant variable into bins.
# train$Ins_Age <- cut(train$Ins_Age,breaks = 8)
# test$Ins_Age <- cut(test$Ins_Age,breaks = 8)


# use -1 or -99 for NA's to not get in the data's range
train[is.na(train)] <- (-1)
test[is.na(test)]<- (-1)


feature.names <- names(train)[2:(ncol(train)-9)]

for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

evalerror <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- ScoreQuadraticWeightedKappa(labels,round(preds))
  return(list(metric = "kappa", value = err))}

xgb_nrounds <- 20

xgb_fit_1<- xgb.cv(data    = data.matrix(train[,feature.names]),
                    label       = train$Response_1,
                    nrounds     = 20,
                    objective   = "binary:logistic",
                    feval = evalerror,
                    metrics=list("rmse","auc"),
                    nfold=5,
                    prediction = TRUE)

xgb_fit_2<- xgboost(data    = data.matrix(train[,feature.names]),
                label       = train$Response_2,
                nrounds     = 200,
                objective   = "binary:logistic",
                eval_metric = evalerror)

xgb_fit_3<- xgboost(data    = data.matrix(train[,feature.names]),
                    label       = train$Response_3,
                    nrounds     = 200,
                    objective   = "binary:logistic",
                    eval_metric = evalerror)

xgb_fit_4<- xgboost(data    = data.matrix(train[,feature.names]),
                    label       = train$Response_4,
                    nrounds     = 200,
                    objective   = "binary:logistic",
                    eval_metric = evalerror)

xgb_fit_5<- xgboost(data    = data.matrix(train[,feature.names]),
                    label       = train$Response_5,
                    nrounds     = 200,
                    objective   = "binary:logistic",
                    eval_metric = evalerror)

xgb_fit_6<- xgboost(data    = data.matrix(train[,feature.names]),
                    label       = train$Response_6,
                    nrounds     = 200,
                    objective   = "binary:logistic",
                    eval_metric = evalerror)

xgb_fit_7<- xgboost(data    = data.matrix(train[,feature.names]),
                    label       = train$Response_7,
                    nrounds     = 200,
                    objective   = "binary:logistic",
                    eval_metric = evalerror)

xgb_fit_8<- xgboost(data    = data.matrix(train[,feature.names]),
                    label       = train$Response_8,
                    nrounds     = 200,
                    objective   = "binary:logistic",
                    eval_metric = evalerror)


submission[is.na(submission)] <- (-1)

submission<-data.frame(Id=submission$Id,Response_1=predict(xgb_fit_1, data.matrix(submission)),
                       Response_2=predict(xgb_fit_2, data.matrix(submission)),
                       Response_3=predict(xgb_fit_3, data.matrix(submission)),
                       Response_4=predict(xgb_fit_4, data.matrix(submission)),
                       Response_5=predict(xgb_fit_5, data.matrix(submission)),
                       Response_6=predict(xgb_fit_6, data.matrix(submission)),
                       Response_7=predict(xgb_fit_7, data.matrix(submission)),
                       Response_8=predict(xgb_fit_8, data.matrix(submission)))

write.csv(submission,file="09022016.csv",row.names= FALSE)
