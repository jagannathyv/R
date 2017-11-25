# All variables from the environment are cleared
rm(list=ls())

# Load necessary libraries
library(caret)
library(bbmle)
library(leaps)
library(plyr)
library(MASS)

# Read the file
consent <- read.csv("C:/Users/Venkata Jagannath/Desktop/Spring 2016/5503/Challenge/consent.csv")

Consent_IDs <- consent[,c("Consent","P_PatientID","D_docid","D_Specialty")]
consent <- consent[,-c(2,8,10)]

# Remove values where Age is null
sum(consent$P_Age==0) # Number of patients with zero age is 85.
consent <- consent[consent$P_Age>0,]


# Partition the data
set.seed(123)
sample <- createDataPartition(consent$Consent,p=0.8,list = FALSE)
train <- consent[sample,]
test <- consent[-sample,]

# CAN THE MODELS BE BUILT AS PER DIFFERENT GENDERS??



# Gender -
# We can see that higher number of women are likely to consent. 
# And higher number of women will consent to share their records with a member of the same sex
# A very low percent of men consent when the doctor is of the opposite gender.

# train["Gender"]<- paste(train$P_Gender,train$D_DocGender,sep="")
# test["Gender"]<- paste(test$P_Gender,test$D_DocGender,sep="")
# train <- train[,-(which(names(train)%in%c("D_DocGender", "P_Gender")))]

# Will the necessity of seeing more number of doctor influence a patient's 
# judgement to give consent to avoid the hassle of carrying the reports?


# More number of doctors implies more hassle
train$P_No.ordering.doctors <- ifelse(train$P_No.ordering.doctors<4,"A","B")
test$P_No.ordering.doctors <- ifelse(test$P_No.ordering.doctors<4,"A","B")

# Does the total number of reports on the patient influence decisions? 
# Since carrying too many is undesirable.

train$P_Total.No..of.Reports <- ifelse(train$P_Total.No..of.Reports<12,"A","B")
test$P_Total.No..of.Reports <- ifelse(test$P_Total.No..of.Reports<12,"A","B")

# Does the distance to the healthcare facility have an impact on patient decisions?
range(train$P_Distance)

train <- train[-119,]
reg <- glm(train$Consent~.,data=train,family = "binomial")

test_check <- data.frame(predict(reg,test,type="response"))

test_check <- ifelse(test_check>0.6,"Y","N")
 
colnames(test_check)<- c("Consent")

confusionMatrix(test_check,test$Consent)

opar <- par()
par(mfrow=c(2,2))
plot(reg)
par(opar)


n = length(train) 
cutoff = 4/n 
z = round(cooks.distance(reg),4) 
View(z[z>cutoff])

plot(reg,which=4,cook.levels = cutoff) 
abline(h=cutoff,col="red")

varImp(reg)






























