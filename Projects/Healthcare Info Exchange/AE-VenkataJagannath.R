# All variables from the environment are cleared
rm(list=ls())

# Load necessary libraries
library(caret)
library(car)
library(MASS)

# Read the file
consent <- read.csv("C:/Users/Venkata Jagannath/Desktop/Spring 2016/5503/Challenge/consent.csv")

Consent_IDs <- consent[,c("Consent","P_PatientID","D_docid","D_Specialty","Referrals.to.doc","no.of.patients")]
consent <- consent[,-c(2,8,10,17,18)]


# Remove values where Age is zero
sum(consent$P_Age==0) # Number of patients with zero age is 85.
consent <- consent[consent$P_Age>0,]

# Feature engineering with Patient and Doctor genders
consent["Gender"]<- paste(consent$P_Gender,consent$D_DocGender,sep="")
consent <- consent[,-(which(names(consent)%in%c("D_DocGender", "P_Gender")))]


# Reducing the data based on Number of ordering doctors
consent <- consent[consent$P_No.ordering.doctors<5,]


# Reducing the data based on Number of Reports
consent <- consent[consent$P_Total.No..of.Reports<13,]

# A generalized regression model is fitted to our data
reg <- glm(consent$Consent~.,data=consent,family = "binomial")

# The prediction is saved to a new column
consent["pred"] <- predict(reg,consent,type="response")

# We use a cutoff of 0.7 to derive our prediction
consent$pred <- ifelse(consent$pred>0.7,"Y","N")

# The function gives all statistics of our prediction
confusionMatrix(consent$Consent,consent$pred)

































