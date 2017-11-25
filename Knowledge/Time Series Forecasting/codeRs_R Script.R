##### ***********  CODERS : Project 2 - Time Series Analysis and Forecasting  *************#####

# Clear the environment
rm(list=ls())

##installs the required packages
install.packages("forecast")
install.packages("fpp")
install.packages("car")


library(forecast)
library(fpp)
library(car)

##########   Dataset description

airline <- read.csv("C:/Users/Venkata Jagannath/Desktop/Spring 2016/5503/Project 2/airline.csv", header=TRUE)
View(airline)
str(airline)
head(airline,5)


##########   1. Exploratory Analysis

## 1.1. creating a time series object for number of passengers

no_of_passengers=ts(airline$Passengers,start =c(1990,1),end =c(2004,2),frequency=12)
x=no_of_passengers

#plotting the time series object

plot(x/1000000, col="gray",
     main="Airline- Number of Passengers over the years",
     xlab="",ylab="",lwd=3,axes=FALSE)
# to customize the display of label values in y axis 
# pretty - computes a sequence of about n+1 equally spaced 'round' values which cover the range of the values in x
axis(1)
pts <- pretty(x / 1000000)
axis(2, at = pts, labels = paste(pts, "M", sep = ""), las = 1)
box()


## 1.2. initial exploratory analysis

# 1.2.1. Fiiting in a regression line over years- upward trend

reg_trend=lm( x/1000000~time(x))
abline(reg_trend,col="red",lwd=2)
text(2002,65,"Upward Trend", font=13, col = "blue")
summary(reg_trend)


# 1.2.2. aggregating the cycles also shows a year on year trend
plot(aggregate(x/1000000,FUN=mean),main="Airline- Average nuumber of Passengers over the years",
     xlab="",ylab="",lwd=3,axes=FALSE, col = "purple")
axis(1)
pts <- pretty(x / 1000000)
axis(2, at = pts, labels = paste(pts, "M", sep = ""), las = 1)
box()


# 1.2.3. boxplot - analyze the mean and variance of time series to get a sense of seasonal effect

boxplot(x/1000000~cycle(x),axes=FALSE,col = "magenta",xlim=c(1,12))
axis(1)
pts <- pretty(x / 1000000)
axis(2, at = pts, labels = paste(pts, "M", sep = ""), las = 1)
box()
title("Seasonal effect: Number of Passengers by Month")


###########  2.   Model decomposition

### 2.2. stl- decomposes the time series data as seasonal,trend and remainder ,follows additive model

#stl : stl( ) performs a seasonal decomposition of a given time series Xt by determining the trend Tt using "loess" regression 
#and then calculating the seasonal component St (and the residuals et) from the differences Xt-Tt

stl_air <- stl(x, s.window=5)
stl_air

#shows the additive decomposition of the data - as components are independent of each other
plot(stl_air$time.series/1000000,main="Model Decomposition: stl()- Number of Passengers over the years",
     lwd=3, col = "black")


## 2.2.1 Trend Component

#shows the original data in gray

plot(x/1000000, col="gray",
     main="AIRLINE - Number of Passengers over the years",
     xlab="",ylab="",lwd=3,axes=FALSE)
axis(1)
pts <- pretty(x / 1000000)
axis(2, at = pts, labels = paste(pts, "M", sep = ""), las = 1)
box()

# shows the trend component in red
lines(stl_air$time.series[,2]/1000000,col="red",lwd=2)
text(1992,65,"stl()- Trend Component",col="blue",font=13)


# 2.2.2 Seasonal pattern

#monthplot - plot seasonal (or other) subseries of a time series. 
#For each season (or other category), a time series is plotted.
monthplot(stl_air$time.series[,"seasonal"]/1000000, main="AIRLINE - Number of passengers in millions By Month",
          col="blue",ylab="Seasonal", lwd=2, axes=FALSE)
text(3,7,"monthplot()- Seasonal Pattern",col="red",font=13)
axis(2)
axis(1,at=c(1:12),las=1)


##  2.2.3 Seasonally adjusted data 

#can be used for building non0-seasonal model if seasonal variation is not our prime interest

# displays the original data in gray

plot(x/1000000, col="gray",
     main="Airline- Number of Passengers over the years",
     xlab="",ylab="",axes=FALSE,lwd=3)
##to customize the display of label values in y axis 
axis(1)
pts <- pretty(x / 1000000)
axis(2, at = pts, labels = paste(pts, "M", sep = ""), las = 1)
box()


# seasadj() - gives seasonally adjusted data =Y(t)-S(t)

lines(seasadj(stl_air)/1000000,col="purple",ylab="Seasonally adjusted",lwd=2)
text(1994,65,"seasadj()- Seasonaly adjusted data",col="red",font=13)



## 2.3. Classific Decomposition

# follows moving average apprach to decompose time series components

# Downsides-not robust,assumptions

# Overcome classic decomposition - stl used

dec_air <- decompose(x/1000000, type="additive")
plot(dec_air, lwd=3, col = "black")



##########  3.  Fitting trend and seasonal models

# 3.1. Only trend
fit_trend <-tslm (x~trend)
plot(forecast(fit_trend,h=100))
#this shows the accuracy of the model after we capture the  seasonality
accuracy(fit_trend)

# 3.2. Only season
fit_season <-tslm (x~season)
plot(forecast(fit_season,h=100))
#this shows the accuracy of the model after we capture the  seasonality
accuracy(fit_season)


# 3.3. for trend and seasonality
fit <-tslm (x~trend+season)
plot(forecast(fit,h=100))
#this shows the accuracy of the model after we capture the trend and seasonality
accuracy(fit)

# 3.4 Intervention
# Plot before intervention
plot.ts(no_of_passengers)

#Put NA to replace the specific event value
no_of_passengers[no_of_passengers==38730506] = NA

#Replace missing value by using na.interp function 
no_of_passengers = (na.interp(no_of_passengers))
plot.ts(no_of_passengers)


####4
# SMOOTHING

# 4.1 Simple Exponential Smoothing
Simple_smoothing <- ses(no_of_passengers, h=100, level=c(80,95),initial="optimal", alpha=NULL)

# Plotting our results and computing accuracy 
plot(Simple_smoothing)
accuracy(Simple_smoothing)

# 4.2 Linear (Holt) Smoothing
# Uses two parameters and captures trend
Linear_Holt <- holt(no_of_passengers, h=100, damped=FALSE, level=c(80,95),initial=c("optimal","simple"), exponential=FALSE,
                    alpha=NULL, beta=NULL)

# Plotting our results and computing accuracy
plot(Linear_Holt)
accuracy(Linear_Holt)

# 4.3 Damped Trend
Damped_Trend <- holt(no_of_passengers, h=100, damped=TRUE, level=c(80,95),initial=c("optimal","simple"), exponential=FALSE,
                     alpha=NULL, beta=NULL)

# Plotting our results and computing accuracy
plot(Damped_Trend)
accuracy(Damped_Trend)

# 4.4 Holt Winters - Additive
# To capture trend, seasonality - 3 parameters

Winters_Additive <- hw(no_of_passengers, h=2*frequency(no_of_passengers), seasonal="additive", damped=FALSE, 
                       level=c(80,95), initial=c("optimal","simple"),exponential=FALSE, alpha=NULL, beta=NULL, gamma=NULL)

# Plotting our results and computing accuracy
plot(Winters_Additive)
accuracy(Winters_Additive)


# 4.5 Holt Winters Multiplicative

Winters_Multiplicative <- hw(no_of_passengers, h=2*frequency(no_of_passengers), seasonal="multiplicative", damped=FALSE, 
                             level=c(80,95), initial=c("optimal","simple"),exponential=FALSE, alpha=NULL, beta=NULL, gamma=NULL)

# Plotting our results and computing accuracy
plot(Winters_Multiplicative)
accuracy(Winters_Multiplicative)

# TEST FOR STATIONARITY
#Augmented Dickey fuller test for stationarity
adf.test(no_of_passengers,alternative = "stationary")

##ACF and PACF
par(mfrow=c(2,1))
Acf(no_of_passengers,plot=TRUE)
Pacf(no_of_passengers,plot=TRUE)



#ARIMA(1,0,0) or AR(1)
after_ar1=arima(no_of_passengers,order=c(1,0,0))
# analysis after arima
par(mfrow=c(2,1))
Acf(resid(after_ar1))
Pacf(resid(after_ar1))
accuracy(after_ar1)



#ARIMA(2,0,0) or AR(2)
after_ar2=arima(no_of_passengers,order=c(2,0,0))
par(mfrow=c(2,1))
Acf(resid(after_ar2))
Pacf(resid(after_ar2))
accuracy(after_ar2)

#ARIMA(0,0,1) or MA(1)
after_ma1=arima(no_of_passengers,order=c(0,0,1))
par(mfrow=c(2,1))
Acf(resid(after_ma1))
Pacf(resid(after_ma1))
accuracy(after_ma1)


# test to estimate the number of seasonal differences
nsdiffs(no_of_passengers, max.D=2)


#Arima with seasonal component(0,1,0) and AR1
after_seasonal_ar12=arima(no_of_passengers,order=c(1,0,0),seasonal=c(0,1,0))
par(mfrow=c(2,1))
Acf(resid(after_seasonal_ar12))
Pacf(resid(after_seasonal_ar12))
accuracy(after_seasonal_ar12)

#Arima with seasonal component(0,1,1) and AR1
after_seasonal_ar13=arima(no_of_passengers,order=c(1,0,0),seasonal=c(0,1,1))
par(mfrow=c(2,1))
Acf(resid(after_seasonal_ar13))
Pacf(resid(after_seasonal_ar13))
accuracy(after_seasonal_ar13)


# returns the best ARIMA Model based on either AIC or BIC
auto.arima(no_of_passengers,max.p = 5,max.q = 5,max.P = 5,max.Q = 5)


#Forecast based on the model developed
Final_Forecast = predict(after_seasonal_ar13,n.ahead=1000)

plot(no_of_passengers)
lines(Final_Forecast$pred,col="blue")
lines(Final_Forecast$pred+2*Final_Forecast$se,col="red")
lines(Final_Forecast$pred-2*Final_Forecast$se,col="red")




