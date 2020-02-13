--
title: "Lending Club"
author: "Siddharth"
date: "May 12th, 2017"
--
install.packages("xgboost")
install.packages("gbm")
install.packages("randonForest")
install.packages("tree")
install.packages("dummies")
install.packages("leaps")
library(leaps)
library(tree)
loan<-read.csv("loan.csv")
library(dummies)
data<-dummy.data.frame(loan,names=c("term", "grade", "emp_length", "home_ownership", "verification_status","purpose"),sep=".")
data1=data[complete.cases(data),]
df<-data1
loan.lstatus <- data1

#Tree
set.seed(12345)
train = sample(nrow(loan.lstatus), nrow(loan.lstatus)*0.5)
loan.lstatus$loan_status <- ifelse(loan.lstatus$loan_status =="Fully Paid", 1, 0)
loan.lstatus <- data.frame(loan.lstatus)
tree.loan=tree(loan_status~.,loan.lstatus,subset=train)
# For comparison
cv.loan=cv.tree(tree.loan)
plot(cv.loan$size,cv.loan$dev,type='b')
prune.loan=prune.tree(tree.loan,best=3)
plot(prune.loan)
text(prune.loan,pretty=0)
yhat=predict(prune.loan,newdata=loan.lstatus[-train,])
yhat <- data.frame(yhat)
predicted.probability <- yhat[,1]
## Generate class predictions using cutoff value
cutoff <- 0.5
Predicted <- ifelse(predicted.probability > cutoff, "Fully Paid", "Charged Off")
loan.test=loan.lstatus[-train,"loan_status"]
(c = table(loan.test,Predicted))
(acc = (c[1,1]+c[2,2])/sum(c))


#Bagging and Random Forests
library(randomForest)
# We first do bagging (which is just RF with m = p)
set.seed(1)
loan.lstatus$loan_status <- as.factor(loan.lstatus$loan_status)
bag.loan=randomForest(loan_status~.,data=loan.lstatus,subset=train,mtry=4,importance=TRUE)
bag.loan
yhat.bag = predict(bag.loan,newdata=loan.lstatus[-train,])
loan.test=loan.lstatus[-train,"loan_status"]
(c = table(loan.test,yhat.bag))
(acc = (c[1,1]+c[2,2])/sum(c))
importance(bag.loan)
varImpPlot(bag.loan)
# Now RF with m = 2
set.seed(1)
loan.lstatus$loan_status <- as.factor(loan.lstatus$loan_status)
bag.loan=randomForest(loan_status~.,data=loan.lstatus,subset=train,mtry=6,importance=TRUE)
bag.loan
yhat.bag = predict(bag.loan,newdata=loan.lstatus[-train,])
loan.test=loan.lstatus[-train,"loan_status"]
(c = table(loan.test,yhat.bag))
(acc = (c[1,1]+c[2,2])/sum(c))
importance(bag.loan)
varImpPlot(bag.loan)


#Boosting
library(gbm)
set.seed(1)
loan.lstatus$loan_status <- as.character(loan.lstatus$loan_status)
boost.loan=gbm(loan_status~.,data=loan.lstatus[train,],distribution="bernoulli",n.trees=1000,interaction.depth=4,shrinkage = 0.1)
summary(boost.loan)
par(mfrow=c(1,2))
plot(boost.loan,i="int_rate")
plot(boost.loan)
yhat.boost=predict(boost.loan,newdata=loan.lstatus[-train,],n.trees=1000,type="response")
predicted <- ifelse(yhat.boost>=0.5,1,0)
yhat.test=loan.lstatus$loan_status[-train]
(c = table(predicted,yhat.test))
(acc = (c[1,1]+c[2,2])/sum(c))

#Forward Selection
leaps <- regsubsets(loan_status~., data = loan.lstatus, nbest = 10, nvmax = 19, method = "forward")
plot(leaps,scale="r2")

#Backward Selection
leaps <- regsubsets(loan_status~., data = loan.lstatus, nbest = 10, nvmax = 19, method = "backward")
plot(leaps,scale="r2")

---
title: "Project"
author: "mengyuanwang"
date: "4/6/2017"
output: html_document
---
#Linear Regression & Logistic Regression
loan<-read.csv("/loan.csv")
set.seed(12345)
library(dummies)
data<-dummy.data.frame(loan,names=c("term", "grade", "emp_length", "home_ownership", "verification_status","purpose"),sep=".")
# nrow(data)
data=data[complete.cases(data),]
n=nrow(data)
train<-sample(n,(0.7*n))
loantrain<-data[train,]
loanvalidation<-data[-train,]
#Linear Regression
loantrain$status = ifelse(loantrain$loan_status=="Charged Off", 0, 1)
loantrain$loan_status=NULL
loantrain[,-55]<- scale(loantrain[,-55])
fit<-lm(status~.,data = loantrain)
summary(fit)
#training
predicted<-predict(fit,loantrain)
pred = ifelse(predicted>0.5,1,0)
confusion=table(loantrain$status,pred)
confusion
acc=(confusion[1,1]+confusion[2,2])/sum(confusion)
acc
#validation
loanvalidation$status = ifelse(loanvalidation$loan_status=="Charged Off", 0, 1)
loanvalidation$loan_status=NULL
loanvalidation[,-55]<- scale(loanvalidation[,-55])
predicted<-predict(fit,loanvalidation)
pred = ifelse(predicted>0.5,1,0)
confusion=table(loanvalidation$status,pred)
confusion
acc=(confusion[1,1]+confusion[2,2])/sum(confusion)
acc


#Logistic Regression
loan<-read.csv("/Users/wangmengyuan/Desktop/loan.csv")
set.seed(12345)
library(dummies)
data<-dummy.data.frame(loan,names=c("term", "grade", "emp_length", "home_ownership", "verification_status","purpose"),sep=".")
# nrow(data)
data=data[complete.cases(data),]
n=nrow(data)
train<-sample(n,(0.7*n))
loantrain<-data[train,]
loanvalidation<-data[-train,]
loantrain[,-55]<- scale(loantrain[,-55])
fit<-glm(loan_status~.,family="binomial",data = loantrain)
summary(fit)
#training
predicted<-predict(fit,loantrain,type="response")
pred = ifelse(predicted>0.5,1,0)
confusion=table(loantrain$loan_status,pred)
confusion
acc=(confusion[1,1]+confusion[2,2])/sum(confusion)
acc
#validation
loanvalidation[,-55]<- scale(loanvalidation[,-55])
predicted<-predict(fit,loanvalidation,type="response")
pred = ifelse(predicted>0.5,1,0)
confusion=table(loanvalidation$loan_status,pred)
confusion
acc=(confusion[1,1]+confusion[2,2])/sum(confusion)
acc



#Neuralnet
library(neuralnet)
Status = ifelse(loantrain$loan_status=="Charged Off", 0, 1)
loantrain1=data.frame(Status)
loantrain1$loan_amnt=loantrain$loan_amnt
loantrain1$month_36=loantrain$`term. 36 months`
loantrain1$month_60=loantrain$`term. 60 months`
loantrain1$gradeA=loantrain$grade.A
loantrain1$intrate=loantrain$int_rate
loantrain1$revol_unitl=loantrain$revol_util
loantrain1$pymnt=loantrain$total_pymnt
loantrain1$totalreclatefee=loantrain$total_rec_late_fee
nn<-neuralnet(Status~loan_amnt+month_36+month_60+gradeA+intrate+revol_unitl+pymnt+totalreclatefee,data=loantrain1,hidden=c(2))
plot(nn)
#trainning data set
predicted<-compute(nn,loantrain1[-1])$net.result
pred<-ifelse(predicted>0.5,1,0)
confusion<-table(loantrain1$Status,pred)
confusion
acc=(confusion[1,1]+confusion[2,2])/sum(confusion)
acc
#validation data set
Status = ifelse(loanvalidation$loan_status=="Charged Off", 0, 1)
loanvalidation1=data.frame(Status)
loanvalidation1$loan_amnt=loanvalidation$loan_amnt
loanvalidation1$month_36=loanvalidation$`term. 36 months`
loanvalidation1$month_60=loanvalidation$`term. 60 months`
loanvalidation1$gradeA=loanvalidation$grade.A
loanvalidation1$intrate=loanvalidation$int_rate
loanvalidation1$revol_unitl=loanvalidation$revol_util
loanvalidation1$pymnt=loanvalidation$total_pymnt
loanvalidation1$totalreclatefee=loanvalidation$total_rec_late_fee
predicted<-compute(nn,loanvalidation1[-1])$net.result
pred<-ifelse(predicted>0.5,1,0)
confusion<-table(loanvalidation1$Status,pred)
confusion
acc=(confusion[1,1]+confusion[2,2])/sum(confusion)
acc


#KNN
fun <- function(x){ 
  a <- mean(x) 
  b <- sd(x) 
  (x - a)/(b) 
} 
loantrain1[,-1] <- apply(loantrain1[,-1], 2, fun)
loanvalidation1[,-1] <- apply(loantrain1[,-1], 2, fun)
#input does not include the prediction line
train_input <- as.matrix(loantrain1[,-1])
train_output <- as.vector(loantrain1[,1]) #vector!!
validate_input <- as.matrix(loanvalidation1[,-1])
library(class)
kmax <- 10
ER1 <- rep(0,kmax)
ER2 <- rep(0,kmax)
library(class)
for (i in 1:kmax){
  prediction <- knn(train_input, train_input,train_output, k=i)
  prediction2 <- knn(train_input, validate_input,train_output, k=i)
  CM1 <- table(loantrain1$Status, prediction)
  ER1[i] <- (CM1[1,2]+CM1[2,1])/sum(CM1) #train predication error
  CM2 <- table(loanvalidation1$Status, prediction2) 
  ER2[i] <- (CM2[1,2]+CM2[2,1])/sum(CM2) #validation predication erro
}
z <- which.min(ER2)
cat("Minimum Validation Error k:", z)
prediction <- knn(train_input, train_input,train_output, k=10)
prediction2 <- knn(train_input, validate_input,train_output, k=10)
table(loantrain1$Status, prediction)
cat("Maxmum Accuracy:", 1-ER1[10])
table(loanvalidation1$Status, prediction2) 
cat("Maxmum Accuracy:", 1-ER2[10])


#Association Rule
loan<-read.csv("loan.csv")
set.seed(12345)
library(dummies)
data<-dummy.data.frame(loan,names=c("term", "grade", "emp_length", "home_ownership", "verification_status","purpose"),sep=".")
# nrow(data)
data=data[complete.cases(data),]
n=nrow(data)
train<-sample(n,(0.7*n))
loantrain<-data[train,]
loanvalidation<-data[-train,]
Status = ifelse(loantrain$loan_status=="Charged Off", 0, 1)
loantrain2=data.frame(Status)
loantrain2$month_36=loantrain$`term. 36 months`
loantrain2$month_60=loantrain$`term. 60 months`
loantrain2$less1y=loantrain$`emp_length.< 1 year`
loantrain2$y1=loantrain$`emp_length.1 year`
loantrain2$y2=loantrain$`emp_length.2 years`
loantrain2$y3=loantrain$`emp_length.3 years`
loantrain2$y4=loantrain$`emp_length.4 years`
loantrain2$y5=loantrain$`emp_length.5 years`
loantrain2$y6=loantrain$`emp_length.6 years`
loantrain2$y7=loantrain$`emp_length.7 years`
loantrain2$y8=loantrain$`emp_length.8 years`
loantrain2$y9=loantrain$`emp_length.9 years`
loantrain2$y10=loantrain$`emp_length.10+ years`
loantrain2$empna=loantrain$`emp_length.n/a`
loantrain2$homemortgage=loantrain$home_ownership.MORTGAGE
loantrain2$homenone=loantrain$home_ownership.NONE
loantrain2$homeother=loantrain$home_ownership.OTHER
loantrain2$homeown=loantrain$home_ownership.OWN
loantrain2$homerent=loantrain$home_ownership.RENT
loantrain2$verifynot=loantrain$`verification_status.Not Verified`
loantrain2$verifysource=loantrain$`verification_status.Source Verified`
loantrain2$verified=loantrain$verification_status.Verified
loantrain2$car=loantrain$purpose.car
loantrain2$creditcard=loantrain$purpose.credit_card
loantrain2$debt=loantrain$purpose.debt_consolidation
loantrain2$edu=loantrain$purpose.educational
loantrain2$homeimp=loantrain$purpose.home_improvement
loantrain2$house=loantrain$purpose.house
loantrain2$purchase=loantrain$purpose.major_purchase
loantrain2$medical=loantrain$purpose.medical
loantrain2$move=loantrain$purpose.moving
loantrain2$other=loantrain$purpose.other
loantrain2$energy=loantrain$purpose.renewable_energy
loantrain2$smallbusiness=loantrain$purpose.small_business
loantrain2$vacation=loantrain$purpose.vacation
loantrain2$wedding=loantrain$purpose.wedding
library(arules)
library(arulesViz)
loan=as.matrix(loantrain2)
rules=apriori(data = loan,parameter = list(supp=0.01, conf=0.8), appearance = list(default="lhs",rhs="Status"),control = list(verbose=F))
summary(rules)
rules=sort(rules,decreasing = TRUE, by="confidence")
inspect(rules[1:10])
plot(rules)
plot(rules[1:10], method="graph", control=list(type="items"))
plot(rules[1:10], method="paracoord", control=list(reorder=TRUE))
---
title: "Project"
author: "huizhu"
date: "4/7/2017"
output: html_document
---
#KMeans
#k=5
#read data
loan<-read.csv("loan.csv")
#set dummy variables
library(dummies)
data<-dummy.data.frame(loan,names=c("term", "grade", "emp_length", "home_ownership", "verification_status","purpose"),sep=".")
#clean missing data
data1=data[complete.cases(data),]
df<-data1
data1[,55] <- NULL
#scale the data
dfsc<-scale(data1)
#set seed
set.seed(12345)
#kmeans
km.out=kmeans(dfsc,5,nstart=20)
km.out
km.out$centers
dist(km.out$centers)
#plot chart
df$Cluster<-km.out$cluster
table(df$Cluster)
barplot(table(df$Cluster)/47891) 
#fully paid rate for each cluster
Cluster1=subset(df,df$Cluster=="1")
Cluster2=subset(df,df$Cluster=="2")
Cluster3=subset(df,df$Cluster=="3")
Cluster4=subset(df,df$Cluster=="4")
Cluster5=subset(df,df$Cluster=="5")
sum(as.numeric(Cluster1$loan_status)-1)/15420
sum(as.numeric(Cluster2$loan_status)-1)/748
sum(as.numeric(Cluster3$loan_status)-1)/10628
sum(as.numeric(Cluster4$loan_status)-1)/10979
sum(as.numeric(Cluster5$loan_status)-1)/10116
#k=4
#read data
loan<-read.csv("loan.csv")
#set dummy variables
library(dummies)
data<-dummy.data.frame(loan,names=c("term", "grade", "emp_length", "home_ownership", "verification_status","purpose"),sep=".")
#clean missing data
data1=data[complete.cases(data),]
df<-data1
data1[,55] <- NULL
#scale the data
dfsc<-scale(data1)
#set seed
set.seed(12345)
#kmeans
km.out=kmeans(dfsc,4,nstart=20)
km.out
km.out$centers
dist(km.out$centers)
#plot the chart
df$Cluster<-km.out$cluster
table(df$Cluster)
barplot(table(df$Cluster)/47891) 
#fully paid rate for each cluster
Cluster1=subset(df,df$Cluster=="1")
Cluster2=subset(df,df$Cluster=="2")
Cluster3=subset(df,df$Cluster=="3")
Cluster4=subset(df,df$Cluster=="4")
sum(as.numeric(Cluster1$loan_status)-1)/10257
sum(as.numeric(Cluster2$loan_status)-1)/15686
sum(as.numeric(Cluster3$loan_status)-1)/10835
sum(as.numeric(Cluster4$loan_status)-1)/11113
#k=6
#read data
loan<-read.csv("loan.csv")
#set dummy variables
library(dummies)
data<-dummy.data.frame(loan,names=c("term", "grade", "emp_length", "home_ownership", "verification_status","purpose"),sep=".")
#clean missing data
data1=data[complete.cases(data),]
df<-data1
data1[,55] <- NULL
#scale data
dfsc<-scale(data1)
#set seed
set.seed(12345)
#kmeans
km.out=kmeans(dfsc,6,nstart=20)
km.out
km.out$centers
dist(km.out$centers)
#plot the chart
df$Cluster<-km.out$cluster
table(df$Cluster)
barplot(table(df$Cluster)/47891) 
#fully paid rate for each cluster
Cluster1=subset(df,df$Cluster=="1")
Cluster2=subset(df,df$Cluster=="2")
Cluster3=subset(df,df$Cluster=="3")
Cluster4=subset(df,df$Cluster=="4")
Cluster5=subset(df,df$Cluster=="5")
Cluster6=subset(df,df$Cluster=="6")
sum(as.numeric(Cluster1$loan_status)-1)/10591
sum(as.numeric(Cluster2$loan_status)-1)/3453
sum(as.numeric(Cluster3$loan_status)-1)/9227
sum(as.numeric(Cluster4$loan_status)-1)/11317
sum(as.numeric(Cluster5$loan_status)-1)/1187
sum(as.numeric(Cluster6$loan_status)-1)/12116


#lasso
library(glmnet)
loan<-read.csv("loan.csv")
set.seed(12345)
library(dummies)
data<-dummy.data.frame(loan,names=c("term", "grade", "emp_length", "home_ownership", "verification_status","purpose"),sep=".")
# nrow(data)
data=data[complete.cases(data),]
n=nrow(data)
train<-sample(n,(0.7*n))
loantrain<-data[train,]
loanvalidation<-data[-train,]
loantrain$loan_status= ifelse(loantrain$loan_status=="Charged Off", 0, 1)
loanvalidation$loan_status= ifelse(loanvalidation$loan_status=="Charged Off", 0, 1)
x=model.matrix(loantrain$loan_status~.,loantrain)[,-55]#train
y=loantrain$loan_status
x1=model.matrix(loanvalidation$loan_status~.,loanvalidation)[,-55]#test
y1=loanvalidation$loan_status
grid=10^seq(10,-2,length=100)
lasso.mod=glmnet(x,y,alpha=1,lambda=grid)#train data set to fit a lasso model
plot(lasso.mod)
cv.out=cv.glmnet(x,y,alpha=1)#lamba figure
plot(cv.out)
#best lambda
bestlam=cv.out$lambda.min
bestlam
lasso.pred=predict(lasso.mod,s=bestlam,newx=x1)
mean((lasso.pred-y1)^2)
#coefficients aat best lambda
out=glmnet(x,y,alpha = 1,lambda = grid)
lasso.coef=predict(out,type="coefficients",s=bestlam)
lasso.coef
#non zero coeeficients
lasso.coef[lasso.coef!=0]


#Naive Bayes
#read data
df<-read.csv("loan.csv")
#set factor
df$term<-as.factor(df$term)
df$grade<-as.factor(df$grade)
df$emp_length<-as.factor(df$emp_length)
df$home_ownership<-as.factor(df$home_ownership)
df$verification_status<-as.factor(df$verification_status)
df$purpose<-as.factor(df$purpose)
#set seed
library("caret")
set.seed(12345)
#split data into train and testing
inTrain <- createDataPartition(df$loan_status, p=0.7, list=FALSE)
dftrain <- data.frame(df[inTrain,])
dfvalidation <- data.frame(df[-inTrain,])
# We require the library e1071
library(e1071)
#naive bays
model <- naiveBayes(loan_status~., data=dftrain)
model
prediction <- predict(model, newdata = dfvalidation[,-20])
table(dfvalidation$loan_status,prediction,dnn=list('actual','predicted'))
model$apriori
predicted.probability <- predict(model, newdata = dfvalidation[,-20], type="raw")
predicted.probability
# The first column is class 0, the second is class 1
PL <- as.numeric(dfvalidation$loan_status)-1
prob <- predicted.probability[,2]
df1 <- data.frame(prediction, PL, prob)
df1
#ROC for NB
cutoff <- seq(0, 1, length = 300)
fpr <- numeric(300)
tpr <- numeric(300)
# We'll collect it in a data frame.  
roc.table <- data.frame(Cutoff = cutoff, FPR = fpr,TPR = tpr)
roc.table
# TPR is the Sensitivity; FPR is 1-Specificity
for (i in 1:300) {
  roc.table$FPR[i] <- sum(df1$prob > cutoff[i] & df1$PL==0)/sum(df1$PL==0)
  roc.table$TPR[i] <- sum(df1$prob> cutoff[i] & df1$PL ==1)/sum(df1$PL == 1)
}
# The first line plots the Sensitivity against 1-Specificity
plot(TPR ~ FPR, data = roc.table, type = "s",xlab="1 - Specificity",ylab="Sensitivity",col="blue" ,main="ROC")
# Next line adds the central daigonal
abline(a = 0, b = 1, lty = 2,col="red")
#Lift chart
df1S <- df1[order(-prob),]
df1S$Gains <- cumsum(df1S$PL)
plot(df1S$Gains,type="n",main="Lift Chart",xlab="Number of Cases",ylab="Cumulative Success")
lines(df1S$Gains,col="blue")
abline(0,sum(df1S$PL)/nrow(df1S),lty = 2, col="red")


#Tree Model
#read data
loan<-read.csv("loan.csv")
#set factor
loan$term=factor(loan$term)
loan$grade=factor(loan$grade)
loan$emp_length=factor(loan$emp_length)
loan$home_ownership=factor(loan$home_ownership)
loan$verification_status=factor(loan$verification_status)
loan$purpose=factor(loan$purpose)
#require package
library(tree)
library(ISLR)
#set seed
set.seed(12345)
#split the data
inTrain <- sample(nrow(loan), 0.7*nrow(loan))
train <- data.frame(loan[inTrain,])
test <- data.frame(loan[-inTrain,])
#tree
tree.credit=tree(loan_status~.,data=train)
summary(tree.credit)
tree.pred=predict(tree.credit,test,type="class")
tree.pred
#accuracy
confusion = table(tree.pred,test$loan_status)
confusion
Error = (confusion[1,2]+confusion[2,1])/sum(confusion)
Error
#prune tree
set.seed(12345)
cv.credit=cv.tree(tree.credit,FUN = prune.misclass)
names(cv.credit)
cv.credit
plot(cv.credit$size,cv.credit$dev,type="b")
prune.credit=prune.misclass(tree.credit,best=7)
plot(prune.credit)
text(prune.credit,pretty=0)
tree.pred=predict(prune.credit,test,type="class")
table(tree.pred,test$loan_status)
#accuracy
confusion = table(tree.pred,test$loan_status)
confusion
Error = (confusion[1,2]+confusion[2,1])/sum(confusion)
1-Error
#91.78