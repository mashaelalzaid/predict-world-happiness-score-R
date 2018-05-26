#install packages
install.packages('mlbench') 
install.packages('caret') 
install.packages('e1071', dependencies=TRUE)
install.packages('caretEnsemble') 
install.packages('gbm')
install.packages('rpart')
install.packages('dplyr')
install.packages('ada') 
install.packages('adabag', dependencies=TRUE)
install.packages('ggplot2') 
install.packages("JOUSBoost")

# Load libraries
library(mlbench)
library(caret)
library(caretEnsemble)
library(gbm)
library("rpart")
library("ada")
library(adabag)
library(dplyr) # for data manipulation
library(caret) # for model-building
library(e1071)#error occurs in caret to solve it we install e1071 package
library(ggplot2)
library("JOUSBoost")

#dataset load
happy<-read.csv("/Users/skydiver/Downloads/wolrd-happiness-data.csv",sep=",")

### First: Data Exploratio ###

#to obtain the dimensions of the data set
dim(happy)
#to check what features the dataset has
names(happy)
#structure
str(happy)
#columns summary, 
summary(happy)
#to check  NAs
is.na(happy)
# delete NAs 
happy<-na.omit((happy))
#ensure there are no NAs anymore
which(is.na(happy))
#set happiness.level variable as factor
happy$happiness.level<-as.factor(happy$happiness.level)
#check the distribution of each class
table(happy$happiness.level)
#check classes distribution
prop.table(table(happy$happiness.level))
#no need for this column
happy$country<-NULL

#find correlation
findCorrelation(cor(happy[,1:15]), cutoff = .70,verbose = FALSE)
# 13  1  2  3  5

###  Second: Preprocessing ###
#delete correleted columns
happy$Standard.deviation.Mean.of.ladder.by.country.year<-NULL
happy$Happiness.score<-NULL
happy$Life.Ladder<-NULL
happy$Log.GDP.per.capita<-NULL
happy$Healthy.life.expectancy.at.birth<-NULL


#data splitting
set.seed(1234)
trainIndices = createDataPartition(happy$happiness.level, p = 0.7, list = F)
happy_train = happy[trainIndices, ]
happy_test = happy[-trainIndices, ]
seed <- 7
metric <- "Accuracy"

### Third:  model building ###

#3.1: Boosting Algorithms
control <- trainControl(method="repeatedcv", number=10, repeats=3)
tcontrol <- rpart.control(cp = -1, maxdepth = 14,maxcompete = 1,xval = 0)
#3.1.1: Random Forest
set.seed(seed)
fit.rf <- train(happiness.level~., data=happy_train, method="rf", metric=metric, trControl=control)
plot(fit.rf)
# 3 .1.2: C5.0
set.seed(seed)
fit.c50 <- train(happiness.level~., data=happy_train, method="C5.0", metric=metric, trControl=control)
plot(fit.c50)

# 3 .1.3: Stochastic Gradient Boosting
set.seed(seed)
fit.gbm <- train(happiness.level~., data=happy_train, method="gbm", metric=metric, trControl=control, verbose=FALSE)
plot(fit.gbm)

### Fourth:(3.1) Model Evaluation ###
# summarize results
boosting_results <- resamples(list(c5.0=fit.c50, gbm=fit.gbm, rf=fit.rf, ada=fit.ada))
summary(boosting_results)
dotplot(boosting_results)

# 3.2 adaboost
Ada_folds<-createFolds(happy$happiness.level, k=10) #creat folds 
Ada_fun <- lapply (Ada_folds, function(x){
  set.seed(seed)
  happy_train = happy[trainIndices, ]
  happy_test = happy[-trainIndices, ]
  happy_test_class <- happy_test[,11]

# AdaBoost - Adaptative Boosting
fit.ada <-boosting(happiness.level~.,  data=happy_train, boos=TRUE, mfinal=20,coeflearn='Breiman')
# predict 
results_ada <- predict(fit.ada, happy_test, type = "class")
results_ada$confusion
return(results_ada$confusion)
})

Ada_sum_matrices <-Reduce('+', Ada_fun)/10 #sum 10 matrices
Ada_final_confusionMatrix<-confusionMatrix(Ada_sum_matrices)
Ada_final_confusionMatrix

