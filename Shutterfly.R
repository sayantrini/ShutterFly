# Importing the dataset
dataset = read.csv('Shutterfly_Data.csv')


# load libraries
#install.packages("h2o", type="source", repos="https://h2o-release.s3.amazonaws.com/h2o/master/3978/R")
library(dplyr)
library(data.table)
library(DMwR)
library(Hmisc)
library(caret)
library(h2o)
library(iml)
#localH2O <- h2o.init(nthreads = -1)
h2o.init()


# Removing the 2015 dates
dataset$order_date= format(strptime(dataset$order_date, format = "%d-%b-%y"), "%d/%m/%y")
dataset$order_date= substr(dataset$order_date,7,8)
filter(dataset, order_date == 14)
library(dplyr)
dataset = filter(dataset, order_date == 14)

# Feature Engineering to get sum(revenue) by customer, highest bought product, highest bought category

dataset <- dataset %>% group_by(customer_id) %>% mutate(Total_revenue = sum(revenue))
dataset <- dataset %>% group_by(customer_id,category_name, product_name) %>% mutate(Category_units = sum(units))


dataset <- dataset %>% group_by(customer_id) %>%
  filter(Category_units == max(Category_units))


input_df <- dataset %>% group_by(customer_id, category_name, product_name, Total_revenue, Category_units) %>%
  filter(order_sequence == max(order_sequence))

# if order sequence > 1 then active, target = 0, else inactive -> target = 1
input_df$target <- ifelse(input_df$order_sequence > 1, 0, 1)

# removing order_id and order_date from dataset

use_df <- input_df %>% select(customer_id, category_name, product_name, Category_units, Total_revenue, target)

use_df <- use_df %>% ungroup()

use_df$target <- as.factor(use_df$target)

# dependent and independent variables
y.dep <- 'target'
x.indep <- names(use_df)[names(use_df) != y.dep & names(use_df) != 'customer_id']

# split into training and test data
set.seed(456)
train_idx <- createDataPartition(use_df$target, p = 0.75, list = FALSE)

trainData <- use_df[train_idx,]
testData <- use_df[-train_idx,]

trainData <- trainData %>% select(-customer_id)

# check frequency of target class
table(trainData$target)
#0     1 
#5798 12308

# balance classes using minority oversampling
SMOTbalanced <- SMOTE(target ~ ., 
                      as.data.frame(trainData), perc.over = 300, perc.under = 150)

table(SMOTbalanced$target)
# 0     1 
# 23192 26091  

train.h2o <- as.h2o(SMOTbalanced)
test.h2o <- as.h2o(testData)

### Training the model ###

#Random Forest Classifier

rforest.model <- h2o.randomForest(y=y.dep, x=x.indep, training_frame = train.h2o, 
                                  ntrees = 1000, mtries = 3, max_depth = 40, seed = 11, 
                                  stopping_rounds = 200, stopping_metric = "AUC",
                                  validation_frame = test.h2o)

h2o.performance(rforest.model)

predict.rforest <- as.data.frame(h2o.predict(rforest.model, test.h2o))
model_rf__pred <- as.data.frame(cbind(testData$customer_id,testData$target, predict.rforest))

#AUC:  0.9829425
#Gini:  0.965885

# Plotting the Confusion Matrix
confusionMatrix(data = model_rf__pred$predict, model_rf__pred$`testData$target`, positive = levels(model_rf__pred$`testData$target`)[2])
cm_testdf <- confusionMatrix( model_rf__pred$predict,model_rf__pred$`testData$target`)
draw_confusion_matrix(cm_testdf)

# Finding out the number of actives and inactives
table(model_rf__pred$`testData$target`)

#GBM - Gradient Boost Classifier

gbm.model <- h2o.gbm(y=y.dep, x=x.indep, training_frame = train.h2o, ntrees = 1000, max_depth = 4, 
                     learn_rate = 0.01, nfolds = 10, stopping_rounds = 100, stopping_metric = "AUC"
                     ,seed = 11)

h2o.performance(gbm.model)

# AUC:  0.9216506
# Gini:  0.8433011

# Plotting the Confusion Matrix
predict.gbm <- as.data.frame(h2o.predict(gbm.model, test.h2o))
cm_gbm <- confusionMatrix(predict.gbm$predict, testData$target)
draw_confusion_matrix(cm_gbm)

# Plotting variables which are important
h2o.varimp_plot(gbm.model, num_of_features = 10)

# Deep learning (Neural Network)
deeplearning.model <- h2o.deeplearning(y = y.dep,
                                       x = x.indep,
                                       training_frame = train.h2o,
                                       epoch = 60,
                                       hidden = c(100,100),
                                       activation = "Rectifier", stopping_metric = "AUC",
                                       seed = 11)
h2o.performance(deeplearning.model)

#AUC:  0.8875251
#Gini:  0.7750503

# Plotting the Confusion Matrix
predict.dl <- as.data.frame(h2o.predict(deeplearning.model, test.h2o))
cm_dl <- confusionMatrix(predict.dl$predict, testData$target)
draw_confusion_matrix(cm_dl)