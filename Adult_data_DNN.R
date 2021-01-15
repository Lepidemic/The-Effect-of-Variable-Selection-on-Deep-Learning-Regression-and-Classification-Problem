library(MASS)
library(reshape2)
library(ggplot2)
library(dplyr)
library(corrplot)
library(vioplot)

library(mlbench)  # Pima data 제공 패키지
library(gmodels)  # Confusion Matrix 작성 도구 제공
library(tictoc)

library(glmnet)  # LASSO, elastic net, ridge
library(agricolae)  # LSD test
library(lawstat)  # levene test

library(ROCR)
###############################################################
# Adult data
# The Adult Data frame has 48842 rows and 14 columns.
# https://archive.ics.uci.edu/ml/datasets/adult
###### 변수 설명 ######

# Listing of attributes:
# 
# >50K, <=50K.
# 
# age: continuous.

# workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov,
#            Local-gov, State-gov, Without-pay, Never-worked.

# fnlwgt: continuous.

# education: Bachelors, Some-college, 11th, HS-grad, Prof-school, 
#            Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters,
#           1st-4th, 10th, Doctorate, 5th-6th, Preschool.

# education-num: continuous.

# marital-status: Married-civ-spouse, Divorced, Never-married,
#                Separated, Widowed, Married-spouse-absent,
#                Married-AF-spouse.

# occupation: Tech-support, Craft-repair, Other-service, Sales,
#            Exec-managerial, Prof-specialty, Handlers-cleaners,
#            Machine-op-inspct, Adm-clerical, Farming-fishing,
#            Transport-moving, Priv-house-serv, Protective-serv,
#            Armed-Forces.

# relationship: Wife, Own-child, Husband, Not-in-family,
#              Other-relative, Unmarried.

# race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.

# sex: Female, Male.

# capital-gain: continuous.

# capital-loss: continuous.

# hours-per-week: continuous.

# native-country: United-States, Cambodia, England, Puerto-Rico,
#            Canada, Germany, Outlying-US(Guam-USVI-etc), India,
#            Japan, Greece, South, China, Cuba, Iran, Honduras,
#            Philippines, Italy, Poland, Jamaica, Vietnam, Mexico,
#            Portugal, Ireland, France, Dominican-Republic, Laos,
#            Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala,
#            Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador,
#            Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
adult_data <- read.table(url, header=FALSE, sep=",", strip.white=TRUE, stringsAsFactors=TRUE, na.strings=c("?"))

names(adult_data) <- c('age', 'workclass', 'fnlwgt', 'education', 
                       'education_num', 'marital_status', 'occupation', 
                       'relationship', 'race', 'sex',
                       'capital_gain', 'capital_loss',
                       'hours_per_week', 'native_country',
                       'wage')


# Adult data structure exploration ( 데이터 구조 탐색 )
head(adult_data)
glimpse(adult_data)  # 32,561 x 15
str(adult_data)  # 32,561 x 15
summary(adult_data)  # There's missing value exist

# missing value plot (https://njtierney.github.io/r/missing%20data/rbloggers/2015/12/01/ggplot-missing-data/)
adult_data_2 <- tibble(adult_data)
adult_data_2

adult_missing <- function(x){
  
  x %>% 
    is.na %>%
    melt %>%
    ggplot(data = .,
           aes(x = Var2,
               y = Var1)) +
    geom_raster(aes(fill = value)) +
    scale_fill_grey(name = "",
                    labels = c("Present","Missing")) +
    theme_minimal() + 
    theme(axis.text.x  = element_text(angle=45, vjust=0.5)) + 
    labs(x = "Variables in Dataset",
         y = "Rows / observations")
}

win.graph()
adult_missing(adult_data_2)  # There's missing value exist via graph

sapply(adult_data, function(x) sum(is.na(x)))
# workclass : 1836 missing values exist
# occupation : 1843 missing values exist
# native_country : 583 missing values exist


##############################
### Missing value handling ###
##############################

# missing value (NA)있는 관측값을 제거함
adult_data_NAdrop <- na.omit(adult_data)

head(adult_data_NAdrop)
glimpse(adult_data_NAdrop)  # 30,162 x 15
str(adult_data_NAdrop)  # 30,162 x 15
summary(adult_data_NAdrop)  # There's no missing value exist


# 변수들 분포 확인.
win.graph()
par(mfrow = c(2, 3))
vioplot(adult_data_NAdrop$age, main = "age")
vioplot(adult_data_NAdrop$fnlwgt, main = "fnlwgt")
vioplot(adult_data_NAdrop$education_num, main = "education_num")
vioplot(adult_data_NAdrop$capital_gain, main = "capital_gain")
vioplot(adult_data_NAdrop$capital_loss, main = "capital_loss")
vioplot(adult_data_NAdrop$hours_per_week, main = "hours_per_week")

# win.graph()
# barplot( prop.table( table(adult_data_NAdrop$workclass) ), ylim = c(0, 1))
# barplot( prop.table( table(adult_data_NAdrop$education) ), ylim = c(0, 1))
# barplot( prop.table( table(adult_data_NAdrop$marital_status) ), ylim = c(0, 1) )
# barplot( prop.table( table(adult_data_NAdrop$occupation) ), ylim = c(0, 1) )
# barplot( prop.table( table(adult_data_NAdrop$relationship) ), ylim = c(0, 1) )
# barplot( prop.table( table(adult_data_NAdrop$race) ), ylim = c(0, 1) )
# barplot( prop.table( table(adult_data_NAdrop$sex) ), ylim = c(0, 1) )
# barplot( prop.table( table(adult_data_NAdrop$native_country) ), ylim = c(0, 1) )
# 
# barplot( prop.table( table(adult_data_NAdrop$wage) ), ylim = c(0, 1) )

win.graph()
gg <- ggplot(data = adult_data_NAdrop, aes(x = workclass))
gg <- gg + geom_bar(aes(y = (..count..)/sum(..count..)))
gg + theme(axis.text.x = element_text(angle = 45, hjust = 1)) + ylab("Proportion")

gg <- ggplot(data = adult_data_NAdrop, aes(x = education))
gg <- gg + geom_bar(aes(y = (..count..)/sum(..count..)))
gg + theme(axis.text.x = element_text(angle = 45, hjust = 1)) + ylab("Proportion")

gg <- ggplot(data = adult_data_NAdrop, aes(x = marital_status))
gg <- gg + geom_bar(aes(y = (..count..)/sum(..count..)))
gg + theme(axis.text.x = element_text(angle = 45, hjust = 1)) + ylab("Proportion")

gg <- ggplot(data = adult_data_NAdrop, aes(x = occupation))
gg <- gg + geom_bar(aes(y = (..count..)/sum(..count..)))
gg + theme(axis.text.x = element_text(angle = 45, hjust = 1)) + ylab("Proportion")

gg <- ggplot(data = adult_data_NAdrop, aes(x = relationship))
gg <- gg + geom_bar(aes(y = (..count..)/sum(..count..)))
gg + theme(axis.text.x = element_text(angle = 45, hjust = 1)) + ylab("Proportion")

gg <- ggplot(data = adult_data_NAdrop, aes(x = race))
gg <- gg + geom_bar(aes(y = (..count..)/sum(..count..)))
gg + theme(axis.text.x = element_text(angle = 45, hjust = 1)) + ylab("Proportion")

gg <- ggplot(data = adult_data_NAdrop, aes(x = sex))
gg <- gg + geom_bar(aes(y = (..count..)/sum(..count..)))
gg + theme(axis.text.x = element_text(angle = 45, hjust = 1)) + ylab("Proportion")

gg <- ggplot(data = adult_data_NAdrop, aes(x = native_country))
gg <- gg + geom_bar(aes(y = (..count..)/sum(..count..)))
gg + theme(axis.text.x = element_text(angle = 45, hjust = 1)) + ylab("Proportion")

gg <- ggplot(data = adult_data_NAdrop, aes(x = wage))
gg <- gg + geom_bar(aes(y = (..count..)/sum(..count..)))
gg + theme(axis.text.x = element_text(angle = 45, hjust = 1)) + ylab("Proportion")

# CorrPlots
# 연속형 설명변수들 사이의 상관관계 확인.
library(corrplot)
win.graph()
corrplot(cor(select(adult_data_NAdrop, -c(wage, workclass, education, marital_status, 
                                          occupation, relationship, race, sex, native_country) ) ) )
corrplot(cor(select(adult_data_NAdrop, -c(wage, workclass, education, marital_status, 
                                          occupation, relationship, race, sex, native_country) ) ), method = "number" )

# 신경망 학습을 위한 범주형(factor) 변수 => 숫자형(numeric) 변수 변환 data set
# 반응변수 ( <=50K : 0, >50K : 1 )로 변환
levels(adult_data_NAdrop$wage)=c("0","1")

adult_data_NAdrop_num <- adult_data_NAdrop

adult_data_NAdrop_num$workclass <- as.numeric(adult_data_NAdrop_num$workclass)
adult_data_NAdrop_num$education <- as.numeric(adult_data_NAdrop_num$education)
adult_data_NAdrop_num$marital_status <- as.numeric(adult_data_NAdrop_num$marital_status)
adult_data_NAdrop_num$occupation <- as.numeric(adult_data_NAdrop_num$occupation)
adult_data_NAdrop_num$relationship <- as.numeric(adult_data_NAdrop_num$relationship)
adult_data_NAdrop_num$race <- as.numeric(adult_data_NAdrop_num$race)
adult_data_NAdrop_num$sex <- as.numeric(adult_data_NAdrop_num$sex)
adult_data_NAdrop_num$native_country <- as.numeric(adult_data_NAdrop_num$native_country)
adult_data_NAdrop_num$wage <- as.numeric(as.character(adult_data_NAdrop_num$wage))

str(adult_data_NAdrop_num)

# Train : Test = 8 : 2 나누기. (100번)
# seed 를 바꿔가며 100개의 서로 다른 train, test data set 생성.
# list() 객체에 저장.

adult_train_x_list <- list()
adult_train_y_list <- list()

adult_test_x_list <- list()
adult_test_y_list <- list()

for(i in 1:100){
  set.seed(i)
  train_index <- sample(1:nrow(adult_data_NAdrop), 0.8 * nrow(adult_data_NAdrop), replace = FALSE)
  adult_train_x <- adult_data_NAdrop[train_index, 1:14]
  adult_train_y <- adult_data_NAdrop[train_index, 15]
  
  adult_test_x <- adult_data_NAdrop[-train_index, 1:14]
  adult_test_y <- adult_data_NAdrop[-train_index, 15]
  
  adult_train_x_list[[i]] <- adult_train_x
  adult_train_y_list[[i]] <- adult_train_y
  
  adult_test_x_list[[i]] <- adult_test_x
  adult_test_y_list[[i]] <- adult_test_y
  
}

# 신경망 학습용 data set
# Train : Test = 8 : 2 나누기. (100번)
# seed 를 바꿔가며 100개의 서로 다른 train, test data set 생성.
# list() 객체에 저장.

adult_train_x_list_num <- list()
adult_train_y_list_num <- list()

adult_test_x_list_num <- list()
adult_test_y_list_num <- list()

for(i in 1:100){
  set.seed(i)
  train_index <- sample(1:nrow(adult_data_NAdrop_num), 0.8 * nrow(adult_data_NAdrop_num), replace = FALSE)
  adult_train_x_num <- adult_data_NAdrop_num[train_index, 1:14]
  adult_train_y_num <- adult_data_NAdrop_num[train_index, 15]
  
  adult_test_x_num <- adult_data_NAdrop_num[-train_index, 1:14]
  adult_test_y_num <- adult_data_NAdrop_num[-train_index, 15]
  
  adult_train_x_list_num[[i]] <- adult_train_x_num
  adult_train_y_list_num[[i]] <- adult_train_y_num
  
  adult_test_x_list_num[[i]] <- adult_test_x_num
  adult_test_y_list_num[[i]] <- adult_test_y_num
  
}


#################
## 신경망 학습 ##
#################
# https://mxnet.apache.org/api/r  참고.
library(mxnet)

adult_model_list <- list()
adult_dnn_test_predict_list <- list()
adult_dnn_test_predict_label_list <- list()
adult_dnn_test_confusion_list <- list()
adult_dnn_test_acc_list <- list() # y_hat 과 test data set의 y 로 Accuracy 저장 list 할당.
adult_dnn_exectime_list <- list()
adult_dnn_train_logger_list <- list()
adult_dnn_test_logger_list <- list()
adult_dnn_test_auc_list <- list()

for(i in 1:100){ # 1:100
  adult_train_x_scale <- scale(adult_train_x_list_num[[i]])
  adult_test_x_scale <- scale(adult_test_x_list_num[[i]])
  
  adult_train_x_datamatrix <- data.matrix(adult_train_x_scale)
  adult_test_x_datamatrix <- data.matrix(adult_test_x_scale)
  
  adult_train_y <- adult_train_y_list_num[[i]]
  adult_test_y <- adult_test_y_list_num[[i]]
  
  mx.set.seed(2020)  # 가중치 초기값 고정.
  
  input <- mx.symbol.Variable("data")
  fc1 <- mx.symbol.FullyConnected(data = input, num.hidden = 5)  # Hyper parameter : 은닉노드 수
  act1 <- mx.symbol.Activation(data = fc1, act_type = "sigmoid")  # Hyper parameter : 활성화 함수 종류
  fc2 <- mx.symbol.FullyConnected(data = act1, num.hidden = 2)
  lro <- mx.symbol.SoftmaxOutput(data = fc2)  # Hyper parameter : 출력노드 함수 종류
  
  tic()
  
  logger <- mx.metric.logger$new()
  model <- mx.model.FeedForward.create(symbol = lro, X = adult_train_x_datamatrix, y = adult_train_y,
                                       eval.data = list(data = adult_test_x_datamatrix, label = adult_test_y),
                                       ctx = mx.gpu(), num.round = 100, optimizer = 'sgd', # Hyper parameter : iter 수, 최적화 함수 종류
                                       array.batch.size = 20, learning.rate = 0.001, momentum = 0.9, eval.metric = mx.metric.accuracy,
                                       verbose = T, epoch.end.callback = mx.callback.log.train.metric(1, logger))  # Hyper parameter : 배치 사이즈, 학습률, 모멘텀 값.
  
  exectime <- toc()
  exectime <- round(exectime$toc - exectime$tic, 5)
  
  adult_model_list[[i]] <- model
  adult_dnn_test_predict_list[[i]] <- predict(adult_model_list[[i]], adult_test_x_datamatrix)
  adult_dnn_test_predict_label_list[[i]] <- max.col(t(adult_dnn_test_predict_list[[i]])) - 1
  adult_dnn_test_confusion_list[[i]] <- CrossTable(x = adult_test_y_list_num[[i]], y = adult_dnn_test_predict_label_list[[i]])
  adult_dnn_test_acc_list[[i]] <- (adult_dnn_test_confusion_list[[i]]$t[1] + adult_dnn_test_confusion_list[[i]]$t[4])  / sum(adult_dnn_test_confusion_list[[i]]$t)
  auc <- performance(prediction( predict(adult_model_list[[i]], adult_test_x_datamatrix)[2, ], adult_test_y_list_num[[i]], label.ordering = c(0, 1)), measure = "auc")
  adult_dnn_test_auc_list[[i]] <- auc@y.values[[1]]
  
  adult_dnn_exectime_list[[i]] <- exectime 
  adult_dnn_train_logger_list[[i]] <- logger$train
  adult_dnn_test_logger_list[[i]] <- logger$eval
}

adult_dnn_test_acc_unlist <- unlist(adult_dnn_test_acc_list)
adult_dnn_test_auc_unlist <- unlist(adult_dnn_test_auc_list)
adult_dnn_exectime_unlist <- unlist(adult_dnn_exectime_list)

adult_dnn_train_logger_unlist <- data.frame( matrix(unlist(adult_dnn_train_logger_list), ncol = 100))
adult_dnn_test_logger_unlist <- data.frame( matrix(unlist(adult_dnn_test_logger_list), ncol = 100))

# 결과저장
# write(t(adult_dnn_test_acc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Adult_DNN_result.txt")
# write(t(adult_dnn_test_auc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/adult_DNN_auc_result.txt")
# write(t(adult_dnn_exectime_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Adult_DNN_time.txt")
# write(t(adult_dnn_train_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/adult_DNN_train_logger.txt")
# write(t(adult_dnn_test_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/adult_DNN_test_logger.txt")
adult_dnn_test_confusion_unlist <- data.frame()
for(i in 1:100){
  for(j in 1:4){
    adult_dnn_test_confusion_unlist[i, j] <- unlist(adult_dnn_test_confusion_list[[i]]$t)[j]
  }
}
adult_dnn_test_confusion_unlist <- cbind(c(1:100), adult_dnn_test_confusion_unlist)
colnames(adult_dnn_test_confusion_unlist) <- c("iter","TN", "FN", "FP", "TP")
head(adult_dnn_test_confusion_unlist)
tail(adult_dnn_test_confusion_unlist)

# write.table(adult_dnn_test_confusion_unlist, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/adult_dnn_confusion.txt", col.names = TRUE)


adult_dnn_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Adult_DNN_result.txt")
adult_dnn_test_acc_unlist <- cbind(adult_dnn_test_acc_unlist, rep("DNN(변수전체)", 100))
names(adult_dnn_test_acc_unlist) <- c("Accuracy", "Model")

adult_dnn_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Adult_DNN_auc_result.txt")
adult_dnn_test_auc_unlist <- cbind(adult_dnn_test_auc_unlist, rep("DNN(변수전체)", 100))
names(adult_dnn_test_auc_unlist) <- c("AUC", "Model")

adult_dnn_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Adult_DNN_time.txt")
adult_dnn_exectime_unlist <- cbind(adult_dnn_exectime_unlist, rep("DNN(변수전체)", 100))
names(adult_dnn_exectime_unlist) <- c("Time", "Model")

# Accuracy 분포 확인.
summary(adult_dnn_test_acc_unlist)

win.graph()
boxplot(adult_dnn_test_acc_unlist[1],  main = "Adult data. DNN 은닉층 1, 은닉노드 5. (전체변수)",
        xlab = "Iter = 100", ylab = "Accuracy", ylim = c(0.5, 1))
vioplot(adult_dnn_test_acc_unlist[1], main = "Adult data. DNN 은닉층 1, 은닉노드 5. (전체변수)",
        xlab = "Iter = 100", ylab = "Accuracy", ylim = c(0.5, 1))
points(mean(adult_dnn_test_acc_unlist$Accuracy), col = "red", pch = 17) # mean 표시

# AUC 분포 확인.
summary(adult_dnn_test_auc_unlist)

win.graph()
boxplot(adult_dnn_test_auc_unlist[1],  main = "Adult data. DNN 은닉층 1, 은닉노드 5. (전체변수)",
        xlab = "Iter = 100", ylab = "AUC", ylim = c(0.5, 1))
vioplot(adult_dnn_test_auc_unlist[1], main = "Adult data. DNN 은닉층 1, 은닉노드 5. (전체변수)",
        xlab = "Iter = 100", ylab = "AUC", ylim = c(0.5, 1))
points(mean(adult_dnn_test_auc_unlist$AUC), col = "red", pch = 17) # mean 표시


# Time 분포 확인.
summary(adult_dnn_exectime_unlist)

win.graph()
boxplot(adult_dnn_exectime_unlist[1],  main = "Adult data. DNN 은닉층 1, 은닉노드 5. (전체변수) 학습시간",
        xlab = "Iter = 100", ylab = "Time(단위 : 초)", ylim = c(100, 130))
vioplot(adult_dnn_exectime_unlist[1], main = "Adult data. DNN 은닉층 1, 은닉노드 5. (전체변수) 학습시간",
        xlab = "Iter = 100", ylab = "Time(단위 : 초)", ylim = c(100, 130))
points(mean(adult_dnn_exectime_unlist$Time), col = "red", pch = 17) # mean 표시


# 2 Adult data  (LASSO 변수선택) 적합 후 Accuracy 분포 확인 (서로다른 100개 data set)
# 2 Adult data  (LASSO 변수선택) 적합 후 Accuracy 분포 확인 (서로다른 100개 data set)
# 2 Adult data  (LASSO 변수선택) 적합 후 Accuracy 분포 확인 (서로다른 100개 data set)
# (Note, LASSO 이용 변수선택)


# LASSO 10-fold CV 방법으로 변수선택 (lambda 1se)

adult_lasso_fit_10fold <- cv.glmnet(model.matrix( ~. -1, adult_train_x_list[[1]]), adult_train_y_list[[1]], type.measure = "auc",
                                    family = "binomial", alpha = 1)  # lambda.1se는 Standard error가 가장 Regularized 된 모델이 되는 람다값을 찾아줌.


win.graph()
plot(adult_lasso_fit_10fold, main = "Adult data (LASSO)")

# LASSO lambda.1se 에서 선택된 변수들 추정회귀계수
adult_lasso_coef <- predict(adult_lasso_fit_10fold, type = "coefficients",
                           s = adult_lasso_fit_10fold$lambda.1se)
adult_lasso_coef

# 범수형 변수 -> 수치형 변수 변환 후
adult_lasso_fit_10fold <- cv.glmnet(model.matrix( ~. -1, adult_train_x_list_num[[1]]), adult_train_y_list_num[[1]], type.measure = "auc",
                                    family = "binomial", alpha = 1)  # lambda.1se는 Standard error가 가장 Regularized 된 모델이 되는 람다값을 찾아줌.

win.graph()
plot(adult_lasso_fit_10fold, main = "Adult data (LASSO)")

# LASSO lambda.1se 에서 선택된 변수들 추정회귀계수
adult_lasso_coef <- predict(adult_lasso_fit_10fold, type = "coefficients",
                            s = adult_lasso_fit_10fold$lambda.1se)
adult_lasso_coef


# adult_lasso_coef_min <- predict(adult_lasso_fit_10fold, type = "coefficients",
#                             s = adult_lasso_fit_10fold$lambda.min)
# adult_lasso_coef_min

############################
## 신경망 학습 (변수선택) ##
############################
# https://mxnet.apache.org/api/r  참고.
# Train : Test = 8 : 2 나누기. (100번)
# seed 를 바꿔가며 100개의 서로 다른 train, test data set 생성.
# list() 객체에 저장.

adult_train_x_list_num <- list()
adult_train_y_list_num <- list()

adult_test_x_list_num <- list()
adult_test_y_list_num <- list()

for(i in 1:100){
  set.seed(i)
  train_index <- sample(1:nrow(adult_data_NAdrop_num), 0.8 * nrow(adult_data_NAdrop_num), replace = FALSE)
  adult_train_x_num <- adult_data_NAdrop_num[train_index, -c(3, 4, 7, 14)]
  adult_train_y_num <- adult_data_NAdrop_num[train_index, 15]
  
  adult_test_x_num <- adult_data_NAdrop_num[-train_index, -c(3, 4, 7, 14)]
  adult_test_y_num <- adult_data_NAdrop_num[-train_index, 15]
  
  adult_train_x_list_num[[i]] <- adult_train_x_num
  adult_train_y_list_num[[i]] <- adult_train_y_num
  
  adult_test_x_list_num[[i]] <- adult_test_x_num
  adult_test_y_list_num[[i]] <- adult_test_y_num
  
}


#################
## 신경망 학습 ##
#################
# https://mxnet.apache.org/api/r  참고.
library(mxnet)

adult_model_selec_list <- list()
adult_dnn_selec_test_predict_list <- list()
adult_dnn_selec_test_predict_label_list <- list()
adult_dnn_selec_test_confusion_list <- list()
adult_dnn_selec_test_acc_list <- list() # y_hat 과 test data set의 y 로 Accuracy 저장 list 할당.
adult_dnn_selec_exectime_list <- list()
adult_dnn_selec_train_logger_list <- list()
adult_dnn_selec_test_logger_list <- list()
adult_dnn_selec_test_auc_list <- list()

for(i in 1:100){ # 1:100
  adult_train_x_scale <- scale(adult_train_x_list_num[[i]])
  adult_test_x_scale <- scale(adult_test_x_list_num[[i]])
  
  adult_train_x_datamatrix <- data.matrix(adult_train_x_scale)
  adult_test_x_datamatrix <- data.matrix(adult_test_x_scale)
  
  adult_train_y <- adult_train_y_list_num[[i]]
  adult_test_y <- adult_test_y_list_num[[i]]
  
  mx.set.seed(2020)  # 가중치 초기값 고정.
  
  input <- mx.symbol.Variable("data")
  fc1 <- mx.symbol.FullyConnected(data = input, num.hidden = 5)  # Hyper parameter : 은닉노드 수
  act1 <- mx.symbol.Activation(data = fc1, act_type = "sigmoid")  # Hyper parameter : 활성화 함수 종류
  fc2 <- mx.symbol.FullyConnected(data = act1, num.hidden = 2)
  lro <- mx.symbol.SoftmaxOutput(data = fc2)  # Hyper parameter : 출력노드 함수 종류
  
  tic()
  
  logger <- mx.metric.logger$new()
  model <- mx.model.FeedForward.create(symbol = lro, X = adult_train_x_datamatrix, y = adult_train_y,
                                       eval.data = list(data = adult_test_x_datamatrix, label = adult_test_y),
                                       ctx = mx.gpu(), num.round = 5, optimizer = 'sgd', # Hyper parameter : iter 수, 최적화 함수 종류
                                       array.batch.size = 20, learning.rate = 0.001, momentum = 0.9, eval.metric = mx.metric.accuracy,
                                       verbose = T, epoch.end.callback = mx.callback.log.train.metric(1, logger))  # Hyper parameter : 배치 사이즈, 학습률, 모멘텀 값.
  
  exectime <- toc()
  exectime <- round(exectime$toc - exectime$tic, 5)
  
  adult_model_selec_list[[i]] <- model
  adult_dnn_selec_test_predict_list[[i]] <- predict(adult_model_selec_list[[i]], adult_test_x_datamatrix)
  adult_dnn_selec_test_predict_label_list[[i]] <- max.col(t(adult_dnn_selec_test_predict_list[[i]])) - 1
  adult_dnn_selec_test_confusion_list[[i]] <- CrossTable(x = adult_test_y_list_num[[i]], y = adult_dnn_selec_test_predict_label_list[[i]])
  adult_dnn_selec_test_acc_list[[i]] <- (adult_dnn_selec_test_confusion_list[[i]]$t[1] + adult_dnn_selec_test_confusion_list[[i]]$t[4])  / sum(adult_dnn_selec_test_confusion_list[[i]]$t)
  auc <- performance(prediction( predict(adult_model_selec_list[[i]], adult_test_x_datamatrix)[2, ], adult_test_y_list_num[[i]], label.ordering = c(0, 1)), measure = "auc")
  adult_dnn_selec_test_auc_list[[i]] <- auc@y.values[[1]]
  
  adult_dnn_selec_exectime_list[[i]] <- exectime
  adult_dnn_selec_train_logger_list[[i]] <- logger$train
  adult_dnn_selec_test_logger_list[[i]] <- logger$eval
}

adult_dnn_selec_test_acc_unlist <- unlist(adult_dnn_selec_test_acc_list)
adult_dnn_selec_test_auc_unlist <- unlist(adult_dnn_selec_test_auc_list)
adult_dnn_selec_exectime_unlist <- unlist(adult_dnn_selec_exectime_list)

adult_dnn_selec_train_logger_unlist <- data.frame( matrix(unlist(adult_dnn_selec_train_logger_list), ncol = 100))
adult_dnn_selec_test_logger_unlist <- data.frame( matrix(unlist(adult_dnn_selec_test_logger_list), ncol = 100))

# 결과저장
# write(t(adult_dnn_selec_test_acc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Adult_DNN_selec_result.txt")
# write(t(adult_dnn_selec_test_auc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/adult_DNN_selec_auc_result.txt")
# write(t(adult_dnn_selec_exectime_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Adult_DNN_selec_time.txt")
# write(t(adult_dnn_selec_train_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/adult_DNN_selec_train_logger.txt")
# write(t(adult_dnn_selec_test_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/adult_DNN_selec_test_logger.txt")
adult_dnn_selec_test_confusion_unlist <- data.frame()
for(i in 1:100){
  for(j in 1:4){
    adult_dnn_selec_test_confusion_unlist[i, j] <- unlist(adult_dnn_selec_test_confusion_list[[i]]$t)[j]
  }
}
adult_dnn_selec_test_confusion_unlist <- cbind(c(1:100), adult_dnn_selec_test_confusion_unlist)
colnames(adult_dnn_selec_test_confusion_unlist) <- c("iter","TN", "FN", "FP", "TP")
head(adult_dnn_selec_test_confusion_unlist)
tail(adult_dnn_selec_test_confusion_unlist)

# write.table(adult_dnn_selec_test_confusion_unlist, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/adult_dnn_selec_confusion.txt", col.names = TRUE)


adult_dnn_selec_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Adult_DNN_selec_result.txt")
adult_dnn_selec_test_acc_unlist <- cbind(adult_dnn_selec_test_acc_unlist, rep("DNN(LASSO 변수선택)", 100))
names(adult_dnn_selec_test_acc_unlist) <- c("Accuracy", "Model")

adult_dnn_selec_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Adult_DNN_selec_auc_result.txt")
adult_dnn_selec_test_auc_unlist <- cbind(adult_dnn_selec_test_auc_unlist, rep("DNN(LASSO 변수선택)", 100))
names(adult_dnn_selec_test_auc_unlist) <- c("AUC", "Model")

adult_dnn_selec_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Adult_DNN_selec_time.txt")
adult_dnn_selec_exectime_unlist <- cbind(adult_dnn_selec_exectime_unlist, rep("DNN(LASSO 변수선택)", 100))
names(adult_dnn_selec_exectime_unlist) <- c("Time", "Model")

# Accuracy 분포 확인.
summary(adult_dnn_selec_test_acc_unlist)

win.graph()
boxplot(adult_dnn_selec_test_acc_unlist[1],  main = "Adult data. DNN 은닉층 1, 은닉노드 5. (LASSO 변수선택)",
        xlab = "Iter = 5", ylab = "Accuracy", ylim = c(0.5, 1))
vioplot(adult_dnn_selec_test_acc_unlist[1], main = "Adult data. DNN 은닉층 1, 은닉노드 5. (LASSO 변수선택)",
        xlab = "Iter = 5", ylab = "Accuracy", ylim = c(0.5, 1))
points(mean(adult_dnn_selec_test_acc_unlist$Accuracy), col = "red", pch = 17) # mean 표시

# AUC 분포 확인.
summary(adult_dnn_selec_test_auc_unlist)

win.graph()
boxplot(adult_dnn_selec_test_auc_unlist[1],  main = "Adult data. DNN 은닉층 1, 은닉노드 5. (LASSO 변수선택)",
        xlab = "Iter = 5", ylab = "AUC", ylim = c(0.5, 1))
vioplot(adult_dnn_selec_test_auc_unlist[1], main = "Adult data. DNN 은닉층 1, 은닉노드 5. (LASSO 변수선택)",
        xlab = "Iter = 5", ylab = "AUC", ylim = c(0.5, 1))
points(mean(adult_dnn_selec_test_auc_unlist$AUC), col = "red", pch = 17) # mean 표시

# Time 분포 확인.
summary(adult_dnn_selec_exectime_unlist)

win.graph()
boxplot(adult_dnn_selec_exectime_unlist[1],  main = "Adult data. DNN 은닉층 1, 은닉노드 5. (LASSO 변수선택) 학습시간",
        xlab = "Iter = 5", ylab = "Time(단위 : 초)", ylim = c(0, 10))
vioplot(adult_dnn_selec_exectime_unlist[1], main = "Adult data. DNN 은닉층 1, 은닉노드 5. (LASSO 변수선택) 학습시간",
        xlab = "Iter = 5", ylab = "Time(단위 : 초)", ylim = c(0, 10))
points(mean(adult_dnn_selec_exectime_unlist$Time), col = "red", pch = 17) # mean 표시








# 3. Adult data Stepwise 변수선택 후 DNN 모형 학습
# 3. Adult data Stepwise 변수선택 후 DNN 모형 학습
# 3. Adult data Stepwise 변수선택 후 DNN 모형 학습

adult_glm_model <- glm(wage ~ ., data = adult_data_NAdrop, family = "binomial"(link = "logit"))
adult_glm_step_model <- step(adult_glm_model, scope = list(upper = adult_glm_model), direction = "both")
summary(adult_glm_step_model)  # education_num 변수 제외한 모든 변수 선택


############################
## 신경망 학습 (변수선택) ##
############################
# https://mxnet.apache.org/api/r  참고.
# Train : Test = 8 : 2 나누기. (100번)
# seed 를 바꿔가며 100개의 서로 다른 train, test data set 생성.
# list() 객체에 저장.

adult_train_x_list_num <- list()
adult_train_y_list_num <- list()

adult_test_x_list_num <- list()
adult_test_y_list_num <- list()

for(i in 1:100){
  set.seed(i)
  train_index <- sample(1:nrow(adult_data_NAdrop_num), 0.8 * nrow(adult_data_NAdrop_num), replace = FALSE)
  adult_train_x_num <- adult_data_NAdrop_num[train_index, -c(5)]
  adult_train_y_num <- adult_data_NAdrop_num[train_index, 15]
  
  adult_test_x_num <- adult_data_NAdrop_num[-train_index, -c(5)]
  adult_test_y_num <- adult_data_NAdrop_num[-train_index, 15]
  
  adult_train_x_list_num[[i]] <- adult_train_x_num
  adult_train_y_list_num[[i]] <- adult_train_y_num
  
  adult_test_x_list_num[[i]] <- adult_test_x_num
  adult_test_y_list_num[[i]] <- adult_test_y_num
  
}


#################
## 신경망 학습 ##
#################
# https://mxnet.apache.org/api/r  참고.
library(mxnet)

adult_model_stepwise_list <- list()
adult_dnn_stepwise_test_predict_list <- list()
adult_dnn_stepwise_test_predict_label_list <- list()
adult_dnn_stepwise_test_confusion_list <- list()
adult_dnn_stepwise_test_acc_list <- list() # y_hat 과 test data set의 y 로 Accuracy 저장 list 할당.
adult_dnn_stepwise_exectime_list <- list()
adult_dnn_stepwise_train_logger_list <- list()
adult_dnn_stepwise_test_logger_list <- list()
adult_dnn_stepwise_test_auc_list <- list()

for(i in 1:100){ # 1:100
  adult_train_x_scale <- scale(adult_train_x_list_num[[i]])
  adult_test_x_scale <- scale(adult_test_x_list_num[[i]])
  
  adult_train_x_datamatrix <- data.matrix(adult_train_x_scale)
  adult_test_x_datamatrix <- data.matrix(adult_test_x_scale)
  
  adult_train_y <- adult_train_y_list_num[[i]]
  adult_test_y <- adult_test_y_list_num[[i]]
  
  mx.set.seed(2020)  # 가중치 초기값 고정.
  
  input <- mx.symbol.Variable("data")
  fc1 <- mx.symbol.FullyConnected(data = input, num.hidden = 5)  # Hyper parameter : 은닉노드 수
  act1 <- mx.symbol.Activation(data = fc1, act_type = "sigmoid")  # Hyper parameter : 활성화 함수 종류
  fc2 <- mx.symbol.FullyConnected(data = act1, num.hidden = 2)
  lro <- mx.symbol.SoftmaxOutput(data = fc2)  # Hyper parameter : 출력노드 함수 종류
  
  tic()
  
  logger <- mx.metric.logger$new()
  model <- mx.model.FeedForward.create(symbol = lro, X = adult_train_x_datamatrix, y = adult_train_y,
                                       eval.data = list(data = adult_test_x_datamatrix, label = adult_test_y),
                                       ctx = mx.gpu(), num.round = 5, optimizer = 'sgd', # Hyper parameter : iter 수, 최적화 함수 종류
                                       array.batch.size = 20, learning.rate = 0.001, momentum = 0.9, eval.metric = mx.metric.accuracy,
                                       verbose = T, epoch.end.callback = mx.callback.log.train.metric(1, logger))  # Hyper parameter : 배치 사이즈, 학습률, 모멘텀 값.
  
  exectime <- toc()
  exectime <- round(exectime$toc - exectime$tic, 5)
  
  adult_model_stepwise_list[[i]] <- model
  adult_dnn_stepwise_test_predict_list[[i]] <- predict(adult_model_stepwise_list[[i]], adult_test_x_datamatrix)
  adult_dnn_stepwise_test_predict_label_list[[i]] <- max.col(t(adult_dnn_stepwise_test_predict_list[[i]])) - 1
  adult_dnn_stepwise_test_confusion_list[[i]] <- CrossTable(x = adult_test_y_list_num[[i]], y = adult_dnn_stepwise_test_predict_label_list[[i]])
  adult_dnn_stepwise_test_acc_list[[i]] <- (adult_dnn_stepwise_test_confusion_list[[i]]$t[1] + adult_dnn_stepwise_test_confusion_list[[i]]$t[4])  / sum(adult_dnn_stepwise_test_confusion_list[[i]]$t)
  auc <- performance(prediction( predict(adult_model_stepwise_list[[i]], adult_test_x_datamatrix)[2, ], adult_test_y_list_num[[i]], label.ordering = c(0, 1)), measure = "auc")
  adult_dnn_stepwise_test_auc_list[[i]] <- auc@y.values[[1]]
  
  adult_dnn_stepwise_exectime_list[[i]] <- exectime
  adult_dnn_stepwise_train_logger_list[[i]] <- logger$train
  adult_dnn_stepwise_test_logger_list[[i]] <- logger$eval
}

adult_dnn_stepwise_test_acc_unlist <- unlist(adult_dnn_stepwise_test_acc_list)
adult_dnn_stepwise_test_auc_unlist <- unlist(adult_dnn_stepwise_test_auc_list)
adult_dnn_stepwise_exectime_unlist <- unlist(adult_dnn_stepwise_exectime_list)

adult_dnn_stepwise_train_logger_unlist <- data.frame( matrix(unlist(adult_dnn_stepwise_train_logger_list), ncol = 100))
adult_dnn_stepwise_test_logger_unlist <- data.frame( matrix(unlist(adult_dnn_stepwise_test_logger_list), ncol = 100))


# 결과저장
# write(t(adult_dnn_stepwise_test_acc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Adult_DNN_stepwise_result.txt")
# write(t(adult_dnn_stepwise_test_auc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/adult_DNN_stepwise_auc_result.txt")
# write(t(adult_dnn_stepwise_exectime_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Adult_DNN_stepwise_time.txt")
# write(t(adult_dnn_stepwise_train_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/adult_DNN_stepwise_train_logger.txt")
# write(t(adult_dnn_stepwise_test_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/adult_DNN_stepwise_test_logger.txt")
adult_dnn_stepwise_test_confusion_unlist <- data.frame()
for(i in 1:100){
  for(j in 1:4){
    adult_dnn_stepwise_test_confusion_unlist[i, j] <- unlist(adult_dnn_stepwise_test_confusion_list[[i]]$t)[j]
  }
}
adult_dnn_stepwise_test_confusion_unlist <- cbind(c(1:100), adult_dnn_stepwise_test_confusion_unlist)
colnames(adult_dnn_stepwise_test_confusion_unlist) <- c("iter","TN", "FN", "FP", "TP")
head(adult_dnn_stepwise_test_confusion_unlist)
tail(adult_dnn_stepwise_test_confusion_unlist)

# write.table(adult_dnn_stepwise_test_confusion_unlist, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/adult_dnn_stepwise_confusion.txt", col.names = TRUE)


adult_dnn_stepwise_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Adult_DNN_stepwise_result.txt")
adult_dnn_stepwise_test_acc_unlist <- cbind(adult_dnn_stepwise_test_acc_unlist, rep("DNN(Stepwise 변수선택)", 100))
names(adult_dnn_stepwise_test_acc_unlist) <- c("Accuracy", "Model")

adult_dnn_stepwise_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Adult_DNN_stepwise_auc_result.txt")
adult_dnn_stepwise_test_auc_unlist <- cbind(adult_dnn_stepwise_test_auc_unlist, rep("DNN(Stepwise 변수선택)", 100))
names(adult_dnn_stepwise_test_auc_unlist) <- c("AUC", "Model")

adult_dnn_stepwise_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Adult_DNN_stepwise_time.txt")
adult_dnn_stepwise_exectime_unlist <- cbind(adult_dnn_stepwise_exectime_unlist, rep("DNN(Stepwise 변수선택택)", 100))
names(adult_dnn_stepwise_exectime_unlist) <- c("Time", "Model")

# Accuracy 분포 확인.
summary(adult_dnn_stepwise_test_acc_unlist)

win.graph()
boxplot(adult_dnn_stepwise_test_acc_unlist[1],  main = "Adult data. DNN 은닉층 1, 은닉노드 5. (Stepwise 변수선택)",
        xlab = "Iter = 5", ylab = "Accuracy", ylim = c(0.5, 1))
vioplot(adult_dnn_stepwise_test_acc_unlist[1], main = "Adult data. DNN 은닉층 1, 은닉노드 5. (Stepwise 변수선택)",
        xlab = "Iter = 5", ylab = "Accuracy", ylim = c(0.5, 1))
points(mean(adult_dnn_stepwise_test_acc_unlist$Accuracy), col = "red", pch = 17) # mean 표시

# AUC 분포 확인.
summary(adult_dnn_stepwise_test_auc_unlist)

win.graph()
boxplot(adult_dnn_stepwise_test_auc_unlist[1],  main = "Adult data. DNN 은닉층 1, 은닉노드 5. (Stepwise 변수선택)",
        xlab = "Iter = 5", ylab = "AUC", ylim = c(0.5, 1))
vioplot(adult_dnn_stepwise_test_auc_unlist[1], main = "Adult data. DNN 은닉층 1, 은닉노드 5. (Stepwise 변수선택)",
        xlab = "Iter = 5", ylab = "AUC", ylim = c(0.5, 1))
points(mean(adult_dnn_stepwise_test_auc_unlist$AUC), col = "red", pch = 17) # mean 표시

# Time 분포 확인.
summary(adult_dnn_stepwise_exectime_unlist)

win.graph()
boxplot(adult_dnn_stepwise_exectime_unlist[1],  main = "Adult data. DNN 은닉층 1, 은닉노드 5. (Stepwise 변수선택) 학습시간",
        xlab = "Iter = 5", ylab = "Time(단위 : 초)", ylim = c(0, 10))
vioplot(adult_dnn_stepwise_exectime_unlist[1], main = "Adult data. DNN 은닉층 1, 은닉노드 5. (Stepwise 변수선택) 학습시간",
        xlab = "Iter = 5", ylab = "Time(단위 : 초)", ylim = c(0, 10))
points(mean(adult_dnn_stepwise_exectime_unlist$Time), col = "red", pch = 17) # mean 표시





