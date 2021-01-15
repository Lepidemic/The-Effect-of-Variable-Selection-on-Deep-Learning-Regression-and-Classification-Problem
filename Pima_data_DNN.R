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

# Pima data
# The Pima data frame has 768 rows and 9 columns.
# https://rdrr.io/cran/mlbench/man/PimaIndiansDiabetes.html
###### 변수 설명 ######
# pregnant
# Number of times pregnant
#
# glucose
# Plasma glucose concentration (glucose tolerance test)
#
# pressure
# Diastolic blood pressure (mm Hg)
#
# triceps
# Triceps skin fold thickness (mm)
#
# insulin
# 2-Hour serum insulin (mu U/ml)
#
# mass
# Body mass index (weight in kg/(height in m)\^2)
#
# pedigree
# Diabetes pedigree function
#
# age
# Age (years)
#
# diabetes
# Class variable (test for diabetes)

data("PimaIndiansDiabetes2",package="mlbench")
pima_data <- PimaIndiansDiabetes2

# Pima data structure exploration ( 데이터 구조 탐색 )
head(pima_data)
glimpse(pima_data)  # 768 x 9
str(pima_data)  # 768 x 9
summary(pima_data)  # There's missing value exist

# missing value plot (https://njtierney.github.io/r/missing%20data/rbloggers/2015/12/01/ggplot-missing-data/)
pima_data_2 <- tibble(PimaIndiansDiabetes2)
pima_data_2

pima_missing <- function(x){
  
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
pima_missing(pima_data_2)  # There's missing value exist via graph

##############################
### Missing value handling ###
##############################
# 변수 insulin: missing value (NA)가 너무 많음 (768개 중 374개) => 설명변수에서 제거함
nrow(pima_data) - length(na.omit(pima_data$insulin))  # 374

# 변수 triceps: missing value (NA)가 너무 많음 (768개 중 227개) => 설명변수에서 제거함
nrow(pima_data) - length(na.omit(pima_data$triceps))  # 227

pima_data_NAdrop <- pima_data[, -c(4, 5)]
# 그외 missing value (NA)있는 관측값을 제거함
pima_data_NAdrop <- na.omit(pima_data_NAdrop)

# missing value 처리 후 data set
head(pima_data_NAdrop)
glimpse(pima_data_NAdrop)  # 724 x 7
str(pima_data_NAdrop)  # 724 x 7
summary(pima_data_NAdrop)  # There's no missing value

# 반응변수 diabetes (neg = 0, pos = 1) 코딩
levels(pima_data_NAdrop$diabetes) <- c("0","1")
pima_data_NAdrop$diabetes <- as.numeric(as.character(pima_data_NAdrop$diabetes))
pima_data_NAdrop$diabetes

table(pima_data_NAdrop$diabetes)

# CorrPlots
library(corrplot)
corrplot(cor(select(pima_data_NAdrop, -diabetes) ) )
corrplot(cor(select(pima_data_NAdrop, -diabetes) ), method = "number")  # 설명변수들 사이의 상관관계 확인.

# Train : Test = 8 : 2 나누기. (100번)
# seed 를 바꿔가며 1000개의 서로 다른 train, test data set 생성.
# list() 객체에 저장.

pima_train_x_list <- list()
pima_train_y_list <- list()

pima_test_x_list <- list()
pima_test_y_list <- list()

for(i in 1:100){
  set.seed(i)
  train_index <- sample(1:nrow(pima_data_NAdrop), 0.8 * nrow(pima_data_NAdrop), replace = FALSE)
  pima_train_x <- pima_data_NAdrop[train_index, 1:6]
  pima_train_y <- pima_data_NAdrop[train_index, 7]
  
  pima_test_x <- pima_data_NAdrop[-train_index, 1:6]
  pima_test_y <- pima_data_NAdrop[-train_index, 7]
  
  pima_train_x_list[[i]] <- pima_train_x
  pima_train_y_list[[i]] <- pima_train_y
  
  pima_test_x_list[[i]] <- pima_test_x
  pima_test_y_list[[i]] <- pima_test_y
  
}

#################
## 신경망 학습 ##
#################
# https://mxnet.apache.org/api/r  참고.
library(mxnet)

pima_model_list <- list()
pima_dnn_test_predict_list <- list()
pima_dnn_test_predict_label_list <- list()
pima_dnn_test_confusion_list <- list()
pima_dnn_test_acc_list <- list() # y_hat 과 test data set의 y 로 Accuracy 저장 list 할당.
pima_dnn_exectime_list <- list()
pima_dnn_train_logger_list <- list()
pima_dnn_test_logger_list <- list()
pima_dnn_test_auc_list <- list()

for(i in 1:100){ # 1:100
  pima_train_x_scale <- scale(pima_train_x_list[[i]])
  pima_test_x_scale <- scale(pima_test_x_list[[i]])
  
  pima_train_x_datamatrix <- data.matrix(pima_train_x_scale)
  pima_test_x_datamatrix <- data.matrix(pima_test_x_scale)
  
  pima_train_y <- pima_train_y_list[[i]]
  pima_test_y <- pima_test_y_list[[i]]
  
  mx.set.seed(2020)  # 가중치 초기값 고정.
  
  input <- mx.symbol.Variable("data")
  fc1 <- mx.symbol.FullyConnected(data = input, num.hidden = 5)  # H    yper parameter : 은닉노드 수
  act1 <- mx.symbol.Activation(data = fc1, act_type = "sigmoid")  # Hyper parameter : 활성화 함수 종류
  fc2 <- mx.symbol.FullyConnected(data = act1, num.hidden = 2)
  lro <- mx.symbol.SoftmaxOutput(data = fc2)  # Hyper parameter : 출력노드 함수 종류
  
  tic()
  
  logger <- mx.metric.logger$new()
  model <- mx.model.FeedForward.create(symbol = lro, X = pima_train_x_datamatrix, y = pima_train_y,
                                       eval.data = list(data = pima_test_x_datamatrix, label = pima_test_y),
                                       ctx = mx.gpu(), num.round = 100, optimizer = 'sgd', # Hyper parameter : iter 수, 최적화 함수 종류
                                       array.batch.size = 10, learning.rate = 0.001, momentum = 0.9, eval.metric = mx.metric.accuracy,
                                       verbose = T, epoch.end.callback = mx.callback.log.train.metric(1, logger))  # Hyper parameter : 배치 사이즈, 학습률, 모멘텀 값.
  
  exectime <- toc()
  exectime <- round(exectime$toc - exectime$tic, 5)
  
  pima_model_list[[i]] <- model
  pima_dnn_test_predict_list[[i]] <- predict(pima_model_list[[i]], pima_test_x_datamatrix)
  pima_dnn_test_predict_label_list[[i]] <- max.col(t(pima_dnn_test_predict_list[[i]])) - 1
  pima_dnn_test_confusion_list[[i]] <- CrossTable(x = pima_test_y_list[[i]], y = pima_dnn_test_predict_label_list[[i]])
  pima_dnn_test_acc_list[[i]] <- (pima_dnn_test_confusion_list[[i]]$t[1] + pima_dnn_test_confusion_list[[i]]$t[4])  / sum(pima_dnn_test_confusion_list[[i]]$t)
  auc <- performance(prediction( predict(pima_model_list[[i]], pima_test_x_datamatrix)[2, ], pima_test_y_list[[i]], label.ordering = c(0, 1)), measure = "auc")
  pima_dnn_test_auc_list[[i]] <- auc@y.values[[1]]
  
  pima_dnn_exectime_list[[i]] <- exectime
  pima_dnn_train_logger_list[[i]] <- logger$train
  pima_dnn_test_logger_list[[i]] <- logger$eval
  
}

pima_dnn_test_acc_unlist <- unlist(pima_dnn_test_acc_list)
pima_dnn_test_auc_unlist <- unlist(pima_dnn_test_auc_list)
pima_dnn_exectime_unlist <- unlist(pima_dnn_exectime_list)

pima_dnn_train_logger_unlist <- data.frame( matrix(unlist(pima_dnn_train_logger_list), ncol = 100))
pima_dnn_test_logger_unlist <- data.frame( matrix(unlist(pima_dnn_test_logger_list), ncol = 100))


# 결과저장
# write(t(pima_dnn_test_acc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Pima_DNN_result.txt")
# write(t(pima_dnn_test_auc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/pima_DNN_auc_result.txt")
# write(t(pima_dnn_exectime_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Pima_DNN_time.txt")
# write(t(pima_dnn_train_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Pima_DNN_train_logger.txt")
# write(t(pima_dnn_test_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Pima_DNN_test_logger.txt")
pima_dnn_test_confusion_unlist <- data.frame()
for(i in 1:100){
  for(j in 1:4){
    pima_dnn_test_confusion_unlist[i, j] <- unlist(pima_dnn_test_confusion_list[[i]]$t)[j]
  }
}
pima_dnn_test_confusion_unlist <- cbind(c(1:100), pima_dnn_test_confusion_unlist)
colnames(pima_dnn_test_confusion_unlist) <- c("iter","TN", "FN", "FP", "TP")
head(pima_dnn_test_confusion_unlist)
tail(pima_dnn_test_confusion_unlist)

# write.table(pima_dnn_test_confusion_unlist, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/pima_dnn_confusion.txt", col.names = TRUE)


pima_dnn_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Pima_DNN_result.txt")
pima_dnn_test_acc_unlist <- cbind(pima_dnn_test_acc_unlist, rep("DNN(변수전체)", 100))
names(pima_dnn_test_acc_unlist) <- c("Accuracy", "Model")

pima_dnn_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Pima_DNN_auc_result.txt")
pima_dnn_test_auc_unlist <- cbind(pima_dnn_test_auc_unlist, rep("DNN(변수전체)", 100))
names(pima_dnn_test_auc_unlist) <- c("AUC", "Model")


pima_dnn_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/pima_DNN_time.txt")
pima_dnn_exectime_unlist <- cbind(pima_dnn_exectime_unlist, rep("DNN(변수전체)", 100))
names(pima_dnn_exectime_unlist) <- c("Time", "Model")


# Accuracy 분포 확인.

summary(pima_dnn_test_acc_unlist)

win.graph()
boxplot(pima_dnn_test_acc_unlist[1],  main = "Pima data. DNN 은닉층 1, 은닉노드 5. (전체변수)",
        xlab = "Iter = 100", ylab = "Accuracy", ylim = c(0, 1))
vioplot(pima_dnn_test_acc_unlist[1], main = "Pima data. DNN 은닉층 1, 은닉노드 5. (전체변수)",
        xlab = "Iter = 100", ylab = "Accuracy", ylim = c(0, 1))
points(mean(pima_dnn_test_acc_unlist$Accuracy), col = "red", pch = 17) # mean 표시

# AUC 분포 확인
summary(pima_dnn_test_auc_unlist)

win.graph()
boxplot(pima_dnn_test_auc_unlist[1],  main = "Pima data. DNN 은닉층 1, 은닉노드 5. (전체변수)",
        xlab = "Iter = 100", ylab = "AUC", ylim = c(0, 1))
vioplot(pima_dnn_test_auc_unlist[1], main = "Pima data. DNN 은닉층 1, 은닉노드 5. (전체변수)",
        xlab = "Iter = 100", ylab = "AUC", ylim = c(0, 1))
points(mean(pima_dnn_test_auc_unlist$AUC), col = "red", pch = 17) # mean 표시


# Time 분포 확인.
summary(pima_dnn_exectime_unlist)

win.graph()
boxplot(pima_dnn_exectime_unlist[1],  main = "Pima data. DNN 은닉층 1, 은닉노드 5. (전체변수) 학습시간",
        xlab = "Iter = 100", ylab = "Time(단위 : 초)", ylim = c(0, 10))
vioplot(pima_dnn_exectime_unlist[1], main = "Pima data. DNN 은닉층 1, 은닉노드 5. (전체변수) 학습시간",
        xlab = "Iter = 100", ylab = "Time(단위 : 초)", ylim = c(0, 10))
points(mean(pima_dnn_exectime_unlist$Time), col = "red", pch = 17) # mean 표시


# 2 Pima data  (LASSO 변수선택) 적합 후 Accuracy 분포 확인 (서로다른 100개 data set)
# 2 Pima data  (LASSO 변수선택) 적합 후 Accuracy 분포 확인 (서로다른 100개 data set)
# 2 Pima data  (LASSO 변수선택) 적합 후 Accuracy 분포 확인 (서로다른 100개 data set)
# (Note, LASSO 이용 변수선택)


# LASSO 10-fold CV 방법으로 변수선택 (lambda 1se)
pima_lasso_fit_10fold <- cv.glmnet(model.matrix( ~. -1, pima_train_x_list[[1]]), pima_train_y_list[[1]], type.measure = "auc",
                                   family = "binomial", alpha = 1)  # lambda.1se는 Standard error가 가장 Regularized 된 모델이 되는 람다값을 찾아줌.

win.graph()
plot(pima_lasso_fit_10fold, main = "Pima data (LASSO)")

# LASSO lambda.1se 에서 선택된 변수들 추정회귀계수
pima_lasso_coef <- predict(pima_lasso_fit_10fold, type = "coefficients",
                           s = pima_lasso_fit_10fold$lambda.1se)
pima_lasso_coef
# 7 x 1 sparse Matrix of class "dgCMatrix"
# 1
# (Intercept) -5.515678855
# pregnant     0.049596959
# glucose      0.024338054
# pressure     .          
# mass         0.043286027
# pedigree     0.258116417
# age          0.003327855

############################
## 신경망 학습 (변수선택) ##
############################
# https://mxnet.apache.org/api/r  참고.
# pressure 변수 제외 (LASSO 10-fold CV 에서 선택X)
# Train : Test = 8 : 2 나누기. (100번)
# seed 를 바꿔가며 1000개의 서로 다른 train, test data set 생성.
# list() 객체에 저장.

pima_train_x_list <- list()
pima_train_y_list <- list()

pima_test_x_list <- list()
pima_test_y_list <- list()

for(i in 1:100){
  set.seed(i)
  train_index <- sample(1:nrow(pima_data_NAdrop), 0.8 * nrow(pima_data_NAdrop), replace = FALSE)
  pima_train_x <- pima_data_NAdrop[train_index, c(1, 2, 4, 5, 6)]
  pima_train_y <- pima_data_NAdrop[train_index, 7]
  
  pima_test_x <- pima_data_NAdrop[-train_index, c(1, 2, 4, 5, 6)]
  pima_test_y <- pima_data_NAdrop[-train_index, 7]
  
  pima_train_x_list[[i]] <- pima_train_x
  pima_train_y_list[[i]] <- pima_train_y
  
  pima_test_x_list[[i]] <- pima_test_x
  pima_test_y_list[[i]] <- pima_test_y
  
}

library(mxnet)

pima_model_selec_list <- list()
pima_dnn_selec_test_predict_list <- list()
pima_dnn_selec_test_predict_label_list <- list()
pima_dnn_selec_test_confusion_list <- list()
pima_dnn_selec_test_acc_list <- list() # y_hat 과 test data set의 y 로 Accuracy 저장 list 할당.
pima_dnn_selec_exectime_list <- list()
pima_dnn_selec_train_logger_list <- list()
pima_dnn_selec_test_logger_list <- list()
pima_dnn_selec_test_auc_list <- list()

for(i in 1:100){ # 1:100
  pima_train_x_scale <- scale(pima_train_x_list[[i]])
  pima_test_x_scale <- scale(pima_test_x_list[[i]])
  
  pima_train_x_datamatrix <- data.matrix(pima_train_x_scale)
  pima_test_x_datamatrix <- data.matrix(pima_test_x_scale)
  
  pima_train_y <- pima_train_y_list[[i]]
  pima_test_y <- pima_test_y_list[[i]]
  
  mx.set.seed(2020)  # 가중치 초기값 고정.
  
  input <- mx.symbol.Variable("data")
  fc1 <- mx.symbol.FullyConnected(data = input, num.hidden = 5)  # Hyper parameter : 은닉노드 수
  act1 <- mx.symbol.Activation(data = fc1, act_type = "sigmoid")  # Hyper parameter : 활성화 함수 종류
  fc2 <- mx.symbol.FullyConnected(data = act1, num.hidden = 2)
  lro <- mx.symbol.SoftmaxOutput(data = fc2)  # Hyper parameter : 출력노드 함수 종류
  
  tic()
  
  logger <- mx.metric.logger$new()
  model <- mx.model.FeedForward.create(symbol = lro, X = pima_train_x_datamatrix, y = pima_train_y,
                                       eval.data = list(data = pima_test_x_datamatrix, label = pima_test_y),
                                       ctx = mx.cpu(), num.round = 100, optimizer = 'sgd', # Hyper parameter : iter 수, 최적화 함수 종류
                                       array.batch.size = 10, learning.rate = 0.001, momentum = 0.9, eval.metric = mx.metric.accuracy,
                                       verbose = T, epoch.end.callback = mx.callback.log.train.metric(1, logger))  # Hyper parameter : 배치 사이즈, 학습률, 모멘텀 값.
  exectime <- toc()
  exectime <- round(exectime$toc - exectime$tic, 5)
  
  pima_model_selec_list[[i]] <- model
  pima_dnn_selec_test_predict_list[[i]] <- predict(pima_model_selec_list[[i]], pima_test_x_datamatrix)
  pima_dnn_selec_test_predict_label_list[[i]] <- max.col(t(pima_dnn_selec_test_predict_list[[i]])) - 1
  pima_dnn_selec_test_confusion_list[[i]] <- CrossTable(x = pima_test_y_list[[i]], y = pima_dnn_selec_test_predict_label_list[[i]])
  pima_dnn_selec_test_acc_list[[i]] <- (pima_dnn_selec_test_confusion_list[[i]]$t[1] + pima_dnn_selec_test_confusion_list[[i]]$t[4])  / sum(pima_dnn_selec_test_confusion_list[[i]]$t)
  auc <- performance(prediction( predict(pima_model_selec_list[[i]], pima_test_x_datamatrix)[2, ], pima_test_y_list[[i]], label.ordering = c(0, 1)), measure = "auc")
  pima_dnn_selec_test_auc_list[[i]] <- auc@y.values[[1]]
  
  pima_dnn_selec_exectime_list[[i]] <- exectime
  pima_dnn_selec_train_logger_list[[i]] <- logger$train
  pima_dnn_selec_test_logger_list[[i]] <- logger$eval
  
}

pima_dnn_selec_test_acc_unlist <- unlist(pima_dnn_selec_test_acc_list)
pima_dnn_selec_test_auc_unlist <- unlist(pima_dnn_selec_test_auc_list)
pima_dnn_selec_exectime_unlist <- unlist(pima_dnn_selec_exectime_list)

pima_dnn_selec_train_logger_unlist <- data.frame( matrix(unlist(pima_dnn_selec_train_logger_list), ncol = 100))
pima_dnn_selec_test_logger_unlist <- data.frame( matrix(unlist(pima_dnn_selec_test_logger_list), ncol = 100))


#  결과저장
# write(t(unlist(pima_dnn_selec_test_acc_list)), ncolumns=1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Pima_DNN_selec_result.txt")
# write(t(pima_dnn_selec_test_auc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/pima_DNN_selec_auc_result.txt")
# write(t(pima_dnn_selec_exectime_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Pima_DNN_selec_time.txt")
# write(t(pima_dnn_selec_train_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Pima_DNN_selec_train_logger.txt")
# write(t(pima_dnn_selec_test_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Pima_DNN_selec_test_logger.txt")
pima_dnn_selec_test_confusion_unlist <- data.frame()
for(i in 1:100){
  for(j in 1:4){
    pima_dnn_selec_test_confusion_unlist[i, j] <- unlist(pima_dnn_selec_test_confusion_list[[i]]$t)[j]
  }
}
pima_dnn_selec_test_confusion_unlist <- cbind(c(1:100), pima_dnn_selec_test_confusion_unlist)
colnames(pima_dnn_selec_test_confusion_unlist) <- c("iter","TN", "FN", "FP", "TP")
pima_dnn_selec_test_confusion_unlist$FP[is.na(pima_dnn_selec_test_confusion_unlist$FP)] <- 0
pima_dnn_selec_test_confusion_unlist$TP[is.na(pima_dnn_selec_test_confusion_unlist$TP)] <- 0

head(pima_dnn_selec_test_confusion_unlist)
tail(pima_dnn_selec_test_confusion_unlist)

# write.table(pima_dnn_selec_test_confusion_unlist, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/pima_dnn_selec_test_confusion_unlist.txt", col.names = TRUE)


# Accuracy 분포 확인.
pima_dnn_selec_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Pima_DNN_selec_result.txt")
pima_dnn_selec_test_acc_unlist <- cbind(pima_dnn_selec_test_acc_unlist, rep("DNN(LASSO & Stepwise변수선택)", 100))
names(pima_dnn_selec_test_acc_unlist) <- c("Accuracy", "Model")

summary(pima_dnn_selec_test_acc_unlist)

win.graph()
boxplot(pima_dnn_selec_test_acc_unlist[1],  main = "Pima data. DNN 은닉층 1, 은닉노드 5. (LASSO & Stepwise변수선택)",
        xlab = "Iter = 100", ylab = "Accuracy", ylim = c(0, 1))
vioplot(pima_dnn_selec_test_acc_unlist[1], main = "Pima data. DNN 은닉층 1, 은닉노드 5. (LASSO & Stepwise변수선택)",
        xlab = "Iter = 100", ylab = "Accuracy", ylim = c(0, 1))

# AUC 분포 확인.
pima_dnn_selec_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Pima_DNN_selec_auc_result.txt")
pima_dnn_selec_test_auc_unlist <- cbind(pima_dnn_selec_test_auc_unlist, rep("DNN(LASSO & Stepwise변수선택)", 100))
names(pima_dnn_selec_test_auc_unlist) <- c("AUC", "Model")

summary(pima_dnn_selec_test_auc_unlist)

win.graph()
boxplot(pima_dnn_selec_test_auc_unlist[1],  main = "Pima data. DNN 은닉층 1, 은닉노드 5. (LASSO & Stepwise변수선택)",
        xlab = "Iter = 100", ylab = "AUC", ylim = c(0, 1))
vioplot(pima_dnn_selec_test_auc_unlist[1], main = "Pima data. DNN 은닉층 1, 은닉노드 5. (LASSO & Stepwise변수선택)",
        xlab = "Iter = 100", ylab = "AUC", ylim = c(0, 1))
points(mean(pima_dnn_selec_test_auc_unlist$AUC), col = "red", pch = 17) # mean 표시

# Time 분포 확인.
pima_dnn_selec_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Pima_DNN_selec_time.txt")
pima_dnn_selec_exectime_unlist <- cbind(pima_dnn_selec_exectime_unlist, rep("DNN(LASSO & Stepwise변수선택)", 100))
names(pima_dnn_selec_exectime_unlist) <- c("Time", "Model")


summary(pima_dnn_selec_exectime_unlist)

win.graph()
boxplot(pima_dnn_selec_exectime_unlist[1],  main = "Pima data. DNN 은닉층 1, 은닉노드 5. (LASSO & Stepwise변수선택) 학습시간",
        xlab = "Iter = 100", ylab = "Time(단위 : 초)", ylim = c(0, 10))
vioplot(pima_dnn_selec_exectime_unlist[1], main = "Pima data. DNN 은닉층 1, 은닉노드 5. (LASSO & Stepwise변수선택) 학습시간",
        xlab = "Iter = 100", ylab = "Time(단위 : 초)", ylim = c(0, 10))
points(mean(pima_dnn_selec_exectime_unlist$Time), col = "red", pch = 17) # mean 표시

# 3. Pima data Stepwise 변수선택 후 DNN 모형 학습
# 3. Pima data Stepwise 변수선택 후 DNN 모형 학습
# 3. Pima data Stepwise 변수선택 후 DNN 모형 학습

glm_model <- glm(diabetes ~ ., data = pima_data_NAdrop, family = "binomial"(link = "logit"))
glm_step_model <- step(glm_model, direction = "both")
summary(glm_step_model)

# Accuracy 분포 확인.
pima_dnn_selec_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Pima_DNN_selec_result.txt")
pima_dnn_selec_test_acc_unlist <- cbind(pima_dnn_selec_test_acc_unlist, rep("DNN(LASSO & Stepwise변수선택)", 100))
names(pima_dnn_selec_test_acc_unlist) <- c("Accuracy", "Model")

summary(pima_dnn_selec_test_acc_unlist)

win.graph()
boxplot(pima_dnn_selec_test_acc_unlist[1],  main = "Pima data. DNN 은닉층 1, 은닉노드 5. (LASSO & Stepwise변수선택)",
        xlab = "Iter = 100", ylab = "Accuracy", ylim = c(0, 1))
vioplot(pima_dnn_selec_test_acc_unlist[1], main = "Pima data. DNN 은닉층 1, 은닉노드 5. (LASSO & Stepwise변수선택)",
        xlab = "Iter = 100", ylab = "Accuracy", ylim = c(0, 1))
points(mean(pima_dnn_selec_test_acc_unlist$Accuracy), col = "red", pch = 17) # mean 표시

