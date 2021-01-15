library(devtools)

# install_github('ramhiser/datamicroarray')
library(datamicroarray)
library(SIS)
library(gmodels)
library(tictoc)
library(vioplot)
library(ROCR)



data('alon', package='datamicroarray')


### 데이터 불러들이기
head(alon)
tail(alon)

str(alon)
dim(alon$x)
alon$y


data.hd = alon$x
dim(data.hd)


y = (alon$y)
levels(y) = c("0","1")
y = as.numeric(as.character(y))
y

alon_data <- cbind(data.hd, y)
alon_data <- data.frame(alon_data)
dim(alon_data)
# 62x2001


summary(alon_data[, 1:15])
summary(alon_data[, 1986:2001])


# 신경망 학습용 data set
# Train : Test = 8 : 2 나누기. (100번)
# seed 를 바꿔가며 100개의 서로 다른 train, test data set 생성.
# list() 객체에 저장.

alon_train_x_list_num <- list()
alon_train_y_list_num <- list()

alon_test_x_list_num <- list()
alon_test_y_list_num <- list()

for(i in 1:100){
  set.seed(i)
  train_index <- sample(1:nrow(alon_data), 0.8 * nrow(alon_data), replace = FALSE)
  alon_train_x_num <- alon_data[train_index, 1:2000]
  alon_train_y_num <- alon_data[train_index, 2001]
  
  alon_test_x_num <- alon_data[-train_index, 1:2000]
  alon_test_y_num <- alon_data[-train_index, 2001]
  
  alon_train_x_list_num[[i]] <- alon_train_x_num
  alon_train_y_list_num[[i]] <- alon_train_y_num
  
  alon_test_x_list_num[[i]] <- alon_test_x_num
  alon_test_y_list_num[[i]] <- alon_test_y_num
  
}


#################
## 신경망 학습 ##
#################
# https://mxnet.apache.org/api/r  참고.
library(mxnet)

alon_model_list <- list()
alon_dnn_test_predict_list <- list()
alon_dnn_test_predict_label_list <- list()
alon_dnn_test_confusion_list <- list()
alon_dnn_test_acc_list <- list() # y_hat 과 test data set의 y 로 Accuracy 저장 list 할당.
alon_dnn_exectime_list <- list()
alon_dnn_train_logger_list <- list()
alon_dnn_test_logger_list <- list()
alon_dnn_test_auc_list <- list()

for(i in 1:100){ # 1:100
  alon_train_x_scale <- scale(alon_train_x_list_num[[i]])
  alon_test_x_scale <- scale(alon_test_x_list_num[[i]])
  
  alon_train_x_datamatrix <- data.matrix(alon_train_x_scale)
  alon_test_x_datamatrix <- data.matrix(alon_test_x_scale)
  
  alon_train_y <- alon_train_y_list_num[[i]]
  alon_test_y <- alon_test_y_list_num[[i]]
  
  mx.set.seed(2020)  # 가중치 초기값 고정.
  
  input <- mx.symbol.Variable("data")
  fc1 <- mx.symbol.FullyConnected(data = input, num.hidden = 20)  # Hyper parameter : 은닉노드 수
  act1 <- mx.symbol.Activation(data = fc1, act_type = "sigmoid")  # Hyper parameter : 활성화 함수 종류
  fc2 <- mx.symbol.FullyConnected(data = act1, num.hidden = 2)
  lro <- mx.symbol.SoftmaxOutput(data = fc2)  # Hyper parameter : 출력노드 함수 종류
  
  tic()
  
  logger <- mx.metric.logger$new()
  model <- mx.model.FeedForward.create(symbol = lro, X = alon_train_x_datamatrix, y = alon_train_y,
                                       eval.data = list(data = alon_test_x_datamatrix, label = alon_test_y),
                                       ctx = mx.gpu(), num.round = 150, optimizer = 'sgd', # Hyper parameter : iter 수, 최적화 함수 종류
                                       array.batch.size = 5, learning.rate = 0.001, momentum = 0.9, eval.metric = mx.metric.accuracy,
                                       verbose = T, epoch.end.callback = mx.callback.log.train.metric(1, logger))  # Hyper parameter : 배치 사이즈, 학습률, 모멘텀 값.
  
  exectime <- toc()
  exectime <- round(exectime$toc - exectime$tic, 5)
  
  alon_model_list[[i]] <- model
  alon_dnn_test_predict_list[[i]] <- predict(alon_model_list[[i]], alon_test_x_datamatrix)
  alon_dnn_test_predict_label_list[[i]] <- max.col(t(alon_dnn_test_predict_list[[i]])) - 1
  alon_dnn_test_confusion_list[[i]] <- CrossTable(x = alon_test_y_list_num[[i]], y = alon_dnn_test_predict_label_list[[i]])
  
  alon_dnn_test_acc_list[[i]] <- (alon_dnn_test_confusion_list[[i]]$t[1] + alon_dnn_test_confusion_list[[i]]$t[4])  / sum(alon_dnn_test_confusion_list[[i]]$t)
  auc <- performance(prediction( predict(alon_model_list[[i]], alon_test_x_datamatrix)[2, ], alon_test_y_list_num[[i]], label.ordering = c(0, 1)), measure = "auc")
  alon_dnn_test_auc_list[[i]] <- auc@y.values[[1]]
  
  alon_dnn_exectime_list[[i]] <- exectime
  alon_dnn_train_logger_list[[i]] <- logger$train
  alon_dnn_test_logger_list[[i]] <- logger$eval
}

alon_dnn_test_acc_unlist <- unlist(alon_dnn_test_acc_list)
alon_dnn_test_auc_unlist <- unlist(alon_dnn_test_auc_list)
alon_dnn_exectime_unlist <- unlist(alon_dnn_exectime_list)

alon_dnn_train_logger_unlist <- data.frame( matrix(unlist(alon_dnn_train_logger_list), ncol = 100))
alon_dnn_test_logger_unlist <- data.frame( matrix(unlist(alon_dnn_test_logger_list), ncol = 100))

# 결과저장
# write(t(alon_dnn_test_acc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_DNN_result.txt")
# write(t(alon_dnn_test_auc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_DNN_auc_result.txt")
# write(t(alon_dnn_exectime_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_DNN_time.txt")
# write(t(alon_dnn_train_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_DNN_train_logger.txt")
# write(t(alon_dnn_test_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_DNN_test_logger.txt")
alon_dnn_test_confusion_unlist <- data.frame()
for(i in 1:100){
  for(j in 1:4){
    alon_dnn_test_confusion_unlist[i, j] <- unlist(alon_dnn_test_confusion_list[[i]]$t)[j]
  }
}
alon_dnn_test_confusion_unlist <- cbind(c(1:100), alon_dnn_test_confusion_unlist)
colnames(alon_dnn_test_confusion_unlist) <- c("iter","TN", "FN", "FP", "TP")
head(alon_dnn_test_confusion_unlist)
tail(alon_dnn_test_confusion_unlist)

# write.table(alon_dnn_test_confusion_unlist, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_dnn_confusion.txt", col.names = TRUE)


alon_dnn_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_DNN_result.txt")
alon_dnn_test_acc_unlist <- cbind(alon_dnn_test_acc_unlist, rep("DNN(변수전체)", 100))
names(alon_dnn_test_acc_unlist) <- c("Accuracy", "Model")

alon_dnn_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_DNN_AUC_result.txt")
alon_dnn_test_auc_unlist <- cbind(alon_dnn_test_auc_unlist, rep("DNN(변수전체)", 100))
names(alon_dnn_test_auc_unlist) <- c("AUC", "Model")

alon_dnn_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_DNN_time.txt")
alon_dnn_exectime_unlist <- cbind(alon_dnn_exectime_unlist, rep("DNN(변수전체)", 100))
names(alon_dnn_exectime_unlist) <- c("Time", "Model")

# Accuracy 분포 확인.
summary(alon_dnn_test_acc_unlist)

win.graph()
boxplot(alon_dnn_test_acc_unlist[1],  main = "Alon data. DNN 은닉층 1, 은닉노드 20. (전체변수)",
        xlab = "Iter = 150", ylab = "Accuracy", ylim = c(0, 1))
vioplot(alon_dnn_test_acc_unlist[1], main = "Alon data. DNN 은닉층 1, 은닉노드 20. (전체변수)",
        xlab = "Iter = 150", ylab = "Accuracy", ylim = c(0, 1))
points(mean(alon_dnn_test_acc_unlist$Accuracy), col = "red", pch = 17) # mean 표시

# AUC 분포 확인.
summary(alon_dnn_test_auc_unlist)

win.graph()
boxplot(alon_dnn_test_auc_unlist[1],  main = "Alon data. DNN 은닉층 1, 은닉노드 20. (전체변수)",
        xlab = "Iter = 150", ylab = "AUC", ylim = c(0, 1))
vioplot(alon_dnn_test_auc_unlist[1], main = "Alon data. DNN 은닉층 1, 은닉노드 20. (전체변수)",
        xlab = "Iter = 150", ylab = "AUC", ylim = c(0, 1))
points(mean(alon_dnn_test_auc_unlist$AUC), col = "red", pch = 17) # mean 표시

# Time 분포 확인.
summary(alon_dnn_exectime_unlist)

win.graph()
boxplot(alon_dnn_exectime_unlist[1],  main = "Alon data. DNN 은닉층 1, 은닉노드 20. (전체변수) 학습시간",
        xlab = "Iter = 150", ylab = "Time(단위 : 초)", ylim = c(0, 5))
vioplot(alon_dnn_exectime_unlist[1], main = "Alon data. DNN 은닉층 1, 은닉노드 20. (전체변수) 학습시간",
        xlab = "Iter = 150", ylab = "Time(단위 : 초)", ylim = c(0, 5))
points(mean(alon_dnn_exectime_unlist$Time), col = "red", pch = 17) # mean 표시

### 2. SIS 변수선택 (alon data) ###
### 2. SIS 변수선택 (alon data) ###
### 2. SIS 변수선택 (alon data) ###

dim(alon_data)
# 62x2001

alon_data_x <- alon_data[ ,-2001]
alon_data_x <- data.matrix(alon_data_x)
alon_data_x <- standardize(alon_data_x)

alon_data_y <- alon_data[ ,2001]

## SCAD ##
## SCAD ##
## SCAD ##
alon_SIS_model_SCAD <- SIS(alon_data_x, alon_data_y, family = 'binomial', tune = 'bic', penalty = "SCAD",
                               perm = TRUE, q = 0.9, greedy = TRUE, seed = 31)

alon_SIS_model_SCAD
alon_SIS_model_SCAD$ix
alon_SIS_model_SCAD$fit$beta


## MCP ##
## MCP ##
## MCP ##
alon_SIS_model_MCP <- SIS(alon_data_x, alon_data_y, family = 'binomial', tune = 'bic', penalty = "MCP",
                              perm = TRUE, q = 0.9, greedy = TRUE, seed = 31)

alon_SIS_model_MCP
alon_SIS_model_MCP$ix
alon_SIS_model_MCP$fit$beta

## LASSO ##
## LASSO ##
## LASSO ##
alon_SIS_model_LASSO <- SIS(alon_data_x, alon_data_y, family = 'binomial', tune = 'bic', penalty = "lasso",
                                perm = TRUE, q = 0.9, greedy = TRUE, seed = 31)

alon_SIS_model_LASSO
alon_SIS_model_LASSO$ix
alon_SIS_model_LASSO$fit$beta

## 각 패널티별 변수선택 확인
alon_SIS_model_SCAD$ix  # 249 1895 1935
alon_SIS_model_MCP$ix  # 249 1895 1935
alon_SIS_model_LASSO$ix  # 177 249 765

### 2.1 SIS SCAD & MCP 변수선택 (alon data) 후 DNN ###
### 2.1 SIS SCAD & MCP 변수선택 (alon data) 후 DNN ###
### 2.1 SIS SCAD & MCP 변수선택 (alon data) 후 DNN ###

# 신경망 학습용 data set
# Train : Test = 8 : 2 나누기. (100번)
# seed 를 바꿔가며 100개의 서로 다른 train, test data set 생성.
# list() 객체에 저장.

alon_SCAD_MCP_train_x_list_num <- list()
alon_SCAD_MCP_train_y_list_num <- list()

alon_SCAD_MCP_test_x_list_num <- list()
alon_SCAD_MCP_test_y_list_num <- list()

for(i in 1:100){
  set.seed(i)
  train_index <- sample(1:nrow(alon_data), 0.8 * nrow(alon_data), replace = FALSE)
  alon_train_x_num <- alon_data[train_index, c(249, 1895, 1935)]
  alon_train_y_num <- alon_data[train_index, 2001]
  
  alon_test_x_num <- alon_data[-train_index, c(249, 1895, 1935)]
  alon_test_y_num <- alon_data[-train_index, 2001]
  
  alon_SCAD_MCP_train_x_list_num[[i]] <- alon_train_x_num
  alon_SCAD_MCP_train_y_list_num[[i]] <- alon_train_y_num
  
  alon_SCAD_MCP_test_x_list_num[[i]] <- alon_test_x_num
  alon_SCAD_MCP_test_y_list_num[[i]] <- alon_test_y_num
  
}


#################
## 신경망 학습 ##
#################
# https://mxnet.apache.org/api/r  참고.
library(mxnet)

alon_SCAD_MCP_model_list <- list()
alon_SCAD_MCP_dnn_test_predict_list <- list()
alon_SCAD_MCP_dnn_test_predict_label_list <- list()
alon_SCAD_MCP_dnn_test_confusion_list <- list()
alon_SCAD_MCP_dnn_test_acc_list <- list() # y_hat 과 test data set의 y 로 Accuracy 저장 list 할당.
alon_SCAD_MCP_dnn_exectime_list <- list()
alon_SCAD_MCP_dnn_train_logger_list <- list()
alon_SCAD_MCP_dnn_test_logger_list <- list()
alon_SCAD_MCP_dnn_test_auc_list <- list()

for(i in 1:100){ # 1:100
  alon_SCAD_MCP_train_x_scale <- scale(alon_SCAD_MCP_train_x_list_num[[i]])
  alon_SCAD_MCP_test_x_scale <- scale(alon_SCAD_MCP_test_x_list_num[[i]])
  
  alon_SCAD_MCP_train_x_datamatrix <- data.matrix(alon_SCAD_MCP_train_x_scale)
  alon_SCAD_MCP_test_x_datamatrix <- data.matrix(alon_SCAD_MCP_test_x_scale)
  
  alon_SCAD_MCP_train_y <- alon_SCAD_MCP_train_y_list_num[[i]]
  alon_SCAD_MCP_test_y <- alon_SCAD_MCP_test_y_list_num[[i]]
  
  mx.set.seed(2020)  # 가중치 초기값 고정.
  
  input <- mx.symbol.Variable("data")
  fc1 <- mx.symbol.FullyConnected(data = input, num.hidden = 20)  # Hyper parameter : 은닉노드 수
  act1 <- mx.symbol.Activation(data = fc1, act_type = "sigmoid")  # Hyper parameter : 활성화 함수 종류
  fc2 <- mx.symbol.FullyConnected(data = act1, num.hidden = 2)
  lro <- mx.symbol.SoftmaxOutput(data = fc2)  # Hyper parameter : 출력노드 함수 종류
  
  tic()
  
  logger <- mx.metric.logger$new()
  model <- mx.model.FeedForward.create(symbol = lro, X = alon_SCAD_MCP_train_x_datamatrix, y = alon_SCAD_MCP_train_y,
                                       eval.data = list(data = alon_SCAD_MCP_test_x_datamatrix, label = alon_SCAD_MCP_test_y),
                                       ctx = mx.gpu(), num.round = 500, optimizer = 'sgd', # Hyper parameter : iter 수, 최적화 함수 종류
                                       array.batch.size = 5, learning.rate = 0.001, momentum = 0.9, eval.metric = mx.metric.accuracy,
                                       verbose = T, epoch.end.callback = mx.callback.log.train.metric(1, logger))  # Hyper parameter : 배치 사이즈, 학습률, 모멘텀 값.
  
  exectime <- toc()
  exectime <- round(exectime$toc - exectime$tic, 5)
  
  alon_SCAD_MCP_model_list[[i]] <- model
  alon_SCAD_MCP_dnn_test_predict_list[[i]] <- predict(alon_SCAD_MCP_model_list[[i]], alon_SCAD_MCP_test_x_datamatrix)
  alon_SCAD_MCP_dnn_test_predict_label_list[[i]] <- max.col(t(alon_SCAD_MCP_dnn_test_predict_list[[i]])) - 1
  alon_SCAD_MCP_dnn_test_confusion_list[[i]] <- CrossTable(x = alon_SCAD_MCP_test_y_list_num[[i]], y = alon_SCAD_MCP_dnn_test_predict_label_list[[i]])
  alon_SCAD_MCP_dnn_test_acc_list[[i]] <- (alon_SCAD_MCP_dnn_test_confusion_list[[i]]$t[1] + alon_SCAD_MCP_dnn_test_confusion_list[[i]]$t[4])  / sum(alon_SCAD_MCP_dnn_test_confusion_list[[i]]$t)
  auc <- performance(prediction( predict(alon_SCAD_MCP_model_list[[i]], alon_SCAD_MCP_test_x_datamatrix)[2, ], alon_SCAD_MCP_test_y_list_num[[i]], label.ordering = c(0, 1)), measure = "auc")
  alon_SCAD_MCP_dnn_test_auc_list[[i]] <- auc@y.values[[1]]
  
  alon_SCAD_MCP_dnn_exectime_list[[i]] <- exectime
  alon_SCAD_MCP_dnn_train_logger_list[[i]] <- logger$train
  alon_SCAD_MCP_dnn_test_logger_list[[i]] <- logger$eval
}

alon_SCAD_MCP_dnn_test_acc_unlist <- unlist(alon_SCAD_MCP_dnn_test_acc_list)
alon_SCAD_MCP_dnn_test_auc_unlist <- unlist(alon_SCAD_MCP_dnn_test_auc_list)
alon_SCAD_MCP_dnn_exectime_unlist <- unlist(alon_SCAD_MCP_dnn_exectime_list)

alon_SCAD_MCP_dnn_train_logger_unlist <- data.frame( matrix(unlist(alon_SCAD_MCP_dnn_train_logger_list), ncol = 100))
alon_SCAD_MCP_dnn_test_logger_unlist <- data.frame( matrix(unlist(alon_SCAD_MCP_dnn_test_logger_list), ncol = 100))

# 결과저장
# write(t(alon_SCAD_MCP_dnn_test_acc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_SCAD_MCP_DNN_result.txt")
# write(t(alon_SCAD_MCP_dnn_test_auc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_SCAD_MCP_DNN_auc_result.txt")
# write(t(alon_SCAD_MCP_dnn_exectime_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_SCAD_MCP_DNN_time.txt")
# write(t(alon_SCAD_MCP_dnn_train_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_SCAD_MCP_DNN_train_logger.txt")
# write(t(alon_SCAD_MCP_dnn_test_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_SCAD_MCP_DNN_test_logger.txt")
alon_SCAD_MCP_dnn_test_confusion_unlist <- data.frame()
for(i in 1:100){
  for(j in 1:4){
    alon_SCAD_MCP_dnn_test_confusion_unlist[i, j] <- unlist(alon_SCAD_MCP_dnn_test_confusion_list[[i]]$t)[j]
  }
}
alon_SCAD_MCP_dnn_test_confusion_unlist <- cbind(c(1:100), alon_SCAD_MCP_dnn_test_confusion_unlist)
colnames(alon_SCAD_MCP_dnn_test_confusion_unlist) <- c("iter","TN", "FN", "FP", "TP")
head(alon_SCAD_MCP_dnn_test_confusion_unlist)
tail(alon_SCAD_MCP_dnn_test_confusion_unlist)

# write.table(alon_SCAD_MCP_dnn_test_confusion_unlist, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_SCAD_MCP_dnn_confusion.txt", col.names = TRUE)


alon_SCAD_MCP_dnn_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_SCAD_MCP_DNN_result.txt")
alon_SCAD_MCP_dnn_test_acc_unlist <- cbind(alon_SCAD_MCP_dnn_test_acc_unlist, rep("DNN(SCAD & MCP 변수선택)", 100))
names(alon_SCAD_MCP_dnn_test_acc_unlist) <- c("Accuracy", "Model")

alon_SCAD_MCP_dnn_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_SCAD_MCP_DNN_auc_result.txt")
alon_SCAD_MCP_dnn_test_auc_unlist <- cbind(alon_SCAD_MCP_dnn_test_auc_unlist, rep("DNN(SCAD 변수선택)", 100))
names(alon_SCAD_MCP_dnn_test_auc_unlist) <- c("AUC", "Model")

alon_SCAD_MCP_dnn_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_SCAD_MCP_DNN_time.txt")
alon_SCAD_MCP_dnn_exectime_unlist <- cbind(alon_SCAD_MCP_dnn_exectime_unlist, rep("DNN(SCAD & MCP 변수선택)", 100))
names(alon_SCAD_MCP_dnn_exectime_unlist) <- c("Time", "Model")

# Accuracy 분포 확인.
summary(alon_SCAD_MCP_dnn_test_acc_unlist)

win.graph()
boxplot(alon_SCAD_MCP_dnn_test_acc_unlist[1],  main = "Alon data. DNN 은닉층 1, 은닉노드 20. (SCAD & MCP 변수선택)",
        xlab = "Iter = 500", ylab = "Accuracy", ylim = c(0, 1))
vioplot(alon_SCAD_MCP_dnn_test_acc_unlist[1], main = "Alon data. DNN 은닉층 1, 은닉노드 20. (SCAD & MCP 변수선택)",
        xlab = "Iter = 500", ylab = "Accuracy", ylim = c(0, 1))
points(mean(alon_SCAD_MCP_dnn_test_acc_unlist$Accuracy), col = "red", pch = 17) # mean 표시

# AUC 분포 확인.
summary(alon_SCAD_MCP_dnn_test_auc_unlist)

win.graph()
boxplot(alon_SCAD_MCP_dnn_test_auc_unlist[1],  main = "Alon data. DNN 은닉층 1, 은닉노드 20. (SCAD & MCP 변수선택)",
        xlab = "Iter = 500", ylab = "AUC", ylim = c(0, 1))
vioplot(alon_SCAD_MCP_dnn_test_auc_unlist[1], main = "Alon data. DNN 은닉층 1, 은닉노드 20. (SCAD & MCP 변수선택)",
        xlab = "Iter = 500", ylab = "AUC", ylim = c(0, 1))
points(mean(alon_SCAD_MCP_dnn_test_auc_unlist$AUC), col = "red", pch = 17) # mean 표시

# Time 분포 확인.
summary(alon_SCAD_MCP_dnn_exectime_unlist)

win.graph()
boxplot(alon_SCAD_MCP_dnn_exectime_unlist[1],  main = "Alon data. DNN 은닉층 1, 은닉노드 20. (SCAD & MCP 변수선택) 학습시간",
        xlab = "Iter = 500", ylab = "Time(단위 : 초)", ylim = c(0, 9))
vioplot(alon_SCAD_MCP_dnn_exectime_unlist[1], main = "Alon data. DNN 은닉층 1, 은닉노드 20. (SCAD & MCP 변수선택) 학습시간",
        xlab = "Iter = 500", ylab = "Time(단위 : 초)", ylim = c(0, 9))
points(mean(alon_SCAD_MCP_dnn_exectime_unlist$Time), col = "red", pch = 17) # mean 표시


### 2.3 SIS LASSO 변수선택 (alon data) 후 DNN ###
### 2.3 SIS LASSO 변수선택 (alon data) 후 DNN ###
### 2.3 SIS LASSO 변수선택 (alon data) 후 DNN ###

# 신경망 학습용 data set
# Train : Test = 8 : 2 나누기. (100번)
# seed 를 바꿔가며 100개의 서로 다른 train, test data set 생성.
# list() 객체에 저장.

alon_LASSO_train_x_list_num <- list()
alon_LASSO_train_y_list_num <- list()

alon_LASSO_test_x_list_num <- list()
alon_LASSO_test_y_list_num <- list()

for(i in 1:100){
  set.seed(i)
  train_index <- sample(1:nrow(alon_data), 0.8 * nrow(alon_data), replace = FALSE)
  alon_train_x_num <- alon_data[train_index, c(177, 249, 765)]
  alon_train_y_num <- alon_data[train_index, 2001]
  
  alon_test_x_num <- alon_data[-train_index, c(177, 249, 765)]
  alon_test_y_num <- alon_data[-train_index, 2001]
  
  alon_LASSO_train_x_list_num[[i]] <- alon_train_x_num
  alon_LASSO_train_y_list_num[[i]] <- alon_train_y_num
  
  alon_LASSO_test_x_list_num[[i]] <- alon_test_x_num
  alon_LASSO_test_y_list_num[[i]] <- alon_test_y_num
  
}


#################
## 신경망 학습 ##
#################
# https://mxnet.apache.org/api/r  참고.
library(mxnet)

alon_LASSO_model_list <- list()
alon_LASSO_dnn_test_predict_list <- list()
alon_LASSO_dnn_test_predict_label_list <- list()
alon_LASSO_dnn_test_confusion_list <- list()
alon_LASSO_dnn_test_acc_list <- list() # y_hat 과 test data set의 y 로 Accuracy 저장 list 할당.
alon_LASSO_dnn_exectime_list <- list()
alon_LASSO_dnn_train_logger_list <- list()
alon_LASSO_dnn_test_logger_list <- list()
alon_LASSO_dnn_test_auc_list <- list()

for(i in 1:100){ # 1:100
  alon_LASSO_train_x_scale <- scale(alon_LASSO_train_x_list_num[[i]])
  alon_LASSO_test_x_scale <- scale(alon_LASSO_test_x_list_num[[i]])
  
  alon_LASSO_train_x_datamatrix <- data.matrix(alon_LASSO_train_x_scale)
  alon_LASSO_test_x_datamatrix <- data.matrix(alon_LASSO_test_x_scale)
  
  alon_LASSO_train_y <- alon_LASSO_train_y_list_num[[i]]
  alon_LASSO_test_y <- alon_LASSO_test_y_list_num[[i]]
  
  mx.set.seed(2020)  # 가중치 초기값 고정.
  
  input <- mx.symbol.Variable("data")
  fc1 <- mx.symbol.FullyConnected(data = input, num.hidden = 20)  # Hyper parameter : 은닉노드 수
  act1 <- mx.symbol.Activation(data = fc1, act_type = "sigmoid")  # Hyper parameter : 활성화 함수 종류
  fc2 <- mx.symbol.FullyConnected(data = act1, num.hidden = 2)
  lro <- mx.symbol.SoftmaxOutput(data = fc2)  # Hyper parameter : 출력노드 함수 종류
  
  tic()
  
  logger <- mx.metric.logger$new()
  model <- mx.model.FeedForward.create(symbol = lro, X = alon_LASSO_train_x_datamatrix, y = alon_LASSO_train_y,
                                       eval.data = list(data = alon_LASSO_test_x_datamatrix, label = alon_LASSO_test_y),
                                       ctx = mx.gpu(), num.round = 370, optimizer = 'sgd', # Hyper parameter : iter 수, 최적화 함수 종류
                                       array.batch.size = 5, learning.rate = 0.001, momentum = 0.9, eval.metric = mx.metric.accuracy,
                                       verbose = T, epoch.end.callback = mx.callback.log.train.metric(1, logger))  # Hyper parameter : 배치 사이즈, 학습률, 모멘텀 값.
  
  exectime <- toc()
  exectime <- round(exectime$toc - exectime$tic, 5)
  
  alon_LASSO_model_list[[i]] <- model
  alon_LASSO_dnn_test_predict_list[[i]] <- predict(alon_LASSO_model_list[[i]], alon_LASSO_test_x_datamatrix)
  alon_LASSO_dnn_test_predict_label_list[[i]] <- max.col(t(alon_LASSO_dnn_test_predict_list[[i]])) - 1
  alon_LASSO_dnn_test_confusion_list[[i]] <- CrossTable(x = alon_LASSO_test_y_list_num[[i]], y = alon_LASSO_dnn_test_predict_label_list[[i]])
  alon_LASSO_dnn_test_acc_list[[i]] <- (alon_LASSO_dnn_test_confusion_list[[i]]$t[1] + alon_LASSO_dnn_test_confusion_list[[i]]$t[4])  / sum(alon_LASSO_dnn_test_confusion_list[[i]]$t)
  auc <- performance(prediction( predict(alon_LASSO_model_list[[i]], alon_LASSO_test_x_datamatrix)[2, ], alon_LASSO_test_y_list_num[[i]], label.ordering = c(0, 1)), measure = "auc")
  alon_LASSO_dnn_test_auc_list[[i]] <- auc@y.values[[1]]
  
  alon_LASSO_dnn_exectime_list[[i]] <- exectime
  alon_LASSO_dnn_train_logger_list[[i]] <- logger$train
  alon_LASSO_dnn_test_logger_list[[i]] <- logger$eval
}

alon_LASSO_dnn_test_acc_unlist <- unlist(alon_LASSO_dnn_test_acc_list)  # 32번째 NA -> 0.3846154 (CrossTable() 함수로 Accuracy 계산 중
alon_LASSO_dnn_test_auc_unlist <- unlist(alon_LASSO_dnn_test_auc_list)  #                         예측을 모두 0 or 1로 하면 코딩과정에서
alon_LASSO_dnn_exectime_unlist <- unlist(alon_LASSO_dnn_exectime_list)  #                         NA가 생성됨.)

alon_LASSO_dnn_train_logger_unlist <- data.frame( matrix(unlist(alon_LASSO_dnn_train_logger_list), ncol = 100))
alon_LASSO_dnn_test_logger_unlist <- data.frame( matrix(unlist(alon_LASSO_dnn_test_logger_list), ncol = 100))

# 결과저장
# write(t(alon_LASSO_dnn_test_acc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_LASSO_DNN_result.txt")
# write(t(alon_LASSO_dnn_test_auc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_LASSO_DNN_auc_result.txt")
# write(t(alon_LASSO_dnn_exectime_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_LASSO_DNN_time.txt")
# write(t(alon_LASSO_dnn_train_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_LASSO_DNN_train_logger.txt")
# write(t(alon_LASSO_dnn_test_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_LASSO_DNN_test_logger.txt")
alon_LASSO_dnn_test_confusion_unlist <- data.frame()
for(i in 1:100){
  for(j in 1:4){
    alon_LASSO_dnn_test_confusion_unlist[i, j] <- unlist(alon_LASSO_dnn_test_confusion_list[[i]]$t)[j]
  }
}
alon_LASSO_dnn_test_confusion_unlist <- cbind(c(1:100), alon_LASSO_dnn_test_confusion_unlist)
colnames(alon_LASSO_dnn_test_confusion_unlist) <- c("iter","TN", "FN", "FP", "TP")
head(alon_LASSO_dnn_test_confusion_unlist)
tail(alon_LASSO_dnn_test_confusion_unlist)  # 32번째에 FP, TP 값이 CrossTable()에서 생략되어 NA로 출력 -> 0 으로 대체

# write.table(alon_LASSO_dnn_test_confusion_unlist, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_LASSO_dnn_confusion.txt", col.names = TRUE)


alon_LASSO_dnn_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_LASSO_DNN_result.txt")
alon_LASSO_dnn_test_acc_unlist <- cbind(alon_LASSO_dnn_test_acc_unlist, rep("DNN(LASSO 변수선택)", 100))
names(alon_LASSO_dnn_test_acc_unlist) <- c("Accuracy", "Model")

alon_LASSO_dnn_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_LASSO_DNN_auc_result.txt")
alon_LASSO_dnn_test_auc_unlist <- cbind(alon_LASSO_dnn_test_auc_unlist, rep("DNN(LASSO 변수선택)", 100))
names(alon_LASSO_dnn_test_auc_unlist) <- c("AUC", "Model")

alon_LASSO_dnn_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_LASSO_DNN_time.txt")
alon_LASSO_dnn_exectime_unlist <- cbind(alon_LASSO_dnn_exectime_unlist, rep("DNN(LASSO 변수선택)", 100))
names(alon_LASSO_dnn_exectime_unlist) <- c("Time", "Model")

# Accuracy 분포 확인.
summary(alon_LASSO_dnn_test_acc_unlist)

win.graph()
boxplot(alon_LASSO_dnn_test_acc_unlist[1],  main = "Alon data. DNN 은닉층 1, 은닉노드 20. (LASSO 변수선택)",
        xlab = "Iter = 370", ylab = "Accuracy", ylim = c(0, 1))
vioplot(alon_LASSO_dnn_test_acc_unlist[1], main = "Alon data. DNN 은닉층 1, 은닉노드 20. (LASSO 변수선택)",
        xlab = "Iter = 370", ylab = "Accuracy", ylim = c(0, 1))
points(mean(alon_LASSO_dnn_test_acc_unlist$Accuracy), col = "red", pch = 17) # mean 표시

# AUC 분포 확인.
summary(alon_LASSO_dnn_test_auc_unlist)

win.graph()
boxplot(alon_LASSO_dnn_test_auc_unlist[1],  main = "Alon data. DNN 은닉층 1, 은닉노드 20. (LASSO 변수선택)",
        xlab = "Iter = 370", ylab = "AUC", ylim = c(0, 1))
vioplot(alon_LASSO_dnn_test_auc_unlist[1], main = "Alon data. DNN 은닉층 1, 은닉노드 20. (LASSO 변수선택)",
        xlab = "Iter = 370", ylab = "AUC", ylim = c(0, 1))
points(mean(alon_LASSO_dnn_test_auc_unlist$AUC), col = "red", pch = 17) # mean 표시

# Time 분포 확인.
summary(alon_LASSO_dnn_exectime_unlist)

win.graph()
boxplot(alon_LASSO_dnn_exectime_unlist[1],  main = "Alon data. DNN 은닉층 1, 은닉노드 20. (LASSO 변수선택) 학습시간",
        xlab = "Iter = 370", ylab = "Time(단위 : 초)", ylim = c(0, 7))
vioplot(alon_LASSO_dnn_exectime_unlist[1], main = "Alon data. DNN 은닉층 1, 은닉노드 20. (LASSO 변수선택) 학습시간",
        xlab = "Iter = 370", ylab = "Time(단위 : 초)", ylim = c(0, 7))
points(mean(alon_LASSO_dnn_exectime_unlist$Time), col = "red", pch = 17) # mean 표시
