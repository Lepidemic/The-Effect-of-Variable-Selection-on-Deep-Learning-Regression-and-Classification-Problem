
### 데이터 로딩

# dim(prostate.train): 102 12601

library(SIS)
library(gmodels)
library(tictoc)
library(vioplot)

data('prostate.train',  package = 'SIS')
dim(prostate.train)
# 102x12601

data('prostate.test',  package = 'SIS')
dim(prostate.test)
# 34x12601

prostate_data <- rbind(prostate.train, prostate.test)
dim(prostate_data)
# 136x12601


# 신경망 학습용 data set
# Train : Test = 8 : 2 나누기. (100번)
# seed 를 바꿔가며 100개의 서로 다른 train, test data set 생성.
# list() 객체에 저장.

prostate_train_x_list_num <- list()
prostate_train_y_list_num <- list()

prostate_test_x_list_num <- list()
prostate_test_y_list_num <- list()

for(i in 1:100){
  set.seed(i)
  train_index <- sample(1:nrow(prostate_data), 0.8 * nrow(prostate_data), replace = FALSE)
  prostate_train_x_num <- prostate_data[train_index, 1:12600]
  prostate_train_y_num <- prostate_data[train_index, 12601]
  
  prostate_test_x_num <- prostate_data[-train_index, 1:12600]
  prostate_test_y_num <- prostate_data[-train_index, 12601]
  
  prostate_train_x_list_num[[i]] <- prostate_train_x_num
  prostate_train_y_list_num[[i]] <- prostate_train_y_num
  
  prostate_test_x_list_num[[i]] <- prostate_test_x_num
  prostate_test_y_list_num[[i]] <- prostate_test_y_num
  
}


#################
## 신경망 학습 ##
#################
# https://mxnet.apache.org/api/r  참고.
library(mxnet)

prostate_model_list <- list()
prostate_dnn_test_predict_list <- list()
prostate_dnn_test_predict_label_list <- list()
prostate_dnn_test_confusion_list <- list()
prostate_dnn_test_acc_list <- list() # y_hat 과 test data set의 y 로 Accuracy 저장 list 할당.
prostate_dnn_exectime_list <- list()
prostate_dnn_train_logger_list <- list()
prostate_dnn_test_logger_list <- list()
prostate_dnn_test_auc_list <- list()

for(i in 1:100){ # 1:100
  prostate_train_x_scale <- scale(prostate_train_x_list_num[[i]])
  prostate_test_x_scale <- scale(prostate_test_x_list_num[[i]])
  
  prostate_train_x_datamatrix <- data.matrix(prostate_train_x_scale)
  prostate_test_x_datamatrix <- data.matrix(prostate_test_x_scale)
  
  prostate_train_y <- prostate_train_y_list_num[[i]]
  prostate_test_y <- prostate_test_y_list_num[[i]]
  
  mx.set.seed(2020)  # 가중치 초기값 고정.
  
  input <- mx.symbol.Variable("data")
  fc1 <- mx.symbol.FullyConnected(data = input, num.hidden = 20)  # Hyper parameter : 은닉노드 수
  act1 <- mx.symbol.Activation(data = fc1, act_type = "sigmoid")  # Hyper parameter : 활성화 함수 종류
  fc2 <- mx.symbol.FullyConnected(data = act1, num.hidden = 2)
  lro <- mx.symbol.SoftmaxOutput(data = fc2)  # Hyper parameter : 출력노드 함수 종류
  
  tic()
  
  logger <- mx.metric.logger$new()
  model <- mx.model.FeedForward.create(symbol = lro, X = prostate_train_x_datamatrix, y = prostate_train_y,
                                       eval.data = list(data = prostate_test_x_datamatrix, label = prostate_test_y),
                                       ctx = mx.gpu(), num.round = 100, optimizer = 'sgd', # Hyper parameter : iter 수, 최적화 함수 종류
                                       array.batch.size = 5, learning.rate = 0.001, momentum = 0.9, eval.metric = mx.metric.accuracy,
                                       verbose = T, epoch.end.callback = mx.callback.log.train.metric(1, logger))  # Hyper parameter : 배치 사이즈, 학습률, 모멘텀 값.
  
  exectime <- toc()
  exectime <- round(exectime$toc - exectime$tic, 5)
  
  prostate_model_list[[i]] <- model
  prostate_dnn_test_predict_list[[i]] <- predict(prostate_model_list[[i]], prostate_test_x_datamatrix)
  prostate_dnn_test_predict_label_list[[i]] <- max.col(t(prostate_dnn_test_predict_list[[i]])) - 1
  prostate_dnn_test_confusion_list[[i]] <- CrossTable(x = prostate_test_y_list_num[[i]], y = prostate_dnn_test_predict_label_list[[i]])
  prostate_dnn_test_acc_list[[i]] <- (prostate_dnn_test_confusion_list[[i]]$t[1] + prostate_dnn_test_confusion_list[[i]]$t[4])  / sum(prostate_dnn_test_confusion_list[[i]]$t)
  auc <- performance(prediction( predict(prostate_model_list[[i]], prostate_test_x_datamatrix)[2, ], prostate_test_y_list_num[[i]], label.ordering = c(0, 1)), measure = "auc")
  prostate_dnn_test_auc_list[[i]] <- auc@y.values[[1]]
  
  prostate_dnn_exectime_list[[i]] <- exectime
  prostate_dnn_train_logger_list[[i]] <- logger$train
  prostate_dnn_test_logger_list[[i]] <- logger$eval
}

prostate_dnn_test_acc_unlist <- unlist(prostate_dnn_test_acc_list)
prostate_dnn_test_auc_unlist <- unlist(prostate_dnn_test_auc_list)
prostate_dnn_exectime_unlist <- unlist(prostate_dnn_exectime_list)

prostate_dnn_train_logger_unlist <- data.frame( matrix(unlist(prostate_dnn_train_logger_list), ncol = 100))
prostate_dnn_test_logger_unlist <- data.frame( matrix(unlist(prostate_dnn_test_logger_list), ncol = 100))

# 결과저장
# write(t(prostate_dnn_test_acc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_DNN_result.txt")
# write(t(prostate_dnn_test_auc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_DNN_auc_result.txt")
# write(t(prostate_dnn_exectime_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_DNN_time.txt")
# write(t(prostate_dnn_train_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_DNN_train_logger.txt")
# write(t(prostate_dnn_test_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_DNN_test_logger.txt")
prostate_dnn_test_confusion_unlist <- data.frame()
for(i in 1:100){
  for(j in 1:4){
    prostate_dnn_test_confusion_unlist[i, j] <- unlist(prostate_dnn_test_confusion_list[[i]]$t)[j]
  }
}
prostate_dnn_test_confusion_unlist <- cbind(c(1:100), prostate_dnn_test_confusion_unlist)
colnames(prostate_dnn_test_confusion_unlist) <- c("iter","TN", "FN", "FP", "TP")
head(prostate_dnn_test_confusion_unlist)
tail(prostate_dnn_test_confusion_unlist)

# write.table(prostate_dnn_test_confusion_unlist, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_dnn_confusion.txt", col.names = TRUE)

prostate_dnn_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_DNN_result.txt")
prostate_dnn_test_acc_unlist <- cbind(prostate_dnn_test_acc_unlist, rep("DNN(변수전체)", 100))
names(prostate_dnn_test_acc_unlist) <- c("Accuracy", "Model")

prostate_dnn_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_DNN_AUC_result.txt")
prostate_dnn_test_auc_unlist <- cbind(prostate_dnn_test_auc_unlist, rep("DNN(변수전체)", 100))
names(prostate_dnn_test_auc_unlist) <- c("AUC", "Model")

prostate_dnn_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_DNN_time.txt")
prostate_dnn_exectime_unlist <- cbind(prostate_dnn_exectime_unlist, rep("DNN(변수전체)", 100))
names(prostate_dnn_exectime_unlist) <- c("Time", "Model")

# Accuracy 분포 확인.
summary(prostate_dnn_test_acc_unlist)

win.graph()
boxplot(prostate_dnn_test_acc_unlist[1],  main = "Prostate data. DNN 은닉층 1, 은닉노드 20. (전체변수)",
        xlab = "Iter = 100", ylab = "Accuracy", ylim = c(0, 1))
vioplot(prostate_dnn_test_acc_unlist[1], main = "Prostate data. DNN 은닉층 1, 은닉노드 20. (전체변수)",
        xlab = "Iter = 100", ylab = "Accuracy", ylim = c(0, 1))
points(mean(prostate_dnn_test_acc_unlist$Accuracy), col = "red", pch = 17) # mean 표시

# AUC 분포 확인.
summary(prostate_dnn_test_auc_unlist)

win.graph()
boxplot(prostate_dnn_test_auc_unlist[1],  main = "Prostate data. DNN 은닉층 1, 은닉노드 20. (전체변수)",
        xlab = "Iter = 100", ylab = "AUC", ylim = c(0, 1))
vioplot(prostate_dnn_test_auc_unlist[1], main = "Prostate data. DNN 은닉층 1, 은닉노드 20. (전체변수)",
        xlab = "Iter = 100", ylab = "AUC", ylim = c(0, 1))
points(mean(prostate_dnn_test_auc_unlist$AUC), col = "red", pch = 17) # mean 표시

# Time 분포 확인.
summary(prostate_dnn_exectime_unlist)

win.graph()
boxplot(prostate_dnn_exectime_unlist[1],  main = "Prostate data. DNN 은닉층 1, 은닉노드 5. (전체변수) 학습시간",
        xlab = "Iter = 100", ylab = "Time(단위 : 초)", ylim = c(0, 5))
vioplot(prostate_dnn_exectime_unlist[1], main = "Prostate data. DNN 은닉층 1, 은닉노드 5. (전체변수) 학습시간",
        xlab = "Iter = 100", ylab = "Time(단위 : 초)", ylim = c(0, 5))
points(mean(prostate_dnn_exectime_unlist$Time), col = "red", pch = 17) # mean 표시


### 2. SIS 변수선택 (prostate data) ###
### 2. SIS 변수선택 (prostate data) ###
### 2. SIS 변수선택 (prostate data) ###

dim(prostate_data)
# 136x12601

prostate_data_x <- prostate_data[ ,-12601]
prostate_data_x <- data.matrix(prostate_data_x)
prostate_data_x <- standardize(prostate_data_x)

prostate_data_y <- prostate_data[ ,12601]

## SCAD ##
## SCAD ##
## SCAD ##
prostate_SIS_model_SCAD <- SIS(prostate_data_x, prostate_data_y, family = 'binomial', tune = 'bic', penalty = "SCAD",
                               perm = TRUE, q = 0.9, greedy = TRUE, seed = 31)

prostate_SIS_model_SCAD
prostate_SIS_model_SCAD$ix
prostate_SIS_model_SCAD$fit$beta


## MCP ##
## MCP ##
## MCP ##
prostate_SIS_model_MCP <- SIS(prostate_data_x, prostate_data_y, family = 'binomial', tune = 'bic', penalty = "MCP",
                              perm = TRUE, q = 0.9, greedy = TRUE, seed = 31)

prostate_SIS_model_MCP
prostate_SIS_model_MCP$ix
prostate_SIS_model_MCP$fit$beta

## LASSO ##
## LASSO ##
## LASSO ##
prostate_SIS_model_LASSO <- SIS(prostate_data_x, prostate_data_y, family = 'binomial', tune = 'bic', penalty = "lasso",
                                perm = TRUE, q = 0.9, greedy = TRUE, seed = 31)

prostate_SIS_model_LASSO
prostate_SIS_model_LASSO$ix
prostate_SIS_model_LASSO$fit$beta

## 각 패널티별 변수선택 확인
prostate_SIS_model_SCAD$ix  # 4483 10431 11052 11200
prostate_SIS_model_MCP$ix  # 4483 11052
prostate_SIS_model_LASSO$ix  # 4483  6151  8610 10431 11052 11200

### 2.1 SIS SCAD 변수선택 (prostate data) 후 DNN ###
### 2.1 SIS SCAD 변수선택 (prostate data) 후 DNN ###
### 2.1 SIS SCAD 변수선택 (prostate data) 후 DNN ###

# 신경망 학습용 data set
# Train : Test = 8 : 2 나누기. (100번)
# seed 를 바꿔가며 100개의 서로 다른 train, test data set 생성.
# list() 객체에 저장.

prostate_SCAD_train_x_list_num <- list()
prostate_SCAD_train_y_list_num <- list()

prostate_SCAD_test_x_list_num <- list()
prostate_SCAD_test_y_list_num <- list()

for(i in 1:100){
  set.seed(i)
  train_index <- sample(1:nrow(prostate_data), 0.8 * nrow(prostate_data), replace = FALSE)
  prostate_train_x_num <- prostate_data[train_index, c(4483, 10431, 11052, 11200)]
  prostate_train_y_num <- prostate_data[train_index, 12601]
  
  prostate_test_x_num <- prostate_data[-train_index, c(4483, 10431, 11052, 11200)]
  prostate_test_y_num <- prostate_data[-train_index, 12601]
  
  prostate_SCAD_train_x_list_num[[i]] <- prostate_train_x_num
  prostate_SCAD_train_y_list_num[[i]] <- prostate_train_y_num
  
  prostate_SCAD_test_x_list_num[[i]] <- prostate_test_x_num
  prostate_SCAD_test_y_list_num[[i]] <- prostate_test_y_num
  
}


#################
## 신경망 학습 ##
#################
# https://mxnet.apache.org/api/r  참고.
library(mxnet)

prostate_model_list <- list()
prostate_SCAD_dnn_test_predict_list <- list()
prostate_SCAD_dnn_test_predict_label_list <- list()
prostate_SCAD_dnn_test_confusion_list <- list()
prostate_SCAD_dnn_test_acc_list <- list() # y_hat 과 test data set의 y 로 Accuracy 저장 list 할당.
prostate_SCAD_dnn_exectime_list <- list()
prostate_SCAD_dnn_train_logger_list <- list()
prostate_SCAD_dnn_test_logger_list <- list()
prostate_SCAD_dnn_test_auc_list <- list()

for(i in 1:100){ # 1:100
  prostate_train_x_scale <- scale(prostate_SCAD_train_x_list_num[[i]])
  prostate_test_x_scale <- scale(prostate_SCAD_test_x_list_num[[i]])
  
  prostate_train_x_datamatrix <- data.matrix(prostate_train_x_scale)
  prostate_test_x_datamatrix <- data.matrix(prostate_test_x_scale)
  
  prostate_train_y <- prostate_SCAD_train_y_list_num[[i]]
  prostate_test_y <- prostate_SCAD_test_y_list_num[[i]]
  
  mx.set.seed(2020)  # 가중치 초기값 고정.
  
  input <- mx.symbol.Variable("data")
  fc1 <- mx.symbol.FullyConnected(data = input, num.hidden = 20)  # Hyper parameter : 은닉노드 수
  act1 <- mx.symbol.Activation(data = fc1, act_type = "sigmoid")  # Hyper parameter : 활성화 함수 종류
  fc2 <- mx.symbol.FullyConnected(data = act1, num.hidden = 2)
  lro <- mx.symbol.SoftmaxOutput(data = fc2)  # Hyper parameter : 출력노드 함수 종류
  
  tic()
  
  logger <- mx.metric.logger$new()
  model <- mx.model.FeedForward.create(symbol = lro, X = prostate_train_x_datamatrix, y = prostate_train_y,
                                       eval.data = list(data = prostate_test_x_datamatrix, label = prostate_test_y),
                                       ctx = mx.gpu(), num.round = 600, optimizer = 'sgd', # Hyper parameter : iter 수, 최적화 함수 종류
                                       array.batch.size = 5, learning.rate = 0.001, momentum = 0.9, eval.metric = mx.metric.accuracy,
                                       verbose = T, epoch.end.callback = mx.callback.log.train.metric(1, logger))  # Hyper parameter : 배치 사이즈, 학습률, 모멘텀 값.
  
  exectime <- toc()
  exectime <- round(exectime$toc - exectime$tic, 5)
  
  prostate_model_list[[i]] <- model
  prostate_SCAD_dnn_test_predict_list[[i]] <- predict(prostate_model_list[[i]], prostate_test_x_datamatrix)
  prostate_SCAD_dnn_test_predict_label_list[[i]] <- max.col(t(prostate_SCAD_dnn_test_predict_list[[i]])) - 1
  prostate_SCAD_dnn_test_confusion_list[[i]] <- CrossTable(x = prostate_test_y_list_num[[i]], y = prostate_SCAD_dnn_test_predict_label_list[[i]])
  prostate_SCAD_dnn_test_acc_list[[i]] <- (prostate_SCAD_dnn_test_confusion_list[[i]]$t[1] + prostate_SCAD_dnn_test_confusion_list[[i]]$t[4])  / sum(prostate_SCAD_dnn_test_confusion_list[[i]]$t)
  auc <- performance(prediction( predict(prostate_model_list[[i]], prostate_test_x_datamatrix)[2, ], prostate_SCAD_test_y_list_num[[i]], label.ordering = c(0, 1)), measure = "auc")
  prostate_SCAD_dnn_test_auc_list[[i]] <- auc@y.values[[1]]
  
  prostate_SCAD_dnn_exectime_list[[i]] <- exectime
  prostate_SCAD_dnn_train_logger_list[[i]] <- logger$train
  prostate_SCAD_dnn_test_logger_list[[i]] <- logger$eval
}

prostate_SCAD_dnn_test_acc_unlist <- unlist(prostate_SCAD_dnn_test_acc_list)
prostate_SCAD_dnn_test_auc_unlist <- unlist(prostate_SCAD_dnn_test_auc_list)
prostate_SCAD_dnn_exectime_unlist <- unlist(prostate_SCAD_dnn_exectime_list)

prostate_SCAD_dnn_train_logger_unlist <- data.frame( matrix(unlist(prostate_SCAD_dnn_train_logger_list), ncol = 100))
prostate_SCAD_dnn_test_logger_unlist <- data.frame( matrix(unlist(prostate_SCAD_dnn_test_logger_list), ncol = 100))

# 결과저장
# write(t(prostate_SCAD_dnn_test_acc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_SCAD_dnn_result.txt")
# write(t(prostate_SCAD_dnn_test_auc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_SCAD_DNN_auc_result.txt")
# write(t(prostate_SCAD_dnn_exectime_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_SCAD_dnn_time.txt")
# write(t(prostate_SCAD_dnn_train_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_SCAD_dnn_train_logger.txt")
# write(t(prostate_SCAD_dnn_test_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_SCAD_dnn_test_logger.txt")
prostate_SCAD_dnn_test_confusion_unlist <- data.frame()
for(i in 1:100){
  for(j in 1:4){
    prostate_SCAD_dnn_test_confusion_unlist[i, j] <- unlist(prostate_SCAD_dnn_test_confusion_list[[i]]$t)[j]
  }
}
prostate_SCAD_dnn_test_confusion_unlist <- cbind(c(1:100), prostate_SCAD_dnn_test_confusion_unlist)
colnames(prostate_SCAD_dnn_test_confusion_unlist) <- c("iter","TN", "FN", "FP", "TP")
head(prostate_SCAD_dnn_test_confusion_unlist)
tail(prostate_SCAD_dnn_test_confusion_unlist)

# write.table(prostate_SCAD_dnn_test_confusion_unlist, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_SCAD_dnn_confusion.txt", col.names = TRUE)


prostate_SCAD_dnn_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_SCAD_DNN_result.txt")
prostate_SCAD_dnn_test_acc_unlist <- cbind(prostate_SCAD_dnn_test_acc_unlist, rep("DNN(SCAD 변수선택)", 100))
names(prostate_SCAD_dnn_test_acc_unlist) <- c("Accuracy", "Model")

prostate_SCAD_dnn_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_SCAD_DNN_auc_result.txt")
prostate_SCAD_dnn_test_auc_unlist <- cbind(prostate_SCAD_dnn_test_auc_unlist, rep("DNN(SCAD 변수선택)", 100))
names(prostate_SCAD_dnn_test_auc_unlist) <- c("AUC", "Model")

prostate_SCAD_dnn_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_SCAD_DNN_time.txt")
prostate_SCAD_dnn_exectime_unlist <- cbind(prostate_SCAD_dnn_exectime_unlist, rep("DNN(SCAD 변수선택)", 100))
names(prostate_SCAD_dnn_exectime_unlist) <- c("Time", "Model")

# Accuracy 분포 확인.
summary(prostate_SCAD_dnn_test_acc_unlist)

win.graph()
boxplot(prostate_SCAD_dnn_test_acc_unlist[1],  main = "Prostate data. DNN 은닉층 1, 은닉노드 20. (SCAD 변수선택)",
        xlab = "Iter = 600", ylab = "Accuracy", ylim = c(0, 1))
vioplot(prostate_SCAD_dnn_test_acc_unlist[1], main = "Prostate data. DNN 은닉층 1, 은닉노드 20. (SCAD 변수선택)",
        xlab = "Iter = 600", ylab = "Accuracy", ylim = c(0, 1))
points(mean(prostate_SCAD_dnn_test_acc_unlist$Accuracy), col = "red", pch = 17) # mean 표시

# AUC 분포 확인.
summary(prostate_SCAD_dnn_test_auc_unlist)

win.graph()
boxplot(prostate_SCAD_dnn_test_auc_unlist[1],  main = "Prostate data. DNN 은닉층 1, 은닉노드 20. (SCAD 변수선택)",
        xlab = "Iter = 600", ylab = "AUC", ylim = c(0, 1))
vioplot(prostate_SCAD_dnn_test_auc_unlist[1], main = "Prostate data. DNN 은닉층 1, 은닉노드 20. (SCAD 변수선택)",
        xlab = "Iter = 600", ylab = "AUC", ylim = c(0, 1))
points(mean(prostate_SCAD_dnn_test_auc_unlist$AUC), col = "red", pch = 17) # mean 표시

# Time 분포 확인.
summary(prostate_SCAD_dnn_exectime_unlist)

win.graph()
boxplot(prostate_SCAD_dnn_exectime_unlist[1],  main = "Prostate data. DNN 은닉층 1, 은닉노드 20. (SCAD 변수선택) 학습시간",
        xlab = "Iter = 600", ylab = "Time(단위 : 초)", ylim = c(15, 25))
vioplot(prostate_SCAD_dnn_exectime_unlist[1], main = "Prostate data. DNN 은닉층 1, 은닉노드 20. (SCAD 변수선택) 학습시간",
        xlab = "Iter = 600", ylab = "Time(단위 : 초)", ylim = c(15, 25))
points(mean(prostate_SCAD_dnn_exectime_unlist$Time), col = "red", pch = 17) # mean 표시

### 2.2 SIS MCP 변수선택 (prostate data) 후 DNN ###
### 2.2 SIS MCP 변수선택 (prostate data) 후 DNN ###
### 2.2 SIS MCP 변수선택 (prostate data) 후 DNN ###

# 신경망 학습용 data set
# Train : Test = 8 : 2 나누기. (100번)
# seed 를 바꿔가며 100개의 서로 다른 train, test data set 생성.
# list() 객체에 저장.

prostate_MCP_train_x_list_num <- list()
prostate_MCP_train_y_list_num <- list()

prostate_MCP_test_x_list_num <- list()
prostate_MCP_test_y_list_num <- list()

for(i in 1:100){
  set.seed(i)
  train_index <- sample(1:nrow(prostate_data), 0.8 * nrow(prostate_data), replace = FALSE)
  prostate_train_x_num <- prostate_data[train_index, c(4483, 11052)]
  prostate_train_y_num <- prostate_data[train_index, 12601]
  
  prostate_test_x_num <- prostate_data[-train_index, c(4483, 11052)]
  prostate_test_y_num <- prostate_data[-train_index, 12601]
  
  prostate_MCP_train_x_list_num[[i]] <- prostate_train_x_num
  prostate_MCP_train_y_list_num[[i]] <- prostate_train_y_num
  
  prostate_MCP_test_x_list_num[[i]] <- prostate_test_x_num
  prostate_MCP_test_y_list_num[[i]] <- prostate_test_y_num
  
}


#################
## 신경망 학습 ##
#################
# https://mxnet.apache.org/api/r  참고.
library(mxnet)

prostate_model_list <- list()
prostate_MCP_dnn_test_predict_list <- list()
prostate_MCP_dnn_test_predict_label_list <- list()
prostate_MCP_dnn_test_confusion_list <- list()
prostate_MCP_dnn_test_acc_list <- list() # y_hat 과 test data set의 y 로 Accuracy 저장 list 할당.
prostate_MCP_dnn_exectime_list <- list()
prostate_MCP_dnn_train_logger_list <- list()
prostate_MCP_dnn_test_logger_list <- list()
prostate_MCP_dnn_test_auc_list <- list()

for(i in 1:100){ # 1:100
  prostate_train_x_scale <- scale(prostate_MCP_train_x_list_num[[i]])
  prostate_test_x_scale <- scale(prostate_MCP_test_x_list_num[[i]])
  
  prostate_train_x_datamatrix <- data.matrix(prostate_train_x_scale)
  prostate_test_x_datamatrix <- data.matrix(prostate_test_x_scale)
  
  prostate_train_y <- prostate_MCP_train_y_list_num[[i]]
  prostate_test_y <- prostate_MCP_test_y_list_num[[i]]
  
  mx.set.seed(2020)  # 가중치 초기값 고정.
  
  input <- mx.symbol.Variable("data")
  fc1 <- mx.symbol.FullyConnected(data = input, num.hidden = 20)  # Hyper parameter : 은닉노드 수
  act1 <- mx.symbol.Activation(data = fc1, act_type = "sigmoid")  # Hyper parameter : 활성화 함수 종류
  fc2 <- mx.symbol.FullyConnected(data = act1, num.hidden = 2)
  lro <- mx.symbol.SoftmaxOutput(data = fc2)  # Hyper parameter : 출력노드 함수 종류
  
  tic()
  
  logger <- mx.metric.logger$new()
  model <- mx.model.FeedForward.create(symbol = lro, X = prostate_train_x_datamatrix, y = prostate_train_y,
                                       eval.data = list(data = prostate_test_x_datamatrix, label = prostate_test_y),
                                       ctx = mx.gpu(), num.round = 600, optimizer = 'sgd', # Hyper parameter : iter 수, 최적화 함수 종류
                                       array.batch.size = 5, learning.rate = 0.001, momentum = 0.9, eval.metric = mx.metric.accuracy,
                                       verbose = T, epoch.end.callback = mx.callback.log.train.metric(1, logger))  # Hyper parameter : 배치 사이즈, 학습률, 모멘텀 값.
  
  exectime <- toc()
  exectime <- round(exectime$toc - exectime$tic, 5)
  
  prostate_model_list[[i]] <- model
  prostate_MCP_dnn_test_predict_list[[i]] <- predict(prostate_model_list[[i]], prostate_test_x_datamatrix)
  prostate_MCP_dnn_test_predict_label_list[[i]] <- max.col(t(prostate_MCP_dnn_test_predict_list[[i]])) - 1
  prostate_MCP_dnn_test_confusion_list[[i]] <- CrossTable(x = prostate_test_y_list_num[[i]], y = prostate_MCP_dnn_test_predict_label_list[[i]])
  prostate_MCP_dnn_test_acc_list[[i]] <- (prostate_MCP_dnn_test_confusion_list[[i]]$t[1] + prostate_MCP_dnn_test_confusion_list[[i]]$t[4])  / sum(prostate_MCP_dnn_test_confusion_list[[i]]$t)
  auc <- performance(prediction( predict(prostate_model_list[[i]], prostate_test_x_datamatrix)[2, ], prostate_MCP_test_y_list_num[[i]], label.ordering = c(0, 1)), measure = "auc")
  prostate_MCP_dnn_test_auc_list[[i]] <- auc@y.values[[1]]
  
  prostate_MCP_dnn_exectime_list[[i]] <- exectime
  prostate_MCP_dnn_train_logger_list[[i]] <- logger$train
  prostate_MCP_dnn_test_logger_list[[i]] <- logger$eval
}

prostate_MCP_dnn_test_acc_unlist <- unlist(prostate_MCP_dnn_test_acc_list)
prostate_MCP_dnn_test_auc_unlist <- unlist(prostate_MCP_dnn_test_auc_list)
prostate_MCP_dnn_exectime_unlist <- unlist(prostate_MCP_dnn_exectime_list)

prostate_MCP_dnn_train_logger_unlist <- data.frame( matrix(unlist(prostate_MCP_dnn_train_logger_list), ncol = 100))
prostate_MCP_dnn_test_logger_unlist <- data.frame( matrix(unlist(prostate_MCP_dnn_test_logger_list), ncol = 100))

# 결과저장
# write(t(prostate_MCP_dnn_test_acc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_MCP_dnn_result.txt")
# write(t(prostate_MCP_dnn_test_auc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_MCP_DNN_auc_result.txt")
# write(t(prostate_MCP_dnn_exectime_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_MCP_dnn_time.txt")
# write(t(prostate_MCP_dnn_train_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_MCP_dnn_train_logger.txt")
# write(t(prostate_MCP_dnn_test_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_MCP_dnn_test_logger.txt")
prostate_MCP_dnn_test_confusion_unlist <- data.frame()
for(i in 1:100){
  for(j in 1:4){
    prostate_MCP_dnn_test_confusion_unlist[i, j] <- unlist(prostate_MCP_dnn_test_confusion_list[[i]]$t)[j]
  }
}
prostate_MCP_dnn_test_confusion_unlist <- cbind(c(1:100), prostate_MCP_dnn_test_confusion_unlist)
colnames(prostate_MCP_dnn_test_confusion_unlist) <- c("iter","TN", "FN", "FP", "TP")
head(prostate_MCP_dnn_test_confusion_unlist)
tail(prostate_MCP_dnn_test_confusion_unlist)

# write.table(prostate_MCP_dnn_test_confusion_unlist, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_MCP_dnn_confusion.txt", col.names = TRUE)


prostate_MCP_dnn_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_MCP_DNN_result.txt")
prostate_MCP_dnn_test_acc_unlist <- cbind(prostate_MCP_dnn_test_acc_unlist, rep("DNN(MCP 변수선택)", 100))
names(prostate_MCP_dnn_test_acc_unlist) <- c("Accuracy", "Model")

prostate_MCP_dnn_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_MCP_DNN_auc_result.txt")
prostate_MCP_dnn_test_auc_unlist <- cbind(prostate_MCP_dnn_test_auc_unlist, rep("DNN(MCP 변수선택)", 100))
names(prostate_MCP_dnn_test_auc_unlist) <- c("AUC", "Model")

prostate_MCP_dnn_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_MCP_DNN_time.txt")
prostate_MCP_dnn_exectime_unlist <- cbind(prostate_MCP_dnn_exectime_unlist, rep("DNN(MCP 변수선택)", 100))
names(prostate_MCP_dnn_exectime_unlist) <- c("Time", "Model")

# Accuracy 분포 확인.
summary(prostate_MCP_dnn_test_acc_unlist)

win.graph()
boxplot(prostate_MCP_dnn_test_acc_unlist[1],  main = "prostate data. DNN 은닉층 1, 은닉노드 20. (MCP 변수선택)",
        xlab = "Iter = 600", ylab = "Accuracy", ylim = c(0, 1))
vioplot(prostate_MCP_dnn_test_acc_unlist[1], main = "prostate data. DNN 은닉층 1, 은닉노드 20. (MCP 변수선택)",
        xlab = "Iter = 600", ylab = "Accuracy", ylim = c(0, 1))
points(mean(prostate_MCP_dnn_test_acc_unlist$Accuracy), col = "red", pch = 17) # mean 표시

# AUC 분포 확인.
summary(prostate_MCP_dnn_test_auc_unlist)

win.graph()
boxplot(prostate_MCP_dnn_test_auc_unlist[1],  main = "prostate data. DNN 은닉층 1, 은닉노드 20. (MCP 변수선택)",
        xlab = "Iter = 600", ylab = "AUC", ylim = c(0, 1))
vioplot(prostate_MCP_dnn_test_auc_unlist[1], main = "prostate data. DNN 은닉층 1, 은닉노드 20. (MCP 변수선택)",
        xlab = "Iter = 600", ylab = "AUC", ylim = c(0, 1))
points(mean(prostate_MCP_dnn_test_auc_unlist$AUC), col = "red", pch = 17) # mean 표시

# Time 분포 확인.
summary(prostate_MCP_dnn_exectime_unlist)

win.graph()
boxplot(prostate_MCP_dnn_exectime_unlist[1],  main = "prostate data. DNN 은닉층 1, 은닉노드 20. (MCP 변수선택) 학습시간",
        xlab = "Iter = 600", ylab = "Time(단위 : 초)", ylim = c(15, 26))
vioplot(prostate_MCP_dnn_exectime_unlist[1], main = "prostate data. DNN 은닉층 1, 은닉노드 20. (MCP 변수선택) 학습시간",
        xlab = "Iter = 600", ylab = "Time(단위 : 초)", ylim = c(15, 26))
points(mean(prostate_MCP_dnn_exectime_unlist$Time), col = "red", pch = 17) # mean 표시

### 2.3 SIS LASSO 변수선택 (prostate data) 후 DNN ###
### 2.3 SIS LASSO 변수선택 (prostate data) 후 DNN ###
### 2.3 SIS LASSO 변수선택 (prostate data) 후 DNN ###

# 신경망 학습용 data set
# Train : Test = 8 : 2 나누기. (100번)
# seed 를 바꿔가며 100개의 서로 다른 train, test data set 생성.
# list() 객체에 저장.

prostate_LASSO_train_x_list_num <- list()
prostate_LASSO_train_y_list_num <- list()

prostate_LASSO_test_x_list_num <- list()
prostate_LASSO_test_y_list_num <- list()

for(i in 1:100){
  set.seed(i)
  train_index <- sample(1:nrow(prostate_data), 0.8 * nrow(prostate_data), replace = FALSE)
  prostate_train_x_num <- prostate_data[train_index, c(4483, 6151, 8610, 10431, 11052, 11200)]
  prostate_train_y_num <- prostate_data[train_index, 12601]
  
  prostate_test_x_num <- prostate_data[-train_index, c(4483, 6151, 8610, 10431, 11052, 11200)]
  prostate_test_y_num <- prostate_data[-train_index, 12601]
  
  prostate_LASSO_train_x_list_num[[i]] <- prostate_train_x_num
  prostate_LASSO_train_y_list_num[[i]] <- prostate_train_y_num
  
  prostate_LASSO_test_x_list_num[[i]] <- prostate_test_x_num
  prostate_LASSO_test_y_list_num[[i]] <- prostate_test_y_num
  
}


#################
## 신경망 학습 ##
#################
# https://mxnet.apache.org/api/r  참고.
library(mxnet)

prostate_model_list <- list()
prostate_LASSO_dnn_test_predict_list <- list()
prostate_LASSO_dnn_test_predict_label_list <- list()
prostate_LASSO_dnn_test_confusion_list <- list()
prostate_LASSO_dnn_test_acc_list <- list() # y_hat 과 test data set의 y 로 Accuracy 저장 list 할당.
prostate_LASSO_dnn_exectime_list <- list()
prostate_LASSO_dnn_train_logger_list <- list()
prostate_LASSO_dnn_test_logger_list <- list()
prostate_LASSO_dnn_test_auc_list <- list()

for(i in 1:100){ # 1:100
  prostate_train_x_scale <- scale(prostate_LASSO_train_x_list_num[[i]])
  prostate_test_x_scale <- scale(prostate_LASSO_test_x_list_num[[i]])
  
  prostate_train_x_datamatrix <- data.matrix(prostate_train_x_scale)
  prostate_test_x_datamatrix <- data.matrix(prostate_test_x_scale)
  
  prostate_train_y <- prostate_LASSO_train_y_list_num[[i]]
  prostate_test_y <- prostate_LASSO_test_y_list_num[[i]]
  
  mx.set.seed(2020)  # 가중치 초기값 고정.
  
  input <- mx.symbol.Variable("data")
  fc1 <- mx.symbol.FullyConnected(data = input, num.hidden = 20)  # Hyper parameter : 은닉노드 수
  act1 <- mx.symbol.Activation(data = fc1, act_type = "sigmoid")  # Hyper parameter : 활성화 함수 종류
  fc2 <- mx.symbol.FullyConnected(data = act1, num.hidden = 2)
  lro <- mx.symbol.SoftmaxOutput(data = fc2)  # Hyper parameter : 출력노드 함수 종류
  
  tic()
  
  logger <- mx.metric.logger$new()
  model <- mx.model.FeedForward.create(symbol = lro, X = prostate_train_x_datamatrix, y = prostate_train_y,
                                       eval.data = list(data = prostate_test_x_datamatrix, label = prostate_test_y),
                                       ctx = mx.gpu(), num.round = 600, optimizer = 'sgd', # Hyper parameter : iter 수, 최적화 함수 종류
                                       array.batch.size = 5, learning.rate = 0.001, momentum = 0.9, eval.metric = mx.metric.accuracy,
                                       verbose = T, epoch.end.callback = mx.callback.log.train.metric(1, logger))  # Hyper parameter : 배치 사이즈, 학습률, 모멘텀 값.
  
  exectime <- toc()
  exectime <- round(exectime$toc - exectime$tic, 5)
  
  prostate_model_list[[i]] <- model
  prostate_LASSO_dnn_test_predict_list[[i]] <- predict(prostate_model_list[[i]], prostate_test_x_datamatrix)
  prostate_LASSO_dnn_test_predict_label_list[[i]] <- max.col(t(prostate_LASSO_dnn_test_predict_list[[i]])) - 1
  prostate_LASSO_dnn_test_confusion_list[[i]] <- CrossTable(x = prostate_test_y_list_num[[i]], y = prostate_LASSO_dnn_test_predict_label_list[[i]])
  prostate_LASSO_dnn_test_acc_list[[i]] <- (prostate_LASSO_dnn_test_confusion_list[[i]]$t[1] + prostate_LASSO_dnn_test_confusion_list[[i]]$t[4])  / sum(prostate_LASSO_dnn_test_confusion_list[[i]]$t)
  auc <- performance(prediction( predict(prostate_model_list[[i]], prostate_test_x_datamatrix)[2, ], prostate_LASSO_test_y_list_num[[i]], label.ordering = c(0, 1)), measure = "auc")
  prostate_LASSO_dnn_test_auc_list[[i]] <- auc@y.values[[1]]
  
  prostate_LASSO_dnn_exectime_list[[i]] <- exectime
  prostate_LASSO_dnn_train_logger_list[[i]] <- logger$train
  prostate_LASSO_dnn_test_logger_list[[i]] <- logger$eval
}

prostate_LASSO_dnn_test_acc_unlist <- unlist(prostate_LASSO_dnn_test_acc_list)
prostate_LASSO_dnn_test_auc_unlist <- unlist(prostate_LASSO_dnn_test_auc_list)
prostate_LASSO_dnn_exectime_unlist <- unlist(prostate_LASSO_dnn_exectime_list)

prostate_LASSO_dnn_train_logger_unlist <- data.frame( matrix(unlist(prostate_LASSO_dnn_train_logger_list), ncol = 100))
prostate_LASSO_dnn_test_logger_unlist <- data.frame( matrix(unlist(prostate_LASSO_dnn_test_logger_list), ncol = 100))

# 결과저장
# write(t(prostate_LASSO_dnn_test_acc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_LASSO_dnn_result.txt")
# write(t(prostate_LASSO_dnn_test_auc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_LASSO_DNN_auc_result.txt")
# write(t(prostate_LASSO_dnn_exectime_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_LASSO_dnn_time.txt")
# write(t(prostate_LASSO_dnn_train_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_LASSO_dnn_train_logger.txt")
# write(t(prostate_LASSO_dnn_test_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_LASSO_dnn_test_logger.txt")
prostate_LASSO_dnn_test_confusion_unlist <- data.frame()
for(i in 1:100){
  for(j in 1:4){
    prostate_LASSO_dnn_test_confusion_unlist[i, j] <- unlist(prostate_LASSO_dnn_test_confusion_list[[i]]$t)[j]
  }
}
prostate_LASSO_dnn_test_confusion_unlist <- cbind(c(1:100), prostate_LASSO_dnn_test_confusion_unlist)
colnames(prostate_LASSO_dnn_test_confusion_unlist) <- c("iter","TN", "FN", "FP", "TP")
head(prostate_LASSO_dnn_test_confusion_unlist)
tail(prostate_LASSO_dnn_test_confusion_unlist)

# write.table(prostate_LASSO_dnn_test_confusion_unlist, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_LASSO_dnn_confusion.txt", col.names = TRUE)


prostate_LASSO_dnn_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_LASSO_DNN_result.txt")
prostate_LASSO_dnn_test_acc_unlist <- cbind(prostate_LASSO_dnn_test_acc_unlist, rep("DNN(LASSO 변수선택)", 100))
names(prostate_LASSO_dnn_test_acc_unlist) <- c("Accuracy", "Model")

prostate_LASSO_dnn_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_LASSO_DNN_auc_result.txt")
prostate_LASSO_dnn_test_auc_unlist <- cbind(prostate_LASSO_dnn_test_auc_unlist, rep("DNN(LASSO 변수선택)", 100))
names(prostate_LASSO_dnn_test_auc_unlist) <- c("AUC", "Model")

prostate_LASSO_dnn_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_LASSO_DNN_time.txt")
prostate_LASSO_dnn_exectime_unlist <- cbind(prostate_LASSO_dnn_exectime_unlist, rep("DNN(LASSO 변수선택)", 100))
names(prostate_LASSO_dnn_exectime_unlist) <- c("Time", "Model")

# Accuracy 분포 확인.
summary(prostate_LASSO_dnn_test_acc_unlist)

win.graph()
boxplot(prostate_LASSO_dnn_test_acc_unlist[1],  main = "prostate data. DNN 은닉층 1, 은닉노드 20. (LASSO 변수선택)",
        xlab = "Iter = 600", ylab = "Accuracy", ylim = c(0, 1))
vioplot(prostate_LASSO_dnn_test_acc_unlist[1], main = "prostate data. DNN 은닉층 1, 은닉노드 20. (LASSO 변수선택)",
        xlab = "Iter = 600", ylab = "Accuracy", ylim = c(0, 1))
points(mean(prostate_LASSO_dnn_test_acc_unlist$Accuracy), col = "red", pch = 17) # mean 표시

# AUC 분포 확인.
summary(prostate_LASSO_dnn_test_auc_unlist)

win.graph()
boxplot(prostate_LASSO_dnn_test_auc_unlist[1],  main = "prostate data. DNN 은닉층 1, 은닉노드 20. (LASSO 변수선택)",
        xlab = "Iter = 600", ylab = "AUC", ylim = c(0, 1))
vioplot(prostate_LASSO_dnn_test_auc_unlist[1], main = "prostate data. DNN 은닉층 1, 은닉노드 20. (LASSO 변수선택)",
        xlab = "Iter = 600", ylab = "AUC", ylim = c(0, 1))
points(mean(prostate_LASSO_dnn_test_auc_unlist$AUC), col = "red", pch = 17) # mean 표시

# Time 분포 확인.
summary(prostate_LASSO_dnn_exectime_unlist)

win.graph()
boxplot(prostate_LASSO_dnn_exectime_unlist[1],  main = "prostate data. DNN 은닉층 1, 은닉노드 20. (LASSO 변수선택) 학습시간",
        xlab = "Iter = 600", ylab = "Time(단위 : 초)", ylim = c(15, 26))
vioplot(prostate_LASSO_dnn_exectime_unlist[1], main = "prostate data. DNN 은닉층 1, 은닉노드 20. (LASSO 변수선택) 학습시간",
        xlab = "Iter = 600", ylab = "Time(단위 : 초)", ylim = c(15, 26))
points(mean(prostate_LASSO_dnn_exectime_unlist$Time), col = "red", pch = 17) # mean 표시
