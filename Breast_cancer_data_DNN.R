library(devtools)

# install_github('ramhiser/datamicroarray')
library(datamicroarray)
library(SIS)
library(gmodels)
library(tictoc)
library(vioplot)



data('chin', package='datamicroarray')


### 데이터 불러들이기
head(chin)
tail(chin)

str(chin)
dim(chin$x)
chin$y


data.hd = chin$x
dim(data.hd)


y = (chin$y)
levels(y) = c("0","1")
y = as.numeric(as.character(y))
y

chin_data <- cbind(data.hd, y)
chin_data <- data.frame(chin_data)
dim(chin_data)
# 118x22216


summary(chin_data[, 1:15])
summary(chin_data[, 22211:22216])


# 신경망 학습용 data set
# Train : Test = 8 : 2 나누기. (100번)
# seed 를 바꿔가며 100개의 서로 다른 train, test data set 생성.
# list() 객체에 저장.

chin_train_x_list_num <- list()
chin_train_y_list_num <- list()

chin_test_x_list_num <- list()
chin_test_y_list_num <- list()

for(i in 1:100){
  set.seed(i)
  train_index <- sample(1:nrow(chin_data), 0.8 * nrow(chin_data), replace = FALSE)
  chin_train_x_num <- chin_data[train_index, 1:22215]
  chin_train_y_num <- chin_data[train_index, 22216]
  
  chin_test_x_num <- chin_data[-train_index, 1:22215]
  chin_test_y_num <- chin_data[-train_index, 22216]
  
  chin_train_x_list_num[[i]] <- chin_train_x_num
  chin_train_y_list_num[[i]] <- chin_train_y_num
  
  chin_test_x_list_num[[i]] <- chin_test_x_num
  chin_test_y_list_num[[i]] <- chin_test_y_num
  
}


#################
## 신경망 학습 ##
#################
# https://mxnet.apache.org/api/r  참고.
library(mxnet)

chin_model_list <- list()
chin_dnn_test_predict_list <- list()
chin_dnn_test_predict_label_list <- list()
chin_dnn_test_confusion_list <- list()
chin_dnn_test_acc_list <- list() # y_hat 과 test data set의 y 로 Accuracy 저장 list 할당.
chin_dnn_exectime_list <- list()
chin_dnn_train_logger_list <- list()
chin_dnn_test_logger_list <- list()
chin_dnn_test_auc_list <- list()

for(i in 1:100){ # 1:100
  chin_train_x_scale <- scale(chin_train_x_list_num[[i]])
  chin_test_x_scale <- scale(chin_test_x_list_num[[i]])
  
  chin_train_x_datamatrix <- data.matrix(chin_train_x_scale)
  chin_test_x_datamatrix <- data.matrix(chin_test_x_scale)
  
  chin_train_y <- chin_train_y_list_num[[i]]
  chin_test_y <- chin_test_y_list_num[[i]]
  
  mx.set.seed(2020)  # 가중치 초기값 고정.
  
  input <- mx.symbol.Variable("data")
  fc1 <- mx.symbol.FullyConnected(data = input, num.hidden = 20)  # Hyper parameter : 은닉노드 수
  act1 <- mx.symbol.Activation(data = fc1, act_type = "sigmoid")  # Hyper parameter : 활성화 함수 종류
  fc2 <- mx.symbol.FullyConnected(data = act1, num.hidden = 2)
  lro <- mx.symbol.SoftmaxOutput(data = fc2)  # Hyper parameter : 출력노드 함수 종류
  
  tic()
  
  logger <- mx.metric.logger$new()
  model <- mx.model.FeedForward.create(symbol = lro, X = chin_train_x_datamatrix, y = chin_train_y,
                                       eval.data = list(data = chin_test_x_datamatrix, label = chin_test_y),
                                       ctx = mx.gpu(), num.round = 30, optimizer = 'sgd', # Hyper parameter : iter 수, 최적화 함수 종류
                                       array.batch.size = 5, learning.rate = 0.001, momentum = 0.9, eval.metric = mx.metric.accuracy,
                                       verbose = T, epoch.end.callback = mx.callback.log.train.metric(1, logger))  # Hyper parameter : 배치 사이즈, 학습률, 모멘텀 값.
  
  exectime <- toc()
  exectime <- round(exectime$toc - exectime$tic, 5)
  
  chin_model_list[[i]] <- model
  chin_dnn_test_predict_list[[i]] <- predict(chin_model_list[[i]], chin_test_x_datamatrix)
  chin_dnn_test_predict_label_list[[i]] <- max.col(t(chin_dnn_test_predict_list[[i]])) - 1
  chin_dnn_test_confusion_list[[i]] <- CrossTable(x = chin_test_y_list_num[[i]], y = chin_dnn_test_predict_label_list[[i]])
  chin_dnn_test_acc_list[[i]] <- (chin_dnn_test_confusion_list[[i]]$t[1] + chin_dnn_test_confusion_list[[i]]$t[4])  / sum(chin_dnn_test_confusion_list[[i]]$t)
  auc <- performance(prediction( predict(chin_model_list[[i]], chin_test_x_datamatrix)[2, ], chin_test_y_list_num[[i]], label.ordering = c(0, 1)), measure = "auc")
  chin_dnn_test_auc_list[[i]] <- auc@y.values[[1]]
  
  chin_dnn_exectime_list[[i]] <- exectime
  chin_dnn_train_logger_list[[i]] <- logger$train
  chin_dnn_test_logger_list[[i]] <- logger$eval
}

chin_dnn_test_acc_unlist <- unlist(chin_dnn_test_acc_list)
chin_dnn_test_auc_unlist <- unlist(chin_dnn_test_auc_list)
chin_dnn_exectime_unlist <- unlist(chin_dnn_exectime_list)

chin_dnn_train_logger_unlist <- data.frame( matrix(unlist(chin_dnn_train_logger_list), ncol = 100))
chin_dnn_test_logger_unlist <- data.frame( matrix(unlist(chin_dnn_test_logger_list), ncol = 100))

# 결과저장
# write(t(chin_dnn_test_acc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_DNN_result.txt")
# write(t(chin_dnn_test_auc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_DNN_auc_result.txt")
# write(t(chin_dnn_exectime_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_DNN_time.txt")
# write(t(chin_dnn_train_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_DNN_train_logger.txt")
# write(t(chin_dnn_test_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_DNN_test_logger.txt")
chin_dnn_test_confusion_unlist <- data.frame()
for(i in 1:100){
  for(j in 1:4){
    chin_dnn_test_confusion_unlist[i, j] <- unlist(chin_dnn_test_confusion_list[[i]]$t)[j]
  }
}
chin_dnn_test_confusion_unlist <- cbind(c(1:100), chin_dnn_test_confusion_unlist)
colnames(chin_dnn_test_confusion_unlist) <- c("iter","TN", "FN", "FP", "TP")
head(chin_dnn_test_confusion_unlist)
tail(chin_dnn_test_confusion_unlist)

# write.table(chin_dnn_test_confusion_unlist, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_dnn_confusion.txt", col.names = TRUE)


chin_dnn_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_DNN_result.txt")
chin_dnn_test_acc_unlist <- cbind(chin_dnn_test_acc_unlist, rep("DNN(변수전체)", 100))
names(chin_dnn_test_acc_unlist) <- c("Accuracy", "Model")

chin_dnn_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_DNN_AUC_result.txt")
chin_dnn_test_auc_unlist <- cbind(chin_dnn_test_auc_unlist, rep("DNN(변수전체)", 100))
names(chin_dnn_test_auc_unlist) <- c("AUC", "Model")

chin_dnn_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_DNN_time.txt")
chin_dnn_exectime_unlist <- cbind(chin_dnn_exectime_unlist, rep("DNN(변수전체)", 100))
names(chin_dnn_exectime_unlist) <- c("Time", "Model")

# Accuracy 분포 확인.
summary(chin_dnn_test_acc_unlist)

win.graph()
boxplot(chin_dnn_test_acc_unlist[1],  main = "Chin data. DNN 은닉층 1, 은닉노드 20. (전체변수)",
        xlab = "Iter = 30", ylab = "Accuracy", ylim = c(0, 1))
vioplot(chin_dnn_test_acc_unlist[1], main = "Chin data. DNN 은닉층 1, 은닉노드 20. (전체변수)",
        xlab = "Iter = 30", ylab = "Accuracy", ylim = c(0, 1))
points(mean(chin_dnn_test_acc_unlist$Accuracy), col = "red", pch = 17) # mean 표시

# AUC 분포 확인.
summary(chin_dnn_test_auc_unlist)

win.graph()
boxplot(chin_dnn_test_auc_unlist[1],  main = "Chin data. DNN 은닉층 1, 은닉노드 20. (전체변수)",
        xlab = "Iter = 30", ylab = "AUC", ylim = c(0, 1))
vioplot(chin_dnn_test_auc_unlist[1], main = "Chin data. DNN 은닉층 1, 은닉노드 20. (전체변수)",
        xlab = "Iter = 30", ylab = "AUC", ylim = c(0, 1))
points(mean(chin_dnn_test_auc_unlist$AUC), col = "red", pch = 17) # mean 표시

# Time 분포 확인.
summary(chin_dnn_exectime_unlist)

win.graph()
boxplot(chin_dnn_exectime_unlist[1],  main = "Chin data. DNN 은닉층 1, 은닉노드 20. (전체변수) 학습시간",
        xlab = "Iter = 30", ylab = "Time(단위 : 초)", ylim = c(0, 3))
vioplot(chin_dnn_exectime_unlist[1], main = "Chin data. DNN 은닉층 1, 은닉노드 20. (전체변수) 학습시간",
        xlab = "Iter = 30", ylab = "Time(단위 : 초)", ylim = c(0, 3))
points(mean(chin_dnn_exectime_unlist$Time), col = "red", pch = 17) # mean 표시

### 2. SIS 변수선택 (chin data) ###
### 2. SIS 변수선택 (chin data) ###
### 2. SIS 변수선택 (chin data) ###

dim(chin_data)
# 118x22216

chin_data_x <- chin_data[ ,-22216]
chin_data_x <- data.matrix(chin_data_x)
chin_data_x <- standardize(chin_data_x)

chin_data_y <- chin_data[ ,22216]

## SCAD ##
## SCAD ##
## SCAD ##
chin_SIS_model_SCAD <- SIS(chin_data_x, chin_data_y, family = 'binomial', tune = 'bic', penalty = "SCAD",
                           perm = TRUE, q = 0.9, greedy = TRUE, seed = 31)

chin_SIS_model_SCAD
chin_SIS_model_SCAD$ix
chin_SIS_model_SCAD$fit$beta


## MCP ##
## MCP ##
## MCP ##
chin_SIS_model_MCP <- SIS(chin_data_x, chin_data_y, family = 'binomial', tune = 'bic', penalty = "MCP",
                          perm = TRUE, q = 0.9, greedy = TRUE, seed = 31)

chin_SIS_model_MCP
chin_SIS_model_MCP$ix
chin_SIS_model_MCP$fit$beta

## LASSO ##
## LASSO ##
## LASSO ##
chin_SIS_model_LASSO <- SIS(chin_data_x, chin_data_y, family = 'binomial', tune = 'bic', penalty = "lasso",
                            perm = TRUE, q = 0.9, greedy = TRUE, seed = 31)

chin_SIS_model_LASSO
chin_SIS_model_LASSO$ix
chin_SIS_model_LASSO$fit$beta

## 각 패널티별 변수선택 확인
chin_SIS_model_SCAD$ix  # 4752
chin_SIS_model_MCP$ix  # 4752
chin_SIS_model_LASSO$ix  # 4752 7404 7454 13543 13795

### 2.1 SIS SCAD & MCP 변수선택 (chin data) 후 DNN ###
### 2.1 SIS SCAD & MCP 변수선택 (chin data) 후 DNN ###
### 2.1 SIS SCAD & MCP 변수선택 (chin data) 후 DNN ###

# 신경망 학습용 data set
# Train : Test = 8 : 2 나누기. (100번)
# seed 를 바꿔가며 100개의 서로 다른 train, test data set 생성.
# list() 객체에 저장.

chin_SCAD_MCP_train_x_list_num <- list()
chin_SCAD_MCP_train_y_list_num <- list()

chin_SCAD_MCP_test_x_list_num <- list()
chin_SCAD_MCP_test_y_list_num <- list()

for(i in 1:100){
  set.seed(i)
  train_index <- sample(1:nrow(chin_data), 0.8 * nrow(chin_data), replace = FALSE)
  chin_train_x_num <- chin_data[train_index, c(4752)]
  chin_train_y_num <- chin_data[train_index, 22216]
  
  chin_test_x_num <- chin_data[-train_index, c(4752)]
  chin_test_y_num <- chin_data[-train_index, 22216]
  
  chin_SCAD_MCP_train_x_list_num[[i]] <- chin_train_x_num
  chin_SCAD_MCP_train_y_list_num[[i]] <- chin_train_y_num
  
  chin_SCAD_MCP_test_x_list_num[[i]] <- chin_test_x_num
  chin_SCAD_MCP_test_y_list_num[[i]] <- chin_test_y_num
  
}


#################
## 신경망 학습 ##
#################
# https://mxnet.apache.org/api/r  참고.
library(mxnet)

chin_SCAD_MCP_model_list <- list()
chin_SCAD_MCP_dnn_test_predict_list <- list()
chin_SCAD_MCP_dnn_test_predict_label_list <- list()
chin_SCAD_MCP_dnn_test_confusion_list <- list()
chin_SCAD_MCP_dnn_test_acc_list <- list() # y_hat 과 test data set의 y 로 Accuracy 저장 list 할당.
chin_SCAD_MCP_dnn_exectime_list <- list()
chin_SCAD_MCP_dnn_train_logger_list <- list()
chin_SCAD_MCP_dnn_test_logger_list <- list()
chin_SCAD_MCP_dnn_test_auc_list <- list()

for(i in 1:100){ # 1:100
  chin_SCAD_MCP_train_x_scale <- scale(chin_SCAD_MCP_train_x_list_num[[i]])
  chin_SCAD_MCP_test_x_scale <- scale(chin_SCAD_MCP_test_x_list_num[[i]])
  
  chin_SCAD_MCP_train_x_datamatrix <- data.matrix(chin_SCAD_MCP_train_x_scale)
  chin_SCAD_MCP_test_x_datamatrix <- data.matrix(chin_SCAD_MCP_test_x_scale)
  
  chin_SCAD_MCP_train_y <- chin_SCAD_MCP_train_y_list_num[[i]]
  chin_SCAD_MCP_test_y <- chin_SCAD_MCP_test_y_list_num[[i]]
  
  mx.set.seed(2020)  # 가중치 초기값 고정.
  
  input <- mx.symbol.Variable("data")
  fc1 <- mx.symbol.FullyConnected(data = input, num.hidden = 20)  # Hyper parameter : 은닉노드 수
  act1 <- mx.symbol.Activation(data = fc1, act_type = "sigmoid")  # Hyper parameter : 활성화 함수 종류
  fc2 <- mx.symbol.FullyConnected(data = act1, num.hidden = 2)
  lro <- mx.symbol.SoftmaxOutput(data = fc2)  # Hyper parameter : 출력노드 함수 종류
  
  tic()
  
  logger <- mx.metric.logger$new()
  model <- mx.model.FeedForward.create(symbol = lro, X = chin_SCAD_MCP_train_x_datamatrix, y = chin_SCAD_MCP_train_y,
                                       eval.data = list(data = chin_SCAD_MCP_test_x_datamatrix, label = chin_SCAD_MCP_test_y),
                                       ctx = mx.gpu(), num.round = 200, optimizer = 'sgd', # Hyper parameter : iter 수, 최적화 함수 종류
                                       array.batch.size = 5, learning.rate = 0.001, momentum = 0.9, eval.metric = mx.metric.accuracy,
                                       verbose = T, epoch.end.callback = mx.callback.log.train.metric(1, logger))  # Hyper parameter : 배치 사이즈, 학습률, 모멘텀 값.
  
  exectime <- toc()
  exectime <- round(exectime$toc - exectime$tic, 5)
  
  chin_SCAD_MCP_model_list[[i]] <- model
  chin_SCAD_MCP_dnn_test_predict_list[[i]] <- predict(chin_SCAD_MCP_model_list[[i]], chin_SCAD_MCP_test_x_datamatrix)
  chin_SCAD_MCP_dnn_test_predict_label_list[[i]] <- max.col(t(chin_SCAD_MCP_dnn_test_predict_list[[i]])) - 1
  chin_SCAD_MCP_dnn_test_confusion_list[[i]] <- CrossTable(x = chin_SCAD_MCP_test_y_list_num[[i]], y = chin_SCAD_MCP_dnn_test_predict_label_list[[i]])
  chin_SCAD_MCP_dnn_test_acc_list[[i]] <- (chin_SCAD_MCP_dnn_test_confusion_list[[i]]$t[1] + chin_SCAD_MCP_dnn_test_confusion_list[[i]]$t[4])  / sum(chin_SCAD_MCP_dnn_test_confusion_list[[i]]$t)
  auc <- performance(prediction( predict(chin_SCAD_MCP_model_list[[i]], chin_SCAD_MCP_test_x_datamatrix)[2, ], chin_SCAD_MCP_test_y_list_num[[i]], label.ordering = c(0, 1)), measure = "auc")
  chin_SCAD_MCP_dnn_test_auc_list[[i]] <- auc@y.values[[1]]
  
  chin_SCAD_MCP_dnn_exectime_list[[i]] <- exectime
  chin_SCAD_MCP_dnn_train_logger_list[[i]] <- logger$train
  chin_SCAD_MCP_dnn_test_logger_list[[i]] <- logger$eval
}

chin_SCAD_MCP_dnn_test_acc_unlist <- unlist(chin_SCAD_MCP_dnn_test_acc_list)
chin_SCAD_MCP_dnn_test_auc_unlist <- unlist(chin_SCAD_MCP_dnn_test_auc_list)
chin_SCAD_MCP_dnn_exectime_unlist <- unlist(chin_SCAD_MCP_dnn_exectime_list)

chin_SCAD_MCP_dnn_train_logger_unlist <- data.frame( matrix(unlist(chin_SCAD_MCP_dnn_train_logger_list), ncol = 100))
chin_SCAD_MCP_dnn_test_logger_unlist <- data.frame( matrix(unlist(chin_SCAD_MCP_dnn_test_logger_list), ncol = 100))

# 결과저장
# write(t(chin_SCAD_MCP_dnn_test_acc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_SCAD_MCP_DNN_result.txt")
# write(t(chin_SCAD_MCP_dnn_test_auc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_SCAD_MCP_DNN_auc_result.txt")
# write(t(chin_SCAD_MCP_dnn_exectime_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_SCAD_MCP_DNN_time.txt")
# write(t(chin_SCAD_MCP_dnn_train_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_SCAD_MCP_DNN_train_logger.txt")
# write(t(chin_SCAD_MCP_dnn_test_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_SCAD_MCP_DNN_test_logger.txt")
chin_SCAD_MCP_dnn_test_confusion_unlist <- data.frame()
for(i in 1:100){
  for(j in 1:4){
    chin_SCAD_MCP_dnn_test_confusion_unlist[i, j] <- unlist(chin_SCAD_MCP_dnn_test_confusion_list[[i]]$t)[j]
  }
}
chin_SCAD_MCP_dnn_test_confusion_unlist <- cbind(c(1:100), chin_SCAD_MCP_dnn_test_confusion_unlist)
colnames(chin_SCAD_MCP_dnn_test_confusion_unlist) <- c("iter","TN", "FN", "FP", "TP")
head(chin_SCAD_MCP_dnn_test_confusion_unlist)
tail(chin_SCAD_MCP_dnn_test_confusion_unlist)

# write.table(chin_SCAD_MCP_dnn_test_confusion_unlist, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_SCAD_MCP_dnn_confusion.txt", col.names = TRUE)


chin_SCAD_MCP_dnn_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_SCAD_MCP_DNN_result.txt")
chin_SCAD_MCP_dnn_test_acc_unlist <- cbind(chin_SCAD_MCP_dnn_test_acc_unlist, rep("DNN(SCAD & MCP 변수선택)", 100))
names(chin_SCAD_MCP_dnn_test_acc_unlist) <- c("Accuracy", "Model")

chin_SCAD_MCP_dnn_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_SCAD_MCP_DNN_auc_result.txt")
chin_SCAD_MCP_dnn_test_auc_unlist <- cbind(chin_SCAD_MCP_dnn_test_auc_unlist, rep("DNN(SCAD 변수선택)", 100))
names(chin_SCAD_MCP_dnn_test_auc_unlist) <- c("AUC", "Model")

chin_SCAD_MCP_dnn_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_SCAD_MCP_DNN_time.txt")
chin_SCAD_MCP_dnn_exectime_unlist <- cbind(chin_SCAD_MCP_dnn_exectime_unlist, rep("DNN(SCAD & MCP 변수선택)", 100))
names(chin_SCAD_MCP_dnn_exectime_unlist) <- c("Time", "Model")

# Accuracy 분포 확인.
summary(chin_SCAD_MCP_dnn_test_acc_unlist)

win.graph()
boxplot(chin_SCAD_MCP_dnn_test_acc_unlist[1],  main = "chin data. DNN 은닉층 1, 은닉노드 20. (SCAD & MCP 변수선택)",
        xlab = "Iter = 200", ylab = "Accuracy", ylim = c(0, 1))
vioplot(chin_SCAD_MCP_dnn_test_acc_unlist[1], main = "chin data. DNN 은닉층 1, 은닉노드 20. (SCAD & MCP 변수선택)",
        xlab = "Iter = 200", ylab = "Accuracy", ylim = c(0, 1))
points(mean(chin_SCAD_MCP_dnn_test_acc_unlist$Accuracy), col = "red", pch = 17) # mean 표시

# AUC 분포 확인.
summary(chin_SCAD_MCP_dnn_test_auc_unlist)

win.graph()
boxplot(chin_SCAD_MCP_dnn_test_auc_unlist[1],  main = "Chin data. DNN 은닉층 1, 은닉노드 20. (SCAD & MCP 변수선택)",
        xlab = "Iter = 200", ylab = "AUC", ylim = c(0, 1))
vioplot(chin_SCAD_MCP_dnn_test_auc_unlist[1], main = "Chin data. DNN 은닉층 1, 은닉노드 20. (SCAD & MCP 변수선택)",
        xlab = "Iter = 200", ylab = "AUC", ylim = c(0, 1))
points(mean(chin_SCAD_MCP_dnn_test_auc_unlist$AUC), col = "red", pch = 17) # mean 표시

# Time 분포 확인.
summary(chin_SCAD_MCP_dnn_exectime_unlist)

win.graph()
boxplot(chin_SCAD_MCP_dnn_exectime_unlist[1],  main = "chin data. DNN 은닉층 1, 은닉노드 20. (SCAD & MCP 변수선택) 학습시간",
        xlab = "Iter = 200", ylab = "Time(단위 : 초)", ylim = c(0, 8))
vioplot(chin_SCAD_MCP_dnn_exectime_unlist[1], main = "chin data. DNN 은닉층 1, 은닉노드 20. (SCAD & MCP 변수선택) 학습시간",
        xlab = "Iter = 200", ylab = "Time(단위 : 초)", ylim = c(0, 8))
points(mean(chin_SCAD_MCP_dnn_exectime_unlist$Time), col = "red", pch = 17) # mean 표시


### 2.3 SIS LASSO 변수선택 (chin data) 후 DNN ###
### 2.3 SIS LASSO 변수선택 (chin data) 후 DNN ###
### 2.3 SIS LASSO 변수선택 (chin data) 후 DNN ###

# 신경망 학습용 data set
# Train : Test = 8 : 2 나누기. (100번)
# seed 를 바꿔가며 100개의 서로 다른 train, test data set 생성.
# list() 객체에 저장.

chin_LASSO_train_x_list_num <- list()
chin_LASSO_train_y_list_num <- list()

chin_LASSO_test_x_list_num <- list()
chin_LASSO_test_y_list_num <- list()

for(i in 1:100){
  set.seed(i)
  train_index <- sample(1:nrow(chin_data), 0.8 * nrow(chin_data), replace = FALSE)
  chin_train_x_num <- chin_data[train_index, c(4752, 7404, 7454, 13543, 13795)]
  chin_train_y_num <- chin_data[train_index, 22216]
  
  chin_test_x_num <- chin_data[-train_index, c(4752, 7404, 7454, 13543, 13795)]
  chin_test_y_num <- chin_data[-train_index, 22216]
  
  chin_LASSO_train_x_list_num[[i]] <- chin_train_x_num
  chin_LASSO_train_y_list_num[[i]] <- chin_train_y_num
  
  chin_LASSO_test_x_list_num[[i]] <- chin_test_x_num
  chin_LASSO_test_y_list_num[[i]] <- chin_test_y_num
  
}


#################
## 신경망 학습 ##
#################
# https://mxnet.apache.org/api/r  참고.
library(mxnet)

chin_LASSO_model_list <- list()
chin_LASSO_dnn_test_predict_list <- list()
chin_LASSO_dnn_test_predict_label_list <- list()
chin_LASSO_dnn_test_confusion_list <- list()
chin_LASSO_dnn_test_acc_list <- list() # y_hat 과 test data set의 y 로 Accuracy 저장 list 할당.
chin_LASSO_dnn_exectime_list <- list()
chin_LASSO_dnn_train_logger_list <- list()
chin_LASSO_dnn_test_logger_list <- list()
chin_LASSO_dnn_test_auc_list <- list()

for(i in 1:100){ # 1:100
  chin_LASSO_train_x_scale <- scale(chin_LASSO_train_x_list_num[[i]])
  chin_LASSO_test_x_scale <- scale(chin_LASSO_test_x_list_num[[i]])
  
  chin_LASSO_train_x_datamatrix <- data.matrix(chin_LASSO_train_x_scale)
  chin_LASSO_test_x_datamatrix <- data.matrix(chin_LASSO_test_x_scale)
  
  chin_LASSO_train_y <- chin_LASSO_train_y_list_num[[i]]
  chin_LASSO_test_y <- chin_LASSO_test_y_list_num[[i]]
  
  mx.set.seed(2020)  # 가중치 초기값 고정.
  
  input <- mx.symbol.Variable("data")
  fc1 <- mx.symbol.FullyConnected(data = input, num.hidden = 20)  # Hyper parameter : 은닉노드 수
  act1 <- mx.symbol.Activation(data = fc1, act_type = "sigmoid")  # Hyper parameter : 활성화 함수 종류
  fc2 <- mx.symbol.FullyConnected(data = act1, num.hidden = 2)
  lro <- mx.symbol.SoftmaxOutput(data = fc2)  # Hyper parameter : 출력노드 함수 종류
  
  tic()
  
  logger <- mx.metric.logger$new()
  model <- mx.model.FeedForward.create(symbol = lro, X = chin_LASSO_train_x_datamatrix, y = chin_LASSO_train_y,
                                       eval.data = list(data = chin_LASSO_test_x_datamatrix, label = chin_LASSO_test_y),
                                       ctx = mx.gpu(), num.round = 200, optimizer = 'sgd', # Hyper parameter : iter 수, 최적화 함수 종류
                                       array.batch.size = 5, learning.rate = 0.001, momentum = 0.9, eval.metric = mx.metric.accuracy,
                                       verbose = T, epoch.end.callback = mx.callback.log.train.metric(1, logger))  # Hyper parameter : 배치 사이즈, 학습률, 모멘텀 값.
  
  exectime <- toc()
  exectime <- round(exectime$toc - exectime$tic, 5)
  
  chin_LASSO_model_list[[i]] <- model
  chin_LASSO_dnn_test_predict_list[[i]] <- predict(chin_LASSO_model_list[[i]], chin_LASSO_test_x_datamatrix)
  chin_LASSO_dnn_test_predict_label_list[[i]] <- max.col(t(chin_LASSO_dnn_test_predict_list[[i]])) - 1
  chin_LASSO_dnn_test_confusion_list[[i]] <- CrossTable(x = chin_LASSO_test_y_list_num[[i]], y = chin_LASSO_dnn_test_predict_label_list[[i]])
  chin_LASSO_dnn_test_acc_list[[i]] <- (chin_LASSO_dnn_test_confusion_list[[i]]$t[1] + chin_LASSO_dnn_test_confusion_list[[i]]$t[4])  / sum(chin_LASSO_dnn_test_confusion_list[[i]]$t)
  auc <- performance(prediction( predict(chin_LASSO_model_list[[i]], chin_LASSO_test_x_datamatrix)[2, ], chin_LASSO_test_y_list_num[[i]], label.ordering = c(0, 1)), measure = "auc")
  chin_LASSO_dnn_test_auc_list[[i]] <- auc@y.values[[1]]
  
  chin_LASSO_dnn_exectime_list[[i]] <- exectime
  chin_LASSO_dnn_train_logger_list[[i]] <- logger$train
  chin_LASSO_dnn_test_logger_list[[i]] <- logger$eval
}

chin_LASSO_dnn_test_acc_unlist <- unlist(chin_LASSO_dnn_test_acc_list)
chin_LASSO_dnn_test_auc_unlist <- unlist(chin_LASSO_dnn_test_auc_list)
chin_LASSO_dnn_exectime_unlist <- unlist(chin_LASSO_dnn_exectime_list)

chin_LASSO_dnn_train_logger_unlist <- data.frame( matrix(unlist(chin_LASSO_dnn_train_logger_list), ncol = 100))
chin_LASSO_dnn_test_logger_unlist <- data.frame( matrix(unlist(chin_LASSO_dnn_test_logger_list), ncol = 100))

# 결과저장
# write(t(chin_LASSO_dnn_test_acc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_LASSO_DNN_result.txt")
# write(t(chin_LASSO_dnn_test_auc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_LASSO_DNN_auc_result.txt")
# write(t(chin_LASSO_dnn_exectime_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_LASSO_DNN_time.txt")
# write(t(chin_LASSO_dnn_train_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_LASSO_DNN_train_logger.txt")
# write(t(chin_LASSO_dnn_test_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_LASSO_DNN_test_logger.txt")
chin_LASSO_dnn_test_confusion_unlist <- data.frame()
for(i in 1:100){
  for(j in 1:4){
    chin_LASSO_dnn_test_confusion_unlist[i, j] <- unlist(chin_LASSO_dnn_test_confusion_list[[i]]$t)[j]
  }
}
chin_LASSO_dnn_test_confusion_unlist <- cbind(c(1:100), chin_LASSO_dnn_test_confusion_unlist)
colnames(chin_LASSO_dnn_test_confusion_unlist) <- c("iter","TN", "FN", "FP", "TP")
head(chin_LASSO_dnn_test_confusion_unlist)
tail(chin_LASSO_dnn_test_confusion_unlist)  

# write.table(chin_LASSO_dnn_test_confusion_unlist, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_LASSO_dnn_confusion.txt", col.names = TRUE)

chin_LASSO_dnn_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_LASSO_DNN_result.txt")
chin_LASSO_dnn_test_acc_unlist <- cbind(chin_LASSO_dnn_test_acc_unlist, rep("DNN(LASSO 변수선택)", 100))
names(chin_LASSO_dnn_test_acc_unlist) <- c("Accuracy", "Model")

chin_LASSO_dnn_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_LASSO_DNN_auc_result.txt")
chin_LASSO_dnn_test_auc_unlist <- cbind(chin_LASSO_dnn_test_auc_unlist, rep("DNN(LASSO 변수선택)", 100))
names(chin_LASSO_dnn_test_auc_unlist) <- c("AUC", "Model")

chin_LASSO_dnn_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_LASSO_DNN_time.txt")
chin_LASSO_dnn_exectime_unlist <- cbind(chin_LASSO_dnn_exectime_unlist, rep("DNN(LASSO 변수선택)", 100))
names(chin_LASSO_dnn_exectime_unlist) <- c("Time", "Model")

# Accuracy 분포 확인.
summary(chin_LASSO_dnn_test_acc_unlist)

win.graph()
boxplot(chin_LASSO_dnn_test_acc_unlist[1],  main = "chin data. DNN 은닉층 1, 은닉노드 20. (LASSO 변수선택)",
        xlab = "Iter = 200", ylab = "Accuracy", ylim = c(0, 1))
vioplot(chin_LASSO_dnn_test_acc_unlist[1], main = "chin data. DNN 은닉층 1, 은닉노드 20. (LASSO 변수선택)",
        xlab = "Iter = 200", ylab = "Accuracy", ylim = c(0, 1))
points(mean(chin_LASSO_dnn_test_acc_unlist$Accuracy), col = "red", pch = 17) # mean 표시

# AUC 분포 확인.
summary(chin_LASSO_dnn_test_auc_unlist)

win.graph()
boxplot(chin_LASSO_dnn_test_auc_unlist[1],  main = "chin data. DNN 은닉층 1, 은닉노드 20. (LASSO 변수선택)",
        xlab = "Iter = 200", ylab = "AUC", ylim = c(0, 1))
vioplot(chin_LASSO_dnn_test_auc_unlist[1], main = "chin data. DNN 은닉층 1, 은닉노드 20. (LASSO 변수선택)",
        xlab = "Iter = 200", ylab = "AUC", ylim = c(0, 1))
points(mean(chin_LASSO_dnn_test_auc_unlist$AUC), col = "red", pch = 17) # mean 표시

# Time 분포 확인.
summary(chin_LASSO_dnn_exectime_unlist)

win.graph()
boxplot(chin_LASSO_dnn_exectime_unlist[1],  main = "chin data. DNN 은닉층 1, 은닉노드 20. (LASSO 변수선택) 학습시간",
        xlab = "Iter = 200", ylab = "Time(단위 : 초)", ylim = c(0, 8))
vioplot(chin_LASSO_dnn_exectime_unlist[1], main = "chin data. DNN 은닉층 1, 은닉노드 20. (LASSO 변수선택) 학습시간",
        xlab = "Iter = 200", ylab = "Time(단위 : 초)", ylim = c(0, 8))
points(mean(chin_LASSO_dnn_exectime_unlist$Time), col = "red", pch = 17) # mean 표시
