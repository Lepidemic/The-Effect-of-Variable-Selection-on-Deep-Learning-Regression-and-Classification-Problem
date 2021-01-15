
library(SIS)
library(gmodels)
library(tictoc)
library(vioplot)


data('leukemia.train', package='SIS')
dim(leukemia.train)
# dim(leukemia.train): 38x7130

data('leukemia.test', package='SIS')
dim(leukemia.test)
# dim(leukemia.test): 34x7130

leukemia_data <- rbind(leukemia.train, leukemia.test)
dim(leukemia_data)
# 72x7130


summary(leukemia_data[, 1:15])
summary(leukemia_data[, 7115:7130])


# 신경망 학습용 data set
# Train : Test = 8 : 2 나누기. (100번)
# seed 를 바꿔가며 100개의 서로 다른 train, test data set 생성.
# list() 객체에 저장.

leukemia_train_x_list_num <- list()
leukemia_train_y_list_num <- list()

leukemia_test_x_list_num <- list()
leukemia_test_y_list_num <- list()

for(i in 1:100){
  set.seed(i)
  train_index <- sample(1:nrow(leukemia_data), 0.8 * nrow(leukemia_data), replace = FALSE)
  leukemia_train_x_num <- leukemia_data[train_index, 1:7129]
  leukemia_train_y_num <- leukemia_data[train_index, 7130]
  
  leukemia_test_x_num <- leukemia_data[-train_index, 1:7129]
  leukemia_test_y_num <- leukemia_data[-train_index, 7130]
  
  leukemia_train_x_list_num[[i]] <- leukemia_train_x_num
  leukemia_train_y_list_num[[i]] <- leukemia_train_y_num
  
  leukemia_test_x_list_num[[i]] <- leukemia_test_x_num
  leukemia_test_y_list_num[[i]] <- leukemia_test_y_num
  
}


#################
## 신경망 학습 ##
#################
# https://mxnet.apache.org/api/r  참고.
library(mxnet)

leukemia_model_list <- list()
leukemia_dnn_test_predict_list <- list()
leukemia_dnn_test_predict_label_list <- list()
leukemia_dnn_test_confusion_list <- list()
leukemia_dnn_test_acc_list <- list() # y_hat 과 test data set의 y 로 Accuracy 저장 list 할당.
leukemia_dnn_exectime_list <- list()
leukemia_dnn_train_logger_list <- list()
leukemia_dnn_test_logger_list <- list()
leukemia_dnn_test_auc_list <- list()

for(i in 1:100){ # 1:100
  leukemia_train_x_scale <- scale(leukemia_train_x_list_num[[i]])
  leukemia_test_x_scale <- scale(leukemia_test_x_list_num[[i]])
  
  leukemia_train_x_datamatrix <- data.matrix(leukemia_train_x_scale)
  leukemia_test_x_datamatrix <- data.matrix(leukemia_test_x_scale)
  
  leukemia_train_y <- leukemia_train_y_list_num[[i]]
  leukemia_test_y <- leukemia_test_y_list_num[[i]]
  
  mx.set.seed(2020)  # 가중치 초기값 고정.
  
  input <- mx.symbol.Variable("data")
  fc1 <- mx.symbol.FullyConnected(data = input, num.hidden = 20)  # Hyper parameter : 은닉노드 수
  act1 <- mx.symbol.Activation(data = fc1, act_type = "sigmoid")  # Hyper parameter : 활성화 함수 종류
  fc2 <- mx.symbol.FullyConnected(data = act1, num.hidden = 2)
  lro <- mx.symbol.SoftmaxOutput(data = fc2)  # Hyper parameter : 출력노드 함수 종류
  
  tic()
  
  logger <- mx.metric.logger$new()
  model <- mx.model.FeedForward.create(symbol = lro, X = leukemia_train_x_datamatrix, y = leukemia_train_y,
                                       eval.data = list(data = leukemia_test_x_datamatrix, label = leukemia_test_y),
                                       ctx = mx.gpu(), num.round = 150, optimizer = 'sgd', # Hyper parameter : iter 수, 최적화 함수 종류
                                       array.batch.size = 5, learning.rate = 0.001, momentum = 0.9, eval.metric = mx.metric.accuracy,
                                       verbose = T, epoch.end.callback = mx.callback.log.train.metric(1, logger))  # Hyper parameter : 배치 사이즈, 학습률, 모멘텀 값.
  
  exectime <- toc()
  exectime <- round(exectime$toc - exectime$tic, 5)
  
  leukemia_model_list[[i]] <- model
  leukemia_dnn_test_predict_list[[i]] <- predict(leukemia_model_list[[i]], leukemia_test_x_datamatrix)
  leukemia_dnn_test_predict_label_list[[i]] <- max.col(t(leukemia_dnn_test_predict_list[[i]])) - 1
  leukemia_dnn_test_confusion_list[[i]] <- CrossTable(x = leukemia_test_y_list_num[[i]], y = leukemia_dnn_test_predict_label_list[[i]])
  leukemia_dnn_test_acc_list[[i]] <- (leukemia_dnn_test_confusion_list[[i]]$t[1] + leukemia_dnn_test_confusion_list[[i]]$t[4])  / sum(leukemia_dnn_test_confusion_list[[i]]$t)
  auc <- performance(prediction( predict(leukemia_model_list[[i]], leukemia_test_x_datamatrix)[2, ], leukemia_test_y_list_num[[i]], label.ordering = c(0, 1)), measure = "auc")
  leukemia_dnn_test_auc_list[[i]] <- auc@y.values[[1]]
  
  leukemia_dnn_exectime_list[[i]] <- exectime
  leukemia_dnn_train_logger_list[[i]] <- logger$train
  leukemia_dnn_test_logger_list[[i]] <- logger$eval
}

leukemia_dnn_test_acc_unlist <- unlist(leukemia_dnn_test_acc_list)
leukemia_dnn_test_auc_unlist <- unlist(leukemia_dnn_test_auc_list)
leukemia_dnn_exectime_unlist <- unlist(leukemia_dnn_exectime_list)

leukemia_dnn_train_logger_unlist <- data.frame( matrix(unlist(leukemia_dnn_train_logger_list), ncol = 100))
leukemia_dnn_test_logger_unlist <- data.frame( matrix(unlist(leukemia_dnn_test_logger_list), ncol = 100))

# 결과저장
# write(t(leukemia_dnn_test_acc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_DNN_result.txt")
# write(t(leukemia_dnn_test_auc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_DNN_auc_result.txt")
# write(t(leukemia_dnn_exectime_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_DNN_time.txt")
# write(t(leukemia_dnn_train_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_DNN_train_logger.txt")
# write(t(leukemia_dnn_test_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_DNN_test_logger.txt")
leukemia_dnn_test_confusion_unlist <- data.frame()
for(i in 1:100){
  for(j in 1:4){
    leukemia_dnn_test_confusion_unlist[i, j] <- unlist(leukemia_dnn_test_confusion_list[[i]]$t)[j]
  }
}
leukemia_dnn_test_confusion_unlist <- cbind(c(1:100), leukemia_dnn_test_confusion_unlist)
colnames(leukemia_dnn_test_confusion_unlist) <- c("iter","TN", "FN", "FP", "TP")
head(leukemia_dnn_test_confusion_unlist)
tail(leukemia_dnn_test_confusion_unlist)

# write.table(leukemia_dnn_test_confusion_unlist, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_dnn_confusion.txt", col.names = TRUE)


leukemia_dnn_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_DNN_result.txt")
leukemia_dnn_test_acc_unlist <- cbind(leukemia_dnn_test_acc_unlist, rep("DNN(변수전체)", 100))
names(leukemia_dnn_test_acc_unlist) <- c("Accuracy", "Model")

leukemia_dnn_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_DNN_auc_result.txt")
leukemia_dnn_test_auc_unlist <- cbind(leukemia_dnn_test_auc_unlist, rep("DNN(변수전체)", 100))
names(leukemia_dnn_test_auc_unlist) <- c("AUC", "Model")

leukemia_dnn_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_DNN_time.txt")
leukemia_dnn_exectime_unlist <- cbind(leukemia_dnn_exectime_unlist, rep("DNN(변수전체)", 100))
names(leukemia_dnn_exectime_unlist) <- c("Time", "Model")

# Accuracy 분포 확인.
summary(leukemia_dnn_test_acc_unlist)

win.graph()
boxplot(leukemia_dnn_test_acc_unlist[1],  main = "Leukemia data. DNN 은닉층 1, 은닉노드 20. (전체변수)",
        xlab = "Iter = 150", ylab = "Accuracy", ylim = c(0, 1))
vioplot(leukemia_dnn_test_acc_unlist[1], main = "Leukemia data. DNN 은닉층 1, 은닉노드 20. (전체변수)",
        xlab = "Iter = 150", ylab = "Accuracy", ylim = c(0, 1))
points(mean(leukemia_dnn_test_acc_unlist$Accuracy), col = "red", pch = 17) # mean 표시

# AUC 분포 확인.
summary(leukemia_dnn_test_auc_unlist)

win.graph()
boxplot(leukemia_dnn_test_auc_unlist[1],  main = "Leukemia data. DNN 은닉층 1, 은닉노드 20. (전체변수)",
        xlab = "Iter = 150", ylab = "AUC", ylim = c(0, 1))
vioplot(leukemia_dnn_test_auc_unlist[1], main = "Leukemia data. DNN 은닉층 1, 은닉노드 20. (전체변수)",
        xlab = "Iter = 150", ylab = "AUC", ylim = c(0, 1))
points(mean(leukemia_dnn_test_auc_unlist$AUC), col = "red", pch = 17) # mean 표시

# Time 분포 확인.
summary(leukemia_dnn_exectime_unlist)

win.graph()
boxplot(leukemia_dnn_exectime_unlist[1],  main = "Leukemia data. DNN 은닉층 1, 은닉노드 20. (전체변수) 학습시간",
        xlab = "Iter = 150", ylab = "Time(단위 : 초)", ylim = c(0, 4))
vioplot(leukemia_dnn_exectime_unlist[1], main = "Leukemia data. DNN 은닉층 1, 은닉노드 20. (전체변수) 학습시간",
        xlab = "Iter = 150", ylab = "Time(단위 : 초)", ylim = c(0, 4))
points(mean(leukemia_dnn_exectime_unlist$Time), col = "red", pch = 17) # mean 표시

### 2. SIS 변수선택 (leukemia data) ###
### 2. SIS 변수선택 (leukemia data) ###
### 2. SIS 변수선택 (leukemia data) ###

dim(leukemia_data)
# 72x7130

leukemia_data_x <- leukemia_data[ ,-7130]
leukemia_data_x <- data.matrix(leukemia_data_x)
leukemia_data_x <- standardize(leukemia_data_x)

leukemia_data_y <- leukemia_data[ ,7130]

## SCAD ##
## SCAD ##
## SCAD ##
leukemia_SIS_model_SCAD <- SIS(leukemia_data_x, leukemia_data_y, family = 'binomial', tune = 'bic', penalty = "SCAD",
               perm = TRUE, q = 0.9, greedy = TRUE, seed = 31)

leukemia_SIS_model_SCAD
leukemia_SIS_model_SCAD$ix
leukemia_SIS_model_SCAD$fit$beta


## MCP ##
## MCP ##
## MCP ##
leukemia_SIS_model_MCP <- SIS(leukemia_data_x, leukemia_data_y, family = 'binomial', tune = 'bic', penalty = "MCP",
                               perm = TRUE, q = 0.9, greedy = TRUE, seed = 31)
 
leukemia_SIS_model_MCP
leukemia_SIS_model_MCP$ix
leukemia_SIS_model_MCP$fit$beta

## LASSO ##
## LASSO ##
## LASSO ##
leukemia_SIS_model_LASSO <- SIS(leukemia_data_x, leukemia_data_y, family = 'binomial', tune = 'bic', penalty = "lasso",
                              perm = TRUE, q = 0.9, greedy = TRUE, seed = 31)

leukemia_SIS_model_LASSO
leukemia_SIS_model_LASSO$ix
leukemia_SIS_model_LASSO$fit$beta

## 각 패널티별 변수선택 확인
leukemia_SIS_model_SCAD$ix  # 1144 2597 4196 4847
leukemia_SIS_model_MCP$ix  # 1144 4847
leukemia_SIS_model_LASSO$ix  # 1144 2684 4847 6855

### 2.1 SIS SCAD 변수선택 (leukemia data) 후 DNN ###
### 2.1 SIS SCAD 변수선택 (leukemia data) 후 DNN ###
### 2.1 SIS SCAD 변수선택 (leukemia data) 후 DNN ###

# 신경망 학습용 data set
# Train : Test = 8 : 2 나누기. (100번)
# seed 를 바꿔가며 100개의 서로 다른 train, test data set 생성.
# list() 객체에 저장.

leukemia_SCAD_train_x_list_num <- list()
leukemia_SCAD_train_y_list_num <- list()

leukemia_SCAD_test_x_list_num <- list()
leukemia_SCAD_test_y_list_num <- list()

for(i in 1:100){
  set.seed(i)
  train_index <- sample(1:nrow(leukemia_data), 0.8 * nrow(leukemia_data), replace = FALSE)
  leukemia_train_x_num <- leukemia_data[train_index, c(1144, 2597, 4196, 4847)]
  leukemia_train_y_num <- leukemia_data[train_index, 7130]
  
  leukemia_test_x_num <- leukemia_data[-train_index, c(1144, 2597, 4196, 4847)]
  leukemia_test_y_num <- leukemia_data[-train_index, 7130]
  
  leukemia_SCAD_train_x_list_num[[i]] <- leukemia_train_x_num
  leukemia_SCAD_train_y_list_num[[i]] <- leukemia_train_y_num
  
  leukemia_SCAD_test_x_list_num[[i]] <- leukemia_test_x_num
  leukemia_SCAD_test_y_list_num[[i]] <- leukemia_test_y_num
  
}


#################
## 신경망 학습 ##
#################
# https://mxnet.apache.org/api/r  참고.
library(mxnet)

leukemia_SCAD_model_list <- list()
leukemia_SCAD_dnn_test_predict_list <- list()
leukemia_SCAD_dnn_test_predict_label_list <- list()
leukemia_SCAD_dnn_test_confusion_list <- list()
leukemia_SCAD_dnn_test_acc_list <- list() # y_hat 과 test data set의 y 로 Accuracy 저장 list 할당.
leukemia_SCAD_dnn_exectime_list <- list()
leukemia_SCAD_dnn_train_logger_list <- list()
leukemia_SCAD_dnn_test_logger_list <- list()
leukemia_SCAD_dnn_test_auc_list <- list()

for(i in 1:100){ # 1:100
  leukemia_SCAD_train_x_scale <- scale(leukemia_SCAD_train_x_list_num[[i]])
  leukemia_SCAD_test_x_scale <- scale(leukemia_SCAD_test_x_list_num[[i]])
  
  leukemia_SCAD_train_x_datamatrix <- data.matrix(leukemia_SCAD_train_x_scale)
  leukemia_SCAD_test_x_datamatrix <- data.matrix(leukemia_SCAD_test_x_scale)
  
  leukemia_SCAD_train_y <- leukemia_SCAD_train_y_list_num[[i]]
  leukemia_SCAD_test_y <- leukemia_SCAD_test_y_list_num[[i]]
  
  mx.set.seed(2020)  # 가중치 초기값 고정.
  
  input <- mx.symbol.Variable("data")
  fc1 <- mx.symbol.FullyConnected(data = input, num.hidden = 20)  # Hyper parameter : 은닉노드 수
  act1 <- mx.symbol.Activation(data = fc1, act_type = "sigmoid")  # Hyper parameter : 활성화 함수 종류
  fc2 <- mx.symbol.FullyConnected(data = act1, num.hidden = 2)
  lro <- mx.symbol.SoftmaxOutput(data = fc2)  # Hyper parameter : 출력노드 함수 종류
  
  tic()
  
  logger <- mx.metric.logger$new()
  model <- mx.model.FeedForward.create(symbol = lro, X = leukemia_SCAD_train_x_datamatrix, y = leukemia_SCAD_train_y,
                                       eval.data = list(data = leukemia_SCAD_test_x_datamatrix, label = leukemia_SCAD_test_y),
                                       ctx = mx.gpu(), num.round = 300, optimizer = 'sgd', # Hyper parameter : iter 수, 최적화 함수 종류
                                       array.batch.size = 5, learning.rate = 0.001, momentum = 0.9, eval.metric = mx.metric.accuracy,
                                       verbose = T, epoch.end.callback = mx.callback.log.train.metric(1, logger))  # Hyper parameter : 배치 사이즈, 학습률, 모멘텀 값.
  
  exectime <- toc()
  exectime <- round(exectime$toc - exectime$tic, 5)
  
  leukemia_SCAD_model_list[[i]] <- model
  leukemia_SCAD_dnn_test_predict_list[[i]] <- predict(leukemia_SCAD_model_list[[i]], leukemia_SCAD_test_x_datamatrix)
  leukemia_SCAD_dnn_test_predict_label_list[[i]] <- max.col(t(leukemia_SCAD_dnn_test_predict_list[[i]])) - 1
  leukemia_SCAD_dnn_test_confusion_list[[i]] <- CrossTable(x = leukemia_SCAD_test_y_list_num[[i]], y = leukemia_SCAD_dnn_test_predict_label_list[[i]])
  leukemia_SCAD_dnn_test_acc_list[[i]] <- (leukemia_SCAD_dnn_test_confusion_list[[i]]$t[1] + leukemia_SCAD_dnn_test_confusion_list[[i]]$t[4])  / sum(leukemia_SCAD_dnn_test_confusion_list[[i]]$t)
  auc <- performance(prediction( predict(leukemia_SCAD_model_list[[i]], leukemia_SCAD_test_x_datamatrix)[2, ], leukemia_SCAD_test_y_list_num[[i]], label.ordering = c(0, 1)), measure = "auc")
  leukemia_SCAD_dnn_test_auc_list[[i]] <- auc@y.values[[1]]
  
  leukemia_SCAD_dnn_exectime_list[[i]] <- exectime
  leukemia_SCAD_dnn_train_logger_list[[i]] <- logger$train
  leukemia_SCAD_dnn_test_logger_list[[i]] <- logger$eval
}

leukemia_SCAD_dnn_test_acc_unlist <- unlist(leukemia_SCAD_dnn_test_acc_list)
leukemia_SCAD_dnn_test_auc_unlist <- unlist(leukemia_SCAD_dnn_test_auc_list)
leukemia_SCAD_dnn_exectime_unlist <- unlist(leukemia_SCAD_dnn_exectime_list)

leukemia_SCAD_dnn_train_logger_unlist <- data.frame( matrix(unlist(leukemia_SCAD_dnn_train_logger_list), ncol = 100))
leukemia_SCAD_dnn_test_logger_unlist <- data.frame( matrix(unlist(leukemia_SCAD_dnn_test_logger_list), ncol = 100))

# 결과저장
# write(t(leukemia_SCAD_dnn_test_acc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_SCAD_DNN_result.txt")
# write(t(leukemia_SCAD_dnn_test_auc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_SCAD_DNN_auc_result.txt")
# write(t(leukemia_SCAD_dnn_exectime_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_SCAD_DNN_time.txt")
# write(t(leukemia_SCAD_dnn_train_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_SCAD_DNN_train_logger.txt")
# write(t(leukemia_SCAD_dnn_test_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_SCAD_DNN_test_logger.txt")
leukemia_SCAD_dnn_test_confusion_unlist <- data.frame()
for(i in 1:100){
  for(j in 1:4){
    leukemia_SCAD_dnn_test_confusion_unlist[i, j] <- unlist(leukemia_SCAD_dnn_test_confusion_list[[i]]$t)[j]
  }
}
leukemia_SCAD_dnn_test_confusion_unlist <- cbind(c(1:100), leukemia_SCAD_dnn_test_confusion_unlist)
colnames(leukemia_SCAD_dnn_test_confusion_unlist) <- c("iter","TN", "FN", "FP", "TP")
head(leukemia_SCAD_dnn_test_confusion_unlist)
tail(leukemia_SCAD_dnn_test_confusion_unlist)

# write.table(leukemia_SCAD_dnn_test_confusion_unlist, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_SCAD_dnn_confusion.txt", col.names = TRUE)


leukemia_SCAD_dnn_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_SCAD_DNN_result.txt")
leukemia_SCAD_dnn_test_acc_unlist <- cbind(leukemia_SCAD_dnn_test_acc_unlist, rep("DNN(SCAD 변수선택)", 100))
names(leukemia_SCAD_dnn_test_acc_unlist) <- c("Accuracy", "Model")

leukemia_SCAD_dnn_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_SCAD_DNN_auc_result.txt")
leukemia_SCAD_dnn_test_auc_unlist <- cbind(leukemia_SCAD_dnn_test_auc_unlist, rep("DNN(SCAD 변수선택)", 100))
names(leukemia_SCAD_dnn_test_auc_unlist) <- c("AUC", "Model")

leukemia_SCAD_dnn_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_SCAD_DNN_time.txt")
leukemia_SCAD_dnn_exectime_unlist <- cbind(leukemia_SCAD_dnn_exectime_unlist, rep("DNN(SCAD 변수선택)", 100))
names(leukemia_SCAD_dnn_exectime_unlist) <- c("Time", "Model")

# Accuracy 분포 확인.
summary(leukemia_SCAD_dnn_test_acc_unlist)

win.graph()
boxplot(leukemia_SCAD_dnn_test_acc_unlist[1],  main = "Leukemia data. DNN 은닉층 1, 은닉노드 20. (SCAD 변수선택)",
        xlab = "Iter = 300", ylab = "Accuracy", ylim = c(0, 1))
vioplot(leukemia_SCAD_dnn_test_acc_unlist[1], main = "Leukemia data. DNN 은닉층 1, 은닉노드 20. (SCAD 변수선택)",
        xlab = "Iter = 300", ylab = "Accuracy", ylim = c(0, 1))
points(mean(leukemia_SCAD_dnn_test_acc_unlist$Accuracy), col = "red", pch = 17) # mean 표시

# AUC 분포 확인.
summary(leukemia_SCAD_dnn_test_auc_unlist)

win.graph()
boxplot(leukemia_SCAD_dnn_test_auc_unlist[1],  main = "Leukemia data. DNN 은닉층 1, 은닉노드 20. (SCAD 변수선택)",
        xlab = "Iter = 300", ylab = "AUC", ylim = c(0, 1))
vioplot(leukemia_SCAD_dnn_test_auc_unlist[1], main = "Leukemia data. DNN 은닉층 1, 은닉노드 20. (SCAD 변수선택)",
        xlab = "Iter = 300", ylab = "AUC", ylim = c(0, 1))
points(mean(leukemia_SCAD_dnn_test_auc_unlist$AUC), col = "red", pch = 17) # mean 표시


# Time 분포 확인.
summary(leukemia_SCAD_dnn_exectime_unlist)

win.graph()
boxplot(leukemia_SCAD_dnn_exectime_unlist[1],  main = "Leukemia data. DNN 은닉층 1, 은닉노드 20. (SCAD 변수선택) 학습시간",
        xlab = "Iter = 300", ylab = "Time(단위 : 초)", ylim = c(0, 9))
vioplot(leukemia_SCAD_dnn_exectime_unlist[1], main = "Leukemia data. DNN 은닉층 1, 은닉노드 20. (SCAD 변수선택) 학습시간",
        xlab = "Iter = 300", ylab = "Time(단위 : 초)", ylim = c(0, 9))
points(mean(leukemia_SCAD_dnn_exectime_unlist$Time), col = "red", pch = 17) # mean 표시

### 2.2 SIS MCP 변수선택 (leukemia data) 후 DNN ###
### 2.2 SIS MCP 변수선택 (leukemia data) 후 DNN ###
### 2.2 SIS MCP 변수선택 (leukemia data) 후 DNN ###

# 신경망 학습용 data set
# Train : Test = 8 : 2 나누기. (100번)
# seed 를 바꿔가며 100개의 서로 다른 train, test data set 생성.
# list() 객체에 저장.

leukemia_MCP_train_x_list_num <- list()
leukemia_MCP_train_y_list_num <- list()

leukemia_MCP_test_x_list_num <- list()
leukemia_MCP_test_y_list_num <- list()

for(i in 1:100){
  set.seed(i)
  train_index <- sample(1:nrow(leukemia_data), 0.8 * nrow(leukemia_data), replace = FALSE)
  leukemia_train_x_num <- leukemia_data[train_index, c(1144, 4847)]
  leukemia_train_y_num <- leukemia_data[train_index, 7130]
  
  leukemia_test_x_num <- leukemia_data[-train_index, c(1144, 4847)]
  leukemia_test_y_num <- leukemia_data[-train_index, 7130]
  
  leukemia_MCP_train_x_list_num[[i]] <- leukemia_train_x_num
  leukemia_MCP_train_y_list_num[[i]] <- leukemia_train_y_num
  
  leukemia_MCP_test_x_list_num[[i]] <- leukemia_test_x_num
  leukemia_MCP_test_y_list_num[[i]] <- leukemia_test_y_num
  
}


#################
## 신경망 학습 ##
#################
# https://mxnet.apache.org/api/r  참고.
library(mxnet)

leukemia_MCP_model_list <- list()
leukemia_MCP_dnn_test_predict_list <- list()
leukemia_MCP_dnn_test_predict_label_list <- list()
leukemia_MCP_dnn_test_confusion_list <- list()
leukemia_MCP_dnn_test_acc_list <- list() # y_hat 과 test data set의 y 로 Accuracy 저장 list 할당.
leukemia_MCP_dnn_exectime_list <- list()
leukemia_MCP_dnn_train_logger_list <- list()
leukemia_MCP_dnn_test_logger_list <- list()
leukemia_MCP_dnn_test_auc_list <- list()

for(i in 1:100){ # 1:100
  leukemia_MCP_train_x_scale <- scale(leukemia_MCP_train_x_list_num[[i]])
  leukemia_MCP_test_x_scale <- scale(leukemia_MCP_test_x_list_num[[i]])
  
  leukemia_MCP_train_x_datamatrix <- data.matrix(leukemia_MCP_train_x_scale)
  leukemia_MCP_test_x_datamatrix <- data.matrix(leukemia_MCP_test_x_scale)
  
  leukemia_MCP_train_y <- leukemia_MCP_train_y_list_num[[i]]
  leukemia_MCP_test_y <- leukemia_MCP_test_y_list_num[[i]]
  
  mx.set.seed(2020)  # 가중치 초기값 고정.
  
  input <- mx.symbol.Variable("data")
  fc1 <- mx.symbol.FullyConnected(data = input, num.hidden = 20)  # Hyper parameter : 은닉노드 수
  act1 <- mx.symbol.Activation(data = fc1, act_type = "sigmoid")  # Hyper parameter : 활성화 함수 종류
  fc2 <- mx.symbol.FullyConnected(data = act1, num.hidden = 2)
  lro <- mx.symbol.SoftmaxOutput(data = fc2)  # Hyper parameter : 출력노드 함수 종류
  
  tic()
  
  logger <- mx.metric.logger$new()
  model <- mx.model.FeedForward.create(symbol = lro, X = leukemia_MCP_train_x_datamatrix, y = leukemia_MCP_train_y,
                                       eval.data = list(data = leukemia_MCP_test_x_datamatrix, label = leukemia_MCP_test_y),
                                       ctx = mx.gpu(), num.round = 400, optimizer = 'sgd', # Hyper parameter : iter 수, 최적화 함수 종류
                                       array.batch.size = 5, learning.rate = 0.001, momentum = 0.9, eval.metric = mx.metric.accuracy,
                                       verbose = T, epoch.end.callback = mx.callback.log.train.metric(1, logger))  # Hyper parameter : 배치 사이즈, 학습률, 모멘텀 값.
  
  exectime <- toc()
  exectime <- round(exectime$toc - exectime$tic, 5)
  
  leukemia_MCP_model_list[[i]] <- model
  leukemia_MCP_dnn_test_predict_list[[i]] <- predict(leukemia_MCP_model_list[[i]], leukemia_MCP_test_x_datamatrix)
  leukemia_MCP_dnn_test_predict_label_list[[i]] <- max.col(t(leukemia_MCP_dnn_test_predict_list[[i]])) - 1
  leukemia_MCP_dnn_test_confusion_list[[i]] <- CrossTable(x = leukemia_MCP_test_y_list_num[[i]], y = leukemia_MCP_dnn_test_predict_label_list[[i]])
  leukemia_MCP_dnn_test_acc_list[[i]] <- (leukemia_MCP_dnn_test_confusion_list[[i]]$t[1] + leukemia_MCP_dnn_test_confusion_list[[i]]$t[4])  / sum(leukemia_MCP_dnn_test_confusion_list[[i]]$t)
  auc <- performance(prediction( predict(leukemia_MCP_model_list[[i]], leukemia_MCP_test_x_datamatrix)[2, ], leukemia_MCP_test_y_list_num[[i]], label.ordering = c(0, 1)), measure = "auc")
  leukemia_MCP_dnn_test_auc_list[[i]] <- auc@y.values[[1]]
  
  leukemia_MCP_dnn_exectime_list[[i]] <- exectime
  leukemia_MCP_dnn_train_logger_list[[i]] <- logger$train
  leukemia_MCP_dnn_test_logger_list[[i]] <- logger$eval
}

leukemia_MCP_dnn_test_acc_unlist <- unlist(leukemia_MCP_dnn_test_acc_list)
leukemia_MCP_dnn_test_auc_unlist <- unlist(leukemia_MCP_dnn_test_auc_list)
leukemia_MCP_dnn_exectime_unlist <- unlist(leukemia_MCP_dnn_exectime_list)

leukemia_MCP_dnn_train_logger_unlist <- data.frame( matrix(unlist(leukemia_MCP_dnn_train_logger_list), ncol = 100))
leukemia_MCP_dnn_test_logger_unlist <- data.frame( matrix(unlist(leukemia_MCP_dnn_test_logger_list), ncol = 100))

# 결과저장
# write(t(leukemia_MCP_dnn_test_acc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_MCP_DNN_result.txt")
# write(t(leukemia_MCP_dnn_test_auc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_MCP_DNN_auc_result.txt")
# write(t(leukemia_MCP_dnn_exectime_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_MCP_DNN_time.txt")
# write(t(leukemia_MCP_dnn_train_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_MCP_DNN_train_logger.txt")
# write(t(leukemia_MCP_dnn_test_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_MCP_DNN_test_logger.txt")
leukemia_MCP_dnn_test_confusion_unlist <- data.frame()
for(i in 1:100){
  for(j in 1:4){
    leukemia_MCP_dnn_test_confusion_unlist[i, j] <- unlist(leukemia_MCP_dnn_test_confusion_list[[i]]$t)[j]
  }
}
leukemia_MCP_dnn_test_confusion_unlist <- cbind(c(1:100), leukemia_MCP_dnn_test_confusion_unlist)
colnames(leukemia_MCP_dnn_test_confusion_unlist) <- c("iter","TN", "FN", "FP", "TP")
head(leukemia_MCP_dnn_test_confusion_unlist)
tail(leukemia_MCP_dnn_test_confusion_unlist)

# write.table(leukemia_MCP_dnn_test_confusion_unlist, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_MCP_dnn_confusion.txt", col.names = TRUE)


leukemia_MCP_dnn_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_MCP_DNN_result.txt")
leukemia_MCP_dnn_test_acc_unlist <- cbind(leukemia_MCP_dnn_test_acc_unlist, rep("DNN(MCP 변수선택)", 100))
names(leukemia_MCP_dnn_test_acc_unlist) <- c("Accuracy", "Model")

leukemia_MCP_dnn_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_MCP_DNN_auc_result.txt")
leukemia_MCP_dnn_test_auc_unlist <- cbind(leukemia_MCP_dnn_test_auc_unlist, rep("DNN(MCP 변수선택)", 100))
names(leukemia_MCP_dnn_test_auc_unlist) <- c("AUC", "Model")

leukemia_MCP_dnn_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_MCP_DNN_time.txt")
leukemia_MCP_dnn_exectime_unlist <- cbind(leukemia_MCP_dnn_exectime_unlist, rep("DNN(MCP 변수선택)", 100))
names(leukemia_MCP_dnn_exectime_unlist) <- c("Time", "Model")

# Accuracy 분포 확인.
summary(leukemia_MCP_dnn_test_acc_unlist)

win.graph()
boxplot(leukemia_MCP_dnn_test_acc_unlist[1],  main = "Leukemia data. DNN 은닉층 1, 은닉노드 20. (MCP 변수선택)",
        xlab = "Iter = 400", ylab = "Accuracy", ylim = c(0, 1))
vioplot(leukemia_MCP_dnn_test_acc_unlist[1], main = "Leukemia data. DNN 은닉층 1, 은닉노드 20. (MCP 변수선택)",
        xlab = "Iter = 400", ylab = "Accuracy", ylim = c(0, 1))
points(mean(leukemia_MCP_dnn_test_acc_unlist$Accuracy), col = "red", pch = 17) # mean 표시

# AUC 분포 확인.
summary(leukemia_MCP_dnn_test_auc_unlist)

win.graph()
boxplot(leukemia_MCP_dnn_test_auc_unlist[1],  main = "Leukemia data. DNN 은닉층 1, 은닉노드 20. (MCP 변수선택)",
        xlab = "Iter = 400", ylab = "AUC", ylim = c(0, 1))
vioplot(leukemia_MCP_dnn_test_auc_unlist[1], main = "Leukemia data. DNN 은닉층 1, 은닉노드 20. (MCP 변수선택)",
        xlab = "Iter = 400", ylab = "AUC", ylim = c(0, 1))
points(mean(leukemia_MCP_dnn_test_auc_unlist$AUC), col = "red", pch = 17) # mean 표시

# Time 분포 확인.
summary(leukemia_MCP_dnn_exectime_unlist)

win.graph()
boxplot(leukemia_MCP_dnn_exectime_unlist[1],  main = "Leukemia data. DNN 은닉층 1, 은닉노드 20. (MCP 변수선택) 학습시간",
        xlab = "Iter = 400", ylab = "Time(단위 : 초)", ylim = c(0, 11))
vioplot(leukemia_MCP_dnn_exectime_unlist[1], main = "Leukemia data. DNN 은닉층 1, 은닉노드 20. (MCP 변수선택) 학습시간",
        xlab = "Iter = 400", ylab = "Time(단위 : 초)", ylim = c(0, 11))
points(mean(leukemia_MCP_dnn_exectime_unlist$Time), col = "red", pch = 17) # mean 표시

### 2.3 SIS LASSO 변수선택 (leukemia data) 후 DNN ###
### 2.3 SIS LASSO 변수선택 (leukemia data) 후 DNN ###
### 2.3 SIS LASSO 변수선택 (leukemia data) 후 DNN ###

# 신경망 학습용 data set
# Train : Test = 8 : 2 나누기. (100번)
# seed 를 바꿔가며 100개의 서로 다른 train, test data set 생성.
# list() 객체에 저장.

leukemia_LASSO_train_x_list_num <- list()
leukemia_LASSO_train_y_list_num <- list()

leukemia_LASSO_test_x_list_num <- list()
leukemia_LASSO_test_y_list_num <- list()

for(i in 1:100){
  set.seed(i)
  train_index <- sample(1:nrow(leukemia_data), 0.8 * nrow(leukemia_data), replace = FALSE)
  leukemia_train_x_num <- leukemia_data[train_index, c(1144, 2684, 4847, 6855)]
  leukemia_train_y_num <- leukemia_data[train_index, 7130]
  
  leukemia_test_x_num <- leukemia_data[-train_index, c(1144, 2684, 4847, 6855)]
  leukemia_test_y_num <- leukemia_data[-train_index, 7130]
  
  leukemia_LASSO_train_x_list_num[[i]] <- leukemia_train_x_num
  leukemia_LASSO_train_y_list_num[[i]] <- leukemia_train_y_num
  
  leukemia_LASSO_test_x_list_num[[i]] <- leukemia_test_x_num
  leukemia_LASSO_test_y_list_num[[i]] <- leukemia_test_y_num
  
}


#################
## 신경망 학습 ##
#################
# https://mxnet.apache.org/api/r  참고.
library(mxnet)

leukemia_LASSO_model_list <- list()
leukemia_LASSO_dnn_test_predict_list <- list()
leukemia_LASSO_dnn_test_predict_label_list <- list()
leukemia_LASSO_dnn_test_confusion_list <- list()
leukemia_LASSO_dnn_test_acc_list <- list() # y_hat 과 test data set의 y 로 Accuracy 저장 list 할당.
leukemia_LASSO_dnn_exectime_list <- list()
leukemia_LASSO_dnn_train_logger_list <- list()
leukemia_LASSO_dnn_test_logger_list <- list()
leukemia_LASSO_dnn_test_auc_list <- list()

for(i in 1:100){ # 1:100
  leukemia_LASSO_train_x_scale <- scale(leukemia_LASSO_train_x_list_num[[i]])
  leukemia_LASSO_test_x_scale <- scale(leukemia_LASSO_test_x_list_num[[i]])
  
  leukemia_LASSO_train_x_datamatrix <- data.matrix(leukemia_LASSO_train_x_scale)
  leukemia_LASSO_test_x_datamatrix <- data.matrix(leukemia_LASSO_test_x_scale)
  
  leukemia_LASSO_train_y <- leukemia_LASSO_train_y_list_num[[i]]
  leukemia_LASSO_test_y <- leukemia_LASSO_test_y_list_num[[i]]
  
  mx.set.seed(2020)  # 가중치 초기값 고정.
  
  input <- mx.symbol.Variable("data")
  fc1 <- mx.symbol.FullyConnected(data = input, num.hidden = 20)  # Hyper parameter : 은닉노드 수
  act1 <- mx.symbol.Activation(data = fc1, act_type = "sigmoid")  # Hyper parameter : 활성화 함수 종류
  fc2 <- mx.symbol.FullyConnected(data = act1, num.hidden = 2)
  lro <- mx.symbol.SoftmaxOutput(data = fc2)  # Hyper parameter : 출력노드 함수 종류
  
  tic()
  
  logger <- mx.metric.logger$new()
  model <- mx.model.FeedForward.create(symbol = lro, X = leukemia_LASSO_train_x_datamatrix, y = leukemia_LASSO_train_y,
                                       eval.data = list(data = leukemia_LASSO_test_x_datamatrix, label = leukemia_LASSO_test_y),
                                       ctx = mx.gpu(), num.round = 400, optimizer = 'sgd', # Hyper parameter : iter 수, 최적화 함수 종류
                                       array.batch.size = 5, learning.rate = 0.001, momentum = 0.9, eval.metric = mx.metric.accuracy,
                                       verbose = T, epoch.end.callback = mx.callback.log.train.metric(1, logger))  # Hyper parameter : 배치 사이즈, 학습률, 모멘텀 값.
  
  exectime <- toc()
  exectime <- round(exectime$toc - exectime$tic, 5)
  
  leukemia_LASSO_model_list[[i]] <- model
  leukemia_LASSO_dnn_test_predict_list[[i]] <- predict(leukemia_LASSO_model_list[[i]], leukemia_LASSO_test_x_datamatrix)
  leukemia_LASSO_dnn_test_predict_label_list[[i]] <- max.col(t(leukemia_LASSO_dnn_test_predict_list[[i]])) - 1
  leukemia_LASSO_dnn_test_confusion_list[[i]] <- CrossTable(x = leukemia_LASSO_test_y_list_num[[i]], y = leukemia_LASSO_dnn_test_predict_label_list[[i]])
  leukemia_LASSO_dnn_test_acc_list[[i]] <- (leukemia_LASSO_dnn_test_confusion_list[[i]]$t[1] + leukemia_LASSO_dnn_test_confusion_list[[i]]$t[4])  / sum(leukemia_LASSO_dnn_test_confusion_list[[i]]$t)
  auc <- performance(prediction( predict(leukemia_LASSO_model_list[[i]], leukemia_LASSO_test_x_datamatrix)[2, ], leukemia_LASSO_test_y_list_num[[i]], label.ordering = c(0, 1)), measure = "auc")
  leukemia_LASSO_dnn_test_auc_list[[i]] <- auc@y.values[[1]]
  
  leukemia_LASSO_dnn_exectime_list[[i]] <- exectime
  leukemia_LASSO_dnn_train_logger_list[[i]] <- logger$train
  leukemia_LASSO_dnn_test_logger_list[[i]] <- logger$eval
}

leukemia_LASSO_dnn_test_acc_unlist <- unlist(leukemia_LASSO_dnn_test_acc_list)
leukemia_LASSO_dnn_test_auc_unlist <- unlist(leukemia_LASSO_dnn_test_auc_list)
leukemia_LASSO_dnn_exectime_unlist <- unlist(leukemia_LASSO_dnn_exectime_list)

leukemia_LASSO_dnn_train_logger_unlist <- data.frame( matrix(unlist(leukemia_LASSO_dnn_train_logger_list), ncol = 100))
leukemia_LASSO_dnn_test_logger_unlist <- data.frame( matrix(unlist(leukemia_LASSO_dnn_test_logger_list), ncol = 100))

# 결과저장
# write(t(leukemia_LASSO_dnn_test_acc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_LASSO_DNN_result.txt")
# write(t(leukemia_LASSO_dnn_test_auc_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_LASSO_DNN_auc_result.txt")
# write(t(leukemia_LASSO_dnn_exectime_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_LASSO_DNN_time.txt")
# write(t(leukemia_LASSO_dnn_train_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_LASSO_DNN_train_logger.txt")
# write(t(leukemia_LASSO_dnn_test_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_LASSO_DNN_test_logger.txt")
leukemia_LASSO_dnn_test_confusion_unlist <- data.frame()
for(i in 1:100){
  for(j in 1:4){
    leukemia_LASSO_dnn_test_confusion_unlist[i, j] <- unlist(leukemia_LASSO_dnn_test_confusion_list[[i]]$t)[j]
  }
}
leukemia_LASSO_dnn_test_confusion_unlist <- cbind(c(1:100), leukemia_LASSO_dnn_test_confusion_unlist)
colnames(leukemia_LASSO_dnn_test_confusion_unlist) <- c("iter","TN", "FN", "FP", "TP")
head(leukemia_LASSO_dnn_test_confusion_unlist)
tail(leukemia_LASSO_dnn_test_confusion_unlist) 

# write.table(leukemia_LASSO_dnn_test_confusion_unlist, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_LASSO_dnn_confusion.txt", col.names = TRUE)

leukemia_LASSO_dnn_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_LASSO_DNN_result.txt")
leukemia_LASSO_dnn_test_acc_unlist <- cbind(leukemia_LASSO_dnn_test_acc_unlist, rep("DNN(LASSO 변수선택)", 100))
names(leukemia_LASSO_dnn_test_acc_unlist) <- c("Accuracy", "Model")

leukemia_LASSO_dnn_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_LASSO_DNN_auc_result.txt")
leukemia_LASSO_dnn_test_auc_unlist <- cbind(leukemia_LASSO_dnn_test_auc_unlist, rep("DNN(LASSO 변수선택)", 100))
names(leukemia_LASSO_dnn_test_auc_unlist) <- c("AUC", "Model")

leukemia_LASSO_dnn_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_LASSO_DNN_time.txt")
leukemia_LASSO_dnn_exectime_unlist <- cbind(leukemia_LASSO_dnn_exectime_unlist, rep("DNN(LASSO 변수선택)", 100))
names(leukemia_LASSO_dnn_exectime_unlist) <- c("Time", "Model")

# Accuracy 분포 확인.
summary(leukemia_LASSO_dnn_test_acc_unlist)

win.graph()
boxplot(leukemia_LASSO_dnn_test_acc_unlist[1],  main = "Leukemia data. DNN 은닉층 1, 은닉노드 20. (LASSO 변수선택)",
        xlab = "Iter = 400", ylab = "Accuracy", ylim = c(0, 1))
vioplot(leukemia_LASSO_dnn_test_acc_unlist[1], main = "Leukemia data. DNN 은닉층 1, 은닉노드 20. (LASSO 변수선택)",
        xlab = "Iter = 400", ylab = "Accuracy", ylim = c(0, 1))
points(mean(leukemia_LASSO_dnn_test_acc_unlist$Accuracy), col = "red", pch = 17) # mean 표시

# AUC 분포 확인.
summary(leukemia_LASSO_dnn_test_auc_unlist)

win.graph()
boxplot(leukemia_LASSO_dnn_test_auc_unlist[1],  main = "Leukemia data. DNN 은닉층 1, 은닉노드 20. (LASSO 변수선택)",
        xlab = "Iter = 400", ylab = "AUC", ylim = c(0, 1))
vioplot(leukemia_LASSO_dnn_test_auc_unlist[1], main = "Leukemia data. DNN 은닉층 1, 은닉노드 20. (LASSO 변수선택)",
        xlab = "Iter = 400", ylab = "AUC", ylim = c(0, 1))
points(mean(leukemia_LASSO_dnn_test_auc_unlist$AUC), col = "red", pch = 17) # mean 표시

# Time 분포 확인.
summary(leukemia_LASSO_dnn_exectime_unlist)

win.graph()
boxplot(leukemia_LASSO_dnn_exectime_unlist[1],  main = "Leukemia data. DNN 은닉층 1, 은닉노드 20. (LASSO 변수선택) 학습시간",
        xlab = "Iter = 400", ylab = "Time(단위 : 초)", ylim = c(0, 10))
vioplot(leukemia_LASSO_dnn_exectime_unlist[1], main = "Leukemia data. DNN 은닉층 1, 은닉노드 20. (LASSO 변수선택) 학습시간",
        xlab = "Iter = 400", ylab = "Time(단위 : 초)", ylim = c(0, 10))
points(mean(leukemia_LASSO_dnn_exectime_unlist$Time), col = "red", pch = 17) # mean 표시
