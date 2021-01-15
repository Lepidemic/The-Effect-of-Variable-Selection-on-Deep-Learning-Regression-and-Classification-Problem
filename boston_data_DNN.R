# 석사 논문 회의 1일차

library(MASS)
library(reshape2)
library(ggplot2)
library(dplyr)
library(corrplot)
library(vioplot)
library(tictoc)
# library(mxnet)

library(glmnet)
library(plotmo) 

library(agricolae)
library(lawstat)

# Boston data
# The Boston data frame has 506 rows and 14 columns. 
###### 변수 설명 ######
# crim
# per capita crime rate by town. 
# (자치시(town) 별 1인당 범죄율)
# 
# zn
# proportion of residential land zoned for lots over 25,000 sq.ft.
# (25,000 평방피트를 초과하는 거주지역의 비율)
# 
# indus
# proportion of non-retail business acres per town.
# (비소매상업지역이 점유하고 있는 토지의 비율)
# 
# chas
# Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
# (찰스강에 대한 더미변수 (강의 경계에 위치한 경우는 1, 아닌 경우 0))
#
# nox
# nitrogen oxides concentration (parts per 10 million).
# (10ppm 당 농축 일산화질소)
#
# rm
# average number of rooms per dwelling.
# (주택 1가구당 평균 방의 개수)
# 
# age
# proportion of owner-occupied units built prior to 1940.
# (1940년 이전에 건축된 소유주택의 비율)
#
# dis
# weighted mean of distances to five Boston employment centres.
# (5개의 보스턴 직업센터까지의 접근성 지수)
#
# rad
# index of accessibility to radial highways.
# (방사형 도로까지의 접근성 지수)
#
# tax
# full-value property-tax rate per \$10,000.
# (10,000 달러 당 재산세율)
#
# ptratio
# pupil-teacher ratio by town.
# (자치시(town)별 학생/교사 비율) 
#
# black
# 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.
# (자치시(town)별 흑인의 비율)
#
# lstat
# lower status of the population (percent).
# (모집단의 하위계층의 비율) 
#
# medv
# median value of owner-occupied homes in \$1000s.
# (본인 소유의 주택가격(중앙값) 단위 : $1,000)
#
data("Boston",package="MASS")
boston_data <- Boston

# Boston data structure exploration ( 데이터 구조 탐색 )
head(boston_data)
glimpse(boston_data)  # 506 x 14
str(boston_data)  # 506 x 14
summary(boston_data)  # There's no missing value

# missing value plot (https://njtierney.github.io/r/missing%20data/rbloggers/2015/12/01/ggplot-missing-data/) 참고
boston_data_2 <- tibble(Boston)
boston_data_2

boston_missing <- function(x){
  
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

boston_missing(boston_data_2)  # There's no missing value via graph

# CorrPlots
library(corrplot)
corrplot(cor(select(boston_data, -medv) ) )
corrplot(cor(select(boston_data, -medv) ), method = "number")  # 설명변수들 사이의 상관관계 확인.

corrplot(cor(boston_data))
corrplot(cor(boston_data), method = "number")  # 반응변수와 설명변수들 사이의 상관관계 확인.

# Train : Test = 8 : 2 나누기. (100번)
# seed 를 바꿔가며 1000개의 서로 다른 train, test data set 생성.
# list() 객체에 저장.

boston_train_x_list <- list()
boston_train_y_list <- list()

boston_test_x_list <- list()
boston_test_y_list <- list()

for(i in 1:100){
  set.seed(i)
  train_index <- sample(1:nrow(boston_data), 0.8 * nrow(boston_data), replace = FALSE)
  boston_train_x <- boston_data[train_index, 1:13]
  boston_train_y <- boston_data[train_index, 14]
  
  boston_test_x <- boston_data[-train_index, 1:13]
  boston_test_y <- boston_data[-train_index, 14]
  
  boston_train_x_list[[i]] <- boston_train_x
  boston_train_y_list[[i]] <- boston_train_y
  
  boston_test_x_list[[i]] <- boston_test_x
  boston_test_y_list[[i]] <- boston_test_y
  
}


## 신경망 학습 ##
# https://mxnet.apache.org/api/r  참고.
library(mxnet)

boston_model_list <- list()
boston_dnn_test_rmse <- list()
boston_dnn_exectime_list <- list()
boston_dnn_train_logger_list <- list()
boston_dnn_test_logger_list <- list()

for(i in 1:100){ # 1:100
  boston_train_x_scale <- scale(boston_train_x_list[[i]])
  boston_test_x_scale <- scale(boston_test_x_list[[i]])
  
  boston_train_x_datamatrix <- data.matrix(boston_train_x_scale)
  boston_test_x_datamatrix <- data.matrix(boston_test_x_scale)
  
  boston_train_y <- boston_train_y_list[[i]]
  boston_test_y <- boston_test_y_list[[i]]
  
  input <- mx.symbol.Variable("data")
  fc1 <- mx.symbol.FullyConnected(data = input, num.hidden = 5)
  act1 <- mx.symbol.Activation(data = fc1, act_type = "sigmoid")
  fc2 <- mx.symbol.FullyConnected(data = act1, num.hidden = 1)
  lro <- mx.symbol.LinearRegressionOutput(data = fc2)
  
  mx.set.seed(2020) # 가중치 초기값 고정.
  
  tic()
  logger <- mx.metric.logger$new()
  model <- mx.model.FeedForward.create(symbol = lro, X = boston_train_x_datamatrix, y = boston_train_y,
                                       eval.data = list(data = boston_test_x_datamatrix, label = boston_test_y),
                                       ctx = mx.gpu(), num.round = 100, optimizer = 'sgd',
                                       array.batch.size = 20, learning.rate = 0.001, momentum = 0.9, eval.metric = mx.metric.rmse,
                                       verbose = T, epoch.end.callback = mx.callback.log.train.metric(1, logger))  # 초매계변수 조율 필요.
  
  exectime <- toc()
  exectime <- round(exectime$toc - exectime$tic, 5)
  
  boston_model_list[[i]] <- model
  boston_dnn_test_rmse[[i]] <- sqrt( mean( (predict(model, boston_test_x_datamatrix) - boston_test_y )^2 ) )
  boston_dnn_exectime_list[[i]] <- exectime
  boston_dnn_train_logger_list[[i]] <- logger$train
  boston_dnn_test_logger_list[[i]] <- logger$eval
  
}

boston_dnn_test_rmse_unlist <- unlist(boston_dnn_test_rmse)
boston_dnn_exectime_unlist <- unlist(boston_dnn_exectime_list)

boston_dnn_train_logger_unlist <- data.frame( matrix(unlist(boston_dnn_train_logger_list), ncol = 100))
boston_dnn_test_logger_unlist <- data.frame( matrix(unlist(boston_dnn_test_logger_list), ncol = 100))

# 결과저장
# write(t(boston_dnn_test_rmse_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Boston_DNN_result.txt")
# write(t(boston_dnn_exectime_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/boston_DNN_time.txt")
# write(t(boston_dnn_train_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/boston_DNN_train_logger.txt")
# write(t(boston_dnn_test_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/boston_DNN_test_logger.txt")

boston_dnn_test_rmse_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Boston_DNN_result.txt")
boston_dnn_test_rmse_unlist <- cbind(boston_dnn_test_rmse_unlist, rep("DNN(변수전체)", 100))
names(boston_dnn_test_rmse_unlist) <- c("RMSE", "Model")

summary(boston_dnn_test_rmse_unlist)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 2.819   3.551   3.897   3.949   4.277   5.476 
win.graph()
boxplot(boston_dnn_test_rmse_unlist[1],  main = "Boston data. DNN 은닉층 1, 은닉노드 5. (전체변수)",
        xlab = "Iter = 100", ylab = "RMSE", ylim = c(0, 10))
vioplot(boston_dnn_test_rmse_unlist[1], main = "Boston data. DNN 은닉층 1, 은닉노드 5. (전체변수)",
        xlab = "Iter = 100", ylab = "RMSE", ylim = c(0, 10))
points(mean(boston_dnn_test_rmse_unlist$RMSE), col = "red", pch = 17) # mean 표시

# 2. Boston data LASSO 이용 변수선택
# 2. Boston data LASSO 이용 변수선택
# 2. Boston data LASSO 이용 변수선택

# LASSO 10-fold CV 방법으로 변수선택 (lambda 1se)
boston_lasso_fit_10fold <- cv.glmnet(as.matrix(boston_data[, -14]), boston_data[, 14], type.measure = "mse",
                                     family = "gaussian", alpha = 1)  # lambda.1se는 Standard error가 가장 Regularized 된 모델이 되는 람다값을 찾아줌.

summary(boston_lasso_fit_10fold)

win.graph()
plot(boston_lasso_fit_10fold, main = "Boston data (LASSO)")

# LASSO lambda.1se 에서 선택된 변수들 추정회귀계수
boston_lasso_coef <- predict(boston_lasso_fit_10fold, type = "coefficients", s = boston_lasso_fit_10fold$lambda.1se)
boston_lasso_coef


# 2. Boston data LASSO 이용 변수선택 후 DNN 모형 학습
# 2. Boston data LASSO 이용 변수선택 후 DNN 모형 학습
# 2. Boston data LASSO 이용 변수선택 후 DNN 모형 학습


# LASSO 에서 선택된 변수들로
# Train : Test = 8 : 2 나누기. (100번)
# seed 를 바꿔가며 100개의 서로 다른 train, test data set 생성.
# list() 객체에 저장.

boston_train_x_selec_list <- list()
boston_train_y_selec_list <- list()

boston_test_x_selec_list <- list()
boston_test_y_selec_list <- list()

for(i in 1:100){
  set.seed(i)
  train_index <- sample(1:nrow(boston_data), 0.8 * nrow(boston_data), replace = FALSE)
  boston_train_x <- boston_data[train_index, c(1, 2, 4, 5, 6, 8, 11, 12, 13)]
  boston_train_y <- boston_data[train_index, 14]
  
  boston_test_x <- boston_data[-train_index, c(1, 2, 4, 5, 6, 8, 11, 12, 13)]
  boston_test_y <- boston_data[-train_index, 14]
  
  boston_train_x_selec_list[[i]] <- boston_train_x
  boston_train_y_selec_list[[i]] <- boston_train_y
  
  boston_test_x_selec_list[[i]] <- boston_test_x
  boston_test_y_selec_list[[i]] <- boston_test_y
  
}


# https://mxnet.apache.org/api/r  참고.
library(mxnet)

boston_model_list <- list()
boston_dnn_selec_test_rmse <- list()
boston_dnn_selec_exectime_list <- list()
boston_dnn_selec_train_logger_list <- list()
boston_dnn_selec_test_logger_list <- list()

for(i in 1:100){ # 1:100
  boston_train_x_selec_scale <- scale(boston_train_x_selec_list[[i]])
  boston_test_x_selec_scale <- scale(boston_test_x_selec_list[[i]])
  
  boston_train_x_selec_datamatrix <- data.matrix(boston_train_x_selec_scale)
  boston_test_x_selec_datamatrix <- data.matrix(boston_test_x_selec_scale)
  
  boston_train_y <- boston_train_y_selec_list[[i]]
  boston_test_y <- boston_test_y_selec_list[[i]]
  
  input <- mx.symbol.Variable("data")
  fc1 <- mx.symbol.FullyConnected(data = input, num.hidden = 5)
  act1 <- mx.symbol.Activation(data = fc1, act_type = "sigmoid")
  fc2 <- mx.symbol.FullyConnected(data = act1, num.hidden = 1)
  lro <- mx.symbol.LinearRegressionOutput(data = fc2)
  
  mx.set.seed(2020) # 가중치 초기값 고정.
  
  tic()
  
  logger <- mx.metric.logger$new()
  model <- mx.model.FeedForward.create(symbol = lro, X = boston_train_x_selec_datamatrix, y = boston_train_y,
                                       eval.data = list(data = boston_test_x_selec_datamatrix, label = boston_test_y),
                                       ctx = mx.gpu(), num.round = 100, optimizer = 'sgd',
                                       array.batch.size = 20, learning.rate = 0.001, momentum = 0.9, eval.metric = mx.metric.rmse,
                                       verbose = T, epoch.end.callback = mx.callback.log.train.metric(1, logger))  # 초매개변수 조율 필요.
  exectime <- toc()
  exectime <- round(exectime$toc - exectime$tic, 5)
  
  boston_model_list[[i]] <- model
  boston_dnn_selec_test_rmse[[i]] <- sqrt( mean( (predict(model, boston_test_x_selec_datamatrix) - boston_test_y )^2 ) )
  boston_dnn_selec_exectime_list[[i]] <- exectime
  boston_dnn_selec_train_logger_list[[i]] <- logger$train
  boston_dnn_selec_test_logger_list[[i]] <- logger$eval
  
}

boston_dnn_selec_test_rmse_unlist <- unlist(boston_dnn_selec_test_rmse)
boston_dnn_selec_exectime_unlist <- unlist(boston_dnn_selec_exectime_list)

boston_dnn_selec_train_logger_unlist <- data.frame( matrix(unlist(boston_dnn_selec_train_logger_list), ncol = 100))
boston_dnn_selec_test_logger_unlist <- data.frame( matrix(unlist(boston_dnn_selec_test_logger_list), ncol = 100))

# 결과저장
# write(t(boston_dnn_selec_test_rmse_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Boston_DNN_selec_result.txt")
# write(t(boston_dnn_selec_exectime_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/boston_DNN_selec_time.txt")
# write(t(boston_dnn_selec_train_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/boston_DNN_selec_train_logger.txt")
# write(t(boston_dnn_selec_test_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/boston_DNN_selec_test_logger.txt")

boston_dnn_selec_test_rmse_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Boston_DNN_selec_result.txt")
boston_dnn_selec_test_rmse_unlist <- cbind(boston_dnn_selec_test_rmse_unlist, rep("DNN(LASSO변수선택)", 100))
names(boston_dnn_selec_test_rmse_unlist) <- c("RMSE", "Model")


summary(boston_dnn_selec_test_rmse_unlist)
# RMSE      
# Min.   :2.709  
# 1st Qu.:3.577  
# Median :4.023  
# Mean   :4.055  
# 3rd Qu.:4.468  
# Max.   :5.651 
win.graph()
boxplot(boston_dnn_selec_test_rmse_unlist[1],  main = "Boston data. DNN 은닉층 1, 은닉노드 5. (LASSO 변수선택)",
        xlab = "Iter = 100", ylab = "RMSE", ylim = c(0, 10))
vioplot(boston_dnn_selec_test_rmse_unlist[1], main = "Boston data. DNN 은닉층 1, 은닉노드 5. (LASSO 변수선택)",
        xlab = "Iter = 100", ylab = "RMSE", ylim = c(0, 10))
points(mean(boston_dnn_selec_test_rmse_unlist$RMSE), col = "red", pch = 17) # mean 표시

# 3. Boston data Stepwise 변수선택 후 DNN 모형 학습
# 3. Boston data Stepwise 변수선택 후 DNN 모형 학습
# 3. Boston data Stepwise 변수선택 후 DNN 모형 학습

lm_model <- lm(medv ~ ., data = boston_data)
lm_step_model <- step(lm_model, direction = "both")
summary(lm_step_model)

#  Stepwise 방법으로 선택된 변수들로
# Train : Test = 8 : 2 나누기. (100번)
# seed 를 바꿔가며 100개의 서로 다른 train, test data set 생성.
# list() 객체에 저장.


boston_train_x_step_list <- list()
boston_train_y_step_list <- list()

boston_test_x_step_list <- list()
boston_test_y_step_list <- list()

for(i in 1:100){
  set.seed(i)
  train_index <- sample(1:nrow(boston_data), 0.8 * nrow(boston_data), replace = FALSE)
  boston_train_x <- boston_data[train_index, -c(3, 7, 14)]
  boston_train_y <- boston_data[train_index, 14]
  
  boston_test_x <- boston_data[-train_index, -c(3, 7, 14)]
  boston_test_y <- boston_data[-train_index, 14]
  
  boston_train_x_step_list[[i]] <- boston_train_x
  boston_train_y_step_list[[i]] <- boston_train_y
  
  boston_test_x_step_list[[i]] <- boston_test_x
  boston_test_y_step_list[[i]] <- boston_test_y
  
}


# https://mxnet.apache.org/api/r  참고.
library(mxnet)

boston_model_list <- list()
boston_dnn_step_test_rmse <- list()
boston_dnn_step_exectime_list <- list()
boston_dnn_step_train_logger_list <- list()
boston_dnn_step_test_logger_list <- list()

for(i in 1:100){ # 1:100
  boston_train_x_step_scale <- scale(boston_train_x_step_list[[i]])
  boston_test_x_step_scale <- scale(boston_test_x_step_list[[i]])
  
  boston_train_x_step_datamatrix <- data.matrix(boston_train_x_step_scale)
  boston_test_x_step_datamatrix <- data.matrix(boston_test_x_step_scale)
  
  boston_train_y <- boston_train_y_step_list[[i]]
  boston_test_y <- boston_test_y_step_list[[i]]
  
  input <- mx.symbol.Variable("data")
  fc1 <- mx.symbol.FullyConnected(data = input, num.hidden = 5)
  act1 <- mx.symbol.Activation(data = fc1, act_type = "sigmoid")
  fc2 <- mx.symbol.FullyConnected(data = act1, num.hidden = 1)
  lro <- mx.symbol.LinearRegressionOutput(data = fc2)
  
  mx.set.seed(2020) # 가중치 초기값 고정.
  
  tic()
  logger <- mx.metric.logger$new()
  model <- mx.model.FeedForward.create(symbol = lro, X = boston_train_x_step_datamatrix, y = boston_train_y,
                                       eval.data = list(data = boston_test_x_step_datamatrix, label = boston_test_y),
                                       ctx = mx.gpu(), num.round = 100, optimizer = 'sgd',
                                       array.batch.size = 20, learning.rate = 0.001, momentum = 0.9, eval.metric = mx.metric.rmse,
                                       verbose = T, epoch.end.callback = mx.callback.log.train.metric(1, logger))  # 초매개변수 조율 필요.
  exectime <- toc()
  exectime <- round(exectime$toc - exectime$tic, 5)
  
  boston_model_list[[i]] <- model
  boston_dnn_step_test_rmse[[i]] <- sqrt( mean( (predict(model, boston_test_x_step_datamatrix) - boston_test_y )^2 ) )
  boston_dnn_step_exectime_list[[i]] <- exectime
  boston_dnn_step_train_logger_list[[i]] <- logger$train
  boston_dnn_step_test_logger_list[[i]] <- logger$eval
  
}

boston_dnn_step_test_rmse_unlist <- unlist(boston_dnn_step_test_rmse)
boston_dnn_step_exectime_unlist <- unlist(boston_dnn_step_exectime_list)

boston_dnn_step_train_logger_unlist <- data.frame( matrix(unlist(boston_dnn_step_train_logger_list), ncol = 100))
boston_dnn_step_test_logger_unlist <- data.frame( matrix(unlist(boston_dnn_step_test_logger_list), ncol = 100))

# 결과저장
# write(t(boston_dnn_step_test_rmse_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Boston_DNN_step_result.txt")
# write(t(boston_dnn_step_exectime_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/boston_DNN_step_time.txt")
# write(t(boston_dnn_step_train_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/boston_DNN_step_train_logger.txt")
# write(t(boston_dnn_step_test_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/boston_DNN_step_test_logger.txt")

boston_dnn_step_test_rmse_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Boston_DNN_step_result.txt")
boston_dnn_step_test_rmse_unlist <- cbind(boston_dnn_step_test_rmse_unlist, rep("DNN(stepwise변수선택)", 100))
names(boston_dnn_step_test_rmse_unlist) <- c("RMSE", "Model")


summary(boston_dnn_step_test_rmse_unlist)
# RMSE                         Model    
# Min.   :2.812   DNN(stepwise변수선택):100  
# 1st Qu.:3.502                              
# Median :3.882                              
# Mean   :3.919                              
# 3rd Qu.:4.244                              
# Max.   :5.275 
win.graph()
boxplot(boston_dnn_step_test_rmse_unlist[1],  main = "Boston data. DNN 은닉층 1, 은닉노드 5. (Stepwise 변수선택)",
        xlab = "Iter = 100", ylab = "RMSE", ylim = c(0, 10))
vioplot(boston_dnn_step_test_rmse_unlist[1], main = "Boston data. DNN 은닉층 1, 은닉노드 5. (Stepwise 변수선택)",
        xlab = "Iter = 100", ylab = "RMSE", ylim = c(0, 10))
points(mean(boston_dnn_step_test_rmse_unlist$RMSE), col = "red", pch = 17) # mean 표시




## 변수선택 전 후 DNN 비교 vioplot##
# 결과 합치기
boston_3model_rmse <- rbind(boston_dnn_test_rmse_unlist, boston_dnn_selec_test_rmse_unlist,
                            boston_dnn_step_test_rmse_unlist)

str(boston_3model_rmse)

# win.graph()
# par(mfrow = c(1, 3))
# vioplot(boston_dnn_test_rmse_unlist$RMSE, main = "Boston data. DNN 은닉층 1, 은닉노드 5. (전체변수)",
#         xlab = "Iter = 100", ylab = "RMSE", ylim = c(0, 10))
# vioplot(boston_dnn_selec_test_rmse_unlist$RMSE, main = "Boston data. DNN 은닉층 1, 은닉노드 5. (LASSO 변수선택)",
#         xlab = "Iter = 100", ylab = "RMSE", ylim = c(0, 10))
# vioplot(boston_dnn_selec_test_rmse_unlist$RMSE, main = "Boston data. DNN 은닉층 1, 은닉노드 5. (Steowise 변수선택)",
#         xlab = "Iter = 100", ylab = "RMSE", ylim = c(0, 10))

win.graph()
vioplot(boston_3model_rmse$RMSE ~ boston_3model_rmse$Model,
        main = "Boston data. DNN 모형 RMSE 비교",
        xlab = "Iter = 100", ylab = "RMSE", ylim = c(0, 10),
        cex.main = 1.4)

## 분산분석 및 사후검정 ##
# 분산분석
boston_rmse_anova <- aov(RMSE ~ Model, data = boston_3model_rmse)
summary(boston_rmse_anova)

boston_rmse_kruskal <- kruskal.test(RMSE ~ Model, data = boston_3model_rmse)
boston_rmse_kruskal

# 정규성 검정
shapiro.test(boston_rmse_anova$residuals)

# 등분산 검정
plot(boston_rmse_anova$fitted.values, boston_rmse_anova$residuals,
     main = "잔차 대 적합값",
     xlab = "적합값",
     ylab = "잔차", cex.lab = 1.5)

# 사후검정
boston_rmse_sceffe <- scheffe.test(boston_rmse_anova, "Model")
boston_rmse_bonferroni <- LSD.test(boston_rmse_anova, "Model", p.adj = "bonferroni")
boston_rmse_Tukey <- HSD.test(boston_rmse_anova, "Model")


boston_rmse_sceffe
boston_rmse_bonferroni
boston_rmse_Tukey

win.graph()
plot(boston_rmse_sceffe, variation = "SD", main = "Scheffe 사후검정 결과")
plot(boston_rmse_bonferroni, variation = "SD", main = "Bonferroni 사후검정 결과")
plot(boston_rmse_Tukey, variation = "SD", main = "Tukey 사후검정 결과")

