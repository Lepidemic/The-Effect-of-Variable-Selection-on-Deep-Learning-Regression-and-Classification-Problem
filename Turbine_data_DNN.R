
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

library(ROCR)
# Gas Turbine data
# Gas Turbine data frame has 7,411 rows and 11 columns. 
###### 변수 설명 ######
# Ambient temperature (AT) 
# 주변 온도

# Ambient pressure (AP) mbar 
# 주변 압력

# Ambient humidity (AH) (%) 
# 주변 습도

# Air filter difference pressure (AFDP) mbar 
# 공기 필터 차압

# Gas turbine exhaust pressure (GTEP) mbar 
# 가스터빈 배기압

# Turbine inlet temperature (TIT) C 
# 터빈 입구 온도

# Turbine after temperature (TAT) C 
# 터빈 후 온도

# Compressor discharge pressure (CDP) mbar 
# 컴프레서 배출 압력

# Turbine energy yield (TEY) MWH 
# 터빈 에너지 산출량

# Carbon monoxide (CO) mg/m3 
# 일산화탄소(CO)

# Nitrogen oxides (NOx) mg/m3 
# 질소산화물(NOx)
#

turbine_data <- read.csv("C:/Users/start/Google 드라이브/석사_졸업논문/데이터셋_설명/data/regression_prob/Gas_Turbine_Co_and_NOx_Emission_Data/gt_2011.csv", header = T)

# Turbine data structure exploration ( 데이터 구조 탐색 )
head(turbine_data)
glimpse(turbine_data)  # 7,411 x 11
str(turbine_data)  # 7,411 x 11
summary(turbine_data)  # There's no missing value

# missing value plot (https://njtierney.github.io/r/missing%20data/rbloggers/2015/12/01/ggplot-missing-data/) 참고
turbine_data_2 <- tibble(turbine_data)
turbine_data_2

turbine_missing <- function(x){
  
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

turbine_missing(turbine_data_2)  # There's no missing value via graph

# CorrPlots
library(corrplot)
corrplot(cor(select(turbine_data, -TEY) ) )
corrplot(cor(select(turbine_data, -TEY) ), method = "number")  # 설명변수들 사이의 상관관계 확인.

corrplot(cor(turbine_data))
corrplot(cor(turbine_data), method = "number")  # 반응변수와 설명변수들 사이의 상관관계 확인.

# Train : Test = 8 : 2 나누기. (100번)
# seed 를 바꿔가며 1000개의 서로 다른 train, test data set 생성.
# list() 객체에 저장.

turbine_train_x_list <- list()
turbine_train_y_list <- list()

turbine_test_x_list <- list()
turbine_test_y_list <- list()

for(i in 1:100){
  set.seed(i)
  train_index <- sample(1:nrow(turbine_data), 0.8 * nrow(turbine_data), replace = FALSE)
  turbine_train_x <- turbine_data[train_index, -8]
  turbine_train_y <- turbine_data[train_index, 8]
  
  turbine_test_x <- turbine_data[-train_index, -8]
  turbine_test_y <- turbine_data[-train_index, 8]
  
  turbine_train_x_list[[i]] <- turbine_train_x
  turbine_train_y_list[[i]] <- turbine_train_y
  
  turbine_test_x_list[[i]] <- turbine_test_x
  turbine_test_y_list[[i]] <- turbine_test_y
  
}


## 신경망 학습 ##
# https://mxnet.apache.org/api/r  참고.
library(mxnet)

turbine_model_list <- list()
turbine_dnn_test_rmse <- list()
turbine_dnn_exectime_list <- list()
turbine_dnn_train_logger_list <- list()
turbine_dnn_test_logger_list <- list()

for(i in 1:100){ # 1:100
  turbine_train_x_scale <- scale(turbine_train_x_list[[i]])
  turbine_test_x_scale <- scale(turbine_test_x_list[[i]])
  
  turbine_train_x_datamatrix <- data.matrix(turbine_train_x_scale)
  turbine_test_x_datamatrix <- data.matrix(turbine_test_x_scale)
  
  turbine_train_y <- turbine_train_y_list[[i]]
  turbine_test_y <- turbine_test_y_list[[i]]
  
  input <- mx.symbol.Variable("data")
  fc1 <- mx.symbol.FullyConnected(data = input, num.hidden = 5)
  act1 <- mx.symbol.Activation(data = fc1, act_type = "sigmoid")
  fc2 <- mx.symbol.FullyConnected(data = act1, num.hidden = 1)
  lro <- mx.symbol.LinearRegressionOutput(data = fc2)
  
  mx.set.seed(2020) # 가중치 초기값 고정.
  
  tic()
  logger <- mx.metric.logger$new()
  model <- mx.model.FeedForward.create(symbol = lro, X = turbine_train_x_datamatrix, y = turbine_train_y,
                                       eval.data = list(data = turbine_test_x_datamatrix, label = turbine_test_y),
                                       ctx = mx.gpu(), num.round = 100, optimizer = 'sgd',
                                       array.batch.size = 20, learning.rate = 0.001, momentum = 0.9, eval.metric = mx.metric.rmse,
                                       verbose = T, epoch.end.callback = mx.callback.log.train.metric(1, logger))  # 초매계변수 조율 필요.
  
  exectime <- toc()
  exectime <- round(exectime$toc - exectime$tic, 5)
  
  turbine_model_list[[i]] <- model
  turbine_dnn_test_rmse[[i]] <- sqrt( mean( (predict(model, turbine_test_x_datamatrix) - turbine_test_y )^2 ) )
  turbine_dnn_exectime_list[[i]] <- exectime
  turbine_dnn_train_logger_list[[i]] <- logger$train
  turbine_dnn_test_logger_list[[i]] <- logger$eval
  
}

turbine_dnn_test_rmse_unlist <- unlist(turbine_dnn_test_rmse)
turbine_dnn_exectime_unlist <- unlist(turbine_dnn_exectime_list)

turbine_dnn_train_logger_unlist <- data.frame( matrix(unlist(turbine_dnn_train_logger_list), ncol = 100))
turbine_dnn_test_logger_unlist <- data.frame( matrix(unlist(turbine_dnn_test_logger_list), ncol = 100))

# 결과저장
# write(t(turbine_dnn_test_rmse_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/turbine_DNN_result.txt")
# write(t(turbine_dnn_exectime_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/turbine_DNN_time.txt")
# write(t(turbine_dnn_train_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/turbine_DNN_train_logger.txt")
# write(t(turbine_dnn_test_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/turbine_DNN_test_logger.txt")

turbine_dnn_test_rmse_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/turbine_DNN_result.txt")
turbine_dnn_test_rmse_unlist <- cbind(turbine_dnn_test_rmse_unlist, rep("DNN(변수전체)", 100))
names(turbine_dnn_test_rmse_unlist) <- c("RMSE", "Model")

summary(turbine_dnn_test_rmse_unlist)
# RMSE                  Model    
# Min.   :0.7261   DNN(변수전체):100  
# 1st Qu.:0.8320                      
# Median :0.9289                      
# Mean   :0.9754                      
# 3rd Qu.:1.1110                      
# Max.   :1.8280
win.graph()
boxplot(turbine_dnn_test_rmse_unlist[1],  main = "Turbine data. DNN 은닉층 1, 은닉노드 5. (전체변수)",
        xlab = "Iter = 100", ylab = "RMSE", ylim = c(0, 2))
vioplot(turbine_dnn_test_rmse_unlist[1], main = "Turbine data. DNN 은닉층 1, 은닉노드 5. (전체변수)",
        xlab = "Iter = 100", ylab = "RMSE", ylim = c(0, 2))
points(mean(turbine_dnn_test_rmse_unlist$RMSE), col = "red", pch = 17) # mean 표시

# 2. Turbine data LASSO 이용 변수선택
# 2. Turbine data LASSO 이용 변수선택
# 2. Turbine data LASSO 이용 변수선택

# LASSO 10-fold CV 방법으로 변수선택 (lambda 1se)
turbine_lasso_fit_10fold <- cv.glmnet(as.matrix(turbine_data[, -8]), turbine_data[, 8], type.measure = "mse",
                                     family = "gaussian", alpha = 1)  # lambda.1se는 Standard error가 가장 Regularized 된 모델이 되는 람다값을 찾아줌.

summary(turbine_lasso_fit_10fold)

win.graph()
plot(turbine_lasso_fit_10fold, main = "Turbine data (LASSO)")

# LASSO lambda.1se 에서 선택된 변수들 추정회귀계수
turbine_lasso_coef <- predict(turbine_lasso_fit_10fold, type = "coefficients", s = turbine_lasso_fit_10fold$lambda.1se)
turbine_lasso_coef


# 2. Turbine data LASSO 이용 변수선택 후 DNN 모형 학습
# 2. Turbine data LASSO 이용 변수선택 후 DNN 모형 학습
# 2. Turbine data LASSO 이용 변수선택 후 DNN 모형 학습


# LASSO 에서 선택된 변수들로
# Train : Test = 8 : 2 나누기. (100번)
# seed 를 바꿔가며 100개의 서로 다른 train, test data set 생성.
# list() 객체에 저장.

turbine_train_x_selec_list <- list()
turbine_train_y_selec_list <- list()

turbine_test_x_selec_list <- list()
turbine_test_y_selec_list <- list()

for(i in 1:100){
  set.seed(i)
  train_index <- sample(1:nrow(turbine_data), 0.8 * nrow(turbine_data), replace = FALSE)
  turbine_train_x <- turbine_data[train_index, -c(8, 10)]
  turbine_train_y <- turbine_data[train_index, 8]
  
  turbine_test_x <- turbine_data[-train_index, -c(8, 10)]
  turbine_test_y <- turbine_data[-train_index, 8]
  
  turbine_train_x_selec_list[[i]] <- turbine_train_x
  turbine_train_y_selec_list[[i]] <- turbine_train_y
  
  turbine_test_x_selec_list[[i]] <- turbine_test_x
  turbine_test_y_selec_list[[i]] <- turbine_test_y
  
}


# https://mxnet.apache.org/api/r  참고.
library(mxnet)

turbine_model_list <- list()
turbine_dnn_selec_test_rmse <- list()
turbine_dnn_selec_exectime_list <- list()
turbine_dnn_selec_train_logger_list <- list()
turbine_dnn_selec_test_logger_list <- list()

for(i in 1:100){ # 1:100
  turbine_train_x_selec_scale <- scale(turbine_train_x_selec_list[[i]])
  turbine_test_x_selec_scale <- scale(turbine_test_x_selec_list[[i]])
  
  turbine_train_x_selec_datamatrix <- data.matrix(turbine_train_x_selec_scale)
  turbine_test_x_selec_datamatrix <- data.matrix(turbine_test_x_selec_scale)
  
  turbine_train_y <- turbine_train_y_selec_list[[i]]
  turbine_test_y <- turbine_test_y_selec_list[[i]]
  
  input <- mx.symbol.Variable("data")
  fc1 <- mx.symbol.FullyConnected(data = input, num.hidden = 5)
  act1 <- mx.symbol.Activation(data = fc1, act_type = "sigmoid")
  fc2 <- mx.symbol.FullyConnected(data = act1, num.hidden = 1)
  lro <- mx.symbol.LinearRegressionOutput(data = fc2)
  
  mx.set.seed(2020) # 가중치 초기값 고정.
  
  tic()
  
  logger <- mx.metric.logger$new()
  model <- mx.model.FeedForward.create(symbol = lro, X = turbine_train_x_selec_datamatrix, y = turbine_train_y,
                                       eval.data = list(data = turbine_test_x_selec_datamatrix, label = turbine_test_y),
                                       ctx = mx.gpu(), num.round = 100, optimizer = 'sgd',
                                       array.batch.size = 20, learning.rate = 0.001, momentum = 0.9, eval.metric = mx.metric.rmse,
                                       verbose = T, epoch.end.callback = mx.callback.log.train.metric(1, logger))  # 초매개변수 조율 필요.
  exectime <- toc()
  exectime <- round(exectime$toc - exectime$tic, 5)
  
  turbine_model_list[[i]] <- model
  turbine_dnn_selec_test_rmse[[i]] <- sqrt( mean( (predict(model, turbine_test_x_selec_datamatrix) - turbine_test_y )^2 ) )
  turbine_dnn_selec_exectime_list[[i]] <- exectime
  turbine_dnn_selec_train_logger_list[[i]] <- logger$train
  turbine_dnn_selec_test_logger_list[[i]] <- logger$eval
  
}

turbine_dnn_selec_test_rmse_unlist <- unlist(turbine_dnn_selec_test_rmse)
turbine_dnn_selec_exectime_unlist <- unlist(turbine_dnn_selec_exectime_list)

turbine_dnn_selec_train_logger_unlist <- data.frame( matrix(unlist(turbine_dnn_selec_train_logger_list), ncol = 100))
turbine_dnn_selec_test_logger_unlist <- data.frame( matrix(unlist(turbine_dnn_selec_test_logger_list), ncol = 100))

# 결과저장
# write(t(turbine_dnn_selec_test_rmse_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/turbine_DNN_selec_result.txt")
# write(t(turbine_dnn_selec_exectime_unlist), ncolumns = 1, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/turbine_DNN_selec_time.txt")
# write(t(turbine_dnn_selec_train_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/turbine_DNN_selec_train_logger.txt")
# write(t(turbine_dnn_selec_test_logger_unlist), ncolumns = 100, "C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/turbine_DNN_selec_test_logger.txt")

turbine_dnn_selec_test_rmse_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/turbine_DNN_selec_result.txt")
turbine_dnn_selec_test_rmse_unlist <- cbind(turbine_dnn_selec_test_rmse_unlist, rep("DNN(LASSO변수선택)", 100))
names(turbine_dnn_selec_test_rmse_unlist) <- c("RMSE", "Model")


summary(turbine_dnn_selec_test_rmse_unlist)
# RMSE                       Model    
# Min.   :0.7008   DNN(LASSO변수선택):100  
# 1st Qu.:0.7877                           
# Median :0.8570                           
# Mean   :0.9197                           
# 3rd Qu.:1.0067                           
# Max.   :1.7875 
win.graph()
boxplot(turbine_dnn_selec_test_rmse_unlist[1],  main = "Turbine data. DNN 은닉층 1, 은닉노드 5. (LASSO 변수선택)",
        xlab = "Iter = 100", ylab = "RMSE", ylim = c(0, 2))
vioplot(turbine_dnn_selec_test_rmse_unlist[1], main = "Turbine data. DNN 은닉층 1, 은닉노드 5. (LASSO 변수선택)",
        xlab = "Iter = 100", ylab = "RMSE", ylim = c(0, 2))
points(mean(turbine_dnn_selec_test_rmse_unlist$RMSE), col = "red", pch = 17) # mean 표시

# 3. Turbine data Stepwise 변수선택 후 DNN 모형 학습
# 3. Turbine data Stepwise 변수선택 후 DNN 모형 학습
# 3. Turbine data Stepwise 변수선택 후 DNN 모형 학습

lm_model <- lm(TEY ~ ., data = turbine_data)
lm_step_model <- step(lm_model, direction = "both")
summary(lm_step_model)  # 모든 변수 선택.



