library(vioplot)
library(agricolae)
library(lawstat)

turbine_dnn_test_rmse_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/turbine_DNN_result.txt")
turbine_dnn_test_rmse_unlist <- cbind(turbine_dnn_test_rmse_unlist, rep("DNN(변수전체)", 100))
names(turbine_dnn_test_rmse_unlist) <- c("RMSE", "Model")

summary(turbine_dnn_test_rmse_unlist)


turbine_dnn_selec_test_rmse_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/turbine_DNN_selec_result.txt")
turbine_dnn_selec_test_rmse_unlist <- cbind(turbine_dnn_selec_test_rmse_unlist, rep("DNN(LASSO변수선택)", 100))
names(turbine_dnn_selec_test_rmse_unlist) <- c("RMSE", "Model")


summary(turbine_dnn_selec_test_rmse_unlist)


turbine_dnn_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/turbine_DNN_time.txt")
turbine_dnn_exectime_unlist <- cbind(turbine_dnn_exectime_unlist, rep("DNN(변수전체)", 100))
names(turbine_dnn_exectime_unlist) <- c("Time", "Model")

turbine_dnn_selec_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/turbine_DNN_selec_time.txt")
turbine_dnn_selec_exectime_unlist <- cbind(turbine_dnn_selec_exectime_unlist, rep("DNN(LASSO 변수선택)", 100))
names(turbine_dnn_selec_exectime_unlist) <- c("Time", "Model")


## 변수선택 전 후 DNN 비교 vioplot##
# 결과 합치기
turbine_2model_rmse <- rbind(turbine_dnn_test_rmse_unlist, turbine_dnn_selec_test_rmse_unlist)

str(turbine_2model_rmse)

# win.graph()
# par(mfrow = c(1, 3))
# vioplot(turbine_dnn_test_rmse_unlist$RMSE, main = "turbine data. DNN 은닉층 1, 은닉노드 5. (전체변수)",
#         xlab = "Iter = 100", ylab = "RMSE", ylim = c(0, 10))
# vioplot(turbine_dnn_selec_test_rmse_unlist$RMSE, main = "turbine data. DNN 은닉층 1, 은닉노드 5. (LASSO 변수선택)",
#         xlab = "Iter = 100", ylab = "RMSE", ylim = c(0, 10))
# vioplot(turbine_dnn_selec_test_rmse_unlist$RMSE, main = "turbine data. DNN 은닉층 1, 은닉노드 5. (Steowise 변수선택)",
#         xlab = "Iter = 100", ylab = "RMSE", ylim = c(0, 10))

win.graph()
vioplot(turbine_2model_rmse$RMSE ~ turbine_2model_rmse$Model,
        main = "Turbine data. DNN 모형 RMSE 비교",
        xlab = "Iter = 100", ylab = "RMSE", ylim = c(0, 2),
        cex.main = 1.4)
points(c(1, 2), c(mean(turbine_2model_rmse$RMSE[turbine_2model_rmse$Model == "DNN(변수전체)"]), 
                     mean(turbine_2model_rmse$RMSE[turbine_2model_rmse$Model == "DNN(LASSO변수선택)"])),
       col = "red", pch = 17)

## 변수선택 전 후 DNN (Time) 비교 vioplot##
# 결과 합치기
turbine_2model_time <- rbind(turbine_dnn_exectime_unlist, turbine_dnn_selec_exectime_unlist)

str(turbine_2model_time)

# 변수전체 Time summary
summary(turbine_dnn_exectime_unlist)
summary(turbine_2model_time$Time[turbine_2model_time$Model == "DNN(변수전체)"])

win.graph()
boxplot(turbine_2model_time$Time[turbine_2model_time$Model == "DNN(변수전체)"],  main = "Turbine data. DNN 은닉층 1, 은닉노드 5. (전체변수)",
        xlab = "Iter = 100", ylab = "Time(단위 : 초)", ylim = c(30, 50))
vioplot(turbine_2model_time$Time[turbine_2model_time$Model == "DNN(변수전체)"],  main = "Turbine data. DNN 은닉층 1, 은닉노드 5. (전체변수)",
        xlab = "Iter = 100", ylab = "Time(단위 : 초)", ylim = c(30, 50))
points(mean(turbine_2model_time$Time[turbine_2model_time$Model == "DNN(변수전체)"]), col = "red", pch = 17) # mean 표시

vioplot(turbine_dnn_exectime_unlist[1],  main = "Turbine data. DNN 은닉층 1, 은닉노드 5. (전체변수)",
        xlab = "Iter = 100", ylab = "Time(단위 : 초)", ylim = c(30, 50))
points(mean(turbine_dnn_exectime_unlist[ ,1]), col = "red", pch = 17) # mean 표시
# LASSO 변수선택 Time summary
summary(turbine_dnn_selec_exectime_unlist)
summary(turbine_2model_time$Time[turbine_2model_time$Model == "DNN(LASSO 변수선택)"])

win.graph()
boxplot(turbine_2model_time$Time[turbine_2model_time$Model == "DNN(LASSO 변수선택)"],  main = "Turbine data. DNN 은닉층 1, 은닉노드 5. (전체변수)",
        xlab = "Iter = 100", ylab = "Time(단위 : 초)", ylim = c(30, 50))
vioplot(turbine_2model_time$Time[turbine_2model_time$Model == "DNN(LASSO 변수선택)"],  main = "Turbine data. DNN 은닉층 1, 은닉노드 5. (전체변수)",
        xlab = "Iter = 100", ylab = "Time(단위 : 초)", ylim = c(30, 50))
points(mean(turbine_2model_time$Time[turbine_2model_time$Model == "DNN(LASSO 변수선택)"]), col = "red", pch = 17) # mean 표시

vioplot(turbine_dnn_selec_exectime_unlist[1],  main = "Turbine data. DNN 은닉층 1, 은닉노드 5. (LASSO 변수선택)",
        xlab = "Iter = 100", ylab = "Time(단위 : 초)", ylim = c(30, 50))
points(mean(turbine_dnn_selec_exectime_unlist[ ,1]), col = "red", pch = 17) # mean 표시


win.graph()
vioplot(turbine_2model_time$Time ~ turbine_2model_time$Model,
        main = "Turbine data. DNN 모형 Time(학습시간) 비교",
        xlab = "", ylab = "Time(학습시간)", ylim = c(30, 50),
        cex.main = 1.4)
points(c(1, 2), c(mean(turbine_2model_time$Time[turbine_2model_time$Model == "DNN(변수전체)"]), 
                     mean(turbine_2model_time$Time[turbine_2model_time$Model == "DNN(LASSO 변수선택)"])), 
       col = "red", pch = 17)

# t-test
t.test(turbine_dnn_test_rmse_unlist[1], turbine_dnn_selec_test_rmse_unlist[1])

# t-test
t.test(turbine_dnn_exectime_unlist[1], turbine_dnn_selec_exectime_unlist[1])

## RMSE 대 Time 산점도 ##
## RMSE 대 Time 산점도 ##
## RMSE 대 Time 산점도 ##
win.graph()
plot(turbine_2model_rmse$RMSE, turbine_2model_time$Time,
     col = turbine_2model_rmse$Model, ylim = c(30, 50), xlim = c(0, 2),
     main = "Turbine data RMSE 대 Time 분포",
     xlab = "RMSE", ylab = "Time (학습시간)", pch = 1, cex = 1.5)
legend("topright", c("DNN(변수전체)", "DNN(LASSO 변수선택)"), 
       col = c(1, 2), cex = 1.3, pch = 1
)

