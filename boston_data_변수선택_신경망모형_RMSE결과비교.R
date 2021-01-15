library(vioplot)
library(agricolae)
library(lawstat)

boston_dnn_test_rmse_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Boston_DNN_result.txt")
boston_dnn_test_rmse_unlist <- cbind(boston_dnn_test_rmse_unlist, rep("DNN(변수전체)", 100))
names(boston_dnn_test_rmse_unlist) <- c("RMSE", "Model")

summary(boston_dnn_test_rmse_unlist)


boston_dnn_selec_test_rmse_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Boston_DNN_selec_result.txt")
boston_dnn_selec_test_rmse_unlist <- cbind(boston_dnn_selec_test_rmse_unlist, rep("DNN(LASSO변수선택)", 100))
names(boston_dnn_selec_test_rmse_unlist) <- c("RMSE", "Model")


summary(boston_dnn_selec_test_rmse_unlist)


boston_dnn_step_test_rmse_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Boston_DNN_step_result.txt")
boston_dnn_step_test_rmse_unlist <- cbind(boston_dnn_step_test_rmse_unlist, rep("DNN(stepwise변수선택)", 100))
names(boston_dnn_step_test_rmse_unlist) <- c("RMSE", "Model")


summary(boston_dnn_step_test_rmse_unlist)


boston_dnn_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/boston_DNN_time.txt")
boston_dnn_exectime_unlist <- cbind(boston_dnn_exectime_unlist, rep("DNN(변수전체)", 100))
names(boston_dnn_exectime_unlist) <- c("Time", "Model")

boston_dnn_selec_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/boston_DNN_selec_time.txt")
boston_dnn_selec_exectime_unlist <- cbind(boston_dnn_selec_exectime_unlist, rep("DNN(LASSO 변수선택)", 100))
names(boston_dnn_selec_exectime_unlist) <- c("Time", "Model")

boston_dnn_stepwise_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/boston_DNN_step_time.txt")
boston_dnn_stepwise_exectime_unlist <- cbind(boston_dnn_stepwise_exectime_unlist, rep("DNN(Stepwise 변수선택)", 100))
names(boston_dnn_stepwise_exectime_unlist) <- c("Time", "Model")

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
points(c(1, 2, 3), c(mean(boston_3model_rmse$RMSE[boston_3model_rmse$Model == "DNN(변수전체)"]), 
                  mean(boston_3model_rmse$RMSE[boston_3model_rmse$Model == "DNN(LASSO변수선택)"]),
                  mean(boston_3model_rmse$RMSE[boston_3model_rmse$Model == "DNN(stepwise변수선택)"])), 
       col = "red", pch = 17)

## 변수선택 전 후 DNN (Time) 비교 vioplot##
# 결과 합치기
boston_3model_time <- rbind(boston_dnn_exectime_unlist, boston_dnn_selec_exectime_unlist,
                           boston_dnn_stepwise_exectime_unlist)

str(boston_3model_time)

# 변수전체 Time summary
summary(boston_dnn_exectime_unlist)
summary(boston_3model_time$Time[boston_3model_time$Model == "DNN(변수전체)"])

win.graph()
boxplot(boston_3model_time$Time[boston_3model_time$Model == "DNN(변수전체)"],  main = "Boston data. DNN 은닉층 1, 은닉노드 5. (전체변수)",
        xlab = "Iter = 100", ylab = "Time(단위 : 초)", ylim = c(0, 5))
vioplot(boston_3model_time$Time[boston_3model_time$Model == "DNN(변수전체)"],  main = "Boston data. DNN 은닉층 1, 은닉노드 5. (전체변수)",
        xlab = "Iter = 100", ylab = "Time(단위 : 초)", ylim = c(0, 5))
points(mean(boston_3model_time$Time[boston_3model_time$Model == "DNN(변수전체)"]), col = "red", pch = 17) # mean 표시

vioplot(boston_dnn_exectime_unlist[1],  main = "Boston data. DNN 은닉층 1, 은닉노드 5. (전체변수)",
        xlab = "Iter = 100", ylab = "Time(단위 : 초)", ylim = c(0, 5))
points(mean(boston_dnn_exectime_unlist[ ,1]), col = "red", pch = 17) # mean 표시
# LASSO 변수선택 Time summary
summary(boston_dnn_selec_exectime_unlist)
summary(boston_3model_time$Time[boston_3model_time$Model == "DNN(LASSO 변수선택)"])

win.graph()
boxplot(boston_3model_time$Time[boston_3model_time$Model == "DNN(LASSO 변수선택)"],  main = "Boston data. DNN 은닉층 1, 은닉노드 5. (전체변수)",
        xlab = "Iter = 100", ylab = "Time(단위 : 초)", ylim = c(0, 5))
vioplot(boston_3model_time$Time[boston_3model_time$Model == "DNN(LASSO 변수선택)"],  main = "Boston data. DNN 은닉층 1, 은닉노드 5. (전체변수)",
        xlab = "Iter = 100", ylab = "Time(단위 : 초)", ylim = c(0, 5))
points(mean(boston_3model_time$Time[boston_3model_time$Model == "DNN(LASSO 변수선택)"]), col = "red", pch = 17) # mean 표시

vioplot(boston_dnn_selec_exectime_unlist[1],  main = "Boston data. DNN 은닉층 1, 은닉노드 5. (LASSO 변수선택)",
        xlab = "Iter = 100", ylab = "Time(단위 : 초)", ylim = c(0, 5))
points(mean(boston_dnn_selec_exectime_unlist[ ,1]), col = "red", pch = 17) # mean 표시
# Stepwise 변수선택 Time summary
summary(boston_dnn_stepwise_exectime_unlist)
summary(boston_3model_time$Time[boston_3model_time$Model == "DNN(Stepwise 변수선택)"])

win.graph()
boxplot(boston_3model_time$Time[boston_3model_time$Model == "DNN(Stepwise 변수선택)"],  main = "Boston data. DNN 은닉층 1, 은닉노드 5. (전체변수)",
        xlab = "Iter = 100", ylab = "Time(단위 : 초)", ylim = c(0, 5))
vioplot(boston_3model_time$Time[boston_3model_time$Model == "DNN(Stepwise 변수선택)"],  main = "Boston data. DNN 은닉층 1, 은닉노드 5. (전체변수)",
        xlab = "Iter = 100", ylab = "Time(단위 : 초)", ylim = c(0, 5))
points(mean(boston_3model_time$Time[boston_3model_time$Model == "DNN(Stepwise 변수선택)"]), col = "red", pch = 17) # mean 표시

vioplot(boston_dnn_stepwise_exectime_unlist[1],  main = "Boston data. DNN 은닉층 1, 은닉노드 5. (Stepwise 변수선택)",
        xlab = "Iter = 100", ylab = "Time(단위 : 초)", ylim = c(0, 5))
points(mean(boston_dnn_stepwise_exectime_unlist[ ,1]), col = "red", pch = 17) # mean 표시


win.graph()
vioplot(boston_3model_time$Time ~ boston_3model_time$Model,
        main = "Boston data. DNN 모형 Time(학습시간) 비교",
        xlab = "", ylab = "Time(학습시간)", ylim = c(0, 5),
        cex.main = 1.4)
points(c(1, 2, 3), c(mean(boston_3model_time$Time[boston_3model_time$Model == "DNN(변수전체)"]), 
                     mean(boston_3model_time$Time[boston_3model_time$Model == "DNN(LASSO 변수선택)"]),
                     mean(boston_3model_time$Time[boston_3model_time$Model == "DNN(Stepwise 변수선택)"])), 
       col = "red", pch = 17)

## 분산분석 및 사후검정 ##
# Accuracy
boston_rmse_anova <- aov(RMSE ~ Model, data = boston_3model_rmse)
summary(boston_rmse_anova)

boston_rmse_kruskal <- kruskal.test(RMSE ~ Model, data = boston_3model_rmse)
boston_rmse_kruskal

# 학습시간 (Time)
boston_exectime_anova <- aov(Time ~ Model, data = boston_3model_time)
summary(boston_exectime_anova)

boston_exectime_kruskal <- kruskal.test(Time ~ Model, data = boston_3model_time)
boston_exectime_kruskal

# 정규성 검정
shapiro.test(boston_exectime_anova$residuals)

# 등분산 검정
plot(boston_exectime_anova$fitted.values, boston_exectime_anova$residuals,
     main = "잔차 대 적합값",
     xlab = "적합값",
     ylab = "잔차", cex.lab = 1.5)

# 사후검정
boston_exectime_sceffe <- scheffe.test(boston_exectime_anova, "Model")
boston_exectime_bonferroni <- LSD.test(boston_exectime_anova, "Model", p.adj = "bonferroni")
boston_exectime_Tukey <- HSD.test(boston_exectime_anova, "Model")


boston_exectime_sceffe
boston_exectime_bonferroni
boston_exectime_Tukey

win.graph()
plot(boston_exectime_sceffe, variation = "SD", main = "Scheffe 사후검정 결과")
plot(boston_exectime_bonferroni, variation = "SD", main = "Bonferroni 사후검정 결과")
plot(boston_exectime_Tukey, variation = "SD", main = "Tukey 사후검정 결과")

