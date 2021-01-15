

adult_dnn_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Adult_DNN_result.txt")
adult_dnn_test_acc_unlist <- cbind(adult_dnn_test_acc_unlist, rep("DNN(변수전체)", 100))
names(adult_dnn_test_acc_unlist) <- c("Accuracy", "Model")

adult_dnn_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Adult_DNN_time.txt")
adult_dnn_exectime_unlist <- cbind(adult_dnn_exectime_unlist, rep("DNN(변수전체)", 100))
names(adult_dnn_exectime_unlist) <- c("Time", "Model")

adult_dnn_selec_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Adult_DNN_selec_result.txt")
adult_dnn_selec_test_acc_unlist <- cbind(adult_dnn_selec_test_acc_unlist, rep("DNN(LASSO 변수선택)", 100))
names(adult_dnn_selec_test_acc_unlist) <- c("Accuracy", "Model")

adult_dnn_selec_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Adult_DNN_selec_time.txt")
adult_dnn_selec_exectime_unlist <- cbind(adult_dnn_selec_exectime_unlist, rep("DNN(LASSO 변수선택)", 100))
names(adult_dnn_selec_exectime_unlist) <- c("Time", "Model")

adult_dnn_stepwise_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Adult_DNN_stepwise_result.txt")
adult_dnn_stepwise_test_acc_unlist <- cbind(adult_dnn_stepwise_test_acc_unlist, rep("DNN(Stepwise 변수선택)", 100))
names(adult_dnn_stepwise_test_acc_unlist) <- c("Accuracy", "Model")

adult_dnn_stepwise_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Adult_DNN_stepwise_time.txt")
adult_dnn_stepwise_exectime_unlist <- cbind(adult_dnn_stepwise_exectime_unlist, rep("DNN(Stepwise 변수선택)", 100))
names(adult_dnn_stepwise_exectime_unlist) <- c("Time", "Model")


## 변수선택 전 후 DNN (Accuracy) 비교 vioplot##
# 결과 합치기
adult_3model_acc <- rbind(adult_dnn_test_acc_unlist, adult_dnn_selec_test_acc_unlist,
                          adult_dnn_stepwise_test_acc_unlist)

str(adult_3model_acc)

# 변수전체 Accuracy summary
summary(adult_3model_acc$Accuracy[adult_3model_acc$Model == "DNN(변수전체)"])

# LASSO 변수선택 Accuracy summary
summary(adult_3model_acc$Accuracy[adult_3model_acc$Model == "DNN(LASSO 변수선택)"])

# Stepwise 변수선택 Accuracy summary
summary(adult_3model_acc$Accuracy[adult_3model_acc$Model == "DNN(Stepwise 변수선택)"])


win.graph()
vioplot(adult_3model_acc$Accuracy ~ adult_3model_acc$Model,
        main = "Adult data. DNN 모형 Accuracy 비교",
         xlab = "", ylab = "Accuracy", ylim = c(0.5, 1),
        cex.main = 1.4)
points(c(1, 2, 3), c(mean(adult_3model_acc$Accuracy[adult_3model_acc$Model == "DNN(변수전체)"]), 
                  mean(adult_3model_acc$Accuracy[adult_3model_acc$Model == "DNN(LASSO 변수선택)"]),
                  mean(adult_3model_acc$Accuracy[adult_3model_acc$Model == "DNN(Stepwise 변수선택)"])), 
        col = "red", pch = 17)

## 변수선택 전 후 DNN (Time) 비교 vioplot##
# 결과 합치기
adult_3model_time <- rbind(adult_dnn_exectime_unlist, adult_dnn_selec_exectime_unlist,
                           adult_dnn_stepwise_exectime_unlist)

str(adult_3model_time)

# 변수전체 Time summary
summary(adult_3model_time$Time[adult_3model_time$Model == "DNN(변수전체)"])

# LASSO 변수선택 Time summary
summary(adult_3model_time$Time[adult_3model_time$Model == "DNN(LASSO 변수선택)"])

# Steapwise 변수선택 Time summary
summary(adult_3model_time$Time[adult_3model_time$Model == "DNN(Stepwise 변수선택)"])


win.graph()
vioplot(adult_3model_time$Time ~ adult_3model_time$Model,
        main = "Adult data. DNN 모형 Time(학습시간) 비교",
        xlab = "", ylab = "Time(학습시간)", ylim = c(0, 130),
        cex.main = 1.4)
points(c(1, 2, 3), c(mean(adult_3model_time$Time[adult_3model_time$Model == "DNN(변수전체)"]), 
                  mean(adult_3model_time$Time[adult_3model_time$Model == "DNN(LASSO 변수선택)"]),
                  mean(adult_3model_time$Time[adult_3model_time$Model == "DNN(Stepwise 변수선택)"])), 
       col = "red", pch = 17)

## 분산분석 및 사후검정 ##
# Accuracy
adult_rmse_anova <- aov(Accuracy ~ Model, data = adult_3model_rmse)
summary(adult_rmse_anova)

adult_rmse_kruskal <- kruskal.test(Accuracy ~ Model, data = adult_3model_rmse)
adult_rmse_kruskal

# 학습시간 (Time)
adult_exectime_anova <- aov(Time ~ Model, data = adult_3model_time)
summary(adult_exectime_anova)

adult_exectime_kruskal <- kruskal.test(Time ~ Model, data = adult_3model_time)
adult_exectime_kruskal

# 정규성 검정
shapiro.test(adult_exectime_anova$residuals)

# 등분산 검정
plot(adult_exectime_anova$fitted.values, adult_exectime_anova$residuals,
     main = "잔차 대 적합값",
     xlab = "적합값",
     ylab = "잔차", cex.lab = 1.5)

# 사후검정
adult_exectime_sceffe <- scheffe.test(adult_exectime_anova, "Model")
adult_exectime_bonferroni <- LSD.test(adult_exectime_anova, "Model", p.adj = "bonferroni")
adult_exectime_Tukey <- HSD.test(adult_exectime_anova, "Model")


adult_exectime_sceffe
adult_exectime_bonferroni
adult_exectime_Tukey

win.graph()
plot(adult_exectime_sceffe, variation = "SD", main = "Scheffe 사후검정 결과")
plot(adult_exectime_bonferroni, variation = "SD", main = "Bonferroni 사후검정 결과")
plot(adult_exectime_Tukey, variation = "SD", main = "Tukey 사후검정 결과")

## Accuracy 대 Time 산점도 ##
## Accuracy 대 Time 산점도 ##
## Accuracy 대 Time 산점도 ##
win.graph()
plot(adult_3model_acc$Accuracy, adult_3model_time$Time,
     col = adult_3model_acc$Model, ylim = c(0, 150), xlim = c(0.5, 1),
     main = "Adult data Accuracy 대 Time 분포",
     xlab = "Accuracy", ylab = "Time (학습시간)", pch = 1, cex = 1.5)
legend("topleft", c("DNN(변수전체)", "DNN(LASSO 변수선택)", "DNN(Stepwise 변수선택)"), 
       col = c(1, 2, 3), cex = 1.3, pch = 1
)
