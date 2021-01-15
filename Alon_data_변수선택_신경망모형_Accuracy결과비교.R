library(agricolae)
library(lawstat)
library(vioplot)

alon_dnn_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_DNN_result.txt")
alon_dnn_test_acc_unlist <- cbind(alon_dnn_test_acc_unlist, rep("DNN(변수전체)", 100))
names(alon_dnn_test_acc_unlist) <- c("Accuracy", "Model")

alon_dnn_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_DNN_time.txt")
alon_dnn_exectime_unlist <- cbind(alon_dnn_exectime_unlist, rep("DNN(변수전체)", 100))
names(alon_dnn_exectime_unlist) <- c("Time", "Model")

alon_SCAD_MCP_dnn_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_SCAD_MCP_DNN_result.txt")
alon_SCAD_MCP_dnn_test_acc_unlist <- cbind(alon_SCAD_MCP_dnn_test_acc_unlist, rep("DNN(SCAD MCP 변수선택)", 100))
names(alon_SCAD_MCP_dnn_test_acc_unlist) <- c("Accuracy", "Model")

alon_SCAD_MCP_dnn_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_SCAD_MCP_DNN_time.txt")
alon_SCAD_MCP_dnn_exectime_unlist <- cbind(alon_SCAD_MCP_dnn_exectime_unlist, rep("DNN(SCAD MCP 변수선택)", 100))
names(alon_SCAD_MCP_dnn_exectime_unlist) <- c("Time", "Model")

alon_LASSO_dnn_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_LASSO_DNN_result.txt")
alon_LASSO_dnn_test_acc_unlist <- cbind(alon_LASSO_dnn_test_acc_unlist, rep("DNN(LASSO 변수선택)", 100))
names(alon_LASSO_dnn_test_acc_unlist) <- c("Accuracy", "Model")

alon_LASSO_dnn_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_LASSO_DNN_time.txt")
alon_LASSO_dnn_exectime_unlist <- cbind(alon_LASSO_dnn_exectime_unlist, rep("DNN(LASSO 변수선택)", 100))
names(alon_LASSO_dnn_exectime_unlist) <- c("Time", "Model")

## 변수선택 전 후 DNN (Accuracy) 비교 vioplot ##
# 결과 합치기
alon_3model_acc <- rbind(alon_dnn_test_acc_unlist, alon_SCAD_MCP_dnn_test_acc_unlist,
                         alon_LASSO_dnn_test_acc_unlist)

str(alon_3model_acc)

win.graph()
vioplot(alon_3model_acc$Accuracy ~ alon_3model_acc$Model,
        main = "Alon data. DNN 모형 Accuracy 비교",
        xlab = "", ylab = "Accuracy", ylim = c(0.4, 1),
        cex.main = 1.4)

points(c(1, 2, 3), c(mean(alon_3model_acc$Accuracy[alon_3model_acc$Model == "DNN(변수전체)"]), 
                     mean(alon_3model_acc$Accuracy[alon_3model_acc$Model == "DNN(SCAD & MCP 변수선택)"]),
                     mean(alon_3model_acc$Accuracy[alon_3model_acc$Model == "DNN(LASSO 변수선택)"])), 
       col = "red", pch = 17)

## 변수선택 전 후 DNN (Time) 비교 vioplot ##
# 결과 합치기
alon_3model_exectime <- rbind(alon_dnn_exectime_unlist, alon_SCAD_MCP_dnn_exectime_unlist,
                              alon_LASSO_dnn_exectime_unlist)

str(alon_3model_exectime)

win.graph()
vioplot(alon_3model_exectime$Time ~ alon_3model_exectime$Model,
        main = "Alon data. DNN 모형 Time(학습시간) 비교",
        xlab = "", ylab = "Time(학습시간)", ylim = c(0, 8),
        cex.main = 1.4)

points(c(1, 2, 3), c(mean(alon_3model_exectime$Time[alon_3model_exectime$Model == "DNN(변수전체)"]), 
                     mean(alon_3model_exectime$Time[alon_3model_exectime$Model == "DNN(SCAD MCP 변수선택)"]),
                     mean(alon_3model_exectime$Time[alon_3model_exectime$Model == "DNN(LASSO 변수선택)"])), 
       col = "red", pch = 17)

## 변수선택 전 후 DNN (Accuracy) ##
## 분산분석 및 사후검정 ##
# 분산분석
alon_acc_anova <- aov(Accuracy ~ Model, data = alon_3model_acc)
summary(alon_acc_anova)

alon_acc_kruskal <- kruskal.test(Accuracy ~ Model, data = alon_3model_acc)
alon_acc_kruskal

## 변수선택 전 후 DNN (Time) ##
## 분산분석 및 사후검정 ##
# 분산분석
alon_exectime_anova <- aov(Time ~ Model, data = alon_3model_exectime)
summary(alon_exectime_anova)

alon_exectime_kruskal <- kruskal.test(Time ~ Model, data = alon_3model_exectime)
alon_exectime_kruskal

# 정규성 검정
shapiro.test(alon_exectime_anova$residuals)

# 등분산 검정
plot(alon_exectime_anova$fitted.values, alon_exectime_anova$residuals,
     main = "잔차 대 적합값",
     xlab = "적합값",
     ylab = "잔차", cex.lab = 1.5)

# 사후검정
alon_exectime_sceffe <- scheffe.test(alon_exectime_anova, "Model")
alon_exectime_bonferroni <- LSD.test(alon_exectime_anova, "Model", p.adj = "bonferroni")
alon_exectime_Tukey <- HSD.test(alon_exectime_anova, "Model")


alon_exectime_sceffe
alon_exectime_bonferroni
alon_exectime_Tukey

win.graph()
plot(alon_exectime_sceffe, variation = "SD", main = "Scheffe 사후검정 결과")
plot(alon_exectime_bonferroni, variation = "SD", main = "Bonferroni 사후검정 결과")
plot(alon_exectime_Tukey, variation = "SD", main = "Tukey 사후검정 결과")


## Accuracy 대 Time 산점도 ##
## Accuracy 대 Time 산점도 ##
## Accuracy 대 Time 산점도 ##
win.graph()
plot(alon_3model_acc$Accuracy, alon_3model_exectime$Time,
     col = alon_3model_acc$Model, ylim = c(0, 20), xlim = c(0.5, 1),
     main = "Alon data Accuracy 대 Time 분포",
     xlab = "Accuracy", ylab = "Time (학습시간)", pch = 1, cex = 1.5)
legend("topright", c("DNN(변수전체)", "DNN(SCAD & MCP 변수선택)", "DNN(LASSO 변수선택)"), 
       col = c(1, 2, 3), cex = 1.3, pch = 1
)
