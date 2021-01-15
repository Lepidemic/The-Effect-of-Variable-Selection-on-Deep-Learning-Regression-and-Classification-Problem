library(agricolae)
library(lawstat)

chin_dnn_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_DNN_result.txt")
chin_dnn_test_acc_unlist <- cbind(chin_dnn_test_acc_unlist, rep("DNN(변수전체)", 100))
names(chin_dnn_test_acc_unlist) <- c("Accuracy", "Model")

chin_dnn_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_DNN_time.txt")
chin_dnn_exectime_unlist <- cbind(chin_dnn_exectime_unlist, rep("DNN(변수전체)", 100))
names(chin_dnn_exectime_unlist) <- c("Time", "Model")

chin_SCAD_MCP_dnn_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_SCAD_MCP_DNN_result.txt")
chin_SCAD_MCP_dnn_test_acc_unlist <- cbind(chin_SCAD_MCP_dnn_test_acc_unlist, rep("DNN(SCAD  MCP 변수선택)", 100))
names(chin_SCAD_MCP_dnn_test_acc_unlist) <- c("Accuracy", "Model")

chin_SCAD_MCP_dnn_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_SCAD_MCP_DNN_time.txt")
chin_SCAD_MCP_dnn_exectime_unlist <- cbind(chin_SCAD_MCP_dnn_exectime_unlist, rep("DNN(SCAD  MCP 변수선택)", 100))
names(chin_SCAD_MCP_dnn_exectime_unlist) <- c("Time", "Model")

chin_LASSO_dnn_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_LASSO_DNN_result.txt")
chin_LASSO_dnn_test_acc_unlist <- cbind(chin_LASSO_dnn_test_acc_unlist, rep("DNN(LASSO 변수선택)", 100))
names(chin_LASSO_dnn_test_acc_unlist) <- c("Accuracy", "Model")

chin_LASSO_dnn_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_LASSO_DNN_time.txt")
chin_LASSO_dnn_exectime_unlist <- cbind(chin_LASSO_dnn_exectime_unlist, rep("DNN(LASSO 변수선택)", 100))
names(chin_LASSO_dnn_exectime_unlist) <- c("Time", "Model")

## 변수선택 전 후 DNN (Accuracy) 비교 vioplot ##
# 결과 합치기
chin_3model_acc <- rbind(chin_dnn_test_acc_unlist, chin_SCAD_MCP_dnn_test_acc_unlist,
                         chin_LASSO_dnn_test_acc_unlist)

str(chin_3model_acc)

win.graph()
vioplot(chin_3model_acc$Accuracy ~ chin_3model_acc$Model,
        main = "Chin data. DNN 모형 Accuracy 비교",
        xlab = "", ylab = "Accuracy", ylim = c(0.5, 1),
        cex.main = 1.4)

points(c(1, 2, 3), c(mean(chin_3model_acc$Accuracy[chin_3model_acc$Model == "DNN(변수전체)"]), 
                     mean(chin_3model_acc$Accuracy[chin_3model_acc$Model == "DNN(SCAD  MCP 변수선택)"]),
                     mean(chin_3model_acc$Accuracy[chin_3model_acc$Model == "DNN(LASSO 변수선택)"])), 
       col = "red", pch = 17)

## 변수선택 전 후 DNN (Time) 비교 vioplot ##
# 결과 합치기
chin_3model_exectime <- rbind(chin_dnn_exectime_unlist, chin_SCAD_MCP_dnn_exectime_unlist,
                              chin_LASSO_dnn_exectime_unlist)

str(chin_3model_exectime)

win.graph()
vioplot(chin_3model_exectime$Time ~ chin_3model_exectime$Model,
        main = "Chin data. DNN 모형 Time(학습시간) 비교",
        xlab = "", ylab = "Time(학습시간)", ylim = c(0, 8),
        cex.main = 1.4)

points(c(1, 2, 3), c(mean(chin_3model_exectime$Time[chin_3model_exectime$Model == "DNN(변수전체)"]), 
                     mean(chin_3model_exectime$Time[chin_3model_exectime$Model == "DNN(SCAD  MCP 변수선택)"]),
                     mean(chin_3model_exectime$Time[chin_3model_exectime$Model == "DNN(LASSO 변수선택)"])), 
       col = "red", pch = 17)

## 변수선택 전 후 DNN (Accuracy) ##
## 분산분석 및 사후검정 ##
# 분산분석
chin_acc_anova <- aov(Accuracy ~ Model, data = chin_3model_acc)
summary(chin_acc_anova)

chin_acc_kruskal <- kruskal.test(Accuracy ~ Model, data = chin_3model_acc)
chin_acc_kruskal

# 정규성 검정
shapiro.test(chin_acc_anova$residuals)

# 등분산 검정
plot(chin_acc_anova$fitted.values, chin_acc_anova$residuals,
     main = "잔차 대 적합값",
     xlab = "적합값",
     ylab = "잔차", cex.lab = 1.5)

# 사후검정
chin_acc_sceffe <- scheffe.test(chin_acc_anova, "Model")
chin_acc_bonferroni <- LSD.test(chin_acc_anova, "Model", p.adj = "bonferroni")
chin_acc_Tukey <- HSD.test(chin_acc_anova, "Model")


chin_acc_sceffe
chin_acc_bonferroni
chin_acc_Tukey

win.graph()
plot(chin_acc_sceffe, variation = "SD", main = "Scheffe 사후검정 결과")
plot(chin_acc_bonferroni, variation = "SD", main = "Bonferroni 사후검정 결과")
plot(chin_acc_Tukey, variation = "SD", main = "Tukey 사후검정 결과")

## 변수선택 전 후 DNN (Time) ##
## 분산분석 및 사후검정 ##
# 분산분석
chin_exectime_anova <- aov(Time ~ Model, data = chin_3model_exectime)
summary(chin_exectime_anova)

chin_exectime_kruskal <- kruskal.test(Time ~ Model, data = chin_3model_exectime)
chin_exectime_kruskal

# 정규성 검정
shapiro.test(chin_exectime_anova$residuals)

# 등분산 검정
plot(chin_exectime_anova$fitted.values, chin_exectime_anova$residuals,
     main = "잔차 대 적합값",
     xlab = "적합값",
     ylab = "잔차", cex.lab = 1.5)

# 사후검정
chin_exectime_sceffe <- scheffe.test(chin_exectime_anova, "Model")
chin_exectime_bonferroni <- LSD.test(chin_exectime_anova, "Model", p.adj = "bonferroni")
chin_exectime_Tukey <- HSD.test(chin_exectime_anova, "Model")


chin_exectime_sceffe
chin_exectime_bonferroni
chin_exectime_Tukey

win.graph()
plot(chin_exectime_sceffe, variation = "SD", main = "Scheffe 사후검정 결과")
plot(chin_exectime_bonferroni, variation = "SD", main = "Bonferroni 사후검정 결과")
plot(chin_exectime_Tukey, variation = "SD", main = "Tukey 사후검정 결과")


## Accuracy 대 Time 산점도 ##
## Accuracy 대 Time 산점도 ##
## Accuracy 대 Time 산점도 ##
win.graph()
plot(chin_3model_acc$Accuracy, chin_3model_exectime$Time,
     col = chin_3model_acc$Model, ylim = c(0, 20), xlim = c(0.5, 1),
     main = "Chin data Accuracy 대 Time 분포",
     xlab = "Accuracy", ylab = "Time (학습시간)", pch = 1, cex = 1.5)
legend("topright", c("DNN(변수전체)", "DNN(SCAD & MCP 변수선택)", "DNN(LASSO 변수선택)"), 
       col = c(1, 2, 3), cex = 1.3, pch = 1
)

