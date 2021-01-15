library(agricolae)
library(lawstat)

leukemia_dnn_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_DNN_result.txt")
leukemia_dnn_test_acc_unlist <- cbind(leukemia_dnn_test_acc_unlist, rep("DNN(변수전체)", 100))
names(leukemia_dnn_test_acc_unlist) <- c("Accuracy", "Model")

leukemia_dnn_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_DNN_time.txt")
leukemia_dnn_exectime_unlist <- cbind(leukemia_dnn_exectime_unlist, rep("DNN(변수전체)", 100))
names(leukemia_dnn_exectime_unlist) <- c("Time", "Model")

leukemia_SCAD_dnn_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_SCAD_DNN_result.txt")
leukemia_SCAD_dnn_test_acc_unlist <- cbind(leukemia_SCAD_dnn_test_acc_unlist, rep("DNN(SCAD 변수선택)", 100))
names(leukemia_SCAD_dnn_test_acc_unlist) <- c("Accuracy", "Model")

leukemia_SCAD_dnn_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_SCAD_DNN_time.txt")
leukemia_SCAD_dnn_exectime_unlist <- cbind(leukemia_SCAD_dnn_exectime_unlist, rep("DNN(SCAD 변수선택)", 100))
names(leukemia_SCAD_dnn_exectime_unlist) <- c("Time", "Model")


leukemia_MCP_dnn_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_MCP_DNN_result.txt")
leukemia_MCP_dnn_test_acc_unlist <- cbind(leukemia_MCP_dnn_test_acc_unlist, rep("DNN(MCP 변수선택)", 100))
names(leukemia_MCP_dnn_test_acc_unlist) <- c("Accuracy", "Model")

leukemia_MCP_dnn_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_MCP_DNN_time.txt")
leukemia_MCP_dnn_exectime_unlist <- cbind(leukemia_MCP_dnn_exectime_unlist, rep("DNN(MCP 변수선택)", 100))
names(leukemia_MCP_dnn_exectime_unlist) <- c("Time", "Model")


leukemia_LASSO_dnn_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_LASSO_DNN_result.txt")
leukemia_LASSO_dnn_test_acc_unlist <- cbind(leukemia_LASSO_dnn_test_acc_unlist, rep("DNN(LASSO 변수선택)", 100))
names(leukemia_LASSO_dnn_test_acc_unlist) <- c("Accuracy", "Model")

leukemia_LASSO_dnn_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_LASSO_DNN_time.txt")
leukemia_LASSO_dnn_exectime_unlist <- cbind(leukemia_LASSO_dnn_exectime_unlist, rep("DNN(LASSO 변수선택)", 100))
names(leukemia_LASSO_dnn_exectime_unlist) <- c("Time", "Model")

## 변수선택 전 후 DNN (Accuracy) 비교 vioplot ##
# 결과 합치기
leukemia_4model_acc <- rbind(leukemia_dnn_test_acc_unlist, leukemia_SCAD_dnn_test_acc_unlist,
                             leukemia_MCP_dnn_test_acc_unlist, leukemia_LASSO_dnn_test_acc_unlist)

str(leukemia_4model_acc)

win.graph()
vioplot(leukemia_4model_acc$Accuracy ~ leukemia_4model_acc$Model,
        main = "Leukemia data. DNN 모형 Accuracy 비교",
        xlab = "", ylab = "Accuracy", ylim = c(0.5, 1),
        cex.main = 1.4)

points(c(1, 2, 3, 4), c(mean(leukemia_4model_acc$Accuracy[leukemia_4model_acc$Model == "DNN(변수전체)"]), 
                  mean(leukemia_4model_acc$Accuracy[leukemia_4model_acc$Model == "DNN(SCAD 변수선택)"]),
                  mean(leukemia_4model_acc$Accuracy[leukemia_4model_acc$Model == "DNN(MCP 변수선택)"]),
                  mean(leukemia_4model_acc$Accuracy[leukemia_4model_acc$Model == "DNN(LASSO 변수선택)"])), 
       col = "red", pch = 17)

## 변수선택 전 후 DNN (Time) 비교 vioplot ##
# 결과 합치기
leukemia_4model_exectime <- rbind(leukemia_dnn_exectime_unlist, leukemia_SCAD_dnn_exectime_unlist,
                             leukemia_MCP_dnn_exectime_unlist, leukemia_LASSO_dnn_exectime_unlist)

str(leukemia_4model_exectime)

win.graph()
vioplot(leukemia_4model_exectime$Time ~ leukemia_4model_exectime$Model,
        main = "Leukemia data. DNN 모형 Time(학습시간) 비교",
        xlab = "", ylab = "Time(학습시간)", ylim = c(0, 11),
        cex.main = 1.4)

points(c(1, 2, 3, 4), c(mean(leukemia_4model_exectime$Time[leukemia_4model_exectime$Model == "DNN(변수전체)"]), 
                        mean(leukemia_4model_exectime$Time[leukemia_4model_exectime$Model == "DNN(SCAD 변수선택)"]),
                        mean(leukemia_4model_exectime$Time[leukemia_4model_exectime$Model == "DNN(MCP 변수선택)"]),
                        mean(leukemia_4model_exectime$Time[leukemia_4model_exectime$Model == "DNN(LASSO 변수선택)"])), 
       col = "red", pch = 17)

## 변수선택 전 후 DNN (Accuracy) ##
## 분산분석 및 사후검정 ##
# 분산분석
leukemia_acc_anova <- aov(Accuracy ~ Model, data = leukemia_4model_acc)
summary(leukemia_acc_anova)

leukemia_acc_kruskal <- kruskal.test(Accuracy ~ Model, data = leukemia_4model_acc)
leukemia_acc_kruskal

## 변수선택 전 후 DNN (Time) ##
## 분산분석 및 사후검정 ##
# 분산분석
leukemia_exectime_anova <- aov(Time ~ Model, data = leukemia_4model_exectime)
summary(leukemia_exectime_anova)

leukemia_exectime_kruskal <- kruskal.test(Time ~ Model, data = leukemia_4model_exectime)
leukemia_exectime_kruskal

# 정규성 검정
shapiro.test(leukemia_exectime_anova$residuals)

# 등분산 검정
plot(leukemia_exectime_anova$fitted.values, leukemia_exectime_anova$residuals,
     main = "잔차 대 적합값",
     xlab = "적합값",
     ylab = "잔차", cex.lab = 1.5)

# 사후검정
leukemia_exectime_sceffe <- scheffe.test(leukemia_exectime_anova, "Model")
leukemia_exectime_bonferroni <- LSD.test(leukemia_exectime_anova, "Model", p.adj = "bonferroni")
leukemia_exectime_Tukey <- HSD.test(leukemia_exectime_anova, "Model")


leukemia_exectime_sceffe
leukemia_exectime_bonferroni
leukemia_exectime_Tukey

win.graph()
plot(leukemia_exectime_sceffe, variation = "SD", main = "Scheffe 사후검정 결과")
plot(leukemia_exectime_bonferroni, variation = "SD", main = "Bonferroni 사후검정 결과")
plot(leukemia_exectime_Tukey, variation = "SD", main = "Tukey 사후검정 결과")

