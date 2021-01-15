library(agricolae)
library(lawstat)

prostate_dnn_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_DNN_result.txt")
prostate_dnn_test_acc_unlist <- cbind(prostate_dnn_test_acc_unlist, rep("DNN(변수전체)", 100))
names(prostate_dnn_test_acc_unlist) <- c("Accuracy", "Model")

prostate_dnn_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_DNN_time.txt")
prostate_dnn_exectime_unlist <- cbind(prostate_dnn_exectime_unlist, rep("DNN(변수전체)", 100))
names(prostate_dnn_exectime_unlist) <- c("Time", "Model")

prostate_SCAD_dnn_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_SCAD_DNN_result.txt")
prostate_SCAD_dnn_test_acc_unlist <- cbind(prostate_SCAD_dnn_test_acc_unlist, rep("DNN(SCAD 변수선택)", 100))
names(prostate_SCAD_dnn_test_acc_unlist) <- c("Accuracy", "Model")

prostate_SCAD_dnn_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_SCAD_DNN_time.txt")
prostate_SCAD_dnn_exectime_unlist <- cbind(prostate_SCAD_dnn_exectime_unlist, rep("DNN(SCAD 변수선택)", 100))
names(prostate_SCAD_dnn_exectime_unlist) <- c("Time", "Model")


prostate_MCP_dnn_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_MCP_DNN_result.txt")
prostate_MCP_dnn_test_acc_unlist <- cbind(prostate_MCP_dnn_test_acc_unlist, rep("DNN(MCP 변수선택)", 100))
names(prostate_MCP_dnn_test_acc_unlist) <- c("Accuracy", "Model")

prostate_MCP_dnn_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_MCP_DNN_time.txt")
prostate_MCP_dnn_exectime_unlist <- cbind(prostate_MCP_dnn_exectime_unlist, rep("DNN(MCP 변수선택)", 100))
names(prostate_MCP_dnn_exectime_unlist) <- c("Time", "Model")


prostate_LASSO_dnn_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_LASSO_DNN_result.txt")
prostate_LASSO_dnn_test_acc_unlist <- cbind(prostate_LASSO_dnn_test_acc_unlist, rep("DNN(LASSO 변수선택)", 100))
names(prostate_LASSO_dnn_test_acc_unlist) <- c("Accuracy", "Model")

prostate_LASSO_dnn_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_LASSO_DNN_time.txt")
prostate_LASSO_dnn_exectime_unlist <- cbind(prostate_LASSO_dnn_exectime_unlist, rep("DNN(LASSO 변수선택)", 100))
names(prostate_LASSO_dnn_exectime_unlist) <- c("Time", "Model")

## 변수선택 전 후 DNN (Accuracy) 비교 vioplot ##
# 결과 합치기
prostate_4model_acc <- rbind(prostate_dnn_test_acc_unlist, prostate_SCAD_dnn_test_acc_unlist,
                             prostate_MCP_dnn_test_acc_unlist, prostate_LASSO_dnn_test_acc_unlist)

str(prostate_4model_acc)

win.graph()
vioplot(prostate_4model_acc$Accuracy ~ prostate_4model_acc$Model,
        main = "Prostate data. DNN 모형 Accuracy 비교",
        xlab = "", ylab = "Accuracy", ylim = c(0.5, 1),
        cex.main = 1.4)

points(c(1, 2, 3, 4), c(mean(prostate_4model_acc$Accuracy[prostate_4model_acc$Model == "DNN(변수전체)"]), 
                        mean(prostate_4model_acc$Accuracy[prostate_4model_acc$Model == "DNN(SCAD 변수선택)"]),
                        mean(prostate_4model_acc$Accuracy[prostate_4model_acc$Model == "DNN(MCP 변수선택)"]),
                        mean(prostate_4model_acc$Accuracy[prostate_4model_acc$Model == "DNN(LASSO 변수선택)"])), 
       col = "red", pch = 17)

## 변수선택 전 후 DNN (Time) 비교 vioplot ##
# 결과 합치기
prostate_4model_exectime <- rbind(prostate_dnn_exectime_unlist, prostate_SCAD_dnn_exectime_unlist,
                                  prostate_MCP_dnn_exectime_unlist, prostate_LASSO_dnn_exectime_unlist)

str(prostate_4model_exectime)

win.graph()
vioplot(prostate_4model_exectime$Time ~ prostate_4model_exectime$Model,
        main = "prostate data. DNN 모형 Time(학습시간) 비교",
        xlab = "", ylab = "Time(학습시간)", ylim = c(0, 28),
        cex.main = 1.4)

points(c(1, 2, 3, 4), c(mean(prostate_4model_exectime$Time[prostate_4model_exectime$Model == "DNN(변수전체)"]), 
                        mean(prostate_4model_exectime$Time[prostate_4model_exectime$Model == "DNN(SCAD 변수선택)"]),
                        mean(prostate_4model_exectime$Time[prostate_4model_exectime$Model == "DNN(MCP 변수선택)"]),
                        mean(prostate_4model_exectime$Time[prostate_4model_exectime$Model == "DNN(LASSO 변수선택)"])), 
       col = "red", pch = 17)

## 변수선택 전 후 DNN (Accuracy) ##
## 분산분석 및 사후검정 ##
# 분산분석
prostate_acc_anova <- aov(Accuracy ~ Model, data = prostate_4model_acc)
summary(prostate_acc_anova)

prostate_acc_kruskal <- kruskal.test(Accuracy ~ Model, data = prostate_4model_acc)
prostate_acc_kruskal

# 정규성 검정
shapiro.test(prostate_acc_anova$residuals)

# 등분산 검정
plot(prostate_acc_anova$fitted.values, prostate_acc_anova$residuals,
     main = "잔차 대 적합값",
     xlab = "적합값",
     ylab = "잔차", cex.lab = 1.5)

# 사후검정
prostate_acc_sceffe <- scheffe.test(prostate_acc_anova, "Model")
prostate_acc_bonferroni <- LSD.test(prostate_acc_anova, "Model", p.adj = "bonferroni")
prostate_acc_Tukey <- HSD.test(prostate_acc_anova, "Model")


prostate_acc_sceffe
prostate_acc_bonferroni
prostate_acc_Tukey

win.graph()
plot(prostate_acc_sceffe, variation = "SD", main = "Scheffe 사후검정 결과")
plot(prostate_acc_bonferroni, variation = "SD", main = "Bonferroni 사후검정 결과")
plot(prostate_acc_Tukey, variation = "SD", main = "Tukey 사후검정 결과")


## 변수선택 전 후 DNN (Time) ##
## 분산분석 및 사후검정 ##
# 분산분석
prostate_exectime_anova <- aov(Time ~ Model, data = prostate_4model_exectime)
summary(prostate_exectime_anova)

prostate_exectime_kruskal <- kruskal.test(Time ~ Model, data = prostate_4model_exectime)
prostate_exectime_kruskal

# 정규성 검정
shapiro.test(prostate_exectime_anova$residuals)

# 등분산 검정
plot(prostate_exectime_anova$fitted.values, prostate_exectime_anova$residuals,
     main = "잔차 대 적합값",
     xlab = "적합값",
     ylab = "잔차", cex.lab = 1.5)

# 사후검정
prostate_exectime_sceffe <- scheffe.test(prostate_exectime_anova, "Model")
prostate_exectime_bonferroni <- LSD.test(prostate_exectime_anova, "Model", p.adj = "bonferroni")
prostate_exectime_Tukey <- HSD.test(prostate_exectime_anova, "Model")


prostate_exectime_sceffe
prostate_exectime_bonferroni
prostate_exectime_Tukey

win.graph()
plot(prostate_exectime_sceffe, variation = "SD", main = "Scheffe 사후검정 결과")
plot(prostate_exectime_bonferroni, variation = "SD", main = "Bonferroni 사후검정 결과")
plot(prostate_exectime_Tukey, variation = "SD", main = "Tukey 사후검정 결과")

