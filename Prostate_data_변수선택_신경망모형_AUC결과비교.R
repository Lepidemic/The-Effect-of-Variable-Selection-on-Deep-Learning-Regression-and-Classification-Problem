library(agricolae)
library(lawstat)

prostate_dnn_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_DNN_auc_result.txt")
prostate_dnn_test_auc_unlist <- cbind(prostate_dnn_test_auc_unlist, rep("DNN(변수전체)", 100))
names(prostate_dnn_test_auc_unlist) <- c("AUC", "Model")

prostate_SCAD_dnn_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_SCAD_DNN_auc_result.txt")
prostate_SCAD_dnn_test_auc_unlist <- cbind(prostate_SCAD_dnn_test_auc_unlist, rep("DNN(SCAD 변수선택)", 100))
names(prostate_SCAD_dnn_test_auc_unlist) <- c("AUC", "Model")

prostate_MCP_dnn_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_MCP_DNN_auc_result.txt")
prostate_MCP_dnn_test_auc_unlist <- cbind(prostate_MCP_dnn_test_auc_unlist, rep("DNN(MCP 변수선택)", 100))
names(prostate_MCP_dnn_test_auc_unlist) <- c("AUC", "Model")

prostate_LASSO_dnn_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/prostate_LASSO_DNN_auc_result.txt")
prostate_LASSO_dnn_test_auc_unlist <- cbind(prostate_LASSO_dnn_test_auc_unlist, rep("DNN(LASSO 변수선택)", 100))
names(prostate_LASSO_dnn_test_auc_unlist) <- c("AUC", "Model")

## 변수선택 전 후 DNN (Accuracy) 비교 vioplot ##
# 결과 합치기
prostate_4model_auc <- rbind(prostate_dnn_test_auc_unlist, prostate_SCAD_dnn_test_auc_unlist,
                             prostate_MCP_dnn_test_auc_unlist, prostate_LASSO_dnn_test_auc_unlist)

str(prostate_4model_auc)

win.graph()
vioplot(prostate_4model_auc$AUC ~ prostate_4model_auc$Model,
        main = "Prostate data. DNN 모형 AUC 비교",
        xlab = "", ylab = "AUC", ylim = c(0.5, 1),
        cex.main = 1.4)

points(c(1, 2, 3, 4), c(mean(prostate_4model_auc$AUC[prostate_4model_auc$Model == "DNN(변수전체)"]), 
                        mean(prostate_4model_auc$AUC[prostate_4model_auc$Model == "DNN(SCAD 변수선택)"]),
                        mean(prostate_4model_auc$AUC[prostate_4model_auc$Model == "DNN(MCP 변수선택)"]),
                        mean(prostate_4model_auc$AUC[prostate_4model_auc$Model == "DNN(LASSO 변수선택)"])), 
       col = "red", pch = 17)


## 변수선택 전 후 DNN (AUC) ##
## 분산분석 및 사후검정 ##
# 분산분석
prostate_auc_anova <- aov(AUC ~ Model, data = prostate_4model_auc)
summary(prostate_auc_anova)

prostate_auc_kruskal <- kruskal.test(AUC ~ Model, data = prostate_4model_auc)
prostate_auc_kruskal

# 정규성 검정
shapiro.test(prostate_auc_anova$residuals)

# 등분산 검정
plot(prostate_auc_anova$fitted.values, prostate_auc_anova$residuals,
     main = "잔차 대 적합값",
     xlab = "적합값",
     ylab = "잔차", cex.lab = 1.5)

# 사후검정
prostate_auc_sceffe <- scheffe.test(prostate_auc_anova, "Model")
prostate_auc_bonferroni <- LSD.test(prostate_auc_anova, "Model", p.adj = "bonferroni")
prostate_auc_Tukey <- HSD.test(prostate_auc_anova, "Model")


prostate_auc_sceffe
prostate_auc_bonferroni
prostate_auc_Tukey

win.graph()
plot(prostate_auc_sceffe, variation = "SD", main = "Scheffe 사후검정 결과")
plot(prostate_auc_bonferroni, variation = "SD", main = "Bonferroni 사후검정 결과")
plot(prostate_auc_Tukey, variation = "SD", main = "Tukey 사후검정 결과")


