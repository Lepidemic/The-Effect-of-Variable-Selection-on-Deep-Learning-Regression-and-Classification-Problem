library(agricolae)
library(lawstat)
library(vioplot)

alon_dnn_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_DNN_auc_result.txt")
alon_dnn_test_auc_unlist <- cbind(alon_dnn_test_auc_unlist, rep("DNN(변수전체)", 100))
names(alon_dnn_test_auc_unlist) <- c("AUC", "Model")

alon_SCAD_MCP_dnn_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_SCAD_MCP_DNN_auc_result.txt")
alon_SCAD_MCP_dnn_test_auc_unlist <- cbind(alon_SCAD_MCP_dnn_test_auc_unlist, rep("DNN(SCAD MCP 변수선택)", 100))
names(alon_SCAD_MCP_dnn_test_auc_unlist) <- c("AUC", "Model")

alon_LASSO_dnn_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/alon_LASSO_DNN_auc_result.txt")
alon_LASSO_dnn_test_auc_unlist <- cbind(alon_LASSO_dnn_test_auc_unlist, rep("DNN(LASSO 변수선택)", 100))
names(alon_LASSO_dnn_test_auc_unlist) <- c("AUC", "Model")

## 변수선택 전 후 DNN (AUC) 비교 vioplot ##
# 결과 합치기
alon_3model_auc <- rbind(alon_dnn_test_auc_unlist, alon_SCAD_MCP_dnn_test_auc_unlist,
                         alon_LASSO_dnn_test_auc_unlist)

str(alon_3model_auc)

win.graph()
vioplot(alon_3model_auc$AUC ~ alon_3model_auc$Model,
        main = "Alon data. DNN 모형 AUC 비교",
        xlab = "", ylab = "AUC", ylim = c(0.4, 1),
        cex.main = 1.4)

points(c(1, 2, 3), c(mean(alon_3model_auc$AUC[alon_3model_auc$Model == "DNN(변수전체)"]), 
                     mean(alon_3model_auc$AUC[alon_3model_auc$Model == "DNN(SCAD MCP 변수선택)"]),
                     mean(alon_3model_auc$AUC[alon_3model_auc$Model == "DNN(LASSO 변수선택)"])), 
       col = "red", pch = 17)


## 변수선택 전 후 DNN (AUC) ##
## 분산분석 및 사후검정 ##
# 분산분석
alon_auc_anova <- aov(AUC ~ Model, data = alon_3model_auc)
summary(alon_auc_anova)

alon_auc_kruskal <- kruskal.test(AUC ~ Model, data = alon_3model_auc)
alon_auc_kruskal

# 정규성 검정
shapiro.test(alon_auc_anova$residuals)

# 등분산 검정
plot(alon_auc_anova$fitted.values, alon_auc_anova$residuals,
     main = "잔차 대 적합값",
     xlab = "적합값",
     ylab = "잔차", cex.lab = 1.5)

# 사후검정
alon_auc_sceffe <- scheffe.test(alon_auc_anova, "Model")
alon_auc_bonferroni <- LSD.test(alon_auc_anova, "Model", p.adj = "bonferroni")
alon_auc_Tukey <- HSD.test(alon_auc_anova, "Model")


alon_auc_sceffe
alon_auc_bonferroni
alon_auc_Tukey

win.graph()
plot(alon_auc_sceffe, variation = "SD", main = "Scheffe 사후검정 결과")
plot(alon_auc_bonferroni, variation = "SD", main = "Bonferroni 사후검정 결과")
plot(alon_auc_Tukey, variation = "SD", main = "Tukey 사후검정 결과")

