library(agricolae)
library(lawstat)

chin_dnn_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_DNN_auc_result.txt")
chin_dnn_test_auc_unlist <- cbind(chin_dnn_test_auc_unlist, rep("DNN(변수전체)", 100))
names(chin_dnn_test_auc_unlist) <- c("AUC", "Model")

chin_SCAD_MCP_dnn_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_SCAD_MCP_DNN_auc_result.txt")
chin_SCAD_MCP_dnn_test_auc_unlist <- cbind(chin_SCAD_MCP_dnn_test_auc_unlist, rep("DNN(SCAD  MCP 변수선택)", 100))
names(chin_SCAD_MCP_dnn_test_auc_unlist) <- c("AUC", "Model")

chin_LASSO_dnn_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/chin_LASSO_DNN_auc_result.txt")
chin_LASSO_dnn_test_auc_unlist <- cbind(chin_LASSO_dnn_test_auc_unlist, rep("DNN(LASSO 변수선택)", 100))
names(chin_LASSO_dnn_test_auc_unlist) <- c("AUC", "Model")

## 변수선택 전 후 DNN (AUC) 비교 vioplot ##
# 결과 합치기
chin_3model_auc <- rbind(chin_dnn_test_auc_unlist, chin_SCAD_MCP_dnn_test_auc_unlist,
                         chin_LASSO_dnn_test_auc_unlist)

str(chin_3model_auc)

win.graph()
vioplot(chin_3model_auc$AUC ~ chin_3model_auc$Model,
        main = "Chin data. DNN 모형 AUC 비교",
        xlab = "", ylab = "AUC", ylim = c(0.5, 1),
        cex.main = 1.4)

points(c(1, 2, 3), c(mean(chin_3model_auc$AUC[chin_3model_auc$Model == "DNN(변수전체)"]), 
                     mean(chin_3model_auc$AUC[chin_3model_auc$Model == "DNN(SCAD  MCP 변수선택)"]),
                     mean(chin_3model_auc$AUC[chin_3model_auc$Model == "DNN(LASSO 변수선택)"])), 
       col = "red", pch = 17)


## 변수선택 전 후 DNN (AUC) ##
## 분산분석 및 사후검정 ##
# 분산분석
chin_auc_anova <- aov(AUC ~ Model, data = chin_3model_auc)
summary(chin_auc_anova)

chin_auc_kruskal <- kruskal.test(AUC ~ Model, data = chin_3model_auc)
chin_auc_kruskal

# 정규성 검정
shapiro.test(chin_auc_anova$residuals)

# 등분산 검정
plot(chin_auc_anova$fitted.values, chin_auc_anova$residuals,
     main = "잔차 대 적합값",
     xlab = "적합값",
     ylab = "잔차", cex.lab = 1.5)

# 사후검정
chin_auc_sceffe <- scheffe.test(chin_auc_anova, "Model")
chin_auc_bonferroni <- LSD.test(chin_auc_anova, "Model", p.adj = "bonferroni")
chin_auc_Tukey <- HSD.test(chin_auc_anova, "Model")


chin_auc_sceffe
chin_auc_bonferroni
chin_auc_Tukey

win.graph()
plot(chin_auc_sceffe, variation = "SD", main = "Scheffe 사후검정 결과")
plot(chin_auc_bonferroni, variation = "SD", main = "Bonferroni 사후검정 결과")
plot(chin_auc_Tukey, variation = "SD", main = "Tukey 사후검정 결과")

