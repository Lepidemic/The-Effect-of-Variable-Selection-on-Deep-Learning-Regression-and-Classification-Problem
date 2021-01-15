library(agricolae)
library(lawstat)

leukemia_dnn_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_DNN_auc_result.txt")
leukemia_dnn_test_auc_unlist <- cbind(leukemia_dnn_test_auc_unlist, rep("DNN(변수전체)", 100))
names(leukemia_dnn_test_auc_unlist) <- c("AUC", "Model")

leukemia_SCAD_dnn_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_SCAD_DNN_auc_result.txt")
leukemia_SCAD_dnn_test_auc_unlist <- cbind(leukemia_SCAD_dnn_test_auc_unlist, rep("DNN(SCAD 변수선택)", 100))
names(leukemia_SCAD_dnn_test_auc_unlist) <- c("AUC", "Model")

leukemia_MCP_dnn_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_MCP_DNN_auc_result.txt")
leukemia_MCP_dnn_test_auc_unlist <- cbind(leukemia_MCP_dnn_test_auc_unlist, rep("DNN(MCP 변수선택)", 100))
names(leukemia_MCP_dnn_test_auc_unlist) <- c("AUC", "Model")

leukemia_LASSO_dnn_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/leukemia_LASSO_DNN_auc_result.txt")
leukemia_LASSO_dnn_test_auc_unlist <- cbind(leukemia_LASSO_dnn_test_auc_unlist, rep("DNN(LASSO 변수선택)", 100))
names(leukemia_LASSO_dnn_test_auc_unlist) <- c("AUC", "Model")

## 변수선택 전 후 DNN (AUC) 비교 vioplot ##
# 결과 합치기
leukemia_4model_auc <- rbind(leukemia_dnn_test_auc_unlist, leukemia_SCAD_dnn_test_auc_unlist,
                             leukemia_MCP_dnn_test_auc_unlist, leukemia_LASSO_dnn_test_auc_unlist)

str(leukemia_4model_auc)

win.graph()
vioplot(leukemia_4model_auc$AUC ~ leukemia_4model_auc$Model,
        main = "Leukemia data. DNN 모형 AUC 비교",
        xlab = "", ylab = "AUC", ylim = c(0.5, 1),
        cex.main = 1.4)

points(c(1, 2, 3, 4), c(mean(leukemia_4model_auc$AUC[leukemia_4model_auc$Model == "DNN(변수전체)"]), 
                        mean(leukemia_4model_auc$AUC[leukemia_4model_auc$Model == "DNN(SCAD 변수선택)"]),
                        mean(leukemia_4model_auc$AUC[leukemia_4model_auc$Model == "DNN(MCP 변수선택)"]),
                        mean(leukemia_4model_auc$AUC[leukemia_4model_auc$Model == "DNN(LASSO 변수선택)"])), 
       col = "red", pch = 17)


## 변수선택 전 후 DNN (AUC) ##
## 분산분석 및 사후검정 ##
# 분산분석
leukemia_auc_anova <- aov(AUC ~ Model, data = leukemia_4model_auc)
summary(leukemia_auc_anova)

leukemia_auc_kruskal <- kruskal.test(AUC ~ Model, data = leukemia_4model_auc)
leukemia_auc_kruskal

