

adult_dnn_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Adult_DNN_auc_result.txt")
adult_dnn_test_auc_unlist <- cbind(adult_dnn_test_auc_unlist, rep("DNN(변수전체)", 100))
names(adult_dnn_test_auc_unlist) <- c("AUC", "Model")

adult_dnn_selec_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Adult_DNN_selec_auc_result.txt")
adult_dnn_selec_test_auc_unlist <- cbind(adult_dnn_selec_test_auc_unlist, rep("DNN(LASSO 변수선택)", 100))
names(adult_dnn_selec_test_auc_unlist) <- c("AUC", "Model")

adult_dnn_stepwise_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Adult_DNN_stepwise_auc_result.txt")
adult_dnn_stepwise_test_auc_unlist <- cbind(adult_dnn_stepwise_test_auc_unlist, rep("DNN(Stepwise 변수선택)", 100))
names(adult_dnn_stepwise_test_auc_unlist) <- c("AUC", "Model")

## 변수선택 전 후 DNN (AUC) 비교 vioplot##
# 결과 합치기
adult_3model_auc <- rbind(adult_dnn_test_auc_unlist, adult_dnn_selec_test_auc_unlist,
                          adult_dnn_stepwise_test_auc_unlist)

str(adult_3model_auc)

# 변수전체 AUC summary
summary(adult_3model_auc$AUC[adult_3model_auc$Model == "DNN(변수전체)"])

# LASSO 변수선택 AUC summary
summary(adult_3model_auc$AUC[adult_3model_auc$Model == "DNN(LASSO 변수선택)"])

# Stepwise 변수선택 AUC summary
summary(adult_3model_auc$AUC[adult_3model_auc$Model == "DNN(Stepwise 변수선택)"])


win.graph()
vioplot(adult_3model_auc$AUC ~ adult_3model_auc$Model,
        main = "Adult data. DNN 모형 AUC 비교",
        xlab = "", ylab = "AUC", ylim = c(0.5, 1),
        cex.main = 1.4)
points(c(1, 2, 3), c(mean(adult_3model_auc$AUC[adult_3model_auc$Model == "DNN(변수전체)"]), 
                     mean(adult_3model_auc$AUC[adult_3model_auc$Model == "DNN(LASSO 변수선택)"]),
                     mean(adult_3model_auc$AUC[adult_3model_auc$Model == "DNN(Stepwise 변수선택)"])), 
       col = "red", pch = 17)

## 분산분석 및 사후검정 ##
# AUC
adult_auc_anova <- aov(AUC ~ Model, data = adult_3model_auc)
summary(adult_auc_anova)

adult_auc_kruskal <- kruskal.test(AUC ~ Model, data = adult_3model_auc)
adult_auc_kruskal

