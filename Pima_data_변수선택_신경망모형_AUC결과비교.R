
pima_dnn_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Pima_DNN_auc_result.txt")
pima_dnn_test_auc_unlist <- cbind(pima_dnn_test_auc_unlist, rep("DNN(변수전체)", 100))
names(pima_dnn_test_auc_unlist) <- c("AUC", "Model")

pima_dnn_selec_test_auc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Pima_DNN_selec_auc_result.txt")
pima_dnn_selec_test_auc_unlist <- cbind(pima_dnn_selec_test_auc_unlist, rep("DNN(LASSO & Stepwise변수선택)", 100))
names(pima_dnn_selec_test_auc_unlist) <- c("AUC", "Model")


## 변수선택 전 후 DNN 비교 vioplot##
# AUC(Area Under a Curve)
# 결과 합치기
pima_3model_auc <- rbind(pima_dnn_test_auc_unlist, pima_dnn_selec_test_auc_unlist)

str(pima_3model_auc)


win.graph()
vioplot(pima_3model_auc$AUC ~ pima_3model_auc$Model,
        main = "Pima data. DNN 모형 AUC 비교",
        xlab = "Iter = 100", ylab = "AUC", ylim = c(0, 1),
        cex.main = 1.4)
points(c(1, 2), c(mean(pima_3model_auc$AUC[pima_3model_auc$Model == "DNN(변수전체)"]), 
                  mean(pima_3model_auc$AUC[pima_3model_auc$Model == "DNN(LASSO & Stepwise변수선택)"])), 
       col = "red", pch = 17)
# t-test
t.test(pima_dnn_test_auc_unlist[1], pima_dnn_selec_test_auc_unlist[1])

