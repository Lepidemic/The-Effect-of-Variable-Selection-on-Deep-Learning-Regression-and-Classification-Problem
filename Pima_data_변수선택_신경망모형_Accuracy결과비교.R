
pima_dnn_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Pima_DNN_result.txt")
pima_dnn_test_acc_unlist <- cbind(pima_dnn_test_acc_unlist, rep("DNN(변수전체)", 100))
names(pima_dnn_test_acc_unlist) <- c("Accuracy", "Model")

pima_dnn_selec_test_acc_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Pima_DNN_selec_result.txt")
pima_dnn_selec_test_acc_unlist <- cbind(pima_dnn_selec_test_acc_unlist, rep("DNN(LASSO & Stepwise변수선택)", 100))
names(pima_dnn_selec_test_acc_unlist) <- c("Accuracy", "Model")

pima_dnn_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/pima_DNN_time.txt")
pima_dnn_exectime_unlist <- cbind(pima_dnn_exectime_unlist, rep("DNN(변수전체)", 100))
names(pima_dnn_exectime_unlist) <- c("Time", "Model")

pima_dnn_selec_exectime_unlist <- read.table("C:/Users/start/Google 드라이브/석사_졸업논문/자과대_발표/result/Pima_DNN_selec_time.txt")
pima_dnn_selec_exectime_unlist <- cbind(pima_dnn_selec_exectime_unlist, rep("DNN(LASSO & Stepwise변수선택)", 100))
names(pima_dnn_selec_exectime_unlist) <- c("Time", "Model")

## 변수선택 전 후 DNN 비교 vioplot##
# Accuracy
# 결과 합치기
pima_3model_acc <- rbind(pima_dnn_test_acc_unlist, pima_dnn_selec_test_acc_unlist)

str(pima_3model_acc)


win.graph()
vioplot(pima_3model_acc$Accuracy ~ pima_3model_acc$Model,
        main = "Pima data. DNN 모형 Accuracy 비교",
        xlab = "Iter = 100", ylab = "Accuracy", ylim = c(0, 1),
        cex.main = 1.4)
points(c(1, 2), c(mean(pima_3model_acc$Accuracy[pima_3model_acc$Model == "DNN(변수전체)"]), 
                  mean(pima_3model_acc$Accuracy[pima_3model_acc$Model == "DNN(LASSO & Stepwise변수선택)"])), 
       col = "red", pch = 17)
# t-test
t.test(pima_dnn_test_acc_unlist[1], pima_dnn_selec_test_acc_unlist[1])

## 변수선택 전 후 DNN 비교 vioplot##
# Time(학습시간)
# 결과 합치기
pima_3model_exectime <- rbind(pima_dnn_exectime_unlist, pima_dnn_selec_exectime_unlist)

str(pima_3model_exectime)

# 전체변수 학습시간 summary

summary(pima_dnn_exectime_unlist)
summary(pima_3model_exectime$Time[pima_3model_exectime$Model == "DNN(변수전체)"])

win.graph()
vioplot(pima_dnn_exectime_unlist[1],  main = "Pima data. DNN 은닉층 1, 은닉노드 5. (전체변수)",
        xlab = "Iter = 100", ylab = "Time(단위 : 초)", ylim = c(0, 12))
points(mean(pima_dnn_exectime_unlist[ ,1]), col = "red", pch = 17) # mean 표시

# LASSO & Stepwise 학습시간 summary
summary(pima_dnn_selec_exectime_unlist)
summary(pima_3model_exectime$Time[pima_3model_exectime$Model == "DNN(LASSO & Stepwise변수선택)"])

win.graph()
vioplot(pima_dnn_selec_exectime_unlist[1],  main = "Pima data. DNN 은닉층 1, 은닉노드 5. (LASSO & Stepwise변수선택)",
        xlab = "Iter = 100", ylab = "Time(단위 : 초)", ylim = c(0, 12))
points(mean(pima_dnn_selec_exectime_unlist[ ,1]), col = "red", pch = 17) # mean 표시



win.graph()
vioplot(pima_3model_exectime$Time ~ pima_3model_exectime$Model,
        main = "Pima data. DNN 모형 학습시간 비교",
        xlab = "Iter = 100", ylab = "Time(단위 : 초)", ylim = c(0, 12),
        cex.main = 1.4)
points(c(1, 2), c(mean(pima_3model_exectime$Time[pima_3model_exectime$Model == "DNN(변수전체)"]), 
                  mean(pima_3model_exectime$Time[pima_3model_exectime$Model == "DNN(LASSO & Stepwise변수선택)"])), 
       col = "red", pch = 17)

# t-test
t.test(pima_dnn_exectime_unlist[1], pima_dnn_selec_exectime_unlist[1])

## Accuracy 대 Time 산점도 ##
## Accuracy 대 Time 산점도 ##
## Accuracy 대 Time 산점도 ##
win.graph()
plot(pima_3model_acc$Accuracy, pima_3model_exectime$Time,
     col = pima_3model_acc$Model, ylim = c(0, 20), xlim = c(0.5, 1),
     main = "Pima data Accuracy 대 Time 분포",
     xlab = "Accuracy", ylab = "Time (학습시간)", pch = 1, cex = 1.5)
legend("topright", c("DNN(변수전체)", "DNN(LASSO & Stepwise변수선택)"), 
       col = c(1, 2), cex = 1.3, pch = 1
)

