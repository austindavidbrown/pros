library(pros)
library(glmnet)

d = read.table("http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/prostate.data")
head(d)
train = d[d$train == TRUE, ]
test = d[d$train == FALSE, ]

X_train = as.matrix(train[, 1:8])
y_train = as.vector(train[, 9])

X_test = as.matrix(test[, 1:8])
y_test = as.vector(test[, 9])

intercept = mean(y_train)

# Center
C = diag(1, nrow(X_train)) - 1/nrow(X_train) * rep(1, nrow(X_train)) %*% t(rep(1, nrow(X_train)))
cX_train = C %*% X_train
cy_train = y_train - rep(mean(y_train), length(y_train))


fit_glmnet = glmnet(cX_train, cy_train, alpha = 1, lambda = .1, intercept = F, standardize = F)
B_glmnet = coef(fit_glmnet)[2:9]
B_glmnet
pred_glmnet = rep(intercept, length(y_test)) + X_test %*% B_glmnet
mean((y_test - pred_glmnet)^2)


fit = pros(cX_train, cy_train, lambda = .1, max_iter = 100000, tolerance = 10^(-3))
B_pros = fit$B
B_pros
pred = rep(intercept, length(y_test)) + X_test %*% B_pros
mean((y_test - pred)^2)


#write.table(X_train, file="prostate_X_train.csv", row.names=FALSE, col.names=FALSE, sep = ",")
#write.table(y_train, file="prostate_y_train.csv", row.names=FALSE, col.names=FALSE, sep = ",")

#write.table(X_test, file="prostate_X_test.csv", row.names=FALSE, col.names=FALSE, sep = ",")
#write.table(y_test, file="prostate_y_test.csv", row.names=FALSE, col.names=FALSE, sep = ",")