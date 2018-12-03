# tools::package_native_routine_registration_skeleton("./", "src/init.c")

devtools::document()
devtools::check(manual = T)


# generate reference manual
R CMD Rd2pdf ./ --force -o pros.pdf


##
# Test
##
devtools::install()
library("pros")

X_train = read.csv("../X_train.csv", header = F)
y_train = read.csv("../y_train.csv", header = F)

X_test = read.csv("../X_test.csv", header = F)
y_test = read.csv("../y_test.csv", header = F)

fit = pros(X_train, y_train, lambda = .1)
pred = predict(fit, X_test)

cv = cv.pros(X_train, y_train)
pred = predict(cv, X_test)
