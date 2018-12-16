# Add registration
tools::package_native_routine_registration_skeleton("./", "src/init.c")

# generate reference manual
R CMD Rd2pdf ./ --force -o pros-manual.pdf

# document
devtools::document()

# check
R CMD check .
devtools::check()

##
# Test
##
devtools::install()
library("pros")

X_train = read.csv("../data/X_train.csv", header = F)
y_train = read.csv("../data/y_train.csv", header = F)

X_test = read.csv("../data/X_test.csv", header = F)
y_test = read.csv("../data/y_test.csv", header = F)

fit = pros(X_train, y_train, lambda = .1, step_size = .01)
pred = predict(fit, X_test)

cv = cv.pros(X_train, y_train, step_size = .01)
pred = predict(cv, X_test)
