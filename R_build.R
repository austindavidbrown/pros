# Build registration
tools::package_native_routine_registration_skeleton("./R-package", "src/init.c")

# Generate reference manual
R CMD Rd2pdf ./R-package --force -o pros-manual.pdf

# Document
R CMD INSTALL ./R-package # needs to install package to document
devtools::document("./R-package")
roxygen2::roxygenise("./R-package")

# Check
R CMD check ./R-package
devtools::check("./R-package")

##
# Test installation
##
devtools::install("R-package")
library("pros")

X_train = read.csv("data/X_train.csv", header = F)
y_train = read.csv("data/y_train.csv", header = F)

X_test = read.csv("data/X_test.csv", header = F)
y_test = read.csv("data/y_test.csv", header = F)

fit = pros(X_train, y_train, lambda = .1, step_size = .001)
pred = predict(fit, X_test)
pred
cv = cv.pros(X_train, y_train, step_size = .001)
pred = predict(cv, X_test)
pred