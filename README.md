STAT8053 Project
===========

This is a project for STAT8053 at the University of Minnesota.

Installation
-------

```r
install.packages("devtools") # Install devtools from CRAN
devtools::install_github("austindavidbrown/pros/R-package")
```

Performance Improvements
-------

Add this to your `~/.R/Makevars` file to improve performance:

```bash
CXXFLAGS = -O3 -march=native -mfpmath=sse
```

Documentation
-------
See the [Reference Manual](https://github.com/austindavidbrown/pros/blob/master/pros-manual.pdf) for the R package.

License
-------
The code is licensed under GPL-3.

References
---------
- This is for a project in STAT8053 at the University of Minnesota.


