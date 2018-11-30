from sympy import symbols
from sympy import plot

x = symbols('x')
y = 0.05*t + 0.2/((t - 5)**2 + 2)

plot(y, (t, -10, 10))