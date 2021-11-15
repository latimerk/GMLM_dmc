function x = boundedExp(x)

MINEXP = -90;
MAXEXP =  90;
x = exp(max(MINEXP,min(MAXEXP,x)));