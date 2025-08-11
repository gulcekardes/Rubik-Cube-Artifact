G := Group<a,a2,b,b2,c,c2,d,d2,e,e2,f,f2 | a^4,b^4,c^4,d^4,e^4,f^4, a^2=a2,b^2=b2,c^2=c2,d^2=d2, e^2=e2,f^2=f2, a*d=d*a, b*e=e*b, c*f=f*c>;
A := AutomaticGroup(G);
f := GrowthFunction(A);
PZ<x> := FunctionField(IntegerRing());
PZ!f;
  (-9*x^2 - 6*x - 1)/(18*x^2 + 12*x - 1)
PR := PolynomialRing( RealField(6) );
[ r[1]^-1 : r in Roots(PR!Denominator(f)) ];
  [ -1.34847, 13.3485 ]
LR<x> := LaurentSeriesRing(Z);
LR!f;
