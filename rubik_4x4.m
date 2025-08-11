G := Group<a,a2,b,b2,c,c2,d,d2,e,e2,f,f2,g,g2,h,h2,i,i2,j,j2,k,k2,l,l2 | a^4, b^4, c^4, d^4, e^4, f^4, g^4, h^4, i^4, j^4, k^4, l^4,

    a^2 = a2, b^2 = b2, c^2 = c2, d^2 = d2, e^2 = e2, 
    f^2 = f2, g^2 = g2, h^2 = h2, i^2 = i2, j^2 = j2, 
    k^2 = k2, l^2 = l2,

    a*d = d*a,
    b*e = e*b,
    c*f = f*c,

    g*j = j*g,
    h*k = k*h,
    i*l = l*i>;
A := AutomaticGroup(G);
f := GrowthFunction(A);
PZ<x> := FunctionField(IntegerRing());
PZ!f; 
PR := PolynomialRing(RealField(6));
[ r[1]^-1 : r in Roots(PR!Denominator(f)) ];
LR<x> := LaurentSeriesRing(IntegerRing(), 30);
LR!f;
