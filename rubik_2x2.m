G_URF := Group< u, u_sq, u_inv,
                 r, r_sq, r_inv,
                 f, f_sq, f_inv |

                 u^4 = 1,
                 u^2 = u_sq,      // u_sq is the 180-degree turn
                 u * u_inv = 1,   // u_inv is the -90 degree turn (u^3)
                 // Redundant but consistent:
                 u_sq * u_sq = 1, // (u^2)^2 = 1
                 u_inv^4 = 1,     // (u^3)^4 = u^12 = 1

                 r^4 = 1,
                 r^2 = r_sq,
                 r * r_inv = 1,
                 r_sq * r_sq = 1,
                 r_inv^4 = 1,

                 f^4 = 1,
                 f^2 = f_sq,
                 f * f_inv = 1,
                 f_sq * f_sq = 1,
                 f_inv^4 = 1    >;
A_URF := AutomaticGroup(G_URF);
f_URF := GrowthFunction(A_URF);
PZ<x> := FunctionField(IntegerRing());
PZ!f_URF;
PR := PolynomialRing(RealField(6));
[ r[1]^-1 : r in Roots(PR!Denominator(f_URF)) ];
LR<x> := LaurentSeriesRing(IntegerRing());
LR!f_URF;
