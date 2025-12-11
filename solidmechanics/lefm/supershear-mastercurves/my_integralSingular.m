function I=my_integralSingular(f,g,a,b)
%---Solves improper integral that has a snigularity of 0<g<1 at on of the
%borders. From Numerical Recipes - improper integrals 4.4
%f is function hamdle, for example f=@(x) 1./x.^g
%[a,b] the lower and upper limit


%------Due to preccision limitations matlab behaves bad for g>0.7 if the singularity is
%not at t=0 change variable before using the function to have the
%singularuty at 0

if (abs(f(a))==inf)
    integrand=@(t) 1/(1-g)*t.^(g/(1-g)).*f(a+t.^(1./(1-g)));
else if(abs(f(b))==inf)
        integrand=@(t) 1/(1-g)*t.^(g/(1-g)).*f(b-t.^(1./(1-g)));
    end
end
I=integral(integrand,0,(b-a)^(1-g));




