function [sol]=Crack_SupershearEqMotion3(k,b)
%Based on Broberg 1994 and Broberg's book.
%This is last version. The difference with Crack_SupershearEqMotion2 is in
%calculation of N(j). This version is also faster.
%Use Crack_SupershearEqMotionCalcV(sol) to calculate Cf for specific parametrs
%k=Cs/Cd
%b=Cf/Cd

%k0577=Crack_SupershearEqMotion3(0.577,round(linspace(0.577*1.01,0.995,400)*1e3)/1e3);

%-----b should be a line vector
if size(b,1)>1
    b=b';
end
b_Vec=b;
clear b;

%--------Works for various cohesive zone models (not for exponential ? - need to check)
D=@(w)(1+w);%Linear Cohesive zone model
ILowerBoundery=-1;%Lower boundary of the integration. cohesive zone size
%D=@(w) 1-(-w).^1.8;% Results in quasi linear slip weakening
%D= @(w) exp(w); %exponential
%ILowerBoundery=-10;%Lower boundary of the integration. cohesive zone size

ap=@(b)(1-b.^2).^0.5;
bs=@(b)((b/k).^2-1).^0.5;
g=@(b) 1/pi*atan(4*ap(b).*bs(b)./(bs(b).^2-1).^2);%g stands for gamma. Note that eq. 34 (book eq6.9.111) is for 1/b

wd=zeros(1,length(b_Vec));
Io=zeros(1,length(b_Vec));
N=zeros(1,length(b_Vec));
A=zeros(1,length(b_Vec));

parfor j=1:length(b_Vec)
    
    b=b_Vec(j);
    
    %---------Calc wd (eq.67 book-eq.6.3.70)
    dz=1E-3;%-3
    z=dz:dz:1-dz; % the integral of I is singular at z=1 but appears to cancel with the prefactord(-z) -> finite limit;
    
    %----------Linear cohesive zone 6.3.70 has two internal integrals,named here by I1  and I2
    % for general cohesive zone this hould be an integral%
    %I1=1/g(b);
    
    %%---calculate I1 for general cohesive zone %Need to check accurecy at various Cf
    I1=zeros(1,length(z));
    for l=1:length(z)
        I1(l)=calcI1sym(D,ILowerBoundery,-z(l),g(b));
    end
    
    %-------calculate I2
    
    I2=zeros(1,length(z));
    
    for l=1:length(z)
        I2_integrand=@(w) 1./(w-z(l))./((w).^(1-g(b)));
        I2(l)=integral(I2_integrand,1,1E7);
    end
    
    wd_integrand=D(-z).*z.^(1-g(b)).*(I1+D(-z).*I2);
    wd(j)=dz*trapz(wd_integrand);
    
    % ------- Calc the integral in eq.70
    %A(j)=1./g(b)./(g(b)+1); %%is solved easily for linear cohesive zone
    % for general cohesize zone
    A_integrand=@(x) D(-x)./( x.^(1-g(b)) ) ;
    A(j)=my_integralSingular(A_integrand,1-g(b),0,1);
    
    %-------------------Calc N eq.47 (A.9)
    
    %--------Calc Io and M
    I=@(w) 2*(w.*g(1./w)-1/b*g(b))./(w.^2-1/b^2);
    Io(j)=integral(I,1,1/b,'RelTol',1e-6,'AbsTol',1e-4);%eq.41 (book eq.6.9.118)
    M=sin(pi*g(b)).*exp(-Io(j)).*b/2.*((1./b-1)./(1./b+1)).^g(b); %eq.46
    
    %-------    
    %There is a power law singularity at s=1/b. the integral can be
    %calculated by using my_integralSingular if the power is calculated
    %first. Instead, I found that using cutoff (1/b-1e-7) works well.
    N(j)=integral(@(s)N_integrand(s,b,g,M),1,1/b-1e-7)+M/g(b)*(2/b/(1/b-1))^g(b);%eq.A9
    
end

d=1;
a=1;
mu=1;
tau_0=1;%(4.595-3.704)/(5.305-3.704);%shear pre stress
tau_d=1;

b=b_Vec;
f1=b.^2.*sin(pi*g(b))./(4*k^2*(1-b.^2).^0.5); %eq.54 very similar to YII
G1=f1.*sin(pi*g(b))*2/pi.*wd;
G2=pi*A.^-1.* b.*exp(-Io)./(2.^(1-g(b)).*g(b).*N).*((1-b)./(1+b)).^g(b);  %A,Io and N are calculated in the for loop
G0=pi*tau_0^2*a/mu/4/(1-k^2);%eq.74
G=(d/a).^(1-2*g(b))/G0.*(G1.*(G2).^2); %eq.68

G_my=a*tau_d^2/mu*(tau_0/tau_d).^(1./g(b)).*G1.*(G2).^(1./g(b));

% Gamma=3;
% tau_d=1.3e6;
% mu=2.1e9;
% a0=Gamma/tau_d^2*mu;
% tau_0=0.26e-3*mu*2;
% a=a0./(sol.G_my.*(tau_0/tau_d).^(1./sol.g) ) ;

sol.k=k;
sol.tau_0=tau_0;%in units of \tau_d
sol.b=b;
sol.g=g(b);
sol.G0=G0;
sol.G1=G1;
sol.G2=G2;
sol.G=G;
sol.G_my=G_my;



function I1=calcI1sym(tau,ILowerBoundery,z,g)
%Integral is done in non dimensional coordinats

I=@(w) -(tau(z)- tau(w))./(w-z)./((-w).^(1-g));

%------Decomposition to sym and Asym
v=@(w) 1/2*( I(w)+ I(2*z-w));
%h=@(w) 1/2*( I(w)- I(2*z-w));
l=min(z-ILowerBoundery,0-z);
%I1=integral(v,z-l/2,z+l/2)+integral(I,-1,z-l/2)+integral(I,z+l/2,0,'RelTol',1e-3,'AbsTol',1e-5);
%I1=integral(I,-1,z-l/2)+2*integral(v,z-l/2,z)+my_integralSingular(I,1-g,z+l/2,0);
I1=integral(I,ILowerBoundery,z-l/2)+2*integral(v,z-l/2,z)+my_integralSingular(I,1-g,z+l/2,0);

%I1=( I1-1i*pi*tau(z)/(-z)^(1-g) );%Second part in (...) needed for Principal value.


function N_integrand=N_integrand(s_Vec,b,g,M)
M_s=zeros(1,length(s_Vec));
for j=1:length(s_Vec)
    
    s=s_Vec(j);
    %-Io is a principle value integral.To solve it I decomposed to
    %symetric and asymetric parts. Asymetric part cancels out.Symetric
    %part is no longer singular.
    I=@(w) 2*(w.*g(1./w)-1/b*g(b))./(w.^2-s^2);
    v=@(w) 1/2*( I(w)+ I(2*s-w));
    %h=@(w) 1/2*( I(w)- I(2*s-w));
    
    if (s<=0.5*(1+1/b))
        Io_s=integral(v,1,2*s-1,'RelTol',1e-6,'AbsTol',1e-4);%eq.6.9.118
        Io_s=Io_s+integral(I,2*s-1,1/b);
    else
        Io_s=integral(v,2*s-1/b,1/b,'RelTol',1e-6,'AbsTol',1e-4);%eq.6.9.118
        Io_s=Io_s+integral(I,1,2*s-1/b);
    end
    
    M_s(j)=sin(pi*g(1/s))*exp(-Io_s)/(1/b+s)*((s-1)/(s+1))^(g(b)/b/s); %eq.46
end
s=s_Vec;
N_integrand=( M*(2/b./(1/b-s)).^g(b)-M_s.*((1/b+s)./(1/b-s)).^(g(b)/b./s) ) ./(1/b-s);


