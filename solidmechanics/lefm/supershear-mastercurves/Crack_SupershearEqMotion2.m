function [sol]=Crack_SupershearEqMotion2(b)
%Based on Broberg 1994.
%This is the most recent version
%Use Crack_SupershearEqMotionCalcV(sol) to calculate Cf for specific
%parametrs


%--------Works for various cohesive zone models (not for exponential)
%D=@(w)(1+w);%Linear Cohesive zone model
%ILowerBoundery=-1;%Lower boundary of the integration. cohesive zone size
%D=@(w) 1-(-w).^1.8;% Results in quasi linear slip weakening
D= @(w) exp(w); %exponential 
ILowerBoundery=-10;%Lower boundary of the integration. cohesive zone size
%--------k=(Cs/Cd);
%k=0.454;
%k= 0.5789;
%k=0.54;
k=0.55;
%k=0.5;

ap=@(b)(1-b.^2).^0.5;
bs=@(b)((b/k).^2-1).^0.5;
g=@(b) 1/pi*atan(4*ap(b).*bs(b)./(bs(b).^2-1).^2);%g stands for gamma. Note that eq. 34 (book eq6.9.111) is for 1/b

b_Vec=b;
%b_Vec=[linspace(k*1.01, k*1.41, 6) linspace(k*1.42, 0.9, 5) 0.92 0.94 0.95 0.96 0.97 0.975 0.98 0.9825 0.985 0.99 ];
%b_Vec=[linspace(k*1.01, k*1.1, 10) linspace(k*1.12,0.98,20) 0.9825 0.985 0.99] ;
%b_Vec=[linspace(k*1.12,0.98,10) 0.9825 0.985 0.99] ;
%b_Vec=[k*1.001 k*1.005 linspace(k*1.01, 0.57, 10) 0.6 0.61 0.63 0.64];
%b_Vec=[ 0.5795    0.5818    0.5847    0.5898    0.5950    0.6001    0.6053    0.6104    0.6156    0.6207    0.6259    0.6310    0.6773    0.7236   ...
 %   0.7699    0.8162    0.8220    0.8415    0.8610    0.8805    0.9000    0.9200    0.9400    0.9500    0.9600    0.9700    0.9750    0.9800    0.9825    0.9850    0.9900];

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
    Io(j)=integral(I,1,1/b,'RelTol',1e-6,'AbsTol',1e-3);%eq.41 (book eq.6.9.118)
    M=sin(pi*g(b)).*exp(-Io(j)).*b/2.*((1./b-1)./(1./b+1)).^g(b); %eq.46
    %
    ds=1e-5;
    s_Vec=1+ds:ds:1/b;
    M_s=zeros(1,length(s_Vec));
    
    for l=1:length(s_Vec)
        s=s_Vec(l);
        %-Io is a principle value integral.To solve it I decomposed to
        %symetric and asymetric parts. Asymetric part cancels out.Symetric
        %part is no longer singular.
        I=@(w) 2*(w.*g(1./w)-1/b*g(b))./(w.^2-s^2);
        v=@(w) 1/2*( I(w)+ I(2*s-w));
        %h=@(w) 1/2*( I(w)- I(2*s-w));
        
        if (s<=0.5*(s_Vec(1)+1/b))
            Io_s=integral(v,1,2*s-1);%eq.6.9.118
            Io_s=Io_s+integral(I,2*s-1,1/b);
        else
            Io_s=integral(v,2*s-1/b,1/b);%eq.6.9.118
            Io_s=Io_s+integral(I,1,2*s-1/b);
        end
        
        M_s(l)=sin(pi*g(1/s))*exp(-Io_s)/(1/b+s)*((s-1)/(s+1))^(g(b)/b/s); %eq.46
    end
    
    s=s_Vec;
    N_integrand=( M*(2/b./(1/b-s)).^g(b)-M_s.*((1/b+s)./(1/b-s)).^(g(b)/b./s) ) ./(1/b-s);
    N(j)=ds*trapz(N_integrand)+M/g(b)*(2/b/(1/b-1))^g(b);%eq.A9
    
     M_s=[];
    
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
