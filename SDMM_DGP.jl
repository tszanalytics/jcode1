#SDMM DGP
using Distributions
include("knn_weight_matrix.jl")
#include("SDMM")

n = 1500;
k = 2;
beta = [-0.5 -0.75 0.5 0.8 -1.0 1.2]'
rho = [0.6 0.2 -0.3]';
gamm = [-0.25 -0.5 0.25 0.5 -0.25 0.25]';
vr = [1 0.75 0.5]';
sigmod = 1.0;

g = 3;
probsi = [0.45 0.35 0.20]
indicp = [0.5 0.5]
tau = zeros(n,g)
indic = zeros(n)
for i = 1:n
  tau[i,:] = rand(Multinomial(1,probsi[:]),1)
  indic[i] = rand(Multinomial(1,indicp[:]),1)[1]
end
#Location Information
latt = randn(n,1) + indic.*(.9*tau*cumsum(ones(g,1)))
long = randn(n,1) + indic.*(.9*tau*cumsum(ones(g,1)))
W = knn_weight_matrix(latt,long,6)
x0 = zeros(n,k);
for ii = 1:n
  x0[ii,:] = 2*randn(1,k)+2;
end
xmat = [x0 W*x0]
#expanding x-matrix by groups
tmat = repmat(xmat,1,g);
tdum = kron(tau,ones(1,k*2));
xt = tmat.*tdum;
aa = [beta'; gamm']
aaa = reshape(aa,6,2)
b_g = vec(aaa')
xtbg = xt*b_g;
#setting up I-psiW
rt = tau*rho;
rt = spdiagm(rt[:])
F = speye(n)-rt*W;
#Error Term
err = ((tau*sqrt(vr)).*(sigmod*randn(n,1)));
#Dependent Variable
y = F\(xtbg+err)
