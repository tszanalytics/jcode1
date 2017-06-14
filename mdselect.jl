using StatsFuns

function mdselect(Y,X,bdraw,sdraw)

#This function Calculates a number of standard model selection criteria
#from MCMC results including: AIC, BIC, DIC, WAIC, Ar2
#These codes are used in Bayesian Predictive Model Selection Essay

#Inputs

#Y: An N x 1 vector of outcomes
#X: An N x K design matrix
#bdraw: An M x K set of MCMC draws for the parameters where M is the
#       post burn-in set of iterations.
#sdraw: An M x 1 set of MCMC draws for the noise parameter.

N = size(X,1)
K = size(X,2)
I = size(bdraw,1)

#Calculating log likelihood at mean of posterior estimates
meanBs = mean(bdraw,1)
̂Y = X*meanBs'
e = Y - ̂Y
epe = e'*e
s2 = epe/(N-K)

#likelihoo calculation.
#eq 7 from Gelman 2013 log( p(y|̂Θ) )
logLik = -(N/2)*log(2*pi) + N*log(sqrt(s2))-1/(2*s2))*eltype

#Deviance Information Criterion
dbar = zeros(I,1)
for s = 1:I;
  e2 = Y-X*bdraw[s,:]'
  epe2 = e2'*e2
  dbar[s] = -(N/2)*log(2*pi)+N*log(sqrt(sdraw[s,:]))-(1/(2*sdraw[s,:]))*epe2
end
Dbar = mean(dbar)
dhat = logLik
pD = dbar - Dhat
dic = 2*Dhat - 2*pD
#Bayesian Information Criterion (BIC)
bic = log(epe/N) + log(N)*(K+1)/N
#Akaike Information Criterion (AIC)
aic = log(epe/N) + 2*(K+1)/N
#Adjusted R-Squared
tss = (Y-mean(Y))'*(Y-mean(Y))
Ar2 = 1 - (epe/(N-K-1))/(tss/(N-1))
#Widely Applicable Information Criterion (WAIC)
logp = zeros(N,I)
trup = zeros(N,I)
lpd  = zeros(N,I)
for ik = 1:N
  for ki = 1:I
    logp[ik,ki] = log(normpdf(X[ik,:]*bdraw[ki,:]',sqrt(sdraw[ki]),Y[ik]))
    trup[ik,ki] = normpdf(X[ik,:]*bdraw[ki,:]',sqrt(sdraw[ki]),Y[ik])
  end
end
lpd = trup
mlogp = mean(logp,2)
mtrup = log(mean(trup,2))
pWAIC = 2*sum(mtrup-mlogp)
lpdht = sum(log(mean(lpd,2)))
waic  = -lpdht + pWAIC

return aic,bic,dic,WAIC,Ar2

end
