using Distributions

function GCDGP(K,T,LY,LX,α,ϕ,β,γ,σ)
#This function creates AR1 lagged simulated data sets. The data comes in
#the form Y = α + ϕ*Yt-1 + X1*β + .... + Xk*βk + σ*ɛ_t.
#         X = X_t-1*γ + ν_t
#         ν = ɛ ~ N(0,1)
#         t = (1,…,T)
#This is useful for working on
#the Predictive Granger Causality stuff. See Ashley 2014.

#Inputs

#K: The number of covariates in the data set.
#T: The number of observations in the data set.
#L: Number of lags for Y
#α: The intercept for the equation above.
#ϕ: The AR1 Co-efficient.
#β: A 1xK vector of betas
#γ: A scalar that governs how much of X is carried through T.
#σ: A scalar for the variance which is drawn from N(0,1).

#Output
#data: This is a matrix which is T x (1+LY+K+LX)
TT = T*2
Lag = LY
Lag1 = LX
#Design Matrix - Additional covariates
X  = zeros(TT,K)
X[1,:] = randn(K,1)
for i = 2:TT
    X[i,:] = X[i-1,:].*γ + randn(K,1);
end
#Dependent Variable
Y1 = zeros(TT,1)
Y1[1:Lag] = α*ones(Lag,1) + X[1:Lag,:]*β + σ*randn(Lag,1)
for i = Lag+1:TT
  B = i-1
  LL = Y1[i-Lag:B]'*ϕ
  Y1[i,:] = α + LL + X[i,:]'*β + σ*randn(1,1)
end
#Setting up the data frame as an array
data = Y1[end-T+1:end]
for i = 1:Lag
  data = [data Y1[(TT-i)-T+1:(TT-i)]]
end
data = [data X[end-T+1:end,:]]

#Output of Function
return data

end
