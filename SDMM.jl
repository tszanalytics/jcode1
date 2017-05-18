function SDMM(y,x,W,g,iter,burn,spec)

#You need to use the following packages with this function in Julia
#-Distributions


n = size(y,1);
n1 = size(x,1);
k = size(x,2)
n2 = size(W,1)
n3 = size(W,2)

#Setting up augmented matrix for SDMM
#You set spec to 1 if you want the SDMM and 0 if you want the SARM
if spec == 1
  x = [x W*x]
  k = size(x,2)
end
#Model Information
gk = g*k

#Storage Matrices
bsave = zeros(iter-burn,gk);
vsave = zeros(iter-burn,g);
rsave = zeros(iter-burn,g);
arate = zeros(iter,g);
gwght = zeros(iter-burn,g);
tsave = zeros(n,g,iter);

#Initial Conditions
adj = 0.0001; #Make sure probabilities are different form 0
acc = zeros(g,1)'
cc = ones(g,1)
cntr = ones(g,1)'
#uninformative group assignment
Puse0 = rand(Uniform(0,1),n,g)
probsi0 = matmul(Puse0,1./sum(Puse0,2));
for pp = 1:n
  tau[pp,:] = rand(Multinomial(1,probsi0[pp,:]),1)
end
P = sum(tau)./sum(sum(tau));
sig_temp = ones(1,g);
rho = ones(g,1)/10;
rmin = -1; #this should technically be the 1/min(eig(W)) but it is slow
rmax = 1;

#Prior Parameters
beta0 = zeros(gk,1)
alphas = 2*ones(g,1)';
inv_parms = ones(gk,1).*1e+3
inv_parms = spdiagm(inv_parms[:])
a1 = a2 = 1.01;
a = 3*ones(g,1)
b = .5*ones(g,1)

#Reshaping and calcing stuff ones, this is just making sure everything is the
#correct size given the number of component distributions and parameters etc.
str = repmat(sig_temp,k,1);
st = str[:]'
std = spdiagm(st[:]);
istd = inv_parms\std;

#Gibbs Sampler
  for j = 2:iter;
    #Expand X matrix with Group Information
    tmat = repmat(x,1,g);
    tdum = kron(tau,ones(1,k));
    xt = tmat.*tdum;
    ck = size(xt,2);
    #Expand Rho for heterogeneity
    tr2 = tau*rho;
    htrho = spdiagm(tr2[:]);
    #Draw Coefficients
    yt = y - htrho*W*y;
    iSig = tau*(1./(sqrt(sig_temp')));
    xtt = matmul(iSig,xt);
    ytt = matmul(iSig,yt);
    D_beta = (xtt'*xtt)+istd;
    d_beta = xtt'*ytt + (istd*beta0);
    H = chol(Hermitian(inv(D_beta)))
    betas = D_beta\d_beta + H'*randn(ck,1);
    xb = xt*betas;
    #Draw Variance
    rbetas = reshape(betas,k,g);
    for l = 1:g;
      points = find(tau[:,l].==1);
      x_use = xt[points,:];
      y_use = yt[points];
      resids = .5*(y_use-x_use*betas)'*(y_use-x_use*betas);
      aa = length(y_use)/2+a[l,1];
      bb = inv(resids + inv(b[l,1]));
      sig_temp[:,l] = rand(InverseGamma(aa,bb[1]),1,1);
    end
    #Draw rho
    for qqq = 1:g;
      rtemp2 = rho;
      accept = 0;
      while accept <= 0;
        ccrnd = (cc[qqq]*randn(1,1));
        rtemp2[qqq] = rtemp2[qqq] + ccrnd[1];
        if (rmin < rtemp2[qqq] && rtemp2[qqq] < rmax)
          accept = 1;
        end
        cntr[qqq] = cntr[qqq]+1;
      end
      #Developing Ratio for M-H

      #Current
      tr3 = tau*rho;
      htrtemp = spdiagm(tr3[:])
      A1= speye(n)-htrtemp*W;
      FAC1 = lufact(A1);
      L1 = FAC1[:L];
      U1 = FAC1[:U];
      s1 = det(L1);
      ldet1 = log(s1*prod(diag(U1)));
      e1 = (y - (htrtemp*W)*y-xt*betas);
      ev1 = matmul(e1,tau*sqrt(1./sig_temp)');
      epe1 = (ev1'*ev1)/2;
      #B1 = (gamma(a1)*gamma(a2))/gamma(a1+a2);
      #num1 = (1+rho).^(a1-1);
      #num1 = num1.*(1-rho).^(a2-1);
      #den = 2^(a1+a2-1);
      #bprior = (1/B1[:])*num[:]/den[:];
      #bprior = beta_prior(rho,1.01,1.01)
      rhox = ldet1 -epe1;

      #Candidate
      tr4 = tau*rtemp2;
      htrtemp2 = spdiagm(tr4[:]);
      A2 = speye(n)-htrtemp2*W;
      FAC2 = lufact(A2)
      L2 = FAC2[:L];
      U2 = FAC2[:U];
      s2 = det(L2);
      ldet2 = log(s2*prod(diag(U2)));
      e2 = (y-(htrtemp2*W)*y-xt*betas);
      ev2 = matmul(e2,tau*sqrt(1./sig_temp)');
      epe2 = (ev2'*ev2)/2;
      #B2 = (gamma(a1)*gamma(a2))/gamma(a1+a2);
      #num2  = (1+rtemp2).^(a1-1);
      #num2  = num2.*(1-rtemp2).^(a2-1);
      #den2 = 2^(a1+a2-1);
      #bprior = (1/B2[:])*num2/den2;
      #bprior = beta_prior(rtemp2,1.01,1.01)
      rhoy = ldet2-epe2;
      #Ratio
      ratio = exp(rhoy-rhox);
      #Evaluation of Ratio
      if ratio[1] > 1;
        p = 1;
      else
        p = min(1,ratio);
      end;
      ru = rand(Uniform(0,1),1,1);
      if ru[1] < p[1]
        rho = rtemp2;
        acc[:,qqq] = acc[:,qqq]+1;
      end
      arate[j,qqq] = acc[qqq]/j;
      if arate[j,qqq] < 0.4
        cc[qqq] = cc[qqq]/1.1;
      end
      if arate[j,qqq] > 0.6
        cc[qqq] = cc[qqq]*1.1;
      end

      #Component Label Vectors
      tempp = zeros(n,g);
      probs = zeros(n,g);
      for l = 1:g;
        xxbb = x*rbetas[:,l];
        ss22 = sqrt(sig_temp[:,l]).*ones(n,1);
        yfilt = (speye(n)-rho[l]*W)*y
        for lk = 1:n
          dnn = Normal(xxbb[lk],ss22[lk])
          tempp[lk,l] = pdf(dnn,yfilt[lk]);
        end
        probs[:,l] = P[1] *tempp[:,l]+adj;
      end
      probs = matmul(probs,1./sum(probs,2));
      for lk = 1:n
      tau[lk,:]   = rand(Multinomial(1,probs[lk,:]),1)
      end

      nn = zeros(g,1)
      for lk = 1:g
        nn[lk] = sum(tau[:,lk]);
      end

      nnp = nn + alphas';
      P = rand(Dirichlet(nnp[:]))
      tsave[:,:,j] = tau;
      if j > burn
        bsave[j-burn,:] = reshape(betas,gk,1)
        gwght[j-burn,:] = P';
        rsave[j-burn,:] = rho;
        vsave[j-burn,:] = sig_temp;
      end
    end
    return bsave, gwght,rsave,vsave
  end
end
