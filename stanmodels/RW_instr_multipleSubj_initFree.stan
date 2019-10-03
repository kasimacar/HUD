data {
  int<lower=1> T; 
  int<lower=1> N;
  int<lower=1,upper=2> stimulus[N,T];     
  int shock[N,T];  // electric shocks 
  real response[N,T];
  int revtrial[N, T];  // reversed trials indicator
}

parameters {
  // Hyper(group)-parameters  
  vector[3] mu_p; // means of Hyperparameters
  vector<lower=0>[3] sigma; // variances of Hyperparameters
  real<lower=0> sigma0; // noise in response
  
  // Subject-response raw parameters (for Matt trick)
  vector[N] A_pr; // learning rate
  vector[N] k_pr; // scaling factor
  vector[N] P_pr; // instruction effect
  vector<lower=0,upper=1> [N] EVinit1;
  vector<lower=0,upper=1> [N] EVinit2;
}

transformed parameters {
  // subject-lresponseel parameters
  vector<lower=0,upper=1>[N] A;
  vector<lower=0,upper=10>[N] k;
  vector<lower=0,upper=1>[N] P;

   for (i in 1:N) {
  A[i]   = Phi_approx( mu_p[1]  + sigma[1]  * A_pr[i]);
  k[i]   = Phi_approx( mu_p[2]  + sigma[2]  * k_pr[i])*10; // scale according to upper of k
  P[i]   = Phi_approx( mu_p[3]  + sigma[3]  * P_pr[i]);
   }
}


model{
  // Hyperparameters:
  mu_p  ~ normal(0, 1); 
  sigma ~ cauchy(0, 5);  
  k_pr ~ normal(0, 1);
  A_pr ~ normal(0, 1);
  P_pr ~ normal(0, 1);
  sigma0 ~ cauchy(0, 10); // depends on scale of response
  // Initial EV-values:
  EVinit1 ~ beta(3,3);
  EVinit2 ~ beta(3,3);

for (i in 1:N) {
    vector[2] EV; // expected value
    vector[2] EV_tmp; // expected value temp
    real PE;      // prediction error
    
    EV[1] = EVinit1[i];
    EV[2] = EVinit2[i];
  for (t in 1:T) {      
    if (revtrial[i,t]==1){
      EV_tmp[1] = P[i] * EV[2] + (1-P[i]) * EV[1];
      EV_tmp[2] = P[i] * EV[1] + (1-P[i]) * EV[2];
      EV[1] = EV_tmp[1];
      EV[2] = EV_tmp[2];
    }
    
    else if (shock[i,t]==0){
    // Response
    response[i,t] ~ normal( EV[stimulus[i,t]] * k[i],sigma0);
    }
    // prediction error 
    PE = shock[i,t] - EV[stimulus[i,t]];
    // value updating (learning) 
   EV[stimulus[i,t]] = EV[stimulus[i,t]] + A[i] * PE; 
   }
   }
}
