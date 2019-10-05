library(lavaan)
library(semPlot)


load("HUD_anonymized_final.rda")

#subset young adults
SCREEN_df <- SCREEN_df[(SCREEN_df$age<36),]
SCREEN_df <- SCREEN_df[(SCREEN_df$age>17),]


#identify and option to remove outliers
outlierKD1 <- function(dt, var) {
  var_name <- eval(substitute(var),eval(dt))
  na1 <- sum(is.na(var_name))
  m1 <- mean(var_name, na.rm = T)
  par(mfrow=c(2, 2), oma=c(0,0,3,0))
  boxplot(var_name, main="With outliers")
  hist(var_name, main="With outliers", xlab=NA, ylab=NA)
  outlier <- boxplot.stats(var_name)$out
  mo <- mean(outlier)
  var_name <- ifelse(var_name %in% outlier, NA, var_name)
  boxplot(var_name, main="Without outliers")
  hist(var_name, main="Without outliers", xlab=NA, ylab=NA)
  title("Outlier Check", outer=TRUE)
  na2 <- sum(is.na(var_name))
  cat("Outliers identified:", na2 - na1, "n")
  cat("Propotion (%) of outliers:", round((na2 - na1) / sum(!is.na(var_name))*100, 1), "n")
  cat("Mean of the outliers:", round(mo, 2), "n")
  m2 <- mean(var_name, na.rm = T)
  cat("Mean without removing outliers:", round(m1, 2), "n")
  cat("Mean if we remove outliers:", round(m2, 2), "n")
  response <- readline(prompt="Do you want to remove outliers and to replace with NA? [yes/no]: ")
  if(response == "y" | response == "yes"){
    dt[as.character(substitute(var))] <- invisible(var_name)
    assign(as.character(as.list(match.call())$dt), dt, envir = .GlobalEnv)
    cat("Outliers successfully removed", "n")
    return(invisible(dt))
  } else{
    cat("Nothing changed", "n")
    return(invisible(var_name))
  }
}
outlierKD1(SCREEN_df, DP) #data, variable



# Whole dataset with log-transforms:

screen_df_RAW <- data.frame(X = as.numeric(SCREEN_df$drug_psychedelics),
                            Y = SCREEN_df$DP,
                            M1 = as.numeric(log1p(SCREEN_df$SEPI_tot)),
                            M2 = as.numeric(log1p(SCREEN_df$SEPI_tot_drug)))

screen_df_ALL <- data.frame(X = as.numeric(SCREEN_df$drug_psychedelics),
                            Y = SCREEN_df$DP,
                            M1 = as.numeric(log1p(SCREEN_df$SEPI_tot)),
                            M2 = as.numeric(log1p(SCREEN_df$SEPI_tot_drug)))


# A subset of those who had both drug-related and endogenously occuring ich-storung
# (then the distributions are close to normality and the assumptions of MLE are met):
screen_df_NORM <- subset(screen_df_RAW, screen_df_RAW$M1>0 & screen_df_RAW$M2>0)

model <- ' # direct effect (X = Psychedelic Use IV, Y = Cognitive Aberrations DV)
             Y ~ c * X
           # mediator 1, M1 = ego-pathology (endogenous)
             M1 ~ a1*X
             Y ~ b1*M1
           # indirect effect of endogenous path
             a1b1 := a1*b1
           # mediator 2, M2 = ego-pathology (drug-induced)
             M2 ~ a2*X
             Y ~ b2*M2
           # indirect effect of drug-induced path
             a2b2 := a2*b2
           # total effect
             total := c + (a1*b1) + (a2*b2)
         '

# Estimate the models:
fitALL_wls <- sem(model, data = screen_df_ALL, se = "bootstrap", estimator='WLS')
fitNORM_ml <- sem(model, data = screen_df_NORM, se = "bootstrap", estimator='ML')



# Summary of the fit
summary(fitALL_wls, fit.measures=T, rsq=T, standardized = TRUE)
summary(fitNORM_ml, fit.measures=T, rsq=T, standardized = TRUE)



#Finally, generate bootstrap confidence intervals:
boot.fit.ALL <- parameterEstimates(fitALL_wls)
boot.fit.NORM <- parameterEstimates(fitNORM_ml)


semPaths(fitALL_wls, "model", "std", residuals = T, exoCov = T)
semPaths(fitNORM_ml, "model", "std", residuals = T, exoCov = T)



#### CONTINOUS ####

#subset of participants aged<36
CONSP_df <- CONSP_df[(CONSP_df$age<36),]
CONSP_df <- CONSP_df[(CONSP_df$age>17),]


#identify and option to remove outliers
outlierKD1 <- function(dt, var) {
  var_name <- eval(substitute(var),eval(dt))
  na1 <- sum(is.na(var_name))
  m1 <- mean(var_name, na.rm = T)
  par(mfrow=c(2, 2), oma=c(0,0,3,0))
  boxplot(var_name, main="With outliers")
  hist(var_name, main="With outliers", xlab=NA, ylab=NA)
  outlier <- boxplot.stats(var_name)$out
  mo <- mean(outlier)
  var_name <- ifelse(var_name %in% outlier, NA, var_name)
  boxplot(var_name, main="Without outliers")
  hist(var_name, main="Without outliers", xlab=NA, ylab=NA)
  title("Outlier Check", outer=TRUE)
  na2 <- sum(is.na(var_name))
  cat("Outliers identified:", na2 - na1, "n")
  cat("Propotion (%) of outliers:", round((na2 - na1) / sum(!is.na(var_name))*100, 1), "n")
  cat("Mean of the outliers:", round(mo, 2), "n")
  m2 <- mean(var_name, na.rm = T)
  cat("Mean without removing outliers:", round(m1, 2), "n")
  cat("Mean if we remove outliers:", round(m2, 2), "n")
  response <- readline(prompt="Do you want to remove outliers and to replace with NA? [yes/no]: ")
  if(response == "y" | response == "yes"){
    dt[as.character(substitute(var))] <- invisible(var_name)
    assign(as.character(as.list(match.call())$dt), dt, envir = .GlobalEnv)
    cat("Outliers successfully removed", "n")
    return(invisible(dt))
  } else{
    cat("Nothing changed", "n")
    return(invisible(var_name))
  }
}
outlierKD1(CONSP_df, DP) #data, variable


screen_df_ALL <- data.frame(X = log1p(CONSP_df$PSY_freqprox),
                            Y = CONSP_df$DP,
                            M1 = as.numeric(log1p(CONSP_df$SEPI_tot)),
                            M2 = as.numeric(log1p(CONSP_df$SEPI_tot_drug)))

screen_df_NORM <- subset(screen_df_ALL, screen_df_ALL$M1>0 & screen_df_ALL$M2>0)


model <- ' # direct effect (X = Psychedelic Use IV, Y = Cognitive Aberrations DV)
             Y ~ c * X
           # mediator 1, M1 = ego-pathology (endogenous)
             M1 ~ a1*X
             Y ~ b1*M1
           # indirect effect of endogenous path
             a1b1 := a1*b1
           # mediator 2, M2 = ego-pathology (drug-induced)
             M2 ~ a2*X
             Y ~ b2*M2
           # covary Ms (uncoment if needed):
           #  M1 ~~ M2
           # indirect effect of drug-induced path
             a2b2 := a2*b2
           # total effect
             total := c + (a1*b1) + (a2*b2)
         '

fitALL_wls <- sem(model, data = screen_df_ALL, se = "bootstrap", estimator='WLS')
fitNORM_ml <- sem(model, data = screen_df_NORM, se = "bootstrap", estimator='ML')

# Summary of the fit
summary(fitALL_wls, fit.measures=T, rsq=T, standardized = TRUE)
summary(fitNORM_ml, fit.measures=T, rsq=T, standardized = TRUE)



#Finally, generate bootstrap confidence intervals:
boot.fit.ALL <- parameterEstimates(fitALL_wls, boot.ci.type="bca.simple")
boot.fit.NORM <- parameterEstimates(fitNORM_ml, boot.ci.type="bca.simple")


semPaths(fitALL_wls, "model", "std", residuals = F, exoCov = F)
semPaths(fitNORM_ml, "model", "std", residuals = F, exoCov = F)

