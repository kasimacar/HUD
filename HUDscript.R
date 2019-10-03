# clear workspace:
rm(list=ls())

# Load libraries:
library(reghelper)
library(ggplot2)
library(ggstance)
library(radarchart)
library(coefplot)
library(psych)
library(effsize)
library(lm.beta)
library(mlogit)
library(jtools)
library(rockchalk)
library(xlsx)
library(stats)
library(compareGroups)

# Load data
load("HUD_anonymized_final.rda")


#Addressing sampling bias, combine levels
scr_cleaned <- read.xlsx2('HUD_anonymized_cleaned.xlsx',1)[,c('ScreenID','surveyfoundout')]
colnames(scr_cleaned)[1] <- 'ID'
SCREEN_df <- merge(SCREEN_df, scr_cleaned, by=c('ID'))

SCREEN_df$surveyfoundout.y <- combineLevels(SCREEN_df$surveyfoundout.y, levs = c("Npv"), newLabel = "NPV")
SCREEN_df$surveyfoundout.y <- combineLevels(SCREEN_df$surveyfoundout.y, levs = c("Facebook Ad Link"), newLabel = "Facebook")
SCREEN_df$surveyfoundout.y <- combineLevels(SCREEN_df$surveyfoundout.y, levs = c("HPPD Support Group"), newLabel = "HPPD")
SCREEN_df$surveyfoundout.y <- combineLevels(SCREEN_df$surveyfoundout.y, levs = c("Internet Search"), newLabel = "InternetSearch")
SCREEN_df$surveyfoundout.y <- combineLevels(SCREEN_df$surveyfoundout.y, levs = c("KI Website"), newLabel = "KI")
SCREEN_df$surveyfoundout.y <- combineLevels(SCREEN_df$surveyfoundout.y, levs = c("Kompis, Student från Lunds universitet"), newLabel = "Other")
SCREEN_df$surveyfoundout.y <- combineLevels(SCREEN_df$surveyfoundout.y, levs = c("En vän tipsade mig."), newLabel = "Other")
SCREEN_df$surveyfoundout.y <- combineLevels(SCREEN_df$surveyfoundout.y, levs = c("Magiska Molekyler"), newLabel = "MagiskaMolekyler")

#subset young adults
SCREEN_df <- SCREEN_df[(SCREEN_df$age<36 & SCREEN_df$age>17),]
CONSP_df <- CONSP_df[(CONSP_df$age<36& CONSP_df$age>17),]
SCRFU_df <- SCRFU_df[(SCRFU_df$age<36 & SCRFU_df$age>17 & !is.na(SCRFU_df$PSY_freqprox)),]

SCREEN_df$sample <- 'scr'
CONSP_df$sample <- 'FU'
SCRFU_df$sample <- 'FU'
HUDMAIN_df$sample <- 'testing'

#identify and option to remove outliers
outlier <- function(dt, var) {
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
outlier(SCREEN_df, DP) #data, variable

# Selecting subsamples of those without/with  psychiatric diagnoses:
SCREEN_df_noPsych <- subset(SCREEN_df, SCREEN_df$PsychDiagAny==0)
CONSP_df_noPsych <- subset(CONSP_df, CONSP_df$PsychDiagAny==0)
SCREEN_df_Psych <- subset(SCREEN_df, SCREEN_df$PsychDiagAny==1)
CONSP_df_Psych <- subset(CONSP_df, CONSP_df$PsychDiagAny==1)
SCRFU_df_noPsych <- subset(SCRFU_df, SCRFU_df$PsychDiagAny==0 )
SCRFU_df_Psych <- subset(SCRFU_df, SCRFU_df$PsychDiagAny==1 )

#################################################
# Checking representativeness of the subsamples #
#################################################

colnames(HUDMAIN_df)[49] <- 'ID'
#HUDMAIN_df_ext <- merge(HUDMAIN_df, SCREEN_df_noPsych[,c('ID', 'DP','OLIFE_totLog', 'PDI_totalLog','ASRSLog', 'raads_anyLog')], by='ID')


repcheck <- rbind(SCREEN_df[,c('DP','OLIFE_totLog', 'PDI_totalLog','ASRSLog', 'raads_anyLog', 'sample')],
                  CONSP_df[,c('DP','OLIFE_totLog', 'PDI_totalLog','ASRSLog', 'raads_anyLog', 'sample')],
                  HUDMAIN_df[,c('DP','OLIFE_totLog', 'PDI_totalLog','ASRSLog', 'raads_anyLog', 'sample')])

repcheck_noPsych <- rbind(SCREEN_df_noPsych[,c('DP','OLIFE_totLog', 'PDI_totalLog','ASRSLog', 'raads_anyLog', 'sample')],
                          CONSP_df_noPsych[,c('DP','OLIFE_totLog', 'PDI_totalLog','ASRSLog', 'raads_anyLog', 'sample')],
                          HUDMAIN_df[,c('DP','OLIFE_totLog', 'PDI_totalLog','ASRSLog', 'raads_anyLog', 'sample')])


repcheck$sample <- as.factor(repcheck$sample)
repcheck_noPsych$sample <- as.factor(repcheck_noPsych$sample)

repcheck$psychDiag <- TRUE
repcheck_noPsych$psychDiag <- FALSE

# Total Sample:
summary(aov(DP~sample, data=repcheck))
summary(aov(OLIFE_totLog~sample, data=repcheck))
summary(aov(PDI_totalLog~sample, data=repcheck))
summary(aov(ASRSLog~sample, data=repcheck))
summary(aov(raads_anyLog~sample, data=repcheck))

# 'noPsych' Sample:
summary(aov(DP~sample, data=repcheck_noPsych))
summary(aov(OLIFE_totLog~sample, data=repcheck_noPsych))
summary(aov(PDI_totalLog~sample, data=repcheck_noPsych))
summary(aov(ASRSLog~sample, data=repcheck_noPsych))
summary(aov(raads_anyLog~sample, data=repcheck_noPsych))

# Total: interaction with psychDiag:
repcheck_tot <- rbind(repcheck,repcheck_noPsych)
summary(aov(DP~sample*psychDiag, data=repcheck_tot))
summary(aov(OLIFE_totLog~sample*psychDiag, data=repcheck_tot))
summary(aov(PDI_totalLog~sample*psychDiag, data=repcheck_tot))
summary(aov(ASRSLog~sample*psychDiag, data=repcheck_tot))
summary(aov(raads_anyLog~sample*psychDiag, data=repcheck_tot))


# Generate descriptive table:
descrTable <- rbind(c(paste(nrow(SCREEN_df_noPsych), '/',nrow(SCREEN_df), sep=''),paste(nrow(SCRFU_df_noPsych), '/', nrow(SCRFU_df), sep=''),nrow(HUDMAIN_df)),
                    
                    c('O-LIFE, PDI, EPI, History of drug use' , 'Drug use (extended)', 'RALT, BADE'),
                    
                    c(paste(median(as.numeric(as.vector(SCREEN_df$education)), na.rm=T), ' (',min(as.numeric(as.vector(SCREEN_df$education)), na.rm=T),'-',max(as.numeric(as.vector(SCREEN_df$education)), na.rm=T),')', sep=''),
                      paste(median(as.numeric(as.vector(SCRFU_df_noPsych$education)), na.rm=T), ' (',min(as.numeric(as.vector(SCRFU_df_noPsych$education)), na.rm=T),'-',max(as.numeric(as.vector(SCRFU_df_noPsych$education)), na.rm=T),')', sep=''),
                      paste(median(as.numeric(as.vector(HUDMAIN_df$education)), na.rm=T), ' (',min(as.numeric(as.vector(HUDMAIN_df$education)), na.rm=T),'-',max(as.numeric(as.vector(HUDMAIN_df$education)), na.rm=T),')', sep='')),
                    
                    c(paste(round(mean(as.numeric(as.vector(SCREEN_df$age)), na.rm=T),2),'±',round(sd(as.numeric(as.vector(SCREEN_df$age)), na.rm=T),2),sep=''),
                      paste(round(mean(as.numeric(as.vector(SCRFU_df_noPsych$age)), na.rm=T),2),'±',round(sd(as.numeric(as.vector(SCRFU_df_noPsych$age)), na.rm=T),2),sep=''),
                      paste(round(mean(as.numeric(as.vector(HUDMAIN_df$age)), na.rm=T),2),'±',round(sd(as.numeric(as.vector(HUDMAIN_df$age)), na.rm=T),2),sep='')),
                    
                    c(paste(median(as.numeric(as.vector(SCREEN_df$OLIFE_tot)), na.rm=T), ' (',min(as.numeric(as.vector(SCREEN_df$OLIFE_tot)), na.rm=T),'-',max(as.numeric(as.vector(SCREEN_df$OLIFE_tot)), na.rm=T),')', sep=''),
                      paste(median(as.numeric(as.vector(SCRFU_df_noPsych$OLIFE_tot)), na.rm=T), ' (',min(as.numeric(as.vector(SCRFU_df_noPsych$OLIFE_tot)), na.rm=T),'-',max(as.numeric(as.vector(SCRFU_df_noPsych$OLIFE_tot)), na.rm=T),')', sep=''),
                      paste(median(as.numeric(as.vector(HUDMAIN_df$OLIFE_tot)), na.rm=T), ' (',min(as.numeric(as.vector(HUDMAIN_df$OLIFE_tot)), na.rm=T),'-',max(as.numeric(as.vector(HUDMAIN_df$OLIFE_tot)), na.rm=T),')', sep='')),
                    
                    c(paste(median(as.numeric(as.vector(SCREEN_df$PDI_total)), na.rm=T), ' (',min(as.numeric(as.vector(SCREEN_df$PDI_total)), na.rm=T),'-',max(as.numeric(as.vector(SCREEN_df$PDI_total)), na.rm=T),')', sep=''),
                      paste(median(as.numeric(as.vector(SCRFU_df_noPsych$PDI_total)), na.rm=T), ' (',min(as.numeric(as.vector(SCRFU_df_noPsych$PDI_total)), na.rm=T),'-',max(as.numeric(as.vector(SCRFU_df_noPsych$PDI_total)), na.rm=T),')', sep=''),
                      paste(median(as.numeric(as.vector(HUDMAIN_df$PDI_total)), na.rm=T), ' (',min(as.numeric(as.vector(HUDMAIN_df$PDI_total)), na.rm=T),'-',max(as.numeric(as.vector(HUDMAIN_df$PDI_total)), na.rm=T),')', sep=''))
)
            
colnames(descrTable) <- c('SCREENING', 'FOLLOW-UP', 'TESTING')
rownames(descrTable) <- c('N (meeting criteria / total)', 'Assessments', 'Education, years: median(range)', 'Age, years: Mean±SD', 'O-LIFE total: median(range)', 'PDI total: median(range)')
write.xlsx2(descrTable, file='descrTable.xlsx')              


#########################
### Group comparisons ###
#########################


### Whole sample:
# T-test: 
t.test(SCREEN_df$DP[SCREEN_df$drug_psychedelics==0],SCREEN_df$DP[SCREEN_df$drug_psychedelics==1])

# effect size:
cohen.d(SCREEN_df$DP[SCREEN_df$drug_psychedelics==0],SCREEN_df$DP[SCREEN_df$drug_psychedelics==1], na.rm = T)


# Plot whole sample
boxplot(SCREEN_df$DP[SCREEN_df$drug_psychedelics==0],SCREEN_df$DP[SCREEN_df$drug_psychedelics==1], ylab="Schizotypy (z-score)", outline = F, col=c('#01bfc4', '#f8766d'))
axis(side = 1, at = 1, labels = 'Non-users')
axis(side = 1, at = 2, labels = 'Psychedelic Drug-users')
points(cbind(jitter(rep(1, table(SCREEN_df$drug_psychedelics==0)[2])), SCREEN_df$DP[SCREEN_df$drug_psychedelics==0]), pch=16)
points(cbind(jitter(rep(2, table(SCREEN_df$drug_psychedelics==1)[2])), SCREEN_df$DP[SCREEN_df$drug_psychedelics==1]), pch=16)

### Only non-psychiatric diagnoses
# Two-sample t-test: 
t.test(SCREEN_df_noPsych$DP[SCREEN_df_noPsych$drug_psychedelics==0],SCREEN_df_noPsych$DP[SCREEN_df_noPsych$drug_psychedelics==1])
# Plot:
boxplot(SCREEN_df_noPsych$DP[SCREEN_df_noPsych$drug_psychedelics==0],SCREEN_df_noPsych$DP[SCREEN_df_noPsych$drug_psychedelics==1], ylab="Schizotypy (Z-score)", outline = F)
axis(side = 1, at = 1, labels = 'Non-users')
axis(side = 1, at = 2, labels = 'Psychedelic Drug-users', pos = -2.68)
points(cbind(jitter(rep(1, table(SCREEN_df_noPsych$drug_psychedelics==0)[2])), SCREEN_df_noPsych$DP[SCREEN_df_noPsych$drug_psychedelics==0]), pch=16)
points(cbind(jitter(rep(2, table(SCREEN_df_noPsych$drug_psychedelics==1)[2])), SCREEN_df_noPsych$DP[SCREEN_df_noPsych$drug_psychedelics==1]), pch=16)

### Only psychiatric diagnoses
t.test(SCREEN_df_Psych$DP[SCREEN_df_Psych$drug_psychedelics==0],SCREEN_df_Psych$DP[SCREEN_df_Psych$drug_psychedelics==1])

### General Linear Modelling
# Whole sample
mydatawhole <- data.frame(Sampling = SCREEN_df$surveyfoundout.y, Psychedelics = SCREEN_df$drug_psychedelics, Opiates = SCREEN_df$drug_opi,
                     MDMA = SCREEN_df$drug_mdma, Alcohol = SCREEN_df$drug_alc,
                     Cannabis = SCREEN_df$drug_cannabis, Tobacco = SCREEN_df$drug_tobacco,
                     Stimulants = SCREEN_df$drug_stim, Schizotypy = SCREEN_df$DP, Age = SCREEN_df$age, Sex = SCREEN_df$sex)

# add "Sampling" to control for sampling bias
All.Subjects <- glm (Schizotypy ~ Age+Sex+Psychedelics+Opiates+MDMA+Alcohol+Cannabis+Tobacco+Stimulants, data= mydatawhole)
summary(All.Subjects)
beta(All.Subjects)
coefplot.glm(All.Subjects, intercept = F, decreasing = T, title = NULL, xlab = "Estimate", color = "black")

summ(All.Subjects)

# Only no psychiatric diagnoses
mydatanodiag <- data.frame(Sampling = SCREEN_df_noPsych$surveyfoundout.y, Psychedelics = SCREEN_df_noPsych$drug_psychedelics, Opiates = SCREEN_df_noPsych$drug_opi,
                          MDMA = SCREEN_df_noPsych$drug_mdma, Alcohol = SCREEN_df_noPsych$drug_alc,
                          Cannabis = SCREEN_df_noPsych$drug_cannabis, Tobacco = SCREEN_df_noPsych$drug_tobacco,
                          Stimulants = SCREEN_df_noPsych$drug_stim, Schizotypy = SCREEN_df_noPsych$DP, Age = SCREEN_df_noPsych$age, Sex = SCREEN_df_noPsych$sex)

# add "Sampling" to control for sampling bias
No.Diagnoses <- glm(Schizotypy ~ Age+Sex+Psychedelics+Opiates+MDMA+Alcohol+Cannabis+Tobacco+Stimulants, data= mydatanodiag)
summary(No.Diagnoses)
beta(No.Diagnoses)
coefplot.glm(No.Diagnoses, intercept = F, decreasing = T, title = NULL, xlab = "Estimate", color = "black")

# Create plot for both models
multiplot(All.Subjects, No.Diagnoses, intercept = F, decreasing = T, title = NULL,
          xlab = "Estimate", numeric = F, zeroColor = "grey", plot.shapes = TRUE)+scale_color_manual(values=c("black", "black"))


###############################
####          BADE         ####
###     Continuous data     ###
###############################

#Standardize EII scores
HUDMAIN_df$EII1scaled <- scale(HUDMAIN_df$EII2)

HUDMAIN_df$EII2scaled <- scale(HUDMAIN_df$EII)

#EII (old)
mydata.EII1 <- data.frame(Psychedelics = HUDMAIN_df$PSY_freqprox,
                       MDMA = HUDMAIN_df$MDMA_freqprox, Alcohol = HUDMAIN_df$ALC_freqprox, Opiates = HUDMAIN_df$OPI_freqprox,
                       Cannabis = HUDMAIN_df$CAN_freqprox, Tobacco = HUDMAIN_df$TOB_freqprox,
                       Stimulants = HUDMAIN_df$STIM_freqprox, EII = HUDMAIN_df$EII1scaled, Age = HUDMAIN_df$age, Sex = HUDMAIN_df$sex)
EII.1 <- glm (EII ~ Age+Sex+Psychedelics+Opiates+MDMA+Alcohol+Cannabis+Tobacco+Stimulants, data= mydata.EII1)
EII.1.S <- lm.beta(EII.1)
summary(EII.1.S)
summary(EII.1)
beta(EII.1)
coefplot.glm(EII.1, intercept = F, decreasing = T, title = NULL, xlab = "Estimate", color = "black")

#EII (new)
mydata.EII2 <- data.frame(Psychedelics = HUDMAIN_df$PSY_freqprox,
                         MDMA = HUDMAIN_df$MDMA_freqprox, Alcohol = HUDMAIN_df$ALC_freqprox, Opiates = HUDMAIN_df$OPI_freqprox,
                         Cannabis = HUDMAIN_df$CAN_freqprox, Tobacco = HUDMAIN_df$TOB_freqprox,
                         Stimulants = HUDMAIN_df$STIM_freqprox, EII = HUDMAIN_df$EII2scaled, Age = HUDMAIN_df$age, Sex = HUDMAIN_df$sex)
EII.2 <- glm (EII ~ Age+Sex+Psychedelics+Opiates+MDMA+Alcohol+Cannabis+Tobacco+Stimulants, data= mydata.EII2)
EII.2.S <- lm.beta(EII.2)
summary(EII.2)
summary(EII.2.S)
beta(EII.2)
coefplot.glm(EII.2, intercept = F, decreasing = T, title = NULL, xlab = "Estimate", color = "black")

# Create plot for both models
multiplot(EII.1, EII.2, intercept = F, decreasing = T, title = NULL,
          xlab = "Estimate", numeric = F, zeroColor = "grey", plot.shapes = TRUE)+scale_color_manual(values = c("black", "black"))

##############################
### Aversive Fear Learning ###
##############################

# T-test alpha between groups
t.test(HUDMAIN_df$alpha[HUDMAIN_df$drug_psychedelics.y==0],HUDMAIN_df$alpha[HUDMAIN_df$drug_psychedelics.y==1])

#effect size:
cohen.d(HUDMAIN_df$alpha[HUDMAIN_df$drug_psychedelics.y==0],HUDMAIN_df$alpha[HUDMAIN_df$drug_psychedelics.y==1], na.rm = T)


# change numbers of plots
par(mfrow=c(1,2))



# boxplot normal
alphaplot <- boxplot(HUDMAIN_df$alpha[HUDMAIN_df$drug_psychedelics.y==0],HUDMAIN_df$alpha[HUDMAIN_df$drug_psychedelics.y==1], ylab="α", outline = F, col=c('#01bfc4', '#f8766d'))
axis(side = 1, at = 1, labels = 'Non-users')
axis(side = 1, at = 2, labels = 'Psychedelic Drug-users')
points(cbind(jitter(rep(1, table(HUDMAIN_df$drug_psychedelics.y==0)[2])), HUDMAIN_df$alpha[HUDMAIN_df$drug_psychedelics.y==0]), pch=16)
points(cbind(jitter(rep(2, table(HUDMAIN_df$drug_psychedelics.y==1)[2])), HUDMAIN_df$alpha[HUDMAIN_df$drug_psychedelics.y==1]), pch=16)


# T-test rho between groups
t.test(HUDMAIN_df$rhos[HUDMAIN_df$drug_psychedelics.y==0],HUDMAIN_df$rhos[HUDMAIN_df$drug_psychedelics.y==1])

#effect size:
cohen.d(HUDMAIN_df$rhos[HUDMAIN_df$drug_psychedelics.y==0],HUDMAIN_df$rhos[HUDMAIN_df$drug_psychedelics.y==1], na.rm = T)


# boxplot normal:
rhoplot <- boxplot(HUDMAIN_df$rhos[HUDMAIN_df$drug_psychedelics.y==0],HUDMAIN_df$rhos[HUDMAIN_df$drug_psychedelics.y==1], ylab="ρ", outline = F, col=c('#01bfc4', '#f8766d'))
axis(side = 1, at = 1, labels = 'Non-users')
axis(side = 1, at = 2, labels = 'Psychedelic Drug-users')
points(cbind(jitter(rep(1, table(HUDMAIN_df$drug_psychedelics.y==0)[2])), HUDMAIN_df$rhos[HUDMAIN_df$drug_psychedelics.y==0]), pch=16)
points(cbind(jitter(rep(2, table(HUDMAIN_df$drug_psychedelics.y==1)[2])), HUDMAIN_df$rhos[HUDMAIN_df$drug_psychedelics.y==1]), pch=16)


#GLM test regression of rho on Psychedelics overall exposure
rhotest <- glm(rhos ~ PSY_freqprox, data = HUDMAIN_df)
summary(rhotest)
beta(rhotest)

######################
### LOG REGRESSION ###
######################

# Change all values above "1" to "1" (binomial)
SCREEN_df$SEPI_tot[SCREEN_df$SEPI_tot > "1"] <- "1"
CONSP_df$SEPI_tot_drug[CONSP_df$SEPI_tot_drug > "1"] <- "1"

#Endogenous ego-pathology
mydatawhole <- data.frame(Psychedelics = SCREEN_df$drug_psychedelics, Opiates = SCREEN_df$drug_opi,
                          MDMA = SCREEN_df$drug_mdma, Alcohol = SCREEN_df$drug_alc,
                          Cannabis = SCREEN_df$drug_cannabis, Tobacco = SCREEN_df$drug_tobacco,
                          Stimulants = SCREEN_df$drug_stim, Sex = SCREEN_df$sex,
                          EgoPathology_Endo = as.factor(SCREEN_df$SEPI_tot), Age = SCREEN_df$age,
                          Sampling = SCREEN_df$surveyfoundout.y)


Endogenous <- glm (EgoPathology_Endo ~ Age+Sex+Psychedelics+Opiates+MDMA+Alcohol+Cannabis+Tobacco+Stimulants,
                   family = "binomial",
                   data= mydatawhole) # add "Sampling" to control for sampling bias

# get summary
summary(Endogenous)
beta(Endogenous)

# plot results
coefplot.glm(Endogenous, intercept = F, decreasing = T, title = NULL, xlab = "Estimate", color = "black")

#odds ratio
exp(coef(Endogenous))

#confidence intervals
exp(confint(Endogenous))

#summary of results
summ(Endogenous)



# Endogenous ego-pathology Non-psychiatric
mydatanodiag <- data.frame(Psychedelics = SCREEN_df_noPsych$drug_psychedelics, Opiates = SCREEN_df_noPsych$drug_opi,
                          MDMA = SCREEN_df_noPsych$drug_mdma, Alcohol = SCREEN_df_noPsych$drug_alc,
                          Cannabis = SCREEN_df_noPsych$drug_cannabis, Tobacco = SCREEN_df_noPsych$drug_tobacco,
                          Stimulants = SCREEN_df_noPsych$drug_stim, Sex = SCREEN_df_noPsych$sex,
                          EgoPathology_Endo = as.factor(SCREEN_df_noPsych$SEPI_tot), Age = SCREEN_df_noPsych$age)


No.Diagnoses <- glm (EgoPathology_Endo ~ Age+Sex+Psychedelics+Opiates+MDMA+Alcohol+Cannabis+Tobacco+Stimulants,
                     family = "binomial",
                     data= mydatanodiag) # add "Sampling" to control for sampling bias

summary(No.Diagnoses)
beta(No.Diagnoses)
coefplot.glm(No.Diagnoses, intercept = F, decreasing = T, title = NULL, xlab = "Estimate", color = "black")

#odds ratio
exp(coef(No.Diagnoses))

#confidence intervals
exp(confint(No.Diagnoses))

#Drug-Induced Ego-Pathology
mydatawhole <- data.frame(Psychedelics = SCREEN_df$drug_psychedelics, Opiates = SCREEN_df$drug_opi,
                          MDMA = SCREEN_df$drug_mdma, Alcohol = SCREEN_df$drug_alc,
                          Cannabis = SCREEN_df$drug_cannabis, Tobacco = SCREEN_df$drug_tobacco,
                          Stimulants = SCREEN_df$drug_stim, Sex = SCREEN_df$sex,
                          EgoPathology_Drug = as.factor(SCREEN_df$SEPI_tot_drug), Age = SCREEN_df$age,
                          Sampling = SCREEN_df$surveyfoundout.y)

Drug.Induced <- glm (EgoPathology_Drug ~ Age+Sex+Psychedelics+Opiates+MDMA+Alcohol+Cannabis+Tobacco+Stimulants,
                     family = "binomial",
                     data= mydatawhole) # add "Sampling" to control for sampling bias

summary(Drug.Induced)
beta(Drug.Induced)
coefplot.glm(Drug.Induced, intercept = F, decreasing = T, title = NULL, xlab = "Estimate", color = "black")

#odds ratio
exp(coef(Drug.Induced))

#confidence intervals
exp(confint(Drug.Induced))


# Drug-induced ego-pathology Non-psychiatric
mydatanodiag <- data.frame(Psychedelics = SCREEN_df_noPsych$drug_psychedelics, Opiates = SCREEN_df_noPsych$drug_opi,
                           MDMA = SCREEN_df_noPsych$drug_mdma, Alcohol = SCREEN_df_noPsych$drug_alc,
                           Cannabis = SCREEN_df_noPsych$drug_cannabis, Tobacco = SCREEN_df_noPsych$drug_tobacco,
                           Stimulants = SCREEN_df_noPsych$drug_stim, Sex = SCREEN_df_noPsych$sex,
                           EgoPathology_Drug = as.factor(SCREEN_df_noPsych$SEPI_tot_drug), Age = SCREEN_df_noPsych$age)


No.Diagnoses <- glm (EgoPathology_Drug ~ Age+Sex+Psychedelics+Opiates+MDMA+Alcohol+Cannabis+Tobacco+Stimulants,
                     data= mydatanodiag, family = "binomial") # add "Sampling" to control for sampling bias

summary(No.Diagnoses)
beta(No.Diagnoses)
coefplot.glm(No.Diagnoses, intercept = F, decreasing = T, title = NULL, xlab = "Estimate", color = "black")

#odds ratio
exp(coef(No.Diagnoses))

#confidence intervals
exp(confint(No.Diagnoses))

#create coeffplot of both models
multiplot(Endogenous, Drug.Induced, intercept = F, decreasing = T, title = NULL,
          xlab = "Estimate", numeric = F, zeroColor = "grey", plot.shapes = TRUE) +scale_color_manual(values = c("black", "black"))




##################
### SUPPLEMENT ###
##################

##############################################################
### Big Table with group differences Psychedelics         ####
##############################################################

#load education.nn
load("~/EDUCATION/Masters Programme in Cognitive Science/4th Semester/Ok Masters Thesis in Cognitive Science 30hp/Data/educationNumber.rda")

#change 0 to no and 1 to yes in drug_psychedelics
SCREEN_df$drug_psychedelics.yn <- combineLevels(SCREEN_df$drug_psychedelics, levs = c("0"), newLabel = "no")
SCREEN_df$drug_psychedelics.yn <- combineLevels(SCREEN_df$drug_psychedelics.yn, levs = c("1"), newLabel = "yes")


#compare groups
dataDescriptives <- data.frame(Psychedelics = SCREEN_df$drug_psychedelics.yn, Age = SCREEN_df$age, Sex = SCREEN_df$sex,
                              Education = SCREEN_df$education.nn, Depression = SCREEN_df$diagMDep,
                              Bipolar = SCREEN_df$diagBP, Schizophrenia = SCREEN_df$diagScz, ADHD = SCREEN_df$diagADHD,
                              Autism = SCREEN_df$diagASD, OCD = SCREEN_df$diagOCD, DiagOther = SCREEN_df$diagOther,
                              Medication = SCREEN_df$medsY1, Alcohol = SCREEN_df$drug_alc, Tobacco = SCREEN_df$drug_tobacco,
                              MDMA = SCREEN_df$drug_mdma, Cannabis = SCREEN_df$drug_cannabis, Stimulants = SCREEN_df$drug_stim,
                              Opiates = SCREEN_df$drug_opi, Schizotypy = SCREEN_df$DP,
                              ASRS = SCREEN_df$ASRS, RAADS = SCREEN_df$raads_any, SEPIendogenous = SCREEN_df$SEPI_tot,
                              SEPIdrug = SCREEN_df$SEPI_tot_drug, OLIFE = SCREEN_df$OLIFE_tot, PDI = SCREEN_df$PDI_total,
                              PsychiatricDiagnosis = SCREEN_df$PsychDiagAny, Survey = SCREEN_df$surveyfoundout.y)

compareGroups(Psychedelics ~ Age + Sex + Education + Depression + Bipolar + Schizophrenia + ADHD + Autism
              + OCD + DiagOther + PsychiatricDiagnosis + Medication + Alcohol + Tobacco + MDMA + Cannabis + Stimulants
              + Opiates + Schizotypy + ASRS + RAADS + SEPIendogenous + SEPIdrug + OLIFE + PDI
              + Survey, data = dataDescriptives)

#create object
res <- compareGroups(Psychedelics ~ Age + Sex + Education + Depression + Bipolar + Schizophrenia + ADHD + Autism
                     + OCD + DiagOther + PsychiatricDiagnosis + Medication + Alcohol + Tobacco + MDMA + Cannabis + Stimulants
                     + Opiates + Schizotypy + ASRS + RAADS + SEPIendogenous + SEPIdrug + OLIFE + PDI
                     + Survey, data = dataDescriptives)

#create table
restab <- createTable(res, hide.no = "2", hide =c(Sex = "man", Alcohol = "0", Tobacco = "0", MDMA = "0",
                                                  Cannabis = "0", Stimulants = "0", Opiates = "0"), show.n = TRUE)

#print table
print(restab, which.table = "descr", header.labels = c(p.overall = "p"))

#export table
export2html(restab, file='descriptives1.html', header.labels = c(p.overall = "p"))

#export Rmarkdown
export2md(restab, header.labels = c(p.overall = "p"))

##############################################################
### EXPERIMENTAL Table with group differences             ####
##############################################################

## Transform education to number and vector
HUDMAIN_df$education <- combineLevels(HUDMAIN_df$education, levs = c("Sjuksköterskeutbildning, 1/2 år / 3 år"), newLabel = "3.5")
HUDMAIN_df$education <- combineLevels(HUDMAIN_df$education, levs = c("Civilingenjör, på 3dje året"), newLabel = "2")
HUDMAIN_df$education <- combineLevels(HUDMAIN_df$education, levs = c("Psykolog, första året"), newLabel = "1")
HUDMAIN_df$education <- combineLevels(HUDMAIN_df$education, levs = c("3 år"), newLabel = "3")
HUDMAIN_df$education <- combineLevels(HUDMAIN_df$education, levs = c("1,5"), newLabel = "1.5")
HUDMAIN_df$education <- combineLevels(HUDMAIN_df$education, levs = c("Civ. Ing. 4,5 år"), newLabel = "4.5")
HUDMAIN_df$education <- combineLevels(HUDMAIN_df$education, levs = c("University,5Years"), newLabel = "5")
HUDMAIN_df$education <- combineLevels(HUDMAIN_df$education, levs = c("0,5"), newLabel = "0.5")
HUDMAIN_df$education <- combineLevels(HUDMAIN_df$education, levs = c("6 years (bachelor and master)"), newLabel = "6")
HUDMAIN_df$education <- combineLevels(HUDMAIN_df$education, levs = c("Masterexamen"), newLabel = "5")
HUDMAIN_df$education <- combineLevels(HUDMAIN_df$education, levs = c("1,5 years"), newLabel = "1.5")
HUDMAIN_df$education <- combineLevels(HUDMAIN_df$education, levs = c("~1"), newLabel = "1")
HUDMAIN_df$education <- combineLevels(HUDMAIN_df$education, levs = c("4,5"), newLabel = "4.5")

HUDMAIN_df$education.n <- as.numeric(as.vector(HUDMAIN_df$education))


#change 0 to no and 1 to yes in drug_psychedelics
HUDMAIN_df$group.yy <- as.factor(HUDMAIN_df$group.y)
HUDMAIN_df$drug_psychedelics.yn <- combineLevels(HUDMAIN_df$group.yy, levs = c("PP"), newLabel = "yes")
HUDMAIN_df$drug_psychedelics.yn <- combineLevels(HUDMAIN_df$drug_psychedelics.yn, levs = c("NP"), newLabel = "no")

#compare groups
dataDescriptivesSmall <- data.frame(Psychedelics = HUDMAIN_df$drug_psychedelics.yn, Age = HUDMAIN_df$age, Sex = HUDMAIN_df$sex,
                               Education = HUDMAIN_df$education.n, Alcohol = HUDMAIN_df$drug_alc, Tobacco = HUDMAIN_df$drug_tobacco,
                               MDMA = HUDMAIN_df$drug_mdma, Cannabis = HUDMAIN_df$drug_cannabis, Stimulants = HUDMAIN_df$drug_stim,
                               Opiates = HUDMAIN_df$drug_opi, Schizotypy = HUDMAIN_df$DP, Alcohol_FreqProx = HUDMAIN_df$ALC_freqprox,
                               Cannabis_FreqProx = HUDMAIN_df$CAN_freqprox, Tobacco_FreqProx = HUDMAIN_df$TOB_freqprox,
                               Stimulants_FreqProx = HUDMAIN_df$STIM_freqprox, Opiates_FreqProx = HUDMAIN_df$OPI_freqprox,
                               ASRS = HUDMAIN_df$ASRS, RAADS = HUDMAIN_df$raads_any, SEPIendogenous = HUDMAIN_df$SEPI_tot,
                               SEPIdrug = HUDMAIN_df$SEPI_tot_drug, OLIFE = HUDMAIN_df$OLIFE_tot, PDI = HUDMAIN_df$PDI_total,
                               MDMA_FreqProx = HUDMAIN_df$MDMA_freqprox, PsychiatricDiagnosis = HUDMAIN_df$PsychDiagAny)

compareGroups(Psychedelics ~ Age + Sex + Education + Alcohol + Tobacco + MDMA + Cannabis + Stimulants
              + Opiates + Schizotypy + ASRS + RAADS + SEPIendogenous + SEPIdrug + OLIFE + PDI + Alcohol_FreqProx
              + Cannabis_FreqProx + Tobacco_FreqProx + Stimulants_FreqProx + Opiates_FreqProx
              + MDMA_FreqProx, data = dataDescriptivesSmall)

#create object
res <- compareGroups(Psychedelics ~ Age + Sex + Education + Alcohol + Tobacco + MDMA + Cannabis + Stimulants
                     + Opiates + Schizotypy + ASRS + RAADS + SEPIendogenous + SEPIdrug + OLIFE + PDI + Alcohol_FreqProx
                     + Cannabis_FreqProx + Tobacco_FreqProx + Stimulants_FreqProx + Opiates_FreqProx
                     + MDMA_FreqProx, data = dataDescriptivesSmall)
#create table
restab <- createTable(res, hide.no = "2", hide =c(Sex = "man", Alcohol = "0", Tobacco = "0", MDMA = "0",
                                                  Cannabis = "0", Stimulants = "0", Opiates = "0"), show.n = TRUE)

#print table
print(restab, which.table = "descr", header.labels = c(p.overall = "p"))

#export table
export2html(restab, file='descriptives1.html', header.labels = c(p.overall = "p"))

#export Rmarkdown
export2md(restab, header.labels = c(p.overall = "p"), format = "markdown")

