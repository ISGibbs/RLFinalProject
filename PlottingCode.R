################ Data Analysis Code For COMP767 Final Project ##############################


############### Plotting the rewards vs number of episodes ##################################
rewardsDQN <- read.csv("/Users/Isaac/Documents/ReinLearnWinter2019/DQNCodeNoScreen/DQNRewards.csv",header=FALSE)[,1]
rewardsCat <- read.csv("/Users/Isaac/Documents/ReinLearnWinter2019/DQNCodeNoScreen/categoricalRewards.csv",header=FALSE)[,1]
rewardsKL <- read.csv("/Users/Isaac/Documents/ReinLearnWinter2019/DQNCodeNoScreen copy/categoricalRewards.csv",header=FALSE)[,1]

### Temporary Line to Fix Mistake
#rewardsCat<-rewardsCat[-(1:(500*6))]

totalEpisodes<-200
numIterations<-32
totalEpisodesKL<-130

rewardsDQNGrouped <-  array(dim=c(totalEpisodes,2))
rewardsCatGrouped <-  array(dim=c(totalEpisodes,2))
rewardsKLGrouped <-  array(dim=c(totalEpisodesKL,2))
temp<-array(dim=c(numIterations,1))
for(i in 1:totalEpisodes){
  for(j in 0:(numIterations-1)){
    temp[j+1,1]<- rewardsDQN[(j*totalEpisodes+i)]
  }
  rewardsDQNGrouped[i,1] <- mean(temp)
  rewardsDQNGrouped[i,2] <- sd(temp)
}
for(i in 1:totalEpisodes){
  for(j in 0:(numIterations-1)){
    temp[j+1,1]<- rewardsCat[(j*totalEpisodes+i)]
  }
  rewardsCatGrouped[i,1] <- mean(temp)
  rewardsCatGrouped[i,2] <- sd(temp)
}
for(i in 1:totalEpisodesKL){
    for(j in 0:(numIterations-1)){
        temp[j+1,1]<- rewardsKL[(j*totalEpisodesKL+i)]
    }
    rewardsKLGrouped[i,1] <- mean(temp)
    rewardsKLGrouped[i,2] <- sd(temp)
}
data<-as.data.frame(rewardsDQNGrouped)
colnames(data)<-c("Mean","SD")
data$Low<-data$Mean-data$SD
data$High<-data$Mean+data$SD
data$Group<-"Expected With Squared Error Loss"
data$ENum<-1:totalEpisodes

data2<-as.data.frame(rewardsCatGrouped)
colnames(data2)<-c("Mean","SD")
data2$Low<-data2$Mean-data2$SD
data2$High<-data2$Mean+data2$SD
data2$Group<-"Distributional With Cramer Distance"
data2$ENum<-1:totalEpisodes

data3<-as.data.frame(rewardsKLGrouped)
colnames(data3)<-c("Mean","SD")
data3$Low<-data3$Mean-data3$SD
data3$High<-data3$Mean+data3$SD
data3$Group<-"Distributional With KL Divergence"
data3$ENum<-1:totalEpisodesKL

data<-rbind(data,data2,data3)

library(ggplot2)
p<-ggplot(data=data, aes(x=ENum, y=Mean, colour=Group))  + geom_line() +  theme(legend.position="none")
p<-p+geom_ribbon(aes(ymin=data$Low, ymax=data$High, fill=data$Group), linetype=0, alpha=0.4)
#p<-p+labs(title="Performance of Algorithms with Expected and Distributional Losses on Cart Pole") + xlab("Episode Number") + ylab("Average Reward") + theme(legend.position="none")
p<-p+ xlab("Episode Number") + ylab("Mean Reward") + theme(legend.position="bottom",text = element_text(size=25))
p
########################## Plotting some sample runs of CatCramer ###########################################
rewardsCatMod<- array(dim=c(totalEpisodes,3))
plot_list<- list()
for(j in 0:(numIterations-1)){
for(i in 1:totalEpisodes){
    rewardsCatMod[i,1]<-rewardsCat[j*totalEpisodes+i]
    rewardsCatMod[i,2]<-j
    rewardsCatMod[i,3]<-i
  }
    rewardsCatMod <- as.data.frame(rewardsCatMod)
    colnames(rewardsCatMod)[1:3]<-c("Rewards","Trial","ENum")
    rewardsCatMod$High<-rewardsCatMod$Rewards
    rewardsCatMod$Low<-rewardsCatMod$Rewards
    plot_list[[j+1]]<-ggplot(data=rewardsCatMod, aes(x=ENum, y=Rewards, colour=Trial)) + geom_point() + geom_line()
}
library(cowplot)
plot_grid(plot_list[[1]],plot_list[[2]],plot_list[[3]],plot_list[[4]],plot_list[[5]],plot_list[[6]], ncol=2,nrow=3)
plot_grid(plot_list[[7]],plot_list[[8]],plot_list[[9]],plot_list[[10]],plot_list[[11]],plot_list[[12]], ncol=2,nrow=3)
plot_grid(plot_list[[13]],plot_list[[14]],plot_list[[15]],plot_list[[16]],plot_list[[17]],plot_list[[18]], ncol=2,nrow=3)
plot_grid(plot_list[[19]],plot_list[[20]],plot_list[[21]],plot_list[[22]],plot_list[[23]],plot_list[[24]], ncol=2,nrow=3)
plot_grid(plot_list[[25]],plot_list[[26]],plot_list[[27]],plot_list[[28]],plot_list[[29]],plot_list[[30]], ncol=2,nrow=3)

############################# Plotting the norms ################################################################

catNorms <- read.csv("/Users/Isaac/Documents/ReinLearnWinter2019/DQNCodeNoScreen/CategoricalNorms.csv",header=FALSE)
DQNNorms <- read.csv("/Users/Isaac/Documents/ReinLearnWinter2019/DQNCodeNoScreen/DQNNorms.csv",header=FALSE)
KLNorms <- read.csv("/Users/Isaac/Documents/ReinLearnWinter2019/DQNCodeNoScreen copy/CategoricalNorms.csv",header=FALSE)

#### Histograms
dat<-as.data.frame(catNorms[,1])
colnames(dat)<-c("Norm")
dat$Group<-"Cat"

dat2<-as.data.frame(DQNNorms[,1])
colnames(dat2)<-c("Norm")
dat2$Group<-"DQN"

dat3<-as.data.frame(KLNorms[,1])
colnames(dat3)<-c("Norm")
dat3$Group<-"KL"

dat<-rbind(dat,dat2,dat3)

#dat<-dat[dat$Norm<10,]

width<-0.01

#ggplot() + geom_histogram(data=dat[dat$Group=="Cat",], aes(x=Norm,y=(..density..)*width),fill = "red", alpha = 0.2,binwidth=width,center=width/2) + geom_histogram(data=dat[dat$Group=="DQN",], aes(x=Norm,y=(..density..)*width),fill = "blue", alpha = 0.2,binwidth=width,center=width/2) + geom_histogram(data=dat[dat$Group=="KL",], aes(x=Norm,y=(..density..)*width),fill = "green", alpha = 0.2,binwidth=width,center=width/2)
library(ggplot2)
p1 <- ggplot() + geom_histogram(data=dat[dat$Norm<2,][dat$Group=="KL",], aes(x=Norm,y=..count../sum(..count..)),fill = "green", alpha = 0.2,binwidth=width,center=width/2)

p2 <- ggplot() + geom_histogram(data=dat[dat$Norm<15,][dat$Group=="DQN",], aes(x=Norm,y=..count../sum(..count..)),fill = "blue", alpha = 0.2,binwidth=width,center=width/2)

p3 <- ggplot() + geom_histogram(data=dat[dat$Norm<1,][dat$Group=="Cat",], aes(x=Norm,y=..count../sum(..count..)),fill = "red", alpha = 0.2,binwidth=width,center=width/2)

library(cowplot)
plot_grid(p1,p2,p3,nrow=1,ncol=3)

### How Norm Evolves Over Time
catNEp<-aggregate(catNorms[,1],list(catNorms[,2]),mean)
catNSD<-aggregate(catNorms[,1],list(catNorms[,2]),sd)
colnames(catNEp)<-c("StepNum","Mean")
catNEp$SD<-catNSD[,2]
catNEp$Low<-catNEp$Mean-catNEp$SD
catNEp$High<-catNEp$Mean+catNEp$SD
catNEp$Group<-"Distributional w/ Cramer Loss"

DQNNEp<-aggregate(DQNNorms[,1],list(DQNNorms[,2]),mean)
DQNNSD<-aggregate(DQNNorms[,1],list(DQNNorms[,2]),sd)
colnames(DQNNEp)<-c("StepNum","Mean")
DQNNEp$SD<-DQNNSD[,2]
DQNNEp$Low<-DQNNEp$Mean-DQNNEp$SD
DQNNEp$High<-DQNNEp$Mean+DQNNEp$SD
DQNNEp$Group<-"Expected w/ Squared Loss"

KLNEp<-aggregate(KLNorms[,1],list(KLNorms[,2]),mean)
KLNSD<-aggregate(KLNorms[,1],list(KLNorms[,2]),sd)
colnames(KLNEp)<-c("StepNum","Mean")
KLNEp$SD<-KLNSD[,2]
KLNEp$Low<-KLNEp$Mean-KLNEp$SD
KLNEp$High<-KLNEp$Mean+KLNEp$SD
KLNEp$Group<-"Distributional w/ KL Divergence"

data<-rbind(catNEp,DQNNEp,KLNEp)

library(ggplot2)
p<-ggplot(data=data, aes(x=StepNum, y=Mean, colour=Group))  + geom_line()
p<-p+geom_ribbon(aes(ymin=data$Low, ymax=data$High, fill=data$Group), linetype=0, alpha=0.4)
#p<-p+labs(title="Norms of the Gradients Produced by Algorithms with Expected and Distributional Losses") + xlab("Step Number") + ylab("Average Norm") + theme(legend.position="none")
p<-p+xlab("Step Number") + ylab("Mean Norm") + theme(legend.position="none",text = element_text(size=25))
p

################################## Plotting Beta Smoothness ################################################################

catSmooth <- read.csv("/Users/Isaac/Documents/ReinLearnWinter2019/DQNCodeNoScreen/CatSmooth.csv",header=FALSE)
DQNSmooth <- read.csv("/Users/Isaac/Documents/ReinLearnWinter2019/DQNCodeNoScreen/DQNSmooth.csv",header=FALSE)
KLSmooth <- read.csv("/Users/Isaac/Documents/ReinLearnWinter2019/DQNCodeNoScreen copy/CatSmooth.csv",header=FALSE)

#### Histograms
dat<-as.data.frame(catSmooth[,1])
colnames(dat)<-c("Smooth")
dat$Group<-"Cat"

dat2<-as.data.frame(DQNSmooth[,1])
colnames(dat2)<-c("Smooth")
dat2$Group<-"DQN"

dat3<-as.data.frame(KLSmooth[,1])
colnames(dat3)<-c("Smooth")
dat3$Group<-"KL"

dat<-rbind(dat,dat2,dat3)

#dat<-dat[dat$Norm<10,]

width<-0.01

#ggplot() + geom_histogram(data=dat[dat$Group=="Cat",], aes(x=Smooth,y=(..density..)*width),fill = "red", alpha = 0.2,binwidth=width,center=width/2) + geom_histogram(data=dat[dat$Group=="DQN",], aes(x=Smooth,y=(..density..)*width),fill = "blue", alpha = 0.2,binwidth=width,center=width/2)
library(ggplot2)
p1 <- ggplot() + geom_histogram(data=dat[dat$Group=="Cat",], aes(x=Smooth,y=..count../sum(..count..)),fill = "green", alpha = 0.2,binwidth=width,center=width/2)

p2 <- ggplot() + geom_histogram(data=dat[dat$Group=="DQN",], aes(x=Smooth,y=..count../sum(..count..)),fill = "blue", alpha = 0.2,binwidth=width,center=width/2)

p3 <- ggplot() + geom_histogram(data=dat[dat$Group=="KL",], aes(x=Smooth,y=..count../sum(..count..)),fill = "red", alpha = 0.2,binwidth=width,center=width/2)

library(cowplot)
plot_grid(p1,p2,p3,nrow=1,ncol=3)

### How Smoothness Evolves Over Time
catSEp<-aggregate(catSmooth[,1],list(catSmooth[,2]),mean)
catSSD<-aggregate(catSmooth[,1],list(catSmooth[,2]),sd)
colnames(catSEp)<-c("StepNum","Mean")
catSEp$SD<-catSSD[,2]
catSEp$Low<-catSEp$Mean-catSEp$SD
catSEp$High<-catSEp$Mean+catSEp$SD
catSEp$Group<-"Distributional w/ Cramer Loss"

DQNSEp<-aggregate(DQNSmooth[,1],list(DQNSmooth[,2]),mean)
DQNSSD<-aggregate(DQNSmooth[,1],list(DQNSmooth[,2]),sd)
colnames(DQNSEp)<-c("StepNum","Mean")
DQNSEp$SD<-DQNSSD[,2]
DQNSEp$Low<-DQNSEp$Mean-DQNSEp$SD
DQNSEp$High<-DQNSEp$Mean+DQNSEp$SD
DQNSEp$Group<-"Expected w/ Squared Loss"

KLSEp<-aggregate(KLSmooth[,1],list(KLSmooth[,2]),mean)
KLSSD<-aggregate(KLSmooth[,1],list(KLSmooth[,2]),sd)
colnames(KLSEp)<-c("StepNum","Mean")
KLSEp$SD<-KLSSD[,2]
KLSEp$Low<-KLSEp$Mean-KLSEp$SD
KLSEp$High<-KLSEp$Mean+KLSEp$SD
KLSEp$Group<-"Distributional w/ KL Divergence"

data<-rbind(catSEp,DQNSEp,KLSEp)

library(ggplot2)
p<-ggplot(data=data, aes(x=StepNum, y=Mean, colour=Group))  + geom_line()
p<-p+geom_ribbon(aes(ymin=data$Low, ymax=data$High, fill=data$Group), linetype=0, alpha=0.4)
#p<-p+labs(title="Data-Driven Approximation of Beta for Distributional and Expected Algorithms") + xlab("Step Number") + ylab("Mean Ratio") + theme(legend.position="none")
p<-p+ xlab("Step Number") + ylab("Mean Ratio")+ theme(legend.position="none",text = element_text(size=25))
p <- p + coord_cartesian(ylim = c(0, 10000))
p

data<-rbind(DQNSEp,KLSEp)

library(ggplot2)
p<-ggplot(data=data, aes(x=StepNum, y=Mean, colour=Group))  + geom_line()
p<-p+geom_ribbon(aes(ymin=data$Low, ymax=data$High, fill=data$Group), linetype=0, alpha=0.4)
p<-p+labs(title="Data-Driven Approximation of Beta for Distributional and Expected Algorithms With No Cramer Loss") + xlab("Step Number") + ylab("Mean Ratio")
p













########################### Plotting Jitters ########################################################################

catJit <- read.csv("/Users/Isaac/Documents/ReinLearnWinter2019/DQNCodeNoScreen/jitteredCategoricalTestRewards.csv",header=FALSE)[,1]
DQNJit <- read.csv("/Users/Isaac/Documents/ReinLearnWinter2019/DQNCodeNoScreen/jitteredTestRewards.csv",header=FALSE)[,1]
KLJit <- read.csv("/Users/Isaac/Documents/ReinLearnWinter2019/DQNCodeNoScreen copy 2/jitteredCategoricalTestRewards.csv",header=FALSE)[,1]

### Temporary Line to Fix Mistake
#catJit<-catJit[-(1:55)]

goodRunsCat<-catJit[seq(1,32*11*30,11)]
goodRunsDQN<-DQNJit[seq(1,32*11*30,11)]
goodRunsKL<-KLJit[seq(1,32*11*30,11)]

catJit<-catJit[-seq(1,32*11*30,11)]
DQNJit<-DQNJit[-seq(1,32*11*30,11)]
KLJit<-KLJit[-seq(1,32*11*30,11)]

### Collect Jitter Data
catJitDat<-array(dim=c(30,2))
DQNJitDat<-array(dim=c(30,2))
KLJitDat<-array(dim=c(30,2))
temp1<-array(dim=c(32,1))
temp2<-array(dim=c(32,1))
temp3<-array(dim=c(32,1))
for(i in 1:30){
    for(j in 1:32){
        temp1[j,1]<-mean(catJit[((j-1)*300 + (i-1)*10 + 1):((j-1)*300 + (i-1)*10 + 10)])
        temp2[j,1]<-mean(DQNJit[((j-1)*300 + (i-1)*10 + 1):((j-1)*300 + (i-1)*10 + 10)])
        temp3[j,1]<-mean(KLJit[((j-1)*300 + (i-1)*10 + 1):((j-1)*300 + (i-1)*10 + 10)])
    }
    catJitDat[i,1] <- mean(temp1[,1])
    catJitDat[i,2] <- sd(temp1[,1])
    DQNJitDat[i,1] <- mean(temp2[,1])
    DQNJitDat[i,2] <- sd(temp2[,1])
    KLJitDat[i,1] <- mean(temp3[,1])
    KLJitDat[i,2] <- sd(temp3[,1])
}

### Format Jitters
data<-as.data.frame(DQNJitDat)
colnames(data)<-c("Mean","SD")
data$Low<-data$Mean-data$SD
data$High<-data$Mean+data$SD
data$Group<-"DQN"
data$ENum<-seq(10,300,10)

data2<-as.data.frame(catJitDat)
colnames(data2)<-c("Mean","SD")
data2$Low<-data2$Mean-data2$SD
data2$High<-data2$Mean+data2$SD
data2$Group<-"CAT"
data2$ENum<-seq(10,300,10)

data3<-as.data.frame(KLJitDat)
colnames(data3)<-c("Mean","SD")
data3$Low<-data3$Mean-data3$SD
data3$High<-data3$Mean+data3$SD
data3$Group<-"KL"
data3$ENum<-seq(10,300,10)

### Collect good Runs
catGoodDat<-array(dim=c(30,2))
DQNGoodDat<-array(dim=c(30,2))
KLGoodDat<-array(dim=c(30,2))
temp1<-array(dim=c(32,1))
temp2<-array(dim=c(32,1))
temp3<-array(dim=c(32,1))
for(i in 1:30){
    for(j in 1:32){
        temp1[j,1]<-mean(goodRunsCat[(j-1)*30 + i])
        temp2[j,1]<-mean(goodRunsDQN[(j-1)*30 + i])
        temp3[j,1]<-mean(goodRunsKL[(j-1)*30 + i])
    }
    catGoodDat[i,1] <- mean(temp1[,1])
    catGoodDat[i,2] <- 0
    DQNGoodDat[i,1] <- mean(temp2[,1])
    DQNGoodDat[i,2] <- 0
    KLGoodDat[i,1] <- mean(temp3[,1])
    KLGoodDat[i,2] <- 0
}

#### Format good runs
data4<-as.data.frame(DQNGoodDat)
colnames(data4)<-c("Mean","SD")
data4$Low<-data4$Mean-data4$SD
data4$High<-data4$Mean+data4$SD
data4$Group<-"DQNGood"
data4$ENum<-seq(10,300,10)

data5<-as.data.frame(catGoodDat)
colnames(data5)<-c("Mean","SD")
data5$Low<-data5$Mean-data5$SD
data5$High<-data5$Mean+data5$SD
data5$Group<-"CatGood"
data5$ENum<-seq(10,300,10)

data6<-as.data.frame(KLGoodDat)
colnames(data6)<-c("Mean","SD")
data6$Low<-data6$Mean-data6$SD
data6$High<-data6$Mean+data6$SD
data6$Group<-"KLGood"
data6$ENum<-seq(10,300,10)


data<-rbind(data,data2,data3,data4,data5,data6)

library(ggplot2)
p<-ggplot(data=data, aes(x=ENum, y=Mean, colour=Group))  + geom_line()
p<-p+geom_ribbon(aes(ymin=data$Low, ymax=data$High, fill=data$Group), linetype=0, alpha=0.4)



#for(i in 1:32){
#    for(j in 1:10){
#        dat$Jit[(i-1)*10+j]<-goodRun$Jit[i] - dat$Jit[(i-1)*10+j]
#        dat$Jit[(i-1)*10+j+320]<-goodRun$Jit[i+32] - dat$Jit[(i-1)*10+j+320]
#        dat$Jit[(i-1)*10+j+640]<-goodRun$Jit[i+64] - dat$Jit[(i-1)*10+j+640]
#    }
#}

width<-2

ggplot() + geom_histogram(data=dat[dat$Group=="Cat",], aes(x=Jit,y=..count../sum(..count..)),fill = "red", alpha = 0.2,binwidth=width,center=width/2) + geom_histogram(data=dat[dat$Group=="DQN",], aes(x=Jit,y=..count../sum(..count..)),fill = "blue", alpha = 0.2,binwidth=width,center=width/2) + geom_histogram(data=dat[dat$Group=="KL",], aes(x=Jit,y=..count../sum(..count..)),fill = "green", alpha = 0.2,binwidth=width,center=width/2)



############################################################# Fourier Basis Plots ##################################################################


############### Plotting the rewards vs number of episodes ##################################
rewardsDQN <- read.csv("/Users/Isaac/Documents/ReinLearnWinter2019/DQNCodeNoScreen copy 2/DQNRewards.csv",header=FALSE)[,1]
rewardsCat <- read.csv("/Users/Isaac/Documents/ReinLearnWinter2019/DQNCodeNoScreen copy 3/categoricalRewards.csv",header=FALSE)[,1]
rewardsKL <- read.csv("/Users/Isaac/Documents/ReinLearnWinter2019/DQNCodeNoScreen copy 2/categoricalRewards.csv",header=FALSE)[,1]

### Temporary Line to Fix Mistake
#rewardsCat<-rewardsCat[-(1:(500*6))]

totalEpisodes<-100
numIterations<-32
totalEpisodesKL<-200
totalEpisodesCR<-200

rewardsDQNGrouped <-  array(dim=c(totalEpisodes,2))
rewardsCatGrouped <-  array(dim=c(totalEpisodesCR,2))
rewardsKLGrouped <-  array(dim=c(totalEpisodesKL,2))
temp<-array(dim=c(numIterations,1))
for(i in 1:totalEpisodes){
    for(j in 0:(numIterations-1)){
        temp[j+1,1]<- rewardsDQN[(j*totalEpisodes+i)]
    }
    rewardsDQNGrouped[i,1] <- mean(temp)
    rewardsDQNGrouped[i,2] <- sd(temp)
}
for(i in 1:totalEpisodesCR){
    for(j in 0:(numIterations-1)){
        temp[j+1,1]<- rewardsCat[(j*totalEpisodesCR+i)]
    }
    rewardsCatGrouped[i,1] <- mean(temp)
    rewardsCatGrouped[i,2] <- sd(temp)
}
for(i in 1:totalEpisodesKL){
    for(j in 0:(numIterations-1)){
        temp[j+1,1]<- rewardsKL[(j*totalEpisodesKL+i)]
    }
    rewardsKLGrouped[i,1] <- mean(temp)
    rewardsKLGrouped[i,2] <- sd(temp)
}
data<-as.data.frame(rewardsDQNGrouped)
colnames(data)<-c("Mean","SD")
data$Low<-data$Mean-data$SD
data$High<-data$Mean+data$SD
data$Group<-"Expected"
data$ENum<-1:totalEpisodes

data2<-as.data.frame(rewardsCatGrouped)
colnames(data2)<-c("Mean","SD")
data2$Low<-data2$Mean-data2$SD
data2$High<-data2$Mean+data2$SD
data2$Group<-"Distributional w/ Cramer Loss"
data2$ENum<-1:totalEpisodesCR

data3<-as.data.frame(rewardsKLGrouped)
colnames(data3)<-c("Mean","SD")
data3$Low<-data3$Mean-data3$SD
data3$High<-data3$Mean+data3$SD
data3$Group<-"Distributional w/ KL Divergence"
data3$ENum<-1:totalEpisodesKL

data<-rbind(data,data2,data3)

library(ggplot2)
p<-ggplot(data=data, aes(x=ENum, y=Mean, colour=Group))  + geom_line() +  theme(legend.position="none")
p<-p+geom_ribbon(aes(ymin=data$Low, ymax=data$High, fill=data$Group), linetype=0, alpha=0.4)
#p<-p+labs(title="Performance of Algorithms with Expected and Distributional Losses on Cart Pole") + xlab("Episode Number") + ylab("Average Reward") + theme(legend.position="none")
p<-p+ xlab("Episode Number") + ylab("Mean Reward") + theme(legend.position="none",text = element_text(size=25))
p



############################# Plotting the norms ################################################################

catNorms <- read.csv("/Users/Isaac/Documents/ReinLearnWinter2019/DQNCodeNoScreen copy 3/CategoricalNorms.csv",header=FALSE)
DQNNorms <- read.csv("/Users/Isaac/Documents/ReinLearnWinter2019/DQNCodeNoScreen copy 2/DQNNorms.csv",header=FALSE)
KLNorms <- read.csv("/Users/Isaac/Documents/ReinLearnWinter2019/DQNCodeNoScreen copy 2/CategoricalNorms.csv",header=FALSE)


### How Norm Evolves Over Time
catNEp<-aggregate(catNorms[,1],list(catNorms[,2]),mean)
catNSD<-aggregate(catNorms[,1],list(catNorms[,2]),sd)
colnames(catNEp)<-c("StepNum","Mean")
catNEp$SD<-catNSD[,2]
catNEp$Low<-catNEp$Mean-catNEp$SD
catNEp$High<-catNEp$Mean+catNEp$SD
catNEp$Group<-"Distributional w/ Cramer Loss"

DQNNEp<-aggregate(DQNNorms[,1],list(DQNNorms[,2]),mean)
DQNNSD<-aggregate(DQNNorms[,1],list(DQNNorms[,2]),sd)
colnames(DQNNEp)<-c("StepNum","Mean")
DQNNEp$SD<-DQNNSD[,2]
DQNNEp$Low<-DQNNEp$Mean-DQNNEp$SD
DQNNEp$High<-DQNNEp$Mean+DQNNEp$SD
DQNNEp$Group<-"Expected w/ Squared Loss"

KLNEp<-aggregate(KLNorms[,1],list(KLNorms[,2]),mean)
KLNSD<-aggregate(KLNorms[,1],list(KLNorms[,2]),sd)
colnames(KLNEp)<-c("StepNum","Mean")
KLNEp$SD<-KLNSD[,2]
KLNEp$Low<-KLNEp$Mean-KLNEp$SD
KLNEp$High<-KLNEp$Mean+KLNEp$SD
KLNEp$Group<-"Distributional w/ KL Divergence"

data<-rbind(catNEp,DQNNEp,KLNEp)

library(ggplot2)
p<-ggplot(data=data, aes(x=StepNum, y=Mean, colour=Group))  + geom_line()
p<-p+geom_ribbon(aes(ymin=data$Low, ymax=data$High, fill=data$Group), linetype=0, alpha=0.4)
#p<-p+labs(title="Norms of the Gradients Produced by Algorithms with Expected and Distributional Losses") + xlab("Step Number") + ylab("Average Norm") + theme(legend.position="none")
p<-p+xlab("Step Number") + ylab("Mean Norm") + theme(legend.position="none",text = element_text(size=25))
p



################################## Plotting Beta Smoothness ################################################################

catSmooth <- read.csv("/Users/Isaac/Documents/ReinLearnWinter2019/DQNCodeNoScreen copy 3/CatSmooth.csv",header=FALSE)
DQNSmooth <- read.csv("/Users/Isaac/Documents/ReinLearnWinter2019/DQNCodeNoScreen copy 2/DQNSmooth.csv",header=FALSE)
KLSmooth <- read.csv("/Users/Isaac/Documents/ReinLearnWinter2019/DQNCodeNoScreen copy 2/CatSmooth.csv",header=FALSE)


### How Smoothness Evolves Over Time
catSEp<-aggregate(catSmooth[,1],list(catSmooth[,2]),mean)
catSSD<-aggregate(catSmooth[,1],list(catSmooth[,2]),sd)
colnames(catSEp)<-c("StepNum","Mean")
catSEp$SD<-catSSD[,2]
catSEp$Low<-catSEp$Mean-catSEp$SD
catSEp$High<-catSEp$Mean+catSEp$SD
catSEp$Group<-"Distributional w/ Cramer Loss"

DQNSEp<-aggregate(DQNSmooth[,1],list(DQNSmooth[,2]),mean)
DQNSSD<-aggregate(DQNSmooth[,1],list(DQNSmooth[,2]),sd)
colnames(DQNSEp)<-c("StepNum","Mean")
DQNSEp$SD<-DQNSSD[,2]
DQNSEp$Low<-DQNSEp$Mean-DQNSEp$SD
DQNSEp$High<-DQNSEp$Mean+DQNSEp$SD
DQNSEp$Group<-"Expected w/ Squared Loss"

KLSEp<-aggregate(KLSmooth[,1],list(KLSmooth[,2]),mean)
KLSSD<-aggregate(KLSmooth[,1],list(KLSmooth[,2]),sd)
colnames(KLSEp)<-c("StepNum","Mean")
KLSEp$SD<-KLSSD[,2]
KLSEp$Low<-KLSEp$Mean-KLSEp$SD
KLSEp$High<-KLSEp$Mean+KLSEp$SD
KLSEp$Group<-"Distributional w/ KL Divergence"

data<-rbind(catSEp,DQNSEp,KLSEp)

library(ggplot2)
p<-ggplot(data=data, aes(x=StepNum, y=Mean, colour=Group))  + geom_line()
p<-p+geom_ribbon(aes(ymin=data$Low, ymax=data$High, fill=data$Group), linetype=0, alpha=0.4)
#p<-p+labs(title="Data-Driven Approximation of Beta for Distributional and Expected Algorithms") + xlab("Step Number") + ylab("Mean Ratio") + theme(legend.position="none")
p<-p+ xlab("Step Number") + ylab("Mean Ratio")+ theme(legend.position="none",text = element_text(size=25))
p

data<-rbind(DQNSEp,KLSEp)

library(ggplot2)
p<-ggplot(data=data, aes(x=StepNum, y=Mean, colour=Group))  + geom_line()
p<-p+geom_ribbon(aes(ymin=data$Low, ymax=data$High, fill=data$Group), linetype=0, alpha=0.4)
#p<-p+labs(title="Data-Driven Approximation of Beta for Distributional and Expected Algorithms With No Cramer Loss") + xlab("Step Number") + ylab("Average Ratio")
p<-p+ xlab("Step Number") + ylab("Average Ratio")+ theme(legend.position="none")
p








