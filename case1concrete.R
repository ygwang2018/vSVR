library(e1071)
library(rpart)
library(neuralnet)
library(readxl)
DataSet<-read_excel('/home/n10141065/mae/10leave/nucase1/Concrete_Data.xls')
#DataSet<-read_excel('C:/Users/n10141065/OneDrive - Queensland University of Technology/Desktop/nu-SVR(collab)/nu_star_experiments/10leave/nucase1/Concrete_Data.xls')

Input<-scale(DataSet[,1:8])

Output<-DataSet[,9]

DataSet<-cbind(Input,Output)

SVs_results<-c()

MSE_results<-c()

MAE_results<-c()

Eps_results<-c()

Optimal_Parameter<-c()

l<-round(0.9*1030)

for (i in c(1:2000)){
  
SVs<-c()

MSE<-c()

MAE<-c()

TotalEps<-c()

count<-1

SampleTrain<-sample(nrow(DataSet),l)

x_train<-DataSet[SampleTrain,1:8]

y_train<-DataSet[SampleTrain,9]

x_test<-DataSet[-SampleTrain,1:8]

y_test<-DataSet[-SampleTrain,9]

N<-8

Gamma_Value<-1/(0.3*N)

GeneralSVR<-svm(x_train,y_train,type='eps-regression',kernel="radial", gamma=Gamma_Value)

res<-GeneralSVR$residuals

epiloss<-function(Para)
  
  # Para<-array(ep,sigma)  
  
  # ep<Para[1]
  
  # sigma<-Para[2]  
  
{
  
  Llog<-ifelse(abs(res/Para[2])<Para[1],0,abs(res/Para[2] )-Para[1])
  
  sum(log(Para[2])+log(2*(1+Para[1]))+Llog)
  
}

optimun<-optim(c(10,2),epiloss,method="L-BFGS-B",lower=c(0.00001,0.01))$par

RobustEps<-optimun[1]

Scale<-optimun[2]

scale_x_train<-x_train

scale_y_train<-y_train/Scale

scale_x_test<-x_test

C2<-quantile(abs(scale_y_train),0.95)

# nu-SVR

C_value <- C2
  
for (cv_nu_Value in c(1:10)/10){ 
  
EpsSVR<-svm(x_train,y_train,type='nu-regression',kernel="radial",scale=FALSE, nu=cv_nu_Value, 
            cost=C_value, gamma=Gamma_Value)

A=as.data.frame(EpsSVR$residuals)

B=as.data.frame(EpsSVR$SV)

a<-which(!rownames(A)%in%rownames(B))

TotalEps[count]<-max(abs(A[a,]))

SVs[count]<-EpsSVR$tot.nSV/l

EpsPreds<-predict(EpsSVR, x_test)

MSE[count]<-sqrt(mean((EpsPreds-y_test)^2))

MAE[count]<-mean(abs(EpsPreds-y_test))

count<-count+1


}

  

##our method


OptimalSVR<-svm(scale_x_train,scale_y_train,type='eps-regression',kernel="radial",scale=FALSE,epsilon=RobustEps, cost=C2, gamma=Gamma_Value)

scale_prediction<-predict(OptimalSVR,scale_x_test)

Prediction<-scale_prediction*Scale

## our optimal output

MSE_Optimal<-sqrt(mean((Prediction-y_test)^2))

MAE_Optimal<-mean(abs(Prediction-y_test))

SVs_Optimal<-OptimalSVR$tot.nSV/l

Eps_Optimal<-RobustEps

Scale_Optimal<-Scale

Nu_Optimal<-log(1+Eps_Optimal)/Eps_Optimal

Optimal_Parameter<-rbind(Optimal_Parameter,cbind(Nu_Optimal,Eps_Optimal,Scale_Optimal,C2))

###

SVs[count+1]<-SVs_Optimal

MSE[count+1]<-MSE_Optimal

MAE[count+1]<-MAE_Optimal

TotalEps[count+1]<-Eps_Optimal

##Banchmark models
SampleTrainData<-cbind(x_train,y_train)

##linear regression
LinearFit=lm(y_train~.,data=SampleTrainData)

LinearPred=predict(LinearFit,x_test)

MSE[count+2]<-sqrt(mean((LinearPred-y_test)^2))

MAE[count+2]<-mean(abs(LinearPred-y_test))

## tree regression
TreeFit=rpart(y_train~.,data=SampleTrainData)

TreePred=predict(TreeFit,x_test)

MSE[count+3]<-sqrt(mean((TreePred-y_test)^2))

MAE[count+3]<-mean(abs(TreePred-y_test))

## neural networks

NNFit=neuralnet(y_train~.,data = SampleTrainData)

NNPred=predict(NNFit,x_test)

MSE[count+4]<-sqrt(mean((NNPred-y_test)^2))

MAE[count+4]<-mean(abs(NNPred-y_test))


SVs_results<-rbind(SVs_results,SVs)

MSE_results<-rbind(MSE_results,MSE)

MAE_results<-rbind(MAE_results,MAE)

Eps_results<-rbind(Eps_results,TotalEps)

}

write.csv(MSE_results,file="concrete_MSE.csv")
write.csv(MAE_results,file="concrete_MAE.csv")

write.csv(SVs_results,file="concrete_SVs.csv")

write.csv(Eps_results,file="concrete_Eps.csv")

write.csv(Optimal_Parameter,file = "concrete_Optimal.csv")