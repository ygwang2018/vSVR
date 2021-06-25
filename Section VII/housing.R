library(scales)
library(e1071)

##Codes for nested cross-validation

DataSet<-read.csv('C:/Users/n10141065/Queensland University of Technology/You-Gan Wang - RyanWu(Jinran)/TNNLS Revised/TNNLS revised iii/Case/nucase_chosen/housing.csv')

#Copy without scale

ResponseIndex<-dim(DataSet)[2]

DataSet1<-DataSet

##Number of folder: 10
MaxY<-max(DataSet[,ResponseIndex])

MinY<-min(DataSet[,ResponseIndex])

DataSet=apply(DataSet,2,rescale)

K<-10

NumObs<-dim(DataSet)[1]

FolderSize<-floor(NumObs/K)

CV_Results<-c()

tt1<-Sys.time()

for (k in c(1:10)){
  
  TestIndex<-c(((k-1)*FolderSize+1):(k*FolderSize))
  
  TestSet<-DataSet[TestIndex,]
  
  InnerK<-setdiff(c(1:10),k)
  
#nu loop
  Average_Valid_Nu_RMSE<-array()
  
  t1<-1
  
  for (nu_cv_Value in c(1:10)/10){
    
    Valid_Nu_RMSE<-array()
    
    t<-1
  
       for (k1 in InnerK){
    
           ValidateIndex<-c(((k1-1)*FolderSize+1):(k1*FolderSize))
    
           ValidateSet<-DataSet[ValidateIndex,]
    
           TrainSet<-DataSet[-rbind(as.matrix(TestIndex),as.matrix(ValidateIndex)),]
           
           ResponseIndex<-dim(DataSet)[2]
           
           x_train<-TrainSet[,1:(ResponseIndex-1)]
             
           y_train<-TrainSet[,ResponseIndex]   
           
           Recom_C<-max(abs(mean(y_train)+3*sd(y_train)),abs(mean(y_train)-3*sd(y_train)))
           
           Gamma_Value<-1/(0.3*(ResponseIndex-1))
             
           NuSVR<-svm(x_train,y_train,type='nu-regression',kernel="radial",scale=FALSE, nu=nu_cv_Value, 
                      cost=Recom_C, gamma=Gamma_Value)
           
           x_valid<-ValidateSet[,1:(ResponseIndex-1)]
           
           y_valid<-ValidateSet[,ResponseIndex] 
           
           y_valid_pred<-predict(NuSVR, x_valid)
           
           Valid_Nu_RMSE[t]<-sqrt(mean((y_valid-y_valid_pred)^2))
           
           t<-t+1
    
       }
    
    Average_Valid_Nu_RMSE[t1]<-mean(Valid_Nu_RMSE)
    
    t1<-t1+1

}

  Nu_Sequence<-c(1:10)/10    
  
  Optimal_Nu<-Nu_Sequence[(rank(Average_Valid_Nu_RMSE)[10])]
  
  #print(Optimal_Nu)
  
  Whole_Train_Set<-DataSet[-TestIndex,]
  
  x_whole_train<-Whole_Train_Set[,1:(ResponseIndex-1)]
  
  y_whole_train<-Whole_Train_Set[,ResponseIndex]  
  
  Optimal_NuSVR<-svm(x_whole_train,y_whole_train,type='nu-regression',kernel="radial",scale=FALSE, nu= Optimal_Nu, 
             cost=Recom_C, gamma=Gamma_Value)
  
  x_test<-TestSet[,1:(ResponseIndex-1)]
  
  Optimal_Preds<-predict(Optimal_NuSVR,x_test)*(MaxY-MinY)+MinY
  
  Y_Test<-DataSet1[TestIndex,ResponseIndex]
  
  Optimal_RMSE<-sqrt(mean((Y_Test-Optimal_Preds)^2))
  
  CV_Results<-cbind(CV_Results,rbind(Optimal_Nu,Optimal_RMSE))
  
}

tt2<-Sys.time()

print(tt2-tt1)

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
# Now, we will conduct our new algorithm with working likelihood method.
###################################################################################
###################################################################################  

InputScale<-scale(DataSet1[,1:(ResponseIndex-1)])  
  
DataSet<-cbind(InputScale,DataSet1[,ResponseIndex])

TuningResults<-c()

for (k in c(1:10)){
  
  TestIndex<-c(((k-1)*FolderSize+1):(k*FolderSize))
  
  TestSet<-DataSet[TestIndex,]
  
  WholeTrainSet<-DataSet[-TestIndex,]
  
  x_wholetrain<-WholeTrainSet[,1:(ResponseIndex-1)]
    
  y_wholetrain<-WholeTrainSet[,ResponseIndex]
  
  x_test<-TestSet[,1:(ResponseIndex-1)]
  
  y_test<-TestSet[,ResponseIndex]
  
  InitialSVR<-svm(x_wholetrain,y_wholetrain,type='eps-regression',kernel="radial", gamma=Gamma_Value)
  
  InitialRes<-InitialSVR$residuals
  
  epiloss<-function(Para)
    
  {
    res<-InitialRes
    
    Llog<-ifelse(abs(res/Para[2])<Para[1],0,abs(res/Para[2] )-Para[1])
    
    sum(log(Para[2])+log(2*(1+Para[1]))+Llog)
    
  }
  
  optimun<-optim(c(10,2),epiloss,method="L-BFGS-B",lower=c(0.00001,0.01))$par
  
  RobustEps<-optimun[1]
  
  Scale<-optimun[2]
  
  scale_y_wholetrain<-y_wholetrain/Scale
  
  Recom_scale_C<-max(abs(mean(scale_y_wholetrain)+3*sd(scale_y_wholetrain)),abs(mean(scale_y_wholetrain)-3*sd(scale_y_wholetrain)))
  
  OptimalSVR<-svm(x_wholetrain,scale_y_wholetrain,type='eps-regression',kernel="radial",scale=FALSE,epsilon=RobustEps, cost=Recom_scale_C, gamma=Gamma_Value)
  
  Update_residuals<-Scale*OptimalSVR$residuals
  
  ##recalculate two parameters
  
  epiloss1<-function(Para)
    
  {
    res<-Update_residuals
    
    Llog<-ifelse(abs(res/Para[2])<Para[1],0,abs(res/Para[2] )-Para[1])
    
    sum(log(Para[2])+log(2*(1+Para[1]))+Llog)
    
  }
  
  
  optimun<-optim(c(10,2),epiloss1,method="L-BFGS-B",lower=c(0.00001,0.01))$par
  
  RobustEps<-optimun[1]
  
  Scale<-optimun[2]
  
  #print(cbind(RobustEps,Scale))
  
  scale_y_wholetrain<-y_wholetrain/Scale
  
  Recom_scale_C<-max(abs(mean(scale_y_wholetrain)+3*sd(scale_y_wholetrain)),abs(mean(scale_y_wholetrain)-3*sd(scale_y_wholetrain)))
  
  OptimalSVR<-svm(x_wholetrain,scale_y_wholetrain,type='eps-regression',kernel="radial",scale=FALSE,epsilon=RobustEps, cost=Recom_scale_C, gamma=Gamma_Value)
  
  TestPreds<-predict(OptimalSVR,x_test)*Scale
  
  Y_Test<-DataSet1[TestIndex,ResponseIndex]
  
  New_RMSE<-sqrt(mean((Y_Test-TestPreds)^2))
  
  TuningResults<-cbind(TuningResults,rbind(RobustEps, Scale, New_RMSE))
  
}  

tt3<-Sys.time()

print(tt3-tt2)

print(TuningResults)

print(CV_Results)


mean(TuningResults[3,])
sd(TuningResults[3,])
mean(CV_Results[2,])
sd(CV_Results[2,])

write.csv(CV_Results,"C:/Users/n10141065/Queensland University of Technology/You-Gan Wang - RyanWu(Jinran)/TNNLS Revised/TNNLS revised iii/Case/nucase_chosen/housing_nucv.csv")

#####Lin

LinResults<-c()

tt5<-Sys.time()

for (k in c(1:10)){
  
  TestIndex<-c(((k-1)*FolderSize+1):(k*FolderSize))
  
  TestSet<-DataSet1[TestIndex,]
  
  WholeTrainSet<-DataSet1[-TestIndex,]
  
  x_wholetrain<-WholeTrainSet[,1:(ResponseIndex-1)]
  
  y_wholetrain<-WholeTrainSet[,ResponseIndex]
  
  x_test<-TestSet[,1:(ResponseIndex-1)]
  
  y_test<-TestSet[,ResponseIndex]
  
  LinSVR<-svm(x_wholetrain,y_wholetrain,type='eps-regression',kernel="radial")
  
  TestPreds<-predict(LinSVR,x_test)
  
  Lin_RMSE<-sqrt(mean((y_test-TestPreds)^2))
  
  LinResults<-cbind(LinResults,Lin_RMSE)
  
}  


tt6<-Sys.time()

##########Ma

MaResults<-c()

for (k in c(1:10)){
  
  TestIndex<-c(((k-1)*FolderSize+1):(k*FolderSize))
  
  TestSet<-DataSet1[TestIndex,]
  
  WholeTrainSet<-DataSet1[-TestIndex,]
  
  x_wholetrain<-WholeTrainSet[,1:(ResponseIndex-1)]
  
  y_wholetrain<-WholeTrainSet[,ResponseIndex]
  
  x_test<-TestSet[,1:(ResponseIndex-1)]
  
  y_test<-TestSet[,ResponseIndex]
  
  LinSVR<-svm(x_wholetrain,y_wholetrain,type='eps-regression',kernel="radial")
  
  Ma_C<-max(abs(mean(y_wholetrain)+3*sd(y_wholetrain)),abs(mean(y_wholetrain)-3*sd(y_wholetrain)))
  
  obs<-length(y_wholetrain)
  
  Ma_Eps<-3*sd(LinSVR$residuals)*sqrt(log(obs)/obs)
  
  MaSVR<-svm(x_wholetrain,y_wholetrain,type='eps-regression',kernel="radial", scale=FALSE, epsilon =  Ma_Eps, cost= Ma_C, gamma=Gamma_Value)
  
  TestPreds<-predict(MaSVR,x_test)
  
  Ma_RMSE<-sqrt(mean((y_test-TestPreds)^2))
  
  MaResults<-cbind(MaResults,Ma_RMSE)
  
} 

tt7<-Sys.time()

print(cbind(LinResults,mean(LinResults),sd(LinResults)))

print(cbind(MaResults,mean(MaResults),sd(MaResults))) 

mean(LinResults)

print(tt6-tt5)

print(tt7-tt6)