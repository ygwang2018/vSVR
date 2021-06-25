library(scales)
library(e1071)

##Codes for nested cross-validation

DataSet<-read.csv('/home/n10141065/espsiloncase_chosen/case3//housing.csv')

#Copy without scale

DataSet1<-DataSet

ResponseIndex<-dim(DataSet)[2]

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
  Average_Valid_Eps_RMSE<-array()
  
  t1<-1
  
  SettingSequences<-c()
  
  for (C_cv_Value in 2^c(-10:19)){
    
    for (Gamma_cv_Value in 2^c(-4:9)){
      
      for (Eps_cv_Value in 2^c(-9:-2)){
  
    Valid_Eps_RMSE<-array()
    
    SettingSequences<-rbind(SettingSequences,cbind(C_cv_Value,Gamma_cv_Value,Eps_cv_Value))
    
    t<-1
  
       for (k1 in InnerK){
    
           ValidateIndex<-c(((k1-1)*FolderSize+1):(k1*FolderSize))
    
           ValidateSet<-DataSet[ValidateIndex,]
    
           TrainSet<-DataSet[-rbind(as.matrix(TestIndex),as.matrix(ValidateIndex)),]
           
           x_train<-TrainSet[,1:(ResponseIndex-1)]
             
           y_train<-TrainSet[,ResponseIndex]   
           
           EpsSVR<-svm(x_train,y_train,type='eps-regression',kernel="radial",scale=FALSE, epsilon =Eps_cv_Value, 
                      cost=C_cv_Value, gamma=Gamma_cv_Value)
           
           x_valid<-ValidateSet[,1:(ResponseIndex-1)]
           
           y_valid<-ValidateSet[,ResponseIndex] 
           
           y_valid_pred<-predict(EpsSVR, x_valid)
           
           Valid_Eps_RMSE[t]<-sqrt(mean((y_valid-y_valid_pred)^2))
           
           t<-t+1
    
       }
    
    Average_Valid_Eps_RMSE[t1]<-mean(Valid_Eps_RMSE)
    
    t1<-t1+1

      }
      
    }
    
  }

  
  Optimal_Sequence<-SettingSequences[(rank(Average_Valid_Eps_RMSE)[3360]),]
  
  #C_cv_Value,Gamma_cv_Value,Eps_cv_Value
  
  print(Optimal_Sequence)
  
  Whole_Train_Set<-DataSet[-TestIndex,]
  
  x_whole_train<-Whole_Train_Set[,1:(ResponseIndex-1)]
  
  y_whole_train<-Whole_Train_Set[,ResponseIndex]  
  
  Optimal_EpsSVR<-svm(x_whole_train,y_whole_train,type='eps-regression',kernel="radial",scale=FALSE, espsilon=  Optimal_Sequence[3], 
             cost=Optimal_Sequence[1], gamma=Optimal_Sequence[2])
  
  x_test<-TestSet[,1:(ResponseIndex-1)]
  
  Optimal_Preds<-predict(Optimal_EpsSVR,x_test)*(MaxY-MinY)+MinY
  
  Y_Test<-DataSet1[TestIndex,ResponseIndex]
  
  Optimal_RMSE<-sqrt(mean((Y_Test-Optimal_Preds)^2))
  
  CV_Results<-rbind(CV_Results,cbind(Optimal_Sequence,Optimal_RMSE))
  
}

tt2<-Sys.time()

GS_Time<-tt2-tt1

print(GS_Time)

write.csv(CV_Results,"CV_Results.csv")


###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
# Now, we will conduct our new algorithm with working likelihood method.
###################################################################################
###################################################################################  

InputScale<-scale(DataSet1[,1:(ResponseIndex-1)])  

DataSet<-cbind(InputScale,DataSet1[,ResponseIndex])

TuningResults<-c()

tt3<-Sys.time()

for (k in c(1:10)){
  
  Gamma_Value<-1/(0.3*(ResponseIndex-1))
  
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
  
  scale_y_wholetrain<-y_wholetrain/Scale
  
  Recom_scale_C<-max(abs(mean(scale_y_wholetrain)+3*sd(scale_y_wholetrain)),abs(mean(scale_y_wholetrain)-3*sd(scale_y_wholetrain)))
  
  OptimalSVR<-svm(x_wholetrain,scale_y_wholetrain,type='eps-regression',kernel="radial",scale=FALSE,epsilon=RobustEps, cost=Recom_scale_C, gamma=Gamma_Value)
  
  TestPreds<-predict(OptimalSVR,x_test)*Scale
  
  Y_Test<-DataSet1[TestIndex,ResponseIndex]
  
  New_RMSE<-sqrt(mean((Y_Test-TestPreds)^2))
  
  TuningResults<-cbind(TuningResults,rbind(RobustEps, Scale, New_RMSE))
  
}  

tt4<-Sys.time()

Op_Time<-tt4-tt3

print(Op_Time)

write.csv(TuningResults,"TuningResults.csv")


