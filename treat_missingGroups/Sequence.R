library(data.table)

test_dt <- fread("test_V2.csv")
subtest<-copy(test_dt[,.(matchId,groupId,kills,killPlace,winPlacepPerc)])
gk <- paste0("killgroup",0:58)
testsortall_dt <- subtest[,.(matchId,groupId,kills,killPlace,winPlacepPerc)]
s=Sys.time()
for( i in (0:58)){
  kk=testsortall_dt$kills==i
  if(sum(kk)!=0){
    temp<-testsortall_dt[,.(matchId,groupId,kills,killPlace)][kk,]
    temp<-temp[!duplicated(temp[,groupId]),][order(killPlace),gk[i+1]:=groupId,by=matchId]
    testsortall_dt<-merge(testsortall_dt, temp,all=T)
  }
}