library(data.table)

result[,`:=`(dif=maxPlace-numGroups,winPlace=as.numeric(winPlace))]
result[dif==0,addedrank:=0]


#"normal-duo", "duo"
k11=0.21997831116930705
k12=0.7800216888306929
mu11=7.148308359782472
mu12=27.223217635025645
sig11=3.466429487643154
sig12=6.514990777110895
#"normal-duo-fpp", "duo-fpp"
k21=0.1
k22=0.9
mu21=6.660831654111273
mu22=28.77210279901127
sig21=2.86201454356337
sig22=6.784061506898811
#"solo","normal-solo"
k31=0.2007918638324044
k32=0.7992081361675956
mu31=9.016640962397442
mu32=32.19306597584008
sig31=3.8218631800981773
sig32=25.224725313115634
#"solo-fpp", "normal-solo-fpp"
k41=0.21713342445721048
k42=0.7828665755427896
mu41=21.390694680773535
mu42=63.69110983442608
sig41=8.48551730041817
sig42=12.41165110642501
#"squad", "normal-squad","flaretpp"
k51=0.07
k52=0.93
mu51=3.1532474986839456
mu52=13.081487058680898
sig51=1.2138821257639176
sig52=4.90649709081083
#"squad-fpp", "normal-squad-fpp"
k61=0.030626102758928537
k62=0.9693738972410716
mu61=2.6283929806089237
mu62=14.748671653335636
sig61=0.6854741457076547
sig62=4.798434331315498
#"flarefpp","crashtpp"
#uniform distribution

#crashfpp
k71=0.12
k72=0.88
mu71=5.460957954273285
mu72=21.76733802960225
sig71=2.2290108994943805
sig72=5.512864275727925

try(result[matchType %in% c("squad-fpp", "normal-squad-fpp"),addedrank:=sum(winPlace>round((k61*rnorm(dif,mean=mu61,sd=sig61)+k62*rnorm(dif,mean=mu62,sd=sig62)))),by=Id])
try(result[matchType %in% c("solo-fpp", "normal-solo-fpp"),addedrank:=sum(winPlace>round((k41*rnorm(dif,mean=mu41,sd=sig41)+k42*rnorm(dif,mean=mu42,sd=sig42)))),by=Id])
try(result[matchType %in% c("normal-duo-fpp", "duo-fpp"),addedrank:=sum(winPlace>round((k21*rnorm(dif,mean=mu21,sd=sig21)+k22*rnorm(dif,mean=mu22,sd=sig22)))),by=Id])
try(result[matchType %in% c("squad", "normal-squad","flaretpp"),addedrank:=sum(winPlace>round((k51*rnorm(dif,mean=mu51,sd=sig51)+k52*rnorm(dif,mean=mu52,sd=sig52)))),by=Id])
try(result[matchType %in% c("solo","normal-solo"),addedrank:=sum(winPlace>round((k31*rnorm(dif,mean=mu31,sd=sig31)+k32*rnorm(dif,mean=mu32,sd=sig32)))),by=Id])
try(result[matchType %in% c("normal-duo", "duo"),addedrank:=sum(winPlace>round((k11*rnorm(dif,mean=mu11,sd=sig11)+k12*rnorm(dif,mean=mu12,sd=sig12)))),by=Id])
try(result[matchType %in% c("crashfpp"),addedrank:=sum(winPlace>round((k71*rnorm(dif,mean=mu71,sd=sig71)+k72*rnorm(dif,mean=mu72,sd=sig72)))),by=Id])
for(i in 1:max(result$dif)){
  try(result[matchType %in% c("flarefpp","crashtpp") & dif==i,addedrank:=as.double(cut(winPlace,(i+1),label=F)-1)])
}

result<-result[,winPlacePerc:=(winPlace+addedrank-1)/(maxPlace-1),by=matchId]

result<-result[,.(Id,winPlacePerc)]

fwrite(result,"result.csv")