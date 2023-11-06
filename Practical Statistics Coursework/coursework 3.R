sweden_ins_data<-read.table("sweden_ins_data.txt",header = TRUE)


#1
plot(sweden_ins_data$claims,sweden_ins_data$payment,main = "The Sweden Insurance Data",xlab = "Claims",ylab = "Payment")


#2
sweden_lm<-function(x,y){
  n=length(x)
  df=n-2
  x_bar=sum(x)/n
  y_bar=sum(y)/n
  Sxx=sum(x^2)-(sum(x))^2/n
  Sxy=sum(x*y)-sum(x)*sum(y)/n
  beta_hat=Sxy/Sxx
  alpha_hat=y_bar-beta_hat*x_bar
  ei_hat=y-alpha_hat-beta_hat*x
  my_list<-list(c(alpha_hat,beta_hat),c(x,y),ei_hat,df)
  return(my_list)
}

sweden_lm(sweden_ins_data$claims,sweden_ins_data$payment)


#3
#(i)
mod1<-lm(sweden_ins_data$payment~sweden_ins_data$claims)
summary(mod1)
abline(mod1)

#(ii)
output=sweden_lm(sweden_ins_data$claims,sweden_ins_data$payment)
ei_hat_list=output[3]
ei_hat=unlist(ei_hat_list)
sse=sum((ei_hat)^2)
n=63
error_variance=sse/(n-2)
error_variance

#(iii)
anova(mod1)

#(iv)
n<-length(sweden_ins_data$payment)
n
summ1<-summary(mod1)$coefficients
est.80<-summ1[1,1]+summ1[2,1]*80
est.80
Sxx<-sum((sweden_ins_data$claims-mean(sweden_ins_data$claims))^2)
Sxx
est.se<-sqrt(error_variance)
est.se
c11<-est.80-qt(0.975,df=n-2)*est.se*sqrt((1/n)+(80-mean(sweden_ins_data$claims))^2/Sxx)
c12<-est.80+qt(0.975,df=n-2)*est.se*sqrt((1/n)+(80-mean(sweden_ins_data$claims))^2/Sxx)
c11;c12


#4
#a)
std.res1<-rstandard(mod1)
plot(mod1$fitted.values,std.res1,main="Standardised residuals vs fitted values")
abline(h=0)

#b)
plot(sweden_ins_data$claims,std.res1,main="Standardised residuals vs covariate values")
abline(h=0)


#5
sample.quantiles<-sort(std.res1)
pn=0
for (k in 1:n) {
  pn[k]=((k-3/8)/(n+1/4))
}
theoretical.quantiles<-qnorm(pn)
windows()
plot(theoretical.quantiles,sample.quantiles,main="Normal Q-Q plot of Sweden Insurance data (n=63)")
abline(a=mean(std.res1), b=sd(std.res1), col="green")


#6
ks.test(x=std.res1, y=pnorm, mean=0, sd=1,alternative = c("two.sided"))

std.res.ord=sort(std.res1)
x.ecdf<-(1:n)/n
y<-pnorm(std.res.ord, mean=0, sd=1)
diff1=x.ecdf-y
md1=max(diff1)
dn.plus=max(md1,0)
dn.plus
x.KS1=std.res.ord[dn.plus==diff1]
diff2=y-x.ecdf
md2=max(diff2)
md2=md2+(1/n)
dn.minus=max(md2,0)
dn.minus
x.KS2=std.res.ord[dn.minus==diff2+(1/n)]
KSstat=max(dn.minus, dn.plus)
KSstat
if(dn.minus < dn.plus) x.KSstat=x.KS1
if(dn.minus > dn.plus) x.KSstat=x.KS2
x.KSstat


#7
min(std.res1)
max(std.res1)

plot.ecdf(std.res1,main="Ecdf and N(0, 1) cdf for the Sweden Insurance data")
x.grid=seq(from=-2.5,to=2.5,length.out=500)
x.grid.pnorm<-pnorm(x.grid,mean = 0,sd=1)
lines(x.grid,x.grid.pnorm,col="green",type="l")
points(x.KSstat,0,pch=15,col="red")
abline(v=x.KSstat,col="blue")


#8
test.sim.fun<-function(N,Nmean,Nsd){
B=2000
test.sim=0
for (i in 1:B) {
  ysim<-rnorm(N,Nmean,Nsd)
  test.sim[i]<-ks.test(x=ysim, y=pnorm,Nmean,Nsd,alternative = c("two.sided"))$statistic
}
return(test.sim)
}

hist(test.sim.fun(63,0,1),freq = F,xlim=c(0,0.25),main = "Histogram of simulated test statistic values (B=2000) with N(0,1) pdf",xlab="Simulation size is B=2000;the curve is the kde of the distribution")
lines(density(test.sim.fun(63,0,1)), col="red")

quantile(test.sim.fun(63,0,1),0.95)