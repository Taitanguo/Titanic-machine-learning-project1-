
library(lattice)
library(MASS) 

mypath <- "/Users/chao/Desktop/machine_learning_2016spring/projects/1_titanic/pics/"
setwd('/Users/chao/Desktop/machine_learning_2016spring/projects/1_titanic/')

df <- read.csv(file ="orig_data.csv")

df <- sapply(df, function(y) { if (y == '') na else y })
dfColName <- colnames(df)

# generate list of colnames for NA datafrome
for(i in c(2:15)){
  colName <- dfColName[i]
  cat('"')
  cat(colName)
  cat('","')
  cat(colName)
  cat('",')
}

# generating the count of missing value for each column
missingCount <- NULL
count <- 0
for(i in c(2:15)){
  if (is.na(sum(df[,i]==''))){
    count <- count
  } else {
    count = count + sum(df[,i]=='')
  }
  if (is.na(sum(is.na.data.frame(df[,i])))) {
    count <- count
  } else {
    count <- count + sum(is.na.data.frame(df[,i]))
  }
  missingCount<-c(missingCount,1309-count,count)
  count <- 0
}

# data frame for missing value stats
stats <- data.frame(  unit  = factor(c("pclass","pclass","survived","survived","name","name",
                                       "sex","sex","age","age","sibsp","sibsp","parch","parch",
                                       "ticket","ticket","fare","fare","cabin","cabin",
                                       "embarked","embarked","boat","boat","body","body",
                                       "home.dest","home.dest")),   
                      counts = factor(c('data','missing','data','missing','data','missing',
                                        'data','missing','data','missing','data','missing',
                                        'data','missing','data','missing','data','missing',
                                        'data','missing','data','missing','data','missing',
                                        'data','missing','data','missing')),   
                      ratio =  missingCount)

library(ggplot2)
imgpath<-file.path(mypath,paste(1,"missing_value_stats",".png",sep=""))
png(file=imgpath,units="in",width=15,height=5,res=300)
ggplot(data=stats, aes(x=unit, y=ratio, fill=counts)) +
  geom_bar(stat="identity") +
  scale_x_discrete(limits=colnames(df)[2:15]) +
  ggtitle ( "Missing value stats")
dev.off()

dfIm <- read.csv(file = "imputed_data.csv")

imgpath<-file.path(mypath,paste(2,"boxplot_pclass_survial",".png",sep=""))
png(file=imgpath,units="in",width=15,height=5,res=300)
bwplot(as.factor(survived)~jitter(pclass), data = dfIm, main = "Box Plots of pclass by survival or not", xlab = "pclass", ylab = "survival")
dev.off()

imgpath<-file.path(mypath,paste(3,"boxplot_sex_survial",".png",sep=""))
png(file=imgpath,units="in",width=15,height=5,res=300)
bwplot(as.factor(sex)~jitter(survived), data = dfIm, main = "Box Plots of gender by survival or not", xlab = "pclass", ylab = "survival")
dev.off()

imgpath<-file.path(mypath,paste(4,"boxplot_age_survial",".png",sep=""))
png(file=imgpath,units="in",width=15,height=5,res=300)
bwplot(as.factor(survived)~jitter(age), data = dfIm, main = "Box Plots of age by survival or not", xlab = "pclass", ylab = "survival")
dev.off()

imgpath<-file.path(mypath,paste(5,"boxplot_sibsp_survial",".png",sep=""))
png(file=imgpath,units="in",width=15,height=5,res=300)
bwplot(as.factor(survived)~jitter(sibsp), data = dfIm, main = "Box Plots of sibsp by survival or not", xlab = "pclass", ylab = "survival")
dev.off()

imgpath<-file.path(mypath,paste(6,"boxplot_parch_survial",".png",sep=""))
png(file=imgpath,units="in",width=15,height=5,res=300)
bwplot(as.factor(survived)~jitter(parch), data = dfIm, main = "Box Plots of parch by survival or not", xlab = "pclass", ylab = "survival")
dev.off()

imgpath<-file.path(mypath,paste(7,"boxplot_fare_survial",".png",sep=""))
png(file=imgpath,units="in",width=15,height=5,res=300)
bwplot(as.factor(survived)~jitter(fare), data = dfIm, main = "Box Plots of fare by survival or not", xlab = "pclass", ylab = "survival")
dev.off()