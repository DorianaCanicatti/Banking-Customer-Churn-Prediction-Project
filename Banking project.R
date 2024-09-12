rm(list=ls())
library(rstudioapi)
current_path<-getActiveDocumentContext()$path
setwd(dirname(current_path))
print(getwd())

library(GGally)
library(tidyverse)
library(ggplot2)

data<-read.csv("Banking_customer_churn.csv")
View(data)
data<-as_tibble(data)
data<-data[,-c(1,2,3)]

######## DATA QUALITY ###########
# Vediamo se ci sono NA
colSums(is.na(data))
# Non ci sono NA

# Vediamo se ci sono righe duplicate
duplicated(data)
# Andiamo a vedere quale riga è duplicata
which(duplicated(data))
# Non ci sono righe duplicate


################## ANALISI ESPLORATIVA ###################

# Vediamo la frequenza assoluta di "Exited"
tab<-data$Exited%>%table()
tab
# Vediamo la frequenza relativa di "Exited"
percentages<-tab%>%prop.table()%>%round(3)*100
percentages

# Raprresentiamo la frequenza relativa di Exited con un grafico a torta
dev.new()
txt<-paste0(names(tab), '\n',percentages, '%')
txt
colors <- c("violet","orange")
pie(tab, labels = txt, col = colors)
legend("topright", legend = names(tab), fill = colors, title = "Exited")

#### Facciamo UNDERSAPMPLING DELLA CLASSE Exited = 0
library(caret)
library(e1071)
library(ROSE)
set.seed(123)
# Definiamo il rapporto di undersampling per ogni classe
undersample_ratio <- 0.52  # Ad esempio, undersampling al 52%

# Eseguiamo undersampling
undersampled_data <- ovun.sample(Exited ~ ., data = data, method = "under", N = undersample_ratio * length(which(data$Exited == 0)))

# Visualizziamo il conteggio delle classi nel dataset undersampled
new_data<-undersampled_data$data
table(new_data$Exited)
percentages<-new_data$Exited%>%table()%>%prop.table()%>%round(3)*100
percentages
# Il dataset a questo punto risulta bilanciato

# BOXPLOT
dev.new()
par(mfrow=c(2,3))
boxplot(new_data$CreditScore, col="violet", main="CreditScore")
boxplot(new_data$Age, col="yellow", main="Age")
boxplot(new_data$Tenure, col="orange", main="Tenure")
boxplot(new_data$Balance, col="red", main="Balance")
boxplot(new_data$NumOfProducts, col="pink", main="NumOfProducts")
boxplot(new_data$EstimatedSalary, col="lightgreen", main="EstimatedSalary")

# Si nota la presenza di outlier per la variabile Age, tuttavia poichè ci sono numerose osservazioni con valori tra i 60 e gli 80 anni
# decidiamo di studiare il comportamento anche di questa categoria di consumatori. Per quanto riguarda l'outlier di NumOfProducts
# essendo l'unico, quest'ultimo non va ad influenzare i valori della variabile.

####### BARPLOT #######
par(mfrow=c(1,1))
# Contiamo il numero di uscite per genere
exit_counts <- table(new_data$Exited, new_data$Gender)
dev.new()
# Creiamo un barplot
barplot(exit_counts, 
        beside = TRUE, 
        col = c("violet","orange"), 
        legend = TRUE,
        names.arg = c("Female", "Male"), 
        xlab = "Gender",
        ylab = "Count",
        main = "Gender vs. Exited Barplot")
total_counts <- colSums(exit_counts)
text(x = barplot(exit_counts, 
                 beside = TRUE, 
                 col = c("violet","orange"), 
                 legend = TRUE,
                 names.arg = c("Female", "Male"), 
                 xlab = "Gender",
                 ylab = "Count",
                 main = "Gender vs. Exited Barplot"),
      y = exit_counts - 0.06 * total_counts,
      labels = exit_counts,
      pos = 3,
      cex = 0.8,
      col = "black")
# Le donne tendono ad uscire più degli uomini

# Contiamo il numero di uscite per HasCrCard
exit_counts <- table(new_data$Exited, new_data$HasCrCard)
dev.new()
# Creiamo un barplot
barplot(exit_counts, 
        beside = TRUE, 
        col = c("violet","orange"), 
        legend = TRUE, 
        names.arg = c("0", "1"),
        xlab = "HasCrCard", 
        ylab = "Count", 
        main = "HasCrCard vs. Exited Barplot")
total_counts <- colSums(exit_counts)
text(x = barplot(exit_counts, 
                 beside = TRUE, 
                 col = c("violet","orange"), 
                 legend = TRUE,
                 names.arg = c("0", "1"), 
                 xlab = "HasCrCard",
                 ylab = "Count",
                 main = "HasCrCard vs. Exited Barplot"),
     y = exit_counts - 0.1 * total_counts,
     labels = exit_counts,
     pos = 3,
     cex = 0.8,
     col = "black")
# La variabile HasCrCard sembra non essere significativa per fare una distinzione tra chi esce o non esce dalla banca

# Contiamo il numero di uscite per IsActiveMember
exit_counts <- table(new_data$Exited, new_data$IsActiveMember)
dev.new()
# Creiamo un barplot
barplot(exit_counts, 
        beside = TRUE, 
        col = c("violet","orange"), 
        legend = TRUE, 
        names.arg = c("0", "1"),
        xlab = "IsActiveMember", 
        ylab = "Count", 
        main = "IsActiveMember vs. Exited Barplot") 
total_counts <- colSums(exit_counts)
text(x = barplot(exit_counts, 
                 beside = TRUE, 
                 col = c("violet","orange"), 
                 legend = TRUE,
                 names.arg = c("0", "1"), 
                 xlab = "IsActiveMember",
                 ylab = "Count",
                 main = "IsActiveMember vs. Exited Barplot"),
     y = exit_counts - 0.06 * total_counts,
     labels = exit_counts,
     pos = 3,
     cex = 0.8,
     col = "black")
# Si nota che i membri più attivi tendono ad uscire meno dalla banca e viceversa.

# Contiamo il numero di uscite per Geography
exit_counts <- table(new_data$Exited, new_data$Geography)
dev.new()
# Creiamo un barplot
barplot(exit_counts, 
        beside = TRUE, 
        col = c("violet","orange"), 
        legend = TRUE, 
        names.arg = c("France", "Germany", "Spain" ),
        xlab = "Geography", 
        ylab = "Count", 
        main = "Geography vs. Exited Barplot") 
total_counts <- colSums(exit_counts)
text(x = barplot(exit_counts, 
                 beside = TRUE, 
                 col = c("violet","orange"), 
                 legend = TRUE,
                 names.arg = c("France", "Germany", "Spain" ), 
                 xlab = "Geography",
                 ylab = "Count",
                 main = "Geography vs. Exited Barplot"),
     y = exit_counts - 0.1 * total_counts,
     labels = exit_counts,
     pos = 3,
     cex = 0.8,
     col = "black")
# In Francia c'è una maggiore tendenza a non uscire dalla propria banca e anche in Spagna.
# Mentre in Germania risulta più evidente che c'è una tendenza maggiore ad uscire dalla banca.

############ DENSITY FUNCTION ###########
# CreditScore vs Exited
ggplot(new_data, aes(x = CreditScore, color = as.factor(Exited))) +
  geom_density(alpha=0.5) +
  scale_fill_manual(values = c("0" = "violet", "1" = "orange")) +
  theme_minimal() + ylab("Density") + labs(color="Exited") + ggtitle("Density function CreditScore vs Exited")
# Il fatto di uscire o non uscire dalla banca sembra non essere influenzato dal CreditScore.

# Tenure vs Exited
ggplot(new_data, aes(x = Tenure, color = as.factor(Exited))) +
  geom_density(alpha=0.5) +
  scale_fill_manual(values = c("0" = "violet", "1" = "orange")) +
  theme_minimal() + ylab("Density") + labs(color="Exited") + ggtitle("Density function Tenure vs Exited")
# Si nota che fino a quasi 4 anni (Tenure) l'individuo tende ad uscire dalla banca mentre per valori maggiori di 6 anni
# tende a rimanerci.

# Balance vs Exited
ggplot(new_data, aes(x = Balance, color = as.factor(Exited))) +
  geom_density(alpha=0.5) +
  scale_fill_manual(values = c("0" = "violet", "1" = "orange")) +
  theme_minimal() + ylab("Density") + labs(color="Exited") + ggtitle("Density function Balance vs Exited")
# Gli individui che hanno valore di Balance tra 75.000 (circa) e 150.000 tendono maggiormente ad uscire dalla banca
# Tuttavia si tratta di una percentuale molto piccola.

# EstimatedSalary vs Exited
ggplot(new_data, aes(x = EstimatedSalary, color = as.factor(Exited))) +
  geom_density(alpha=0.5) +
  scale_fill_manual(values = c("0" = "violet", "1" = "orange")) +
  theme_minimal() + ylab("Density") + labs(color="Exited") + ggtitle("Density function EstimatedSalary vs Exited")
# La variabile EstimatedSalary sembra non essere così rilevante per discriminare chi esce o meno dalla banca.

# NumOfProducts vs Exited
ggplot(new_data, aes(x = NumOfProducts, color = as.factor(Exited))) +
  geom_density(alpha=0.5) +
  scale_fill_manual(values = c("0" = "violet", "1" = "orange")) +
  theme_minimal() + ylab("Density") + labs(color="Exited") + ggtitle("Density function NumOfProducts vs Exited")
# Un individuo che possiede 2 prodotti è più probabile che rimanga nella banca. Per 1 prodotto è indifferente, mentre
# per 3 o 4 prodotti è poco più probabile che si lasci la banca.

# Age vs Exited
ggplot(new_data, aes(x = Age, color = as.factor(Exited))) +
  geom_density(alpha=0.5) +
  scale_fill_manual(values = c("0" = "violet", "1" = "orange")) +
  theme_minimal() + ylab("Density") + labs(color="Exited") + ggtitle("Density function Age vs Exited")
# Tra 40 e 70 anni c'è una prevalenza di chi esce dalla banca mentre con un'età inferiore a 40 anni è più probabile
# che si rimanga nella banca

# CORRELAZIONI
new_data$Gender<-ifelse(new_data$Gender == "Male", 1, 0)
# Trasformiamo Geography in una variabile factor numerica ("France"=0; "Germany"=1; "Spain"=2)
new_data$Geography<-as.factor(new_data$Geography)
new_data$Geography<-ifelse(new_data$Geography == "France", 0, ifelse(new_data$Geography == "Germany", 1, 2 ))
dev.new()
matrcorr=cor(new_data)

#library(ggcorrplot)
#ggcorrplot(matrcorr)
library(corrplot)
corrplot(matrcorr, method = "color", 
         addCoef.col = "black", # colore dei coefficienti
         number.cex = 0.7)

# Dalla matrice di correlazione si evince che nessuna coppia di variabili è fortemente correlata
# ciò significa che c'è assenza di multicollinearità e di conseguenza utilizzare la PCA come tecnica di 
# dimensionality reduction non porterebbe ad una riduzione considerevole del numero di variabili
# Le uniche correlazione degne di nota sono quelle tra NumOfProducts e Balance, IsActiveMember ed Exited e Age ed Exited

## SUMMARY STATISTICS
summary(new_data)
# La variabile Gender ha una media che si assesta a 0.51, ciò significa che ci sono (per poco) più uomini che donne nel nostro dataset.
# La variabile HasCRCard ha una media intorno a 0.71, ciò significa che più della metà delle persone ha la carta di credito.
# La variabile IsActiveMember ha una media pari a 0.46, ciò significa che la maggior degli individui non è un membro attivo nella banca
# La variabile Exited ha una media pari a 0.49, ciò significa che la percentuale di individui che nel dataset rimangono o escono dalla banca è la stessa.

######## PCA ##########
pc<-prcomp(new_data,scale=TRUE)
summary(pc)
# Si raggiunge il 70% della varianza spiegata alla settima componente principale
## rappresentiamo percentuale varianza spiegata
library(factoextra)
dev.new()
fviz_eig(pc, choice = "variance", ncp=11, barfill=rgb(1, 0.8, 0.8))

########### REGRESSIONE LOGISTICA #############

# Dividiamo il set di dati in training e test set
library(caTools)
set.seed(123)
sample_data<-sample.split(new_data, SplitRatio = 0.75)
train_data<-subset(new_data, sample_data==TRUE)
test_data<-subset(new_data, sample_data==FALSE)
prop.table(table(train_data$Exited))
prop.table(table(test_data$Exited))

library(lmtest)
mod0=glm( Exited ~ CreditScore + Geography + Gender + Age + Tenure + Balance + NumOfProducts + HasCrCard + IsActiveMember + EstimatedSalary, family = binomial( link = logit ) , data = train_data)
summary(mod0)

mod1=glm( Exited ~ CreditScore + Geography + Gender + Age + Tenure + Balance + NumOfProducts + HasCrCard + IsActiveMember, family = binomial( link = logit ) , data = train_data)
summary(mod1)
lrtest(mod0,mod1) # accettiamo H0 ossia il modello con meno regressori (modello 1)

mod2=glm( Exited ~ CreditScore + Geography + Gender + Age + Tenure + Balance + HasCrCard + IsActiveMember, family = binomial( link = logit ) , data = train_data)
summary(mod2)
lrtest(mod1,mod2) # accettiamo H0 (modello 2 è più significativo)

mod3=glm( Exited ~ CreditScore + Gender + Age + Tenure + Balance + HasCrCard + IsActiveMember, family = binomial( link = logit ) , data = train_data)
summary(mod3)
lrtest(mod2,mod3) # accettiamo H0 (il modello 3 è più significativo)

mod4=glm( Exited ~ CreditScore + Gender + Age + Balance + HasCrCard + IsActiveMember, family = binomial( link = logit ) , data = train_data)
summary(mod4)
lrtest(mod3,mod4) # accettiamo H0 (il modello 4 è più significativo)

mod5=glm( Exited ~ CreditScore + Gender + Age + Balance + IsActiveMember,family = binomial( link = logit ) , data = train_data)
summary(mod5)

lrtest(mod4,mod5) # accettiamo H0 (il modello 5 è più significativo)

mod6=glm( Exited ~ Gender + Age + Balance + IsActiveMember,family = binomial( link = logit ) , data = train_data)
summary(mod6)
lrtest(mod5,mod6)

# MODELLO N. 5 è IL DEFINITIVO

# All'aumentare di CreditScore l'individuo tende a non uscire dalla banca
# La donna tende a cambiare banca più dell'uomo
# Andando avanti con l'età (Age) si tende ad uscire dalla banca
# All'aumentare di Balance (saldo) si tende ad uscire dalla banca
# Se l'individuo è membro attivo nella banca tende a rimanerci

# AIC pari a 3580.7 del modello finale. Nel modello 4 AIC era leggermente più basso (3580.5) ma scegliamo comunque il modello 5 per lrtest
# e perchè nel modello 4 è presente la variabile Geography che non è significativa quanto le altre ed inoltre dall'analisi esplorativa risultava
# poco rilevante ai fini dell'uscita o meno dalla banca.
# Inoltre scegliamo il modello 5 rispetto al modello 6 perchè AIC del modello 5 è più basso del modello 6 ed inoltre nel modello 5 è presente
# la variabile CreditScore che per il momento riteniamo opportuno tenere nel modello in quanto ha comunque una significatività non bassa.

# Vediamo se c'è overfitting
# Accuracy del training
predictions_train<-predict(mod5, train_data, type="response")

predictions.hat_train<-ifelse(predictions_train>0.5,1,0)

accuracy<-table(predictions.hat_train, train_data$Exited)
accuracy
sum(diag(accuracy))/sum(accuracy)*100 # accuracy=69.49%

# Accuracy del testing
predictions<-predict(mod5, test_data, type="response")

predictions.hat<-ifelse(predictions>0.5,1,0)

accuracy<-table(predictions.hat, test_data$Exited)
accuracy
sum(diag(accuracy))/sum(accuracy)*100

## ACCURACY = 69.14894 (% correttamente predetti)
## MISCLASSIFICATION ERROR RATE = 1-69.148946 = 0.30851054 (% non correttamente predetti)

# L'accuracy è del 69.1% quindi è buona ma il modello non è del tutto affidabile (non c'è overfitting)

### Grafici
## influential values
dev.new()
plot(mod5, which=4, id.n = 10)

library(car)
dev.new()
influencePlot(mod5)

summary(train_data)
# Dal grafico si nota che  l'osservazione n. 851 ha elevata Distanza di Cook. Infatti andando ad analizzarla
# si vede che presenta valori marginali per alcune variabili


############## LDA #############
### Vediamo come si distribuiscono singolarmente le variabili, se non sono normali proviamo a trasformarle tramite il logaritmo
plot(density(new_data$CreditScore)) #si
plot(density(new_data$Geography)) # no
plot(density(log(new_data$Gender))) # si
plot(density(log(new_data$Age))) # si
plot(density(log(new_data$Tenure))) #no
plot(density(log(new_data$Balance))) #no
plot(density(log(new_data$NumOfProducts))) #no
plot(density(log(new_data$HasCrCard))) # si
plot(density(log(new_data$IsActiveMember))) # si
plot(density(log(new_data$EstimatedSalary))) # no
plot(density(log(new_data$Exited))) # si
# ALCUNE NON SONO NORMALI

new_data_lda=new_data[,c(1,3,4,6,9,11)]
new_data_lda$CreditScore=log(new_data_lda$CreditScore)
new_data_lda$Gender=log(new_data_lda$Gender)
new_data_lda$Age=log(new_data_lda$Age)
new_data_lda$Balance=log(new_data_lda$Balance)
new_data_lda$IsActiveMember=log(new_data_lda$IsActiveMember)
new_data_lda$Exited=log(new_data_lda$Exited)

## Mardia's test
## H0: p (predittori) si distribuiscono come una normale multivariata
## H1: p (predittori) NON distribuiti come una normale multivariata
library(QuantPsyc)
mult.norm(new_data_lda[new_data_lda$Exited==0,][,c(1:5)])$mult.test
mult.norm(new_data_lda[new_data_lda$Exited==1,][,c(1:5)])$mult.test
# Rifiutiamo H0
# l'assunzione della LDA NON è rispettata

####### NEAREST NEIGHBOR ####################
library(tidyverse)
library(mlbench)
library(caret)
library(caTools)
set.seed(123)
sample_data<-sample.split(new_data, SplitRatio = 0.75)
train_data<-subset(new_data, sample_data==TRUE)
test_data<-subset(new_data, sample_data==FALSE)
prop.table(table(train_data$Exited))
prop.table(table(test_data$Exited))

# Creiamo i k-folds fissando k=10
train_index=createFolds(train_data$Exited, k=10) 
train_index
knnFit=train_data%>%train(as.factor(Exited) ~ CreditScore + Gender + Age + Balance + IsActiveMember, method="knn", data=., preProcess = "scale", metric = "Accuracy", 
                         tuneGrid = data.frame(k=1:10), trControl=trainControl(method="cv", 
                                                                               indexOut = train_index))
knnFit

# Scelgliamo il modello con un vicino in base all'accuracy più alta
knnFit$finalModel # 1 nearest neighbor model

# Vediamo se c'è overfitting
# Accuracy del training
pr_knn_train=predict(knnFit, train_data)
pr_knn_train
# Otteniamo l'overall accuracy
confusionMatrix(pr_knn_train, reference=as.factor(train_data$Exited)) # 99.9% accuracy
#accuracy
sum(diag(table(pr_knn_train, as.factor(train_data$Exited))))/sum(table(pr_knn_train, as.factor(train_data$Exited)))
#####
# Lo utilizziamo per fare la nostra previsione
pr_knn=predict(knnFit, test_data)
pr_knn
# Otteniamo l'overall accuracy
confusionMatrix(pr_knn, reference=as.factor(test_data$Exited)) #65.9% accuracy
# Altro metodo per calcolare l'accuracy
sum(diag(table(pr_knn, as.factor(test_data$Exited))))/sum(table(pr_knn, as.factor(test_data$Exited)))
## Il modello non è molto buono poichè c'è overfitting, quindi si adatta troppo bene al training ma non al testing

##################### SUPPORT VECTOR MACHINE #####################################################
library(caTools)
set.seed(123)
sample_data<-sample.split(new_data, SplitRatio = 0.75)
train_data<-subset(new_data, sample_data==TRUE)
test_data<-subset(new_data, sample_data==FALSE)
prop.table(table(train_data$Exited))
prop.table(table(test_data$Exited))

svmFit=train_data%>%train(as.factor(Exited) ~ CreditScore + Gender + Age + Balance + IsActiveMember, method="svmLinear", 
                         data = ., trControl=trainControl(method="cv", indexOut = train_index)) 
svmFit 
svmFit$finalModel

## Accuracy per training (vediamo se c'è overfitting)
pr_svm_train=predict(svmFit, train_data)
sum(diag(table(pr_svm_train, as.factor(train_data$Exited)))/sum(table(pr_svm_train, as.factor(train_data$Exited)))) # 70.28%

## Accuracy per testing
pr_svm=predict(svmFit, test_data)
sum(diag(table(pr_svm, as.factor(test_data$Exited)))/sum(table(pr_svm, as.factor(test_data$Exited))))

# accuracy 70.03%
# Non c'è overfitting

############# DECISION TREE #############
library(rpart)
library(caTools)
set.seed(123)
sample_data<-sample.split(new_data, SplitRatio = 0.75)
train_data<-subset(new_data, sample_data==TRUE)
test_data<-subset(new_data, sample_data==FALSE)
prop.table(table(train_data$Exited))
prop.table(table(test_data$Exited))
# Trasformo Gender e IsActiveMember in factor
train_data$Gender<-as.factor(train_data$Gender)
train_data$IsActiveMember<-as.factor(train_data$IsActiveMember)
test_data$Gender<-as.factor(test_data$Gender)
test_data$IsActiveMember<-as.factor(test_data$IsActiveMember)
# tree construction
tree<-rpart(as.factor(Exited) ~ CreditScore + Gender + Age + Balance + IsActiveMember, data = train_data, control = rpart.control(cp=0.0001),
            parms = list(spli="gini"))
dev.new()
plotcp(tree)
# La dimensione dell'albero è pari a 10 (numero dei nodi terminali) quindi il numero degli split(livelli)=9
printcp(tree)
bestcp<-tree$cptable[which.min(tree$cptable[,"xerror"]), "CP"]
bestcp
# Chiediamo di farci vedere il cp che minimizza l'x error
# Tagliamo l'albero in corrispondenza del best cp ossia di un numero di livelli pari a 9
tree.pruned<-prune(tree, cp=bestcp)
tree.pruned 

library(caret)
library(rpart.plot)
library(rattle)

dev.new()
fancyRpartPlot(tree.pruned, main="Decision Tree", palettes = "PuRd")

# Vediamo se c'è overfitting
conf.matrix<-table(train_data$Exited, predict(tree.pruned, type = "class"))
rownames(conf.matrix)<-paste("Actual", rownames(conf.matrix), sep = ".")
colnames(conf.matrix)<-paste("Pred", colnames(conf.matrix), sep = ".")
conf.matrix
sum(diag(conf.matrix))/sum(conf.matrix)*100 #accuracy=73.04%
predict_unseen<-predict(tree.pruned, test_data, type = "class")
table_mat<-table(test_data$Exited, predict_unseen)
table_mat
sum(diag(table_mat))/sum(table_mat) # accuracy = 71.28%

# Non ci sono problemi di overfitting
# Infatti l'albero si adatta abbastanza bene sia al training sia al testing

# Trasformo nuovamente in numeric Gender e IsActiveMember
train_data$Gender<-as.numeric(train_data$Gender)
train_data$IsActiveMember<-as.numeric(train_data$IsActiveMember)
test_data$Gender<-as.numeric(test_data$Gender)
test_data$IsActiveMember<-as.numeric(test_data$IsActiveMember)

#################### RANDOM FOREST #############
# Proviamo a vedere se l'accuracy migliora tramite il metodo del random forest
library(randomForest)
library(caTools)
set.seed(123)
sample_data<-sample.split(new_data, SplitRatio = 0.75)
train_data<-subset(new_data, sample_data==TRUE)
test_data<-subset(new_data, sample_data==FALSE)
prop.table(table(train_data$Exited))
prop.table(table(test_data$Exited))
model<-randomForest(formula=as.factor(Exited) ~ CreditScore + Gender + Age + Balance + IsActiveMember, data=train_data, importance = TRUE)
model

conf.matrix<-table(train_data$Exited, model$predicted)
rownames(conf.matrix)<-paste("Actual", rownames(conf.matrix), sep = ".")
colnames(conf.matrix)<-paste("Pred", colnames(conf.matrix), sep = ".")
conf.matrix
sum(diag(conf.matrix))/sum(conf.matrix)*100 #accuracy=70.42%

predict_unseen<-predict(model, test_data, type = "class")
table_mat<-table(test_data$Exited, predict_unseen)
table_mat
sum(diag(table_mat))/sum(table_mat) # accuracy = 70.57%

# Si nota che l'accuracy del random forest è minore rispetto a quella del decision tree pertanto non è il miglior
# algoritmo per classification task per il nostro dataset.

############### NEURAL NETWORK #################

library(neuralnet)
library(caTools)
set.seed(123)
sample_data<-sample.split(new_data, SplitRatio = 0.75)
train_data<-subset(new_data, sample_data==TRUE)
test_data<-subset(new_data, sample_data==FALSE)
prop.table(table(train_data$Exited))
prop.table(table(test_data$Exited))

## min-max method
# ridimensioniamo i dati del train sull'intervallo 0-1
normalize<-function(x){
  return((x-min(x))/(max(x)-min(x)))
}

maxmindf_train<-as.data.frame(lapply(train_data, normalize))
str(maxmindf_train)

# Ridimensioniamo i dati del test sull'intervallo 0-1
maxmindf_test<-as.data.frame(lapply(test_data, normalize))
str(maxmindf_test)

# Facciamo alcune prove considerando di volta in volta un numero diverso di nodi negli strati di hidden
nn<-neuralnet(Exited ~ CreditScore + Gender + Age + Balance + IsActiveMember, data = maxmindf_train,
              hidden=c(3,1), linear.output = F, threshold = 0.01)
# Decidiamo di tenere questo modello poichè ha un errore minore (anche se accuracy più bassa di mezzo punto percentuale)
nn<-neuralnet(Exited ~ CreditScore + Gender + Age + Balance + IsActiveMember, data = maxmindf_train,
              hidden=c(2,1), linear.output = F, threshold = 0.01)

nn<-neuralnet(Exited ~ CreditScore + Gender + Age + Balance + IsActiveMember, data = maxmindf_train,
              hidden=c(3,2), linear.output = F, threshold = 0.01)

nn<-neuralnet(Exited ~ CreditScore + Gender + Age + Balance + IsActiveMember, data = maxmindf_train,
              hidden=c(2,2), linear.output = F, threshold = 0.01)
# Non converge nel numero massimo di step
nn<-neuralnet(Exited ~ CreditScore + Gender + Age + Balance + IsActiveMember, data = maxmindf_train,
              hidden=c(4,2), linear.output = F, threshold = 0.01)
# Non converge nel numero massimo di step
# Facciamo il grafico della rete neurale scelta
dev.new()
plot(nn)
nn$result.matrix

# Accuracy training
nn.results<-compute(nn, train_data)
results<-data.frame(actual=train_data$Exited, prediction=nn.results$net.result)
results
roundedresults<-sapply(results, round, digits=0)
roundedresultsdf<-data.frame(roundedresults)
# Confusion matrix
tab<-table(roundedresultsdf$actual, roundedresultsdf$prediction)
sum(diag(tab))/sum(tab)*100 #accuracy = 56.34%

# Accuracy testing
nn.results<-compute(nn, test_data)
results<-data.frame(actual=test_data$Exited, prediction=nn.results$net.result)
results
roundedresults<-sapply(results, round, digits=0)
roundedresultsdf<-data.frame(roundedresults)
# Confusion matrix
tab1<-table(roundedresultsdf$actual, roundedresultsdf$prediction)
sum(diag(tab1))/sum(tab1)*100 #accuracy = 58.86 %
## Non c'è overfitting, tuttavia l'accuracy del modello è troppo bassa per poterlo considerare un modello di prediction affidabile

#### Il modello che risulta migliore in termini di accuracy è quello del decision tree. Il peggiore è il neural network
