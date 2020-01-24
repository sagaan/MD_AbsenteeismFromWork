#Librerias
library("caret");
library("randomForest");
library("rJava");
library("party");
library("RWeka");
library("FSelector");
library("class");
library("dplyr");
library("e1071");

#Lectura Datos
data <- read.csv("~/1 ESCUELA/Uni/Semestre 7/Mineria de Datos/Actividad 5/Absenteeism/Absenteeism_at_work.csv", sep=";");

#Eliminado por ser el identificador
data$ID <- NULL;

#Limpieza de datos
 #Se eliminan renglones con clases invalidas
data=data[which(data$Absenteeism.time.in.hours!=0),];

  #Datos vacios
data$Month.of.absence[which(data$Month.of.absence==0)] <- median(data$Month.of.absence);

  #Dataframe sin factores
data.nf = data;

 #Conversion a Factores
data$Reason.for.absence <- factor(data$Reason.for.absence, levels = c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28), 
                                  labels = c("Certain infectious and parasitic diseases","Neoplasms","Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism",
                                             "Endocrine, nutritional and metabolic diseases","Mental and behavioural disorders","Diseases of the nervous system","Diseases of the eye and adnexa",
                                             "Diseases of the ear and mastoid process","Diseases of the circulatory system","Diseases of the respiratory system","Diseases of the digestive system",
                                             "Diseases of the skin and subcutaneous tissue","Diseases of the musculoskeletal system and connective tissue","Diseases of the genitourinary system",
                                             "Pregnancy, childbirth and the puerperium","Certain conditions originating in the perinatal period","Congenital malformations, deformations and chromosomal abnormalities",
                                             "Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified","Injury, poisoning and certain other consequences of external causes",
                                             "External causes of morbidity and mortality","Factors influencing health status and contact with health services","patient follow-up","medical consultation",
                                             "blood donation","laboratory examination","unjustified absence","physiotherapy","dental consultation"));
data$Month.of.absence <- factor(data$Month.of.absence, levels = c(1,2,3,4,5,6,7,8,9,10,11,12), labels = c("January","February","March","April","May","June","July","August","September","October","November","December"));
data$Day.of.the.week <- factor(data$Day.of.the.week, levels = c(2,3,4,5,6), labels = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday"));
data$Seasons <- factor(data$Seasons, levels = c(1,2,3,4), labels = c("Summer", "Autumn", "Winter", "Spring"));
data$Education <- factor(data$Education, levels = c(1,2,3,4), labels = c("High School", "Graduate", "Postgraduate", "Master and Doctor"));
data$Social.drinker <- factor(data$Social.drinker, levels = c(0,1), labels = c("No", "Yes"));
data$Social.smoker <- factor(data$Social.smoker, levels = c(0,1), labels = c("No", "Yes"));
data$Disciplinary.failure <- factor(data$Disciplinary.failure, levels=c(0,1), labels = c("No", "Yes"));
data$Absenteeism.time.in.hours <- as.factor(data$Absenteeism.time.in.hours);

information.gain(Absenteeism.time.in.hours~.,data = data);

#Eliminado en base a la ganancia de informacion
data$Transportation.expense <- NULL;
data$Distance.from.Residence.to.Work <- NULL;
data$Service.time <- NULL;
data$Age <- NULL;
data$Work.load.Average.day <- NULL;
data$Hit.target <- NULL;
data$Son <- NULL;
data$Pet <- NULL;
data$Weight <- NULL;
data$Height <- NULL;
data$Body.mass.index <- NULL;

#Eliminado por tener el mismo dato en toda la columna
data$Disciplinary.failure <- NULL;

#Eliminacion de datos (Dataset sin factores)
data.nf$Transportation.expense <- NULL;
data.nf$Distance.from.Residence.to.Work <- NULL;
data.nf$Service.time <- NULL;
data.nf$Age <- NULL;
data.nf$Work.load.Average.day <- NULL;
data.nf$Hit.target <- NULL;
data.nf$Son <- NULL;
data.nf$Pet <- NULL;
data.nf$Weight <- NULL;
data.nf$Height <- NULL;
data.nf$Body.mass.index <- NULL;
data.nf$Disciplinary.failure <- NULL;
#Clases como factores
data.nf$Absenteeism.time.in.hours <- as.factor(data$Absenteeism.time.in.hours);

#Arbol Binario - J48
 #Particion datos sin factores
set.seed(5)
index.J48 = createDataPartition(data.nf$Absenteeism.time.in.hours, p = 0.8, list = FALSE);
train.J48 = data.nf[index.J48,];
test.J48 = data.nf[-index.J48,];

modelo.J48 = J48(Absenteeism.time.in.hours~., data = train.J48);
plot(modelo.J48);
matriz.J48 = summary(modelo.J48)$confusionMatrix;
precision.J48 = ((matriz.J48[1,1] + matriz.J48[2,2] + matriz.J48[3,3] + matriz.J48[4,4] + matriz.J48[5,5] + matriz.J48[6,6] + matriz.J48[7,7] + matriz.J48[8,8] + 
                   matriz.J48[9,9] + matriz.J48[10,10] + matriz.J48[11,11] + matriz.J48[12,12] + matriz.J48[13,13] + matriz.J48[14,14] + matriz.J48[15,15] +
                   matriz.J48[16,16] + matriz.J48[17,17] + matriz.J48[18,18])/sum(matriz.J48))*100;

#Particion Datos normales
set.seed(5);
index = createDataPartition(data$Absenteeism.time.in.hours, p = 0.8, list = FALSE);
train = data[index,];
test = data[-index,];

#RandomForest
modelo.rf = randomForest(Absenteeism.time.in.hours~., data=train, importance=TRUE);

matriz.rf = modelo.rf$confusion;
precision.rf = ((matriz.rf[1,1] + matriz.rf[2,2] + matriz.rf[3,3] + matriz.rf[4,4] + matriz.rf[5,5] + matriz.rf[6,6] + matriz.rf[7,7] + matriz.rf[8,8] + 
                   matriz.rf[9,9] + matriz.rf[10,10] + matriz.rf[11,11] + matriz.rf[12,12] + matriz.rf[13,13] + matriz.rf[14,14] + matriz.rf[15,15] +
                   matriz.rf[16,16] + matriz.rf[17,17] + matriz.rf[18,18])/sum(matriz.rf[,-19]))*100;

#NaiveBayes
modelo.nb = naiveBayes(Absenteeism.time.in.hours~., train);
prediccion = predict(modelo.nb,test[-8]);
matriz.nb = table(prediccion,test$Absenteeism.time.in.hours);
precision.nb = ((matriz.nb[1,1] + matriz.nb[2,2] + matriz.nb[3,3] + matriz.nb[4,4] + matriz.nb[5,5] + matriz.nb[6,6] + matriz.nb[7,7] + matriz.nb[8,8] + 
               matriz.nb[9,9] + matriz.nb[10,10] + matriz.nb[11,11] + matriz.nb[12,12] + matriz.nb[13,13] + matriz.nb[14,14] + matriz.nb[15,15] +
               matriz.nb[16,16] + matriz.nb[17,17] + matriz.nb[18,18])/sum(matriz.nb))*100;

#KNN
  #Normalizacion
normalizacion = function(x){ 
  (x -min(x))/(max(x)-min(x))   
}

class.temp = data.nf$Absenteeism.time.in.hours;
data.nf = as.data.frame(lapply(data.nf[,seq(1,7)], normalizacion));
data.nf$Absenteeism.time.in.hours = class.temp;

  #Particionamiento datos normalizados
set.seed(5);
index.normal = createDataPartition(data.nf$Absenteeism.time.in.hours, p=0.8, list = FALSE);
train.normal = data.nf[index.normal,];
test.normal = data.nf[-index.normal,];
  
  #Modelo
modelo.knn = knn(train.normal[-8],test.normal[-8],train.normal$Absenteeism.time.in.hours,k=26);
summary(modelo.knn);
matriz.knn = table(modelo.knn,test$Absenteeism.time.in.hours);
precision.knn = ((matriz.knn[1,1] + matriz.knn[2,2] + matriz.knn[3,3] + matriz.knn[4,4] + matriz.knn[5,5] + matriz.knn[6,6] + matriz.knn[7,7] + matriz.knn[8,8] + 
                   matriz.knn[9,9] + matriz.knn[10,10] + matriz.knn[11,11] + matriz.knn[12,12] + matriz.knn[13,13] + matriz.knn[14,14] + matriz.knn[15,15] +
                   matriz.knn[16,16] + matriz.knn[17,17] + matriz.knn[18,18])/sum(matriz.knn))*100;