# Capstone Course

# HYPOTHESIS:::Does a greater dependency on tourism mean a greater response to covid,

# Define Tourism calculation - Strong tourism, Percentage tourism, Tourism dependent
# Define dependency and greater/less dependency - Very dependent - not very dependent
# Define Covid Response as well as strenght Greater/lesser - large protocols

###   DATA SOURCES   ###
# Datasets are available from the links below approriate dataset name
#Countries_usefulFeatures
#https://www.kaggle.com/ishivinal/covid19-useful-features-by-country
#Countries Useful feature dataset provides generic geographic information such as Population, placement, first case/death,etc.,

# COVID_19_ContainmentMeasuresData
#https://www.kaggle.com/paultimothymooney/covid19-containment-and-mitigation-measures
#Containment measures give us the details on the responses/restrictions countries had to the COVID-19 pandemic

# econominExposure
#https://www.kaggle.com/mpwolke/cusersmarildownloadsexposurecsv
#economic Exposure gives us relative econmic indicators of countries impacted by COVID

# Country_TopTourismArrivals1995_2020
#https://www.kaggle.com/mathurinache/topcountrybytourismnumberofarrivals19952020
#Tourism arrival data will be used to examine time series data such as seasonality.

#install.packages("tm")  # for text mining
#install.packages("SnowballC") # for text stemming
#install.packages("wordcloud") # word-cloud generator 
#install.packages("RColorBrewer") # color palettes
#install.packages("syuzhet") # for sentiment analysis
###   Code Begins   ###
###   Libraries    ###
#if(!require(devtools)) install.packages("devtools")
#devtools::install_github("kassambara/ggcorrplot")
#install.packages("textdata")
library(ggcorrplot)
library(ggstatsplot)
library(fpp2)
library(funModeling)
library(readxl)
library(skimr)
library(magrittr)
library(broom)
library(lubridate)
library(desc)
library(Hmisc)
library(ggplot2)
library(GGally)
library(corrplot)
library(InformationValue)
library(psych)
library(car)
library(devtools)
library(stringr)
library(tidytext)
library(textdata)
library(topicmodels)
library(tidyverse) 
library(tidyr)
library(tm)  
library(SnowballC) 
library(wordcloud) 
library(RColorBrewer) 
library(syuzhet)
library(dplyr)
library(ggbiplot)
library(doParallel)
library(stats)
library(factoextra)
library(cluster)
library(e1071)
library(caret)
library(xgboost)
library(DiagrammeR)
library(ROCR)
###Check/set working directory
getwd()
setwd("C:/Users/Alexl/Documents/JWU/Capstone/datasets")

############################
###   Load in the data   ###
############################

# Countries_usefulFeatures
Countries_usefulFeatures <- read.csv("Countries_usefulFeatures.csv")
head(Countries_usefulFeatures)
glimpse(Countries_usefulFeatures)
# COVID_19_ContainmentMeasuresData
ContainmentMeasuresData <- read.csv("COVID_19_ContainmentMeasuresData.csv",sep=",")
head(ContainmentMeasuresData)
glimpse(ContainmentMeasuresData)

# economicExposure
# try read_delim function 
# only values for rows with a country entry
economicExposure <- read_delim("C:/Users/Alexl/Documents/JWU/Capstone/datasets/economicExposure.csv", 
                               ";", escape_double = FALSE, trim_ws = TRUE)
glimpse(economicExposure)
head(economicExposure)

# Country_TopTourismArrivals1995_2020
TourismArrivals <- read.csv("Country_TopTourismArrivals1995_2020.csv", header = TRUE, na.strings=c(""," ","NA"))
head(TourismArrivals)


##################################################
###   Cleaning the data/ data transformation   ###
##################################################

### Economic Exposure
# Check each column
glimpse(economicExposure)


economicExposure <- economicExposure %>% 
  mutate_all(funs(str_replace(., ",", "."))) #Remove all ,'s  to .'s

# rest of the dataset should be in numerical values
economicExposure <- economicExposure%>%
  mutate_at(vars( -`Income classification according to WB`, -country, -GHRP),#convert everything but these columns
            as.numeric) #convert to Numeric class

# Excess rows due to infinite ;;; (they are registered as NA's) 
economicExposure <- economicExposure %>%
  drop_na(country)
tail(economicExposure) #should contain no names, Djibouti is the last country listed

# Convert to Factors:: GHRP (Global Humanitarian Response Plan) && Income classification,
economicExposure$GHRP <- as.factor(ifelse(economicExposure$GHRP  == "yes",1,0))
economicExposure$`Income classification according to WB`<- as.factor(economicExposure$`Income classification according to WB`)
# double check dataset is converted correctly
glimpse(economicExposure)
view(economicExposure)

### Countries_usefulFeatures
# Again country may need to be a factor for model later
glimpse(Countries_usefulFeatures)
describe(Countries_usefulFeatures)

# Make a Factor - Lockdown_Type
Countries_usefulFeatures$Lockdown_Type <- as.factor(Countries_usefulFeatures$Lockdown_Type) 

### ContainmentMeasuresData
glimpse(ContainmentMeasuresData)
describe(ContainmentMeasuresData)
# Columns that may not be needed, redundant: 
ContainmentMeasuresData <- ContainmentMeasuresData %>%
  select(-Implementing.City, 
         -Implementing.State.Province, 
         -Target.city, 
         -Target.country, 
         -Target.region, 
         -Target.state,
         -Applies.To)
glimpse(ContainmentMeasuresData)
# Within the country column the US is seperated by state as different obsrvations. 
# I believe all should be brought in together to look at the data from country relative perspective
ContainmentMeasuresData %>%
  distinct(Country)

### REMOVE ALL THAT START WITH THE US:
# the Untied Sates is its own observation. Or we can kep California and NYC if needed, Hawaii may make a case as well 
ContainmentMeasuresData2 <- ContainmentMeasuresData %>% filter(str_detect(Country, "US:"))
ContainmentMeasuresData2 %>%
  distinct(Country)
ContainmentMeasuresData_tidy <- anti_join(ContainmentMeasuresData, ContainmentMeasuresData2, by="Country")
ContainmentMeasuresData_tidy%>%
  distinct(Country)

### TourismArrivals
# region, Image url and X1995.1 columns not needed
names(TourismArrivals)
TourismArrivals <- TourismArrivals %>%
  select(-"Image.URL",-"region", -"X1995.1")
glimpse(TourismArrivals)



#############################
###   JOINING DATASETS   ####
#############################

t1 <- economicExposure %>% 
  group_by(country) %>% #Linked on country
  dplyr:::mutate(id = row_number())
           
glimpse(t1) #checking the dataset
unique(t1$country)
names(t1)
head(t1)
names(Countries_usefulFeatures)#review names of columns where countries are listed

# join Countries_usefulFeatures to joined dataset T1 above
t2 <- left_join(t1 %>%
                  group_by(country) %>%
                  dplyr:::mutate(id = row_number()),
                Countries_usefulFeatures %>%
                  group_by(Country_Region) %>%
                  dplyr:::mutate(id = row_number()),
                by = c("country" = "Country_Region"))
names(TourismArrivals)  #review names of columns where countries are listed

# join TourismArrivals to t2 joined dataset
t3 <- left_join(t2 %>%
                  group_by(country) %>%
                  dplyr:::mutate(id = row_number()),
                TourismArrivals %>%
                  group_by("Country.Name") %>%
                  dplyr:::mutate(id = row_number()),
                by = c("country" = "Country.Name"))
t3$country <- as.factor(t3$country)   #convert countries to factors

# review to make sure everthing is well with t3 dataset
glimpse(t3)
summary(t3)
str(t3)
names(t3)

# Creating a numerical variable table
t3Numerical <- select_if(t3, is.numeric)
t3Numerical <- na.omit(t3Numerical)
t3Numerical[is.na(t3Numerical)] <- 0 #convert all NAs into zeros, no meaningful Zeros

t3Numerical <- t3Numerical %>%
  select(-21,-24,-25,-27,-28,-29) #remove non - insightful numerical values
names(t3Numerical)
str(t3Numerical)

t3Numerical <- lapply(t3Numerical, as.numeric) # convert everything into a numeric type over a int
str(t3Numerical)                               # Double check
class(t3Numerical)                             # shows as a list
t3Numerical <- as.data.frame(t3Numerical)      # converts into a proper data frame
glimpse(t3Numerical)


#############################
###     Correlation       ###
#############################

# Corr Plot prep
corr <-  round(cor(t3Numerical))
p.mat <- cor_pmat(t3Numerical)

summary(p.mat)
# Get the upper triangle viz.,
ggcorrplot(corr, 
           hc.order = TRUE, 
           type = "lower",
           p.mat = p.mat, 
           insig = "blank")

# to get the correlation analyses results in a dataframe
corr_DF <-  ggstatsplot::ggcorrmat(
    data = t3Numerical,
    output = "dataframe"
  )
glimpse(corr_DF)

# Tourism and Tourism dependence will be calculated elsewhere
# Reundencies in the data, removing overcorrelated factors.
t3Numerical_Corr2 <- t3Numerical%>%
  select(-"Covid_19_Economic_exposure_index_Ex_aid_and_FDI_and_food_import",
         -"Covid_19_Economic_exposure_index_Ex_aid_and_FDI",
         -"Covid_19_Economic_exposure_index",
         -"Total.reserves.in.months.of.imports.2018",
         -"Net_ODA_received_perc_of_GNI",
         -"General.government.gross.debt.Percent.of.GDP.2019",
         -"Foreign.direct.investment..net.inflows.percent.of.GDP",
         -"tourism.dependence",
         -"Volume.of.remittances.in.USD.as.a.proportion.of.total.GDP.percent.2014.18",
         -"Fuels.ores.and.metals.exports.percent.of.total.merchandise.exports",
         -"Food.imports.percent.of.total.merchandise.exports",
         -"Tourism")  

names(t3Numerical_Corr2)
# Corr Prep
corr <-  round(cor(t3Numerical_Corr2))
p.mat <- cor_pmat(t3Numerical_Corr2)

summary(p.mat)
# Get the upper triangle viz.,
ggcorrplot(corr, 
           hc.order = TRUE, 
           type = "lower",
           p.mat = p.mat, 
           insig = "blank")

# to get the correlation analyses results in a dataframe
corr_DF2 <-  ggstatsplot::ggcorrmat(
  data = t3Numerical_Corr2,
  output = "dataframe"
)
glimpse(corr_DF2)




############### 
###   PCA   ###
###############

#use second corr plot for PCA
t3Numerical2 <- t3Numerical_Corr2 %>%
  select(-1)%>%
  glimpse()

t3.pca <- prcomp(t3Numerical2, center = TRUE, scale = TRUE)         #PCA on numerical dataset
summary(t3.pca)               #review results of PCA
str(t3.pca)                   #review structure
t3.pca$scale
t3.pca$rotation
ggbiplot(t3.pca)               #bi plot pca results for viz.,
ggbiplot(t3.pca, labels=rownames(t3.pca))
# PC 1 -6 explain 81% of the varience 
names(t3.pca)
head(t3.pca$scale^2, n=8) #review and compare variables

#Most explanatory variables:::
# Mean_Age
# Tourism.as.percentage.of.GDP
# Aid_Dependence
# food.import.depedence
# Foreign Direct Investment 


###############
###   LDA   ###
###############

# LDA Prep.
names(ContainmentMeasuresData_tidy)       #going to grab the country and keywords columns
countryKeywords <- ContainmentMeasuresData_tidy %>%     #set up df for LDA prep
  select(Country, Keywords)
head(countryKeywords)   #review
count(is.na(countryKeywords$Keywords))  #no NA records
#view(countryKeywords)    #last check to review

# Split into words - unnest tokens
library(tidytext)
library(dplyr)
countryKeywords <- as.data.frame(countryKeywords) #convert to a dataframe
glimpse(countryKeywords)  #review
countryKeywords_word <- countryKeywords%>%
    tidytext::unnest_tokens(word, Keywords)

# Find document-word counts
word_counts <- countryKeywords_word %>%
  anti_join(stop_words) %>%     #remmove stop words if anny
  dplyr:::count(word, sort=TRUE) %>%    #count the number of words in the document
  ungroup()
word_counts   # review the wordcounts
#top words by count not seperated
# international, countries, closure, isolation, travel
# Word counts seperated by country
word_counts <- countryKeywords_word %>%
  anti_join(stop_words) %>% #remmove stop words if anny
  dplyr:::count(Country, word, sort=TRUE) %>% #count the number of words by county
  ungroup()
word_counts

total_words <- word_counts %>%  #create total words columns
  group_by(Country) %>% 
  summarize(total = sum(n))
total_words

# tf-IDF
keywords_tf_idf <- word_counts %>%
  bind_tf_idf(word, Country, n)
keywords_tf_idf %>%
  names()

#review tf-idf
keywords_tf_idf %>%
  arrange(desc(tf_idf)) #descending order

#create Document term matrix
keywords_dtm <- word_counts %>%
  cast_dtm(Country, word, n)
keywords_dtm

class(keywords_dtm)

#LDA on all keywords as a whole
# Starting with a K of 3 on the whole dataset
keywords_lda <- LDA(keywords_dtm, k = 3, control = list(seed = 1234))
keywords_lda
keywordTopics <- tidy(keywords_lda, matrix = "beta") #per topic word probability
keywordTopics

# review the top 10 topics
top_terms <- keywordTopics %>%
  group_by(topic) %>%
  top_n(10, beta) %>% #show only the top 10 based on beta scores
  ungroup() %>%
  arrange(topic, -beta)
top_terms # print the top terms

# data viz., to plot terms of topics
top_terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered()
#groups seem to be about case isolation, social distancing, travel banning/testing.
# Social Response, Public Health respose, Travel Response

countries_gamma <- tidy(keywords_lda, matrix = "gamma")
countries_gamma
# reorder titles in order of topic 1, topic 2, etc before plotting
countries_gamma %>%
  mutate(title = reorder(document, gamma * topic)) %>%
  ggplot(aes(factor(topic), gamma)) +
  geom_boxplot() +
  facet_wrap(~ title) +
  labs(x = "topic", y = expression(gamma))
# group by country and review each countrys tf-idf or gamma score
# the top wthin the group could have said to have "strongest" response of that type
names(countries_gamma)
dim(countries_gamma)  

# 288 among 3 groups, 288/3 = 96, we want the top 20%
# 96 * .20 = 19 = top 20 for the sake of simplicity instead of rounding up.
# viewing top 10 first to compare
countries_gamma %>%
  group_by(topic)%>%
  top_n(10, gamma) %>% # 10 first for display purposes
  mutate(document = reorder_within(document, gamma, topic)) %>%
  ggplot(aes(gamma, document, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered() +
  ggtitle("Top 10 Countries Per Group")
# List version
countries_gamma %>%
  group_by(topic)%>%
  top_n(10, gamma) 

#Bottom 10
countries_gamma %>%
  group_by(topic)%>%
  top_n(-10, gamma) %>% # 10 first for display purposes
  mutate(document = reorder_within(document, gamma, topic)) %>%
  ggplot(aes(gamma, document, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered() +
  ggtitle("Bottom 10")
# List version
countries_gamma %>%
  group_by(topic)%>%
  top_n(-10, gamma)

# top 25 variation of above.
countries_gamma %>%
  group_by(topic)%>%
  top_n(19, gamma) %>% # 10 first for display purposes
  mutate(document = reorder_within(document, gamma, topic)) %>%
  ggplot(aes(gamma, document, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered() +
  ggtitle("Top 19 Countries of Each Group")
# List version
countries_gamma %>%
  group_by(topic)%>%
  top_n(19, gamma)

# Bottom Up
# top 25 variation of above.
countries_gamma %>%
  group_by(topic)%>%
  top_n(-19, gamma) %>% # 10 first for display purposes
  mutate(document = reorder_within(document, gamma, topic)) %>%
  ggplot(aes(gamma, document, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered() +
  ggtitle("Bottom 19")
# List version
countries_gamma %>%
  group_by(topic)%>%
  top_n(-19, gamma)


#save as a group of repsonses
Strongest_responses <- countries_gamma %>%
  group_by(topic)%>%
  top_n(19, gamma)

Weakest_responses <- countries_gamma %>%
  group_by(topic)%>%
  top_n(-19, gamma)

glimpse(Weakest_responses)  # Document = Country

t3$StrongResponse <- ifelse(t3$country %in% Strongest_responses$document, 1, 0)
t3$Weakest_responses <- ifelse(t3$country %in% Weakest_responses$document, 1, 0)

# Strong Catagory responses: Logical 1,0 Yes or No
# Did those with Stong scores in 2 have lower scores in 1 and 3 due to stongr measures
# SARS pademic and those with strong catagory 2 responses

# review the quartile ranges of each countries tourism dependence
# information is normalized on a scale of 1 - 10
summary(t3$`tourism dependence`)
# First Quartile: 2.525
# 3rd Quartile: 6.075
t3 <- t3%>% 
  mutate(TD_Cat = case_when(`tourism dependence` >= 6.075 ~ "High",
                            `tourism dependence` < 2.525 ~ "Low",
                            `tourism dependence` > 2.525 & `tourism dependence` < 6.075  ~ "Normal"))  

#View(t3)#review data
names(t3)

#Create the sae IQG logic factors for :
#  "Population_Size"  
summary(t3$`Population_Size`)
# First Quartile: 2.818e+06
# 3rd Quartile: 2.934e+07
t3 <- t3%>% 
  mutate(Population_Cat = case_when(`Population_Size` >= 2.934e+07 ~ "High",
                            `Population_Size` < 2.818e+06 ~ "Low",
                            `Population_Size` > 2.818e+06 & `Population_Size` < 2.934e+07  ~ "Normal"))  

#  "Tourism"
summary(t3$`Tourism`)
# First Quartile: 58500
# 3rd Quartile: 7403750
t3 <- t3%>% 
  mutate(Tourism_Cat = case_when(`Tourism` >= 7403750 ~ "High",
                                    `Tourism` < 58500 ~ "Low",
                                    `Tourism` > 58500 & `Tourism` < 7403750  ~ "Normal")) 
names(t3)


# review statistical variences of these components
# calculating the approprate bin width for histograms
# bw <- 2 * IQR(t3Numerical$) / length(t3Numerical$)^(1/3) 
# Freedman Diaconis rule for calculating bins


#_____________________________________________________________
#What is Tourism Relience - Define
#who is Tourism Relient - list
#What is a strong Covid respone
#Who had a strong covid response
#Who had both Strong Relience/response - MLR


#FINAL DATASET:
#tourism Cat
#Population Cat
#TD_Cat

## Mean_Age
# Tourism.as.percentage.of.GDP
# Aid_Dependence
# food.import.depedence
# Foreign Direct Investment 
names(t3)
d1 <- t3%>%
  select('Tourism_Cat',
         'Population_Cat',
         'TD_Cat',
         'Mean_Age',
         'tourism as percentage of GDP',
         'Aid dependence',
         'food import dependence',
         'Foreign direct investment',
         "Weakest_responses",
         "StrongResponse")
View(d1)
dim(d1)
# Interesting that there are countries with both strong and weak responses.

# need to seperate by reponse saccording to groups.
#Create logic based on indivdual groups
Strong1 <- Strongest_responses %>%
  filter(topic==1)
View(Strong1)
Strong2 <- Strongest_responses %>%
  filter(topic==2)
Strong3 <- Strongest_responses %>%
  filter(topic==3)

d1$Strong1 <- ifelse(t3$country %in% Strong1$document, 1, 0)
d1$Strong2 <- ifelse(t3$country %in% Strong2$document, 1, 0)
d1$Strong3 <- ifelse(t3$country %in% Strong3$document, 1, 0)

# Weak Responses per group
Weak1 <- Weakest_responses %>%
  filter(topic==1)
Weak2 <- Weakest_responses %>%
  filter(topic==2)
Weak3 <- Weakest_responses %>%
  filter(topic==3)

d1$Weak1 <- ifelse(t3$country %in% Weak1$document, 1, 0)
d1$Weak2 <- ifelse(t3$country %in% Weak2$document, 1, 0)
d1$Weak3 <- ifelse(t3$country %in% Weak3$document, 1, 0)

View(d1)

# review statistical variences of these components
# calculating the approprate bin width for histograms
# bw <- 2 * IQR(t3Numerical$) / length(t3Numerical$)^(1/3) 
# Freedman Diaconis rule for calculating bins
names(d1)
d1 %>%
  filter(Tourism_Cat =="High" & StrongResponse ==1)%>%
  View() # 28/ 48 strong responses have high tourism

d1 %>%
  filter(Tourism_Cat =="High" & StrongResponse ==0)%>%
  View() # 13/28 high tourism did not have a strong response



d1 %>%  # everyoe with a strong 2 response did not have other strong rates
  filter( Strong2 ==1)%>%
  View()

d1 %>%
  filter(TD_Cat =="High" & StrongResponse ==1)%>%
  View() # 9/48 TD_Cat =="High" & StrongResponse ==1
d1 %>%
  filter(TD_Cat =="High" & StrongResponse ==0)%>%
  View() # 35/143 TD_Cat =="High" & StrongResponse ==0

d1 %>%
  filter(TD_Cat =="High" & StrongResponse ==1 & Tourism_Cat =="High" )%>%
  View() # Mexico, Morroco, Thailand
#those who have a strong tourism sector and stong responses

#(9 - 3 + 28)/48
#34/48 of strong responses have either High tourism or tourism dependence and strong responses

d1 %>%
  filter(StrongResponse ==1)%>%
  View()
d1 %>%
  filter(Weakest_responses ==1)%>%
  View()


d1 %>%
  filter(StrongResponse ==1 & Weakest_responses ==1)%>%
  View()

d1 %>%
  filter(StrongResponse ==0 & Weakest_responses ==0)%>%
  View() #most of the data points

d1 %>%
  filter(TD_Cat =="High" & Weak2 ==1)%>%
  View()

d1%>%
  filter(StrongResponse ==1 & Strong2 == 1)%>%
  View()

d1%>%
  filter(Strong1 ==1 )%>%
  View()


### Prep dataframe for XG Boost
names(d1)
dxg <- d1%>%
  select(2:9, Strong2)   # add dsired variable here, for emaple  strong 2


# Run a simple regresion Model
# to try to predict "__________"

#Can we predict if it is strong response or not
# XG Boost on clumns or cat 1,2,3,

## 70% of the sample size
smp_size <- floor(0.70 * nrow(dxg))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(dxg)), size = smp_size)
train <- dxg[train_ind, ]
test <- dxg[-train_ind, ]
names(train)

trainlabel <- as.numeric(as.factor(train$Strong2))-1 # set training label
testlabel <- as.numeric(as.factor(test$Strong2))-1   #set test Label

train$Strong2 <- NULL   # remove frm data
test$Strong2 <- NULL

trainmat <- data.matrix(train)
testmat <- data.matrix(test)

##put our testing & training data into seperate Dmatrixs objects
dtrain <- xgb.DMatrix(data = trainmat, label= trainlabel)
dtest <- xgb.DMatrix(data = testmat, label= testlabel)

# Run XG boost model  on dtrained dataset aboce
xgmodel <- xgboost(data = dtrain, # the data   
                   nround = 50,
                   max.depth = 3,# boosting iterations
                   objective = "binary:logistic")  # the objective function
pred <- predict(xgmodel, dtest)

#error rate
err <- mean(as.numeric(pred > 0.5) != testlabel)
print(paste("test-error=", err))

# get the number of negative & positive cases in our data
negative_cases <- sum(trainlabel == 0)
postive_cases <- sum(testlabel == 1)

model_tuned <- xgboost(data = dtrain, # the data           
                       max.depth = 5, # the maximum depth of each decision tree
                       nround = 50, # number of boosting rounds
                       early_stopping_rounds = 3, # if we dont see an improvement in this many rounds, stop
                       objective = "binary:logistic", # the objective function
                       #scale_pos_weight = negative_cases/postive_cases, # control for imbalanced classes
                       gamma = 1) # add a regularization term

pred <- predict(model_tuned, dtest)

#error rate
err <- mean(as.numeric(pred > 0.5) != testlabel)
print(paste("test-error=", err))

# plot the features
xgb.plot.multi.trees(feature_names = names(trainmat), 
                     model = model_tuned)
# convert log odds to probability
odds_to_probs <- function(odds){
  return(exp(odds)/ (1 + exp(odds)))
}
# probability of top leaf
odds_to_probs(1.8052)    #0.8587807

# get information on how important each feature is
importance_matrix <- xgb.importance(names(trainmat), model = model_tuned)
#plotting importance
xgb.plot.importance(importance_matrix)



##################################
###     XG BOOST STRONG 3     ####
#################################

### Prep dataframe for XG Boost
names(d1)
dxg <- d1%>%
  select(2:9, Strong3)  
## 70% of the sample size
smp_size <- floor(0.70 * nrow(dxg))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(dxg)), size = smp_size)
train <- dxg[train_ind, ]
test <- dxg[-train_ind, ]
names(train)

trainlabel <- as.numeric(as.factor(train$Strong3))-1 # set training label
testlabel <- as.numeric(as.factor(test$Strong3))-1   #set test Label

train$Strong3 <- NULL   # remove frm data
test$Strong3 <- NULL

trainmat <- data.matrix(train)
testmat <- data.matrix(test)

##put our testing & training data into seperate Dmatrixs objects
dtrain <- xgb.DMatrix(data = trainmat, label= trainlabel)
dtest <- xgb.DMatrix(data = testmat, label= testlabel)

# Run XG boost model  on dtrained dataset aboce
xgmodel <- xgboost(data = dtrain, # the data   
                   nround = 50,
                   max.depth = 3,# boosting iterations
                   objective = "binary:logistic")  # the objective function
pred <- predict(xgmodel, dtest)

#error rate
err <- mean(as.numeric(pred > 0.5) != testlabel)
print(paste("test-error=", err))

# get the number of negative & positive cases in our data
negative_cases <- sum(trainlabel == 0)
postive_cases <- sum(testlabel == 1)

model_tuned <- xgboost(data = dtrain, # the data           
                       max.depth = 5, # the maximum depth of each decision tree
                       nround = 50, # number of boosting rounds
                       early_stopping_rounds = 3, # if we dont see an improvement in this many rounds, stop
                       objective = "binary:logistic", # the objective function
                       #scale_pos_weight = negative_cases/postive_cases, # control for imbalanced classes
                       gamma = 1) # add a regularization term

pred <- predict(model_tuned, dtest)

#error rate
err <- mean(as.numeric(pred > 0.5) != testlabel)
print(paste("test-error=", err))

# plot the features
xgb.plot.multi.trees(feature_names = names(trainmat), 
                     model = model_tuned)
# convert log odds to probability
odds_to_probs <- function(odds){
  return(exp(odds)/ (1 + exp(odds)))
}
# probability of top leaf
odds_to_probs(1.8052)    #0.8587807

# get information on how important each feature is
importance_matrix <- xgb.importance(names(trainmat), model = model_tuned)
#plotting importance
xgb.plot.importance(importance_matrix)

##################################
###     XG BOOST STRONG 1    ####
#################################

### Prep dataframe for XG Boost
names(d1)
dxg <- d1%>%
  select(2:9, Strong1)  
## 70% of the sample size
smp_size <- floor(0.70 * nrow(dxg))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(dxg)), size = smp_size)
train <- dxg[train_ind, ]
test <- dxg[-train_ind, ]
names(train)

trainlabel <- as.numeric(as.factor(train$Strong1))-1 # set training label
testlabel <- as.numeric(as.factor(test$Strong1))-1   #set test Label

train$Strong1 <- NULL   # remove frm data
test$Strong1 <- NULL

trainmat <- data.matrix(train)
testmat <- data.matrix(test)

##put our testing & training data into seperate Dmatrixs objects
dtrain <- xgb.DMatrix(data = trainmat, label= trainlabel)
dtest <- xgb.DMatrix(data = testmat, label= testlabel)

# Run XG boost model  on dtrained dataset aboce
xgmodel <- xgboost(data = dtrain, # the data   
                   nround = 50,
                   max.depth = 3,# boosting iterations
                   objective = "binary:logistic")  # the objective function
pred <- predict(xgmodel, dtest)

#error rate
err <- mean(as.numeric(pred > 0.5) != testlabel)
print(paste("test-error=", err))

# get the number of negative & positive cases in our data
negative_cases <- sum(trainlabel == 0)
postive_cases <- sum(testlabel == 1)

model_tuned <- xgboost(data = dtrain, # the data           
                       max.depth = 5, # the maximum depth of each decision tree
                       nround = 50, # number of boosting rounds
                       early_stopping_rounds = 3, # if we dont see an improvement in this many rounds, stop
                       objective = "binary:logistic", # the objective function
                       #scale_pos_weight = negative_cases/postive_cases, # control for imbalanced classes
                       gamma = 1) # add a regularization term

pred <- predict(model_tuned, dtest)

#error rate
err <- mean(as.numeric(pred > 0.5) != testlabel)
print(paste("test-error=", err))

# plot the features
xgb.plot.multi.trees(feature_names = names(trainmat), 
                     model = model_tuned)
# convert log odds to probability
odds_to_probs <- function(odds){
  return(exp(odds)/ (1 + exp(odds)))
}
# probability of top leaf
odds_to_probs(1.8052)    #0.8587807

# get information on how important each feature is
importance_matrix <- xgb.importance(names(trainmat), model = model_tuned)
#plotting importance
xgb.plot.importance(importance_matrix)

##################################
###     XG BOOST Weak 2    ####
#################################

### Prep dataframe for XG Boost
names(d1)
dxg <- d1%>%
  select(2:9, Weak2)  
## 70% of the sample size
smp_size <- floor(0.70 * nrow(dxg))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(dxg)), size = smp_size)
train <- dxg[train_ind, ]
test <- dxg[-train_ind, ]
names(train)

trainlabel <- as.numeric(as.factor(train$Weak2))-1 # set training label
testlabel <- as.numeric(as.factor(test$Weak2))-1   #set test Label

train$Weak2 <- NULL   # remove frm data
test$Weak2 <- NULL

trainmat <- data.matrix(train)
testmat <- data.matrix(test)

##put our testing & training data into seperate Dmatrixs objects
dtrain <- xgb.DMatrix(data = trainmat, label= trainlabel)
dtest <- xgb.DMatrix(data = testmat, label= testlabel)

# Run XG boost model  on dtrained dataset aboce
xgmodel <- xgboost(data = dtrain, # the data   
                   nround = 50,
                   max.depth = 3,# boosting iterations
                   objective = "binary:logistic")  # the objective function
pred <- predict(xgmodel, dtest)

#error rate
err <- mean(as.numeric(pred > 0.5) != testlabel)
print(paste("test-error=", err))

# get the number of negative & positive cases in our data
negative_cases <- sum(trainlabel == 0)
postive_cases <- sum(testlabel == 1)

model_tuned <- xgboost(data = dtrain, # the data           
                       max.depth = 5, # the maximum depth of each decision tree
                       nround = 50, # number of boosting rounds
                       early_stopping_rounds = 3, # if we dont see an improvement in this many rounds, stop
                       objective = "binary:logistic", # the objective function
                       #scale_pos_weight = negative_cases/postive_cases, # control for imbalanced classes
                       gamma = 1) # add a regularization term

pred <- predict(model_tuned, dtest)

#error rate
err <- mean(as.numeric(pred > 0.5) != testlabel)
print(paste("test-error=", err))

# plot the features
xgb.plot.multi.trees(feature_names = names(trainmat), 
                     model = model_tuned)
# convert log odds to probability
odds_to_probs <- function(odds){
  return(exp(odds)/ (1 + exp(odds)))
}
# probability of top leaf
odds_to_probs(1.8052)    #0.8587807

# get information on how important each feature is
importance_matrix <- xgb.importance(names(trainmat), model = model_tuned)
#plotting importance
xgb.plot.importance(importance_matrix)

###############################
###     Strong Respone      ###
###############################
### Prep dataframe for XG Boost
names(d1)
dxg <- d1%>%
  select(2:9, StrongResponse)  
## 70% of the sample size
smp_size <- floor(0.70 * nrow(dxg))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(dxg)), size = smp_size)
train <- dxg[train_ind, ]
test <- dxg[-train_ind, ]
names(train)

trainlabel <- as.numeric(as.factor(train$StrongResponse))-1 # set training label
testlabel <- as.numeric(as.factor(test$StrongResponse))-1   #set test Label

train$StrongResponse <- NULL   # remove frm data
test$StrongResponse <- NULL

trainmat <- data.matrix(train)
testmat <- data.matrix(test)

##put our testing & training data into seperate Dmatrixs objects
dtrain <- xgb.DMatrix(data = trainmat, label= trainlabel)
dtest <- xgb.DMatrix(data = testmat, label= testlabel)

# Run XG boost model  on dtrained dataset aboce
xgmodel <- xgboost(data = dtrain, # the data   
                   nround = 50,
                   max.depth = 3,# boosting iterations
                   objective = "binary:logistic")  # the objective function
pred <- predict(xgmodel, dtest)

#error rate
err <- mean(as.numeric(pred > 0.5) != testlabel)
print(paste("test-error=", err))

# get the number of negative & positive cases in our data
negative_cases <- sum(trainlabel == 0)
postive_cases <- sum(testlabel == 1)

model_tuned <- xgboost(data = dtrain, # the data           
                       max.depth = 5, # the maximum depth of each decision tree
                       nround = 50, # number of boosting rounds
                       early_stopping_rounds = 3, # if we dont see an improvement in this many rounds, stop
                       objective = "binary:logistic", # the objective function
                       #scale_pos_weight = negative_cases/postive_cases, # control for imbalanced classes
                       gamma = 1) # add a regularization term

pred <- predict(model_tuned, dtest)

#error rate
err <- mean(as.numeric(pred > 0.5) != testlabel)
print(paste("test-error=", err))

# plot the features
xgb.plot.multi.trees(feature_names = names(trainmat), 
                     model = model_tuned)
# convert log odds to probability
odds_to_probs <- function(odds){
  return(exp(odds)/ (1 + exp(odds)))
}
# probability of top leaf
odds_to_probs(1.8052)    #0.8587807

# get information on how important each feature is
importance_matrix <- xgb.importance(names(trainmat), model = model_tuned)
#plotting importance
xgb.plot.importance(importance_matrix)


#############################
###     Weak Respone      ###
#############################

### Prep dataframe for XG Boost
names(d1)
dxg <- d1%>%
  select(2:9, Weakest_responses)  
## 70% of the sample size
smp_size <- floor(0.70 * nrow(dxg))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(dxg)), size = smp_size)
train <- dxg[train_ind, ]
test <- dxg[-train_ind, ]
names(train)

trainlabel <- as.numeric(as.factor(train$Weakest_responses))-1 # set training label
testlabel <- as.numeric(as.factor(test$Weakest_responses))-1   #set test Label

train$Weakest_responses <- NULL   # remove frm data
test$Weakest_responses <- NULL

trainmat <- data.matrix(train)
testmat <- data.matrix(test)

##put our testing & training data into seperate Dmatrixs objects
dtrain <- xgb.DMatrix(data = trainmat, label= trainlabel)
dtest <- xgb.DMatrix(data = testmat, label= testlabel)

# Run XG boost model  on dtrained dataset aboce
xgmodel <- xgboost(data = dtrain, # the data   
                   nround = 50,
                   max.depth = 3,# boosting iterations
                   objective = "binary:logistic")  # the objective function
pred <- predict(xgmodel, dtest)

#error rate
err <- mean(as.numeric(pred > 0.5) != testlabel)
print(paste("test-error=", err))

# get the number of negative & positive cases in our data
negative_cases <- sum(trainlabel == 0)
postive_cases <- sum(testlabel == 1)

model_tuned <- xgboost(data = dtrain, # the data           
                       max.depth = 5, # the maximum depth of each decision tree
                       nround = 50, # number of boosting rounds
                       early_stopping_rounds = 3, # if we dont see an improvement in this many rounds, stop
                       objective = "binary:logistic", # the objective function
                       #scale_pos_weight = negative_cases/postive_cases, # control for imbalanced classes
                       gamma = 1) # add a regularization term

pred <- predict(model_tuned, dtest)

#error rate
err <- mean(as.numeric(pred > 0.5) != testlabel)
print(paste("test-error=", err))

# plot the features
xgb.plot.multi.trees(feature_names = names(trainmat), 
                     model = model_tuned)
# convert log odds to probability
odds_to_probs <- function(odds){
  return(exp(odds)/ (1 + exp(odds)))
}
# probability of top leaf
odds_to_probs(1.8052)    #0.8587807

# get information on how important each feature is
importance_matrix <- xgb.importance(names(trainmat), model = model_tuned)
#plotting importance
xgb.plot.importance(importance_matrix)


# MLR Model to try to predict the typeor response based on other variables within the dataset

###Evaluation Metrics
# Classification 
  # Accuracy
  # Precision
  # Recall
  # F1 Score
  # AUC Area under the curve

#Regression 
  # MAE meanabsolute error
  # MSE Mean squared error
  # Root mean Square Erreor RMSE
  # R squared (R^2)
