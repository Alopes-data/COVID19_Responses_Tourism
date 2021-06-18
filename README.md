# COVID19_Responses_Tourism

### Abstract
With this pandemic shutting down international tourism in an effort to contain the spread. Did those countries/economies more reliant on tourism have stronger responses to the COVID-19 pandemic. We use a combination of Principal component analysis and a latent dirichlet allocation to define our measures as well as predict the responses of various countries using and XG Boost algorithm. We were able to determine that those with a greater tourism or tourism dependence had strong responses to the COVID-19 pandemic.

### Introduction
COVID-19 is the most recent pandemic to impact the world starting in early 2020 as a global pandemic. Countries have implemented different responses such as shutting down their economies and/or implemented a form of travel restriction to deal with the spread impacting global tourism more than any other since 1995 such as the September 11 attacks, 2003 SARS  outbreak, 2009 global recession and the 2015 MERS outbreak (Gössling et al., 2020) as lower income countries that have a greater reliance on tourism as well as others for their resources suffered greatly from the indirect and direct impacts from the pandemic. Some types of responses include school closings, travel restrictions, bans on public gatherings, emergency investments in healthcare facilities, new forms of social welfare provision, and contact tracing (Hale et al., 2020). These responses all come from different intentions as some countries may have had experience with a similar event, some are looking to protect their economies as well as citizens causing tourism destinations in enact specific border strictions impacting supply chains as well as global relations (Seyfi et al., 2020). Before the pandemic tourism was thought as a resilient industry used to help foster relationships and build capacity for economic development as, “…tourism dependent countries do not face real exchange rate distortion and deindustrialization but higher than average economic growth rates. Investment in physical capital, such as for instance transport infrastructure, is complementary to investment in tourism.” (Holzner, 2011) since the 1950s (Jenkins, 2015).

### Methods
Calculating and defining the tourism levels, dependency, and COVID-19 response among different countries required multiple datasets from World Bank data sources. The first dataset we started with provides generic geographic information such as Population, placement, first case/death,etc. know as the “Countries Useful Features”. The second dataset gives us the details on the responses/restrictions countries had to the COVID-19 pandemic known as “Containment measures” and the final dataset “economic Exposure” gives us relative economic indicators of countries impacted by COVID. Cleaning the data, removing the US: separated as a state only retaining the whole United States as a single observation. Joining the datasets together by their country, retaining the numerical values and removing all correlated factors to reduce multicollinearity introduced to our models producing the final dataset show in Figure 1.




Figure 1. Correlation Plot
		 
Avoiding introducing Multi Collinearity to our models

These uncorrelated numerical variables will be used in our principal component analysis to analyze which of these variables explains the most variance in our dataset and to what degree show in Figure 2. This is how we will measure the economic differences from our countries listed finding the most explanatory variables to be mean age, tourism as a percentage of GDP, aid dependence, food import dependence, and foreign direct investment.



Figure 2. Principal Component Analysis Plot
 
Variables that explain the most Variance and their direction

I continued with a latent dirichlet allocation, LDA,  model to contextualize the responses of each country within the dataset. The responses were split by their words and cleaned to create a term frequency inverse document frequency highlighting the most used words such as international, countries, closure isolation and travel all with over 150 out of 4515 occurrences allowing us to use to sperate the data based on countries use the data in a document term matrix and sequentially running our LDA on it. 
From the LDA model we can derive that the first group involves Social Responses. These are responses that could be considered public health but rely on individuals following the rules or constraints on matters usually outside of the public sphere. An examples can be highlighted by School and Business closures making it as top terms for this group. The second group can be considered inside the public health sphere directly as it related to matter such as isolation, contact tracing, and hospital but with words like pubic and compulsory we can designate this as direct public health responses where medical practitioners are directly involved. The third and final groups pertains to the movement of people more specifically internationally. With the top words being International, countries, travel, and risk followed by ban, traveler, and testing we can assume this is where the closing of borders and limiting tourism responses are grouped.
The top countries of each group created from our LDA were considered those that had a “strong” response and those on the bottom percentile received a “weak” response. We had these responses individually defined for each group in case there were observations having a strong response in one group but weak in another. 







Figure 3. Top Terms per LDA Group
 
Strongest words in each created group, Contextualizes LDA groups.






Figure 4. Countries Ranked in each LDA Group
 
Strongest responders at the top.







Figure 5. Bottom Courtiers in each LDA Group
 
Weakest Responders at the Top.

	Next continuing with defining if a country was Tourism dependent, its Population Level and level of tourism by reviewing the top and bottom of each category making the data relative to the observations within the dataset. We combined these variables indicated as factors and joined it with the variables selected from out PCA model to explain the economic variance and the different response rates to each group designated in our LDA model also listed as individual factors. This dataset was used in our XGBoost model to predict the if a county is going to have a strong response or not in a specific category.

### Limitations further research
	COVID-19 landscape and available data constantly changing. As countries update their responses and events unfold we are seeing different pivots. Also the data available was before vaccine rollouts started in many western countries which may change how each country sits within the responses. The United states could be evaluated with specific states such as New York, Florida, and California with the rest of the world instead of calculating the United States as a whole observation.  Using the PCA analysis to remove some variables could produce different results if different variables were included over others such as primary commodity export dependence.
	
### Conclusions and Discussions
With our XGBoost model in determining who had a Strong response with group 2, our direct public health group, the mean age, tourism category and GDP as well as the Food import dependence were the most important factors with a model error rate of .08. 







Figure 6. XG Boost Variable Importance – Strong 2
 


“The first four polities share experience in fighting similar infectious respiratory diseases…” (An & Tang, 2020) referring to the SARS and MERS pandemics that they to face  requiring these countries to update their health systems. All of these countries aside from Taiwan had a strong group 2 score relative to their other scores in each group showing that in our data they also had strong direct public health responses. All five including Taiwan did make it to the 19 within group 2 responses. This can be attributed to the infrastructure in place from dealing with the past pandemics as even though Japan hadn’t dealt with those in the same way, they still had experience from the swine flu (H1N1) pandemic in 2009 which allowed for not only institutional infrastructure but also public support and cooperation (An & Tang, 2020). The public support and cooperation is another key factor as measuring the responses on the institutional strength itself cannot clearly tell the responses as indicated by (Greer et al., 2020).  Even Though Taiwan’s Score may be low relative to other measures it made this could be due to already having a strong infrastructure to not need additional response as they focused more on group 3 indicating focusing on restricting mobility with other countries. Canada is another country that had a significant SARS outbreak not listed in this article but they also had a high category 2 response (Gössling et al., 2020). Many on the countries had a high mean age as well which is a strong correlating factor with a countries development as the more developed countries in the world are considered below replacement in birth rates.
An interesting point with food independence being a high determining factor with those with strong group 2 responses is the link between COVID-19 or global pandemics in general, tourism, and food sources such as, “…there is much evidence that food production patterns are responsible for repeated outbreaks of the corona virus, including SARS, MERS and COVID-19 (Pongsiri et al., 2009; Labonte et al., 2011)…” (Gössling et al., 2020).





Figure 7. Distribution of groups among countries.
 


	Within our dataset all of the countries with a high level of tourism dependence did not have weak group 2 responses and all were considered countries with strong responses showing that those with a strong dependence on tourism did have a stronger response to tourism. Those who relied on tourism had a harder time dealing with the pandemic impacts as a greater reliance on tourism directly reduces the rate of is resiliency specifically regional economic resilience (Watson & Deller, 2021), as, “show that countries exposed to high flows of international tourism are more prone to cases and deaths caused by the COVID-19 outbreak...This study examines the relationship between international tourism and COVID-19 cases and associated deaths in more than 90 nations.” (Farzanegan et al., 2020). This direct relation to deaths from COVID-19 and tourism contextualizes some of the responses from High tourism or tourism dependent countries, “The collapse of tourism resulting from the COVID-19 pandemic will have a profound impact on the Asia-Pacific small island developing States because of their high reliance on tourism rents.” (Tateno & Bolesta, 2020). This information correlates with our group 2 results of countries in group 2 having a higher rate of food independence which is similar to one of the negative impacts of tourism when compared with manufacturing and agriculture industry economic development, as tourism reliance has a moderating effect on the relationship between tourism development and economic development (Bojanic & Lo, 2016). These factors linking food independence and the connecting relationship to tourism is apparent in our strong 3 group, international mobility responses where we see that food import dependence is the largest determining factor with a error rate of .13.
Figure 8. XG Boost Variable Importance – Group 3
 


 	Reviewing those who had Strong group 1 responses, our XG Boost model found Aid dependence to be the most important factor followed closely by food and tourism levels with an error rate or .15. This appears to be the column with the most mixed responses. I believe this is due to most countries having strong group 2 or 3 responses to cover most of their issues and any other types of responses on their public after dealing with perceived external causes will be grouped in group 1. 

Figure 9. XG Boost Variable Importance –  Strong 1
 

The best results were on determining who had a weak group 2 response which our model had an error rate of .03, where again tourism and mean age were the biggest determining factor followed by food independence. 

Figure 10. XG Boost Variable Importance –  Weak 2
 








### References
An, B. Y., & Tang, S.-Y. (2020). Lessons From COVID-19 Responses in East Asia: Institutional Infrastructure and Enduring Policy Instruments. The American Review of Public Administration, 50(6-7), 790–800. https://doi.org/10.1177/0275074020943707
Bojanic, D. C., & Lo, M. (2016). A comparison of the moderating effect of tourism reliance on the economic development for islands and other countries. Tourism Management, 53, 207–214. https://doi.org/10.1016/j.tourman.2015.10.006
Farzanegan, M. R., Gholipour, H. F., Feizi, M., Nunkoo, R., & Andargoli, A. E. (2020). International Tourism and Outbreak of Coronavirus (COVID-19): A Cross-Country Analysis. Journal of Travel Research, 004728752093159. https://doi.org/10.1177/0047287520931593
Gössling, S., Scott, D., & Hall, C. M. (2020). Pandemics, tourism and global change: a rapid assessment of COVID-19. Journal of Sustainable Tourism, 29(1), 1–20. https://doi.org/10.1080/09669582.2020.1758708
Greer, S. L., King, E. J., da Fonseca, E. M., & Peralta-Santos, A. (2020). The comparative politics of COVID-19: The need to understand government responses. Global Public Health, 15(9), 1–4. https://doi.org/10.1080/17441692.2020.1783340
Hale, T., Angrist, N., Cameron-Blake, E., Hallas, L., Kira, B., Majumdar, S., Petherick, A., Phillips, T., Tatlow, H., & Webster, S. (2020). Variation in government responses to COVID-19 BSG Working Paper Series Providing access to the latest policy-relevant research. https://www.bsg.ox.ac.uk/sites/default/files/2020-09/BSG-WP-2020-032-v7.0.pdf
Holzner, M. (2011). Tourism and economic development: The beach disease? Tourism Management, 32(4), 922–933. https://doi.org/10.1016/j.tourman.2010.08.007
Jenkins, C. L. (2015). Tourism policy and planning for developing countries: some critical issues. Tourism Recreation Research, 40(2), 144–156. https://doi.org/10.1080/02508281.2015.1045363
Seyfi, S., Hall, C. M., & Shabani, B. (2020). COVID-19 and international travel restrictions: the geopolitics of health and tourism. Tourism Geographies, 1–17. https://doi.org/10.1080/14616688.2020.1833972
Tateno, Y., & Bolesta, A. (2020, May 1). Addressing the impact of the pandemic on tourism in Asia-Pacific small island developing States. Ideas.repec.org. https://ideas.repec.org/p/unt/pbmpdd/pb111.html
Watson, P., & Deller, S. (2021). Tourism and economic resilience. Tourism Economics, 135481662199094. https://doi.org/10.1177/1354816621990943

