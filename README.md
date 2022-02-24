# Project NephronAI
*Early detection and etiology of chronic kidney disease using deep learning*
***<br>Ayush Anand***

### Table of Contents
- [Abstract](#Abstract)
- [Question](#Question)
- [Hypothesis](#Hypothesis)
- [Background Research](#Background)
- [Procedure](#Procedure)
- [Data Analysis](#Data)
- [Functioning](#Functioning)
- [Conclusion](#Conclusion)
- [Future Research](#Future)
- [Ackowledgements](#Acknowledgements)
- [Contribute to Project](#Contribute)
- [Links to live demo](#Links)
- [External References](#References)



## Abstract
The kidneys are complex and resilient organs, each roughly the size of a fist. But many medical conditions can strain the kidneys, including diabetes, and high blood pressure causing chronic kidney disease. Chronic Kidney Disease is a major public health problem that affects 1.6 billion people globally each year. Around 10% of world population suffers from the disease and can cause complications including kidney failures, an increased risk of heart disease, high blood pressure, bone disease, and anemia. Chronic Kidney disease is a five-stage disease, often has no symptoms in its early stages and can go undetected until it is very advanced (For this reason, Chronic Kidney Disease is also referred to as a “silent disease”). Kidney disease causes gradual loss of kidney function over a period of months to years.
Each year, chronic kidney disease kills more people than breast or prostate cancer. And patient awareness is less than 10% in stages 1 to 3. Unfortunately, there has not been much research done on fighting the disease and there is an apparent lack of seriousness among public health officials against this silent healthcare crisis.
Chronic kidney disease can be treated. With early diagnosis and treatment, it's possible to slow or stop the progression of kidney disease.
Existing research works (even though very scarce) have focused only on detection of the disease. However, in order to tackle a disease on such a large scale a more comprehensive work must be done. Therefore, I came up with the idea to tackle chronic kidney disease using a three step approach – early detection, knowing the underlying cause and recommending a lifestyle approach to minimize risks – using one of the most advanced technology in computational intelligence, ‘deep learning’. The work has been implemented to make a web-based solution – NephronAI, which is accessible to anybody for free with the ease of a web browser.
A typical prediction using NephronAI (diagnosis, etiology, and creation of recommended approach) takes less than 10 minutes for a new user, without costing any money. Therefore, it proves to be a zero-cost, accessible, and comprehensive, tool to tackle Chronic Kidney Disease.


## Question
*How can we fight Chronic Kidney Disease using deep learning?*<br>
*How to create an AI system that not just only detects diseases but also understand the underlying cause?*


## Hypothesis
I was successful in making a web-based solution, ***NephronAI*** for **“Early detection** and **understanding of** underlying **cause** of a **Chronic Kidney Disease** using **deep learning**"


## Background Research
Kidneys are essentials organs that filter some 140 litres of blood each day, leaving behind a litre or two of water and waste in the form of urine. Each kidney features a latticework of roughly one million tiny filtering units, called nephrons. Blood entering a nephron passes through a cluster of tiny vessels called the glomerulus. The thin walls of the glomerulus enable waste, water and other small molecules to pass through, while blocking larger ones such as proteins and blood cells. From there, the filtered fluid flows into kidney tubules, where the balance of minerals, water, salts and glucose is calibrated and molecules necessary for bodily functions are reabsorbed into the bloodstream. But many medical conditions are depleting the function of kidneys. Diabetic people, and those suffering from hypertension are more likely to develop chronic kidney disease. In the most advanced stages of Chronic Kidney Disease the survival rate is less than 5 years- shorter than for many cancers.
The treatment is also very costly, as much as $91,000 annually per patient in the United States. And as well as using a lot of water, the current approach consumes vast quantities of power and materials such as plastics. However, if we increase patient awareness, make self-testing of kidney disease easy for vulnerable groups than we can save millions of lives and billions of dollars each year.
The diagnosis of chronic kidney disease typically involves a blood test to measure the glomerular filtration rate (to check how well your kidneys are filtering your blood) and a urine test to check for albumin. If your glomerular filtration rate is lower than normal, that indicates chronic kidney disease. While access to physicians and high-grade healthcare facilities is still not available to many poor people, plus some physicians will turn away patients not because they do not need care but to reduce lab costs or because their insurers pay relatively less. A computational approach can effectively address all the above issues- cost, accessibility, and accuracy. 


## Procedure
The project entailed using deep learning for fighting chronic kidney disease through a three phased approach – detecting the disease, finding the underlying cause and then recommending some changes to lifestyle in order to reduce the effects of the disease. This section describes the proposed methodology for designing and building the application and has been divided into 3 sections.


**A | Diagnosis of the disease** 
For the diagnosis of the disease a deep learning neural network using a dataset on chronic kidney disease has been used. The obtained dataset was converted to a machine readable (CSV) format. This dataset contains 24 features + class = 25 columns with around 400 rows of data most of which are clinical in nature. Using such a high number of parameters on a very small database would result in underfitting i.e. less accuracy in real world. Therefore, to fit for the data only significant columns must be kept and insignificants removed. This is known as Feature selection and is an extensively used data preprocessing technique in data mining. It not only improves performance of the model but also reduces training time and better correlation.
Features were selected using one correlation-based algorithm with eight different searching techniques. To start with it, we completely randomized the data sets with missing records. Missing tuples were excluded leaving 158 data sets for use in this work without missing values. Correlation-based feature subset selection (CFS) technique was used to do the task, which says that a feature of a subset is considered good which are highly correlated with the class but may be uncorrelated with other features of the class. The process was long and can be understood with the help of the following flowchart. 

We apply CFS with the eight search techniques (namely, Best First, Exhaustive Search, Genetic Search, Greedy Stepwise, Linear Forward Selection, Random Search, Scatter Search, and Subset Size Forward Selection) using a free software named WEKA. Out of the eight search algorithms, six algorithms, a majority, suggested eight common reduced attributes – specific gravity, albumin, serum creatinine, hemoglobin, packed cell volume, white blood cell count, red blood cell count, and hypertension, plus the label. These six attributes were included, and rest removed from the original dataset. This new dataset which contains just 8 parameters and the label (whether chronic kidney disease or not) is now saved and converted to CSV format for easy visualization. The dataset however contains And, to prevent under-fitting the missing values must be eliminated using a suitable algorithm (MICE in this case). 
Benefits of using MICE: For each missing value, this method assigns a new value, which is calculated by using a method described in the statistical literature as "Multivariate Imputation using Chained Equations" or "Multiple Imputation by Chained Equations". With a multiple imputation method, each variable with missing data is modeled conditionally using the other variables in the data before filling in the missing values.
Using seaborn module in Python3, this new data was visualized to identify trends in the features compared to the label. The trends have been discussed in the “Data Analysis” section. The cleaned data is then split into training set (70%) and validation or testing set (30%). A Two class decision Tree algorithm was applied on the cleaned dataset (with feature selection). Performance measures such as accuracy, false positives, false negatives, specificity, AUC (Area Under Curve) for Receiver Operating Characteristics (ROC) curve are used to evaluate the trained model.
This trained model is then designed to function as a REST API. A new web application is then created in a way to take inputs from the user on the 8 attributes and integrated with the API to return a predicted value (chronic kidney disease or no chronic kidney disease).

**B | Knowing the cause of the disease** 
Doctors often ask their patients a series of questions in order to narrow down the set of plausible conditions matching the observed symptoms. This section of the application was designed to investigate the cause for the kidney disease based on the answers provided by the user to some pre-defined questions.
For this experiment, A QA system such as a symptom checker which enables the emulation of this conventional approach by asking the relevant questions in order to refine the differential diagnosis is the closest approach to look at. The dataset used for this section contains a list of simple and complex questions with their answers and the associated label (being the cause). An RNN based neural network to find correlation between the response to the last questions with a reinforcement learning (RL) methodology is used. A reward function is used to measure the similarities between the candidate (machine generated) and abstract answers(data). 
The basic principle of working of the algorithm is very simple. Let there be N causes and M questions, here the neural network is used to evaluate weights for each question (using RNN) and multiply it with the value of answer it receives from the user for a question. Users answer the particular question, in the form of yes/no/don't know/, and their value is some float in the interval [0,1] (a form of fuzzy logic; (0 being less likely and 1 being very likely). This value is multiplied by the weights for the particular question and this score is recorded for each cause. 
At the end, when the user has answered all questions the result set containing evaluated scores for all causes is obtained and the highest ranked amongst them, which needs to be greater than a set minimum value is returned as the potential cause. If the highest ranked score does not pass the above test, then the algorithm returns ‘other causes’ as the result.

**C | Recommending the lifestyle change** 
The application does not stop here, it also recommends a change in lifestyle to ease the ill-effects of the disease. It lays out the biological principle how the prime cause, as supplied by the previous layer of the application, causes chronic kidney disease in a simplified manner. Plus, it also outlines what additional steps the end user must take to fight the disease based on the questions answered by the user in the previous step. These recommended steps include information on diet intake, exercise, health habits, etc.
All three layers/sections of the application are integrated together to create a comprehensive and immersive solution to help fight chronic kidney disease. The UI of the application is also intuitive and easy to use with instructions in basic English.



## Data Analysis
**Results for deep learning model:** Chronic Kidney Disease detection
In this section the results for the experiment done for chronic kidney disease detection are discussed. The table below results of the performance measures for the model such as accuracy, false positives, false negatives, specificity, AUC (Area Under Curve) for Receiver Operating Characteristics (ROC) curve are shown.


**Trends in data: Chronic Kidney Disease detection**<br>
Data for detection of chronic kidney disease clearly show correlation between parameters used to assess and the probability of being affected. For the diagnosis of chronic kidney disease a comprehensive data containing specific gravity[1], albumin[2] (in urine), serum creatinine[3], hemoglobin[4], packed cell volume[5], white blood cell count[6], red blood count[7], hypertension[8] from the lab reports of patients were analyzed. The trend between these values and the diagnosis is outlined below.

**Trends in data: Causes behind Chronic Kidney Disease**<br>
From data available at the US NIH, the fact that some medical conditions account for development of chronic kidney disease in many patients is evident.

Individuals who are diabetic (both type I and II) the risk for developing chronic kidney disease is very high. The following data shows the rate per million population of those requiring hemodialysis (treatment in advanced stages of chronic kidney disease).


## Functioning
The application works via a three-phase approach – diagnosis or detection of disease, etiology or finding the underlying cause, and the last step recommending a lifestyle approach to reduce complications based on the cause detected.

The following graphic shows the way the application works.
Requirements to use the service:
-	A web browser (mobile, or desktop, etc.), and
-	Internet Connectivity.


## Conclusion
The result of the experimental project demonstrates that the proposed methodology can not only effectively learn how to detect chronic kidney disease but also investigate its cause. For instance, the neural network to diagnose patients clocked an accuracy of 99% over test data. Upon large-scale implementation, the result will prove to be superior to any previous screening methods because,
+	it does not require any hefty scientific or technical investment and can be used from comfort of any home around the world “without” investing on any third-party software or integrating the main the main network.
+	this project aims at targeting a huge set of affected individuals, who are either unaware of their disease or lie in the more vulnerable group.
+	using even lesser parameters than a physician would require, this solution produced an effective accuracy of 99% in detecting chronic kidney disease over the test data.
+	it does not require any specialized training or any kind of technical knowledge to use and user friendly.
+	it is “completely free”, which makes it available to people all around the world without any regard to financial status.
+	the project is embodying a more comprehensive approach to tackle chronic kidney disease. No other research in the field have focused on knowing the cause or even recommending the patient a lifestyle change.
+	this project not only addresses those patients whose primary diagnosis come out to be positive but also those who do not seem to be affected yet. “Individuals who have the disease get the cause and a lifestyle recommendation, but those who do not have the disease too get a checklist so that they stay alert and do not develop the disease in future.”
	


## Future Research
I would wish to pursue further research on the project with the following objectives:
+ How artificial intelligence can be applied to benefit the process of ‘hemodialysis’ (the treatment to chronic kidney disease).
+ Investigate into other parameters which may help us in more accurate and feasible diagnosis of chronic kidney disease.
+ How to fully integrate this project into real-world healthcare systems to increase adoption and accessibility.

**Commercial prospects:** There are no prospects of monetizing or earning profits from this service anytime in the future. Healthcare institutions, public bodies, and/or other concerned organizations are welcome to pair up with this project to deliver this project to the end-user on a mass scale.


## Acknowledgements
I would like to show my gratitude to the anonymous reviewers for sharing their pearls of wisdom with us during the course of documenting this research, and I thank my 3 friends for their insights in order to improve user experience with the project interface.

We are accepting reviews and open-source contributions for the project. We'll be glad to hear your feedback.

## Contribute to the Project
If you would like to contribute to this open project, it would be really great. You could help the world in fighting chronic kidney disease plus your name will be featured here.
The following steps must be used to contribute
+ Fork the repository
+ Change the code or add files
+ Create a new issue
+ Provide detailed summary of the changes in the commit as:
````
The commit message must have three parts,
Feat: *The title of the commit (what are you going to change)*
Desc: *Detail the changes you have brought*
Info: *Provide references to the issue*
````
+ Start a pull request

If you would rather like to contribute to the project other than coding (grants, promotion, or guidance), [then please write to us here](https://ayushanand18.github.io/nephron-ai/contribute).
We will get back to you as soon as possible.
All your criticism and feedback is welcome, and we are equally eager to hear them.


### Links
The Service can be accessed online through any device using a web browser. [Check it out](https://ayushanand18.github.io/nephron-ai/)


### References
References have been taken through various resources from the internet which are listed below. We thank all the resource owners, we don't claim their ownership nor authenticity.

[Rajesh Misir, Malay Mitra, and Ranjit Kumar Samanta, 2017 June 19](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5497482/)

[Ajay K Singh Youssef MK Farag and Mohan M Rajapurkar, 2013 May 28](https://bmcnephrol.biomedcentral.com/articles/10.1186/1471-2369-14-114)

[Causes - Chronic Kidney Disease](https://www.niddk.nih.gov/health-information/kidney-disease/chronic-kidney-disease-ckd/causes)

[Heavy Metal Poisoning](https://rarediseases.info.nih.gov/diseases/6577/heavy-metal-poisoning)

[High Blood Pressure](https://www.webmd.com/hypertension-high-blood-pressure/guide/hypertension-symptoms-high-blood-pressure)

[Diabetes](https://www.niddk.nih.gov/health-information/diabetes/overview/symptoms-causes)

[Webmd - Heavy Metal Poisoning](https://www.webmd.com/a-to-z-guides/what-is-heavy-metal-poisoning#1)

[How to fight Diabetes](https://www.webmd.com/diabetes/tips-diabetes-lifestyle#1)

[How to lower blood sugar level](https://www.healthline.com/nutrition/15-ways-to-lower-blood-sugar#section15)

[How to lower High Blood Pressure](https://www.healthline.com/health/high-blood-pressure-hypertension/lower-it-fast#8)

[How to cure Heavy Metal Poisoning](https://www.healthline.com/health/heavy-metal-poisoning)

[How to prevent Kidney Failure](https://www.healthline.com/health/kidney-health/how-to-prevent-kidney-failure#11-tips)

[Data for the Machine Learning project](http://archive.ics.uci.edu/ml)

Dataset Attribution:
```
@misc{Dua:2019 ,
author = "Dua, Dheeru and Graff, Casey",
year = "2017",
title = "{UCI} Machine Learning Repository",
url = "http://archive.ics.uci.edu/ml",
institution = "University of California, Irvine, School of Information and Computer Sciences" }
```
<br><br><br>
We thank everybody whose even indirect contribution has helped this project, <br>
With &hearts; from <br>
**Ayush Anand** <br>
`&copy; 2022`
----

PS. To contact the repository owner or to get more information on how to contribute, please write [us here](https://ayushanand18.github.io/nephron-ai/contribute)

*<small> Based on a small test dataset from Apollo Hospitals publicly available on the web.