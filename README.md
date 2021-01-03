# FlipRobo-Assignments
•	Business Problem Framing
This is a classic Business problem which helps Micro Financing Institutions and other Lending companies reduce Credit risks by recognizing potential Defaulters. 
•	Conceptual Background of the Loan Delinquency	
Before advancement of Data Science, loan lending companies used to risk a high rate of defaulting. Many a times a perfect candidate would display erratic financial and repayment behaviour after being approved for loan. Machine Learning can help lenders predict potential defaulters before approving their candidature using their past data. The candidates’ income, past debt and repayment behaviour can be important metrics for the same.  
•	Review of Literature
1.	Data Exploration and Cleaning On data exploration, I found that the dataset was imbalanced for the target feature (87.5% for Non-defaulters and 12.5% for Defaulters). Also, I found that the data had some very unrealistic values such as 999860 days which is not possible. Also, there were negative values for variables which must not have one (example: frequency, amount of recharge etc…). All these unrealistic values were dropped which caused a data loss of 8% only.
2.	Feature Selection Since there were 36 features, many of which I suspected were redundant because of the data duplication. It was imperative to select only most significant of them to make ML models more efficient and cost effective. The method used was 'Univariate Selection' using chi-square test. I selected top 20 features which were highly significant.
3.	Data Visualization On visualizing data, there were two important insights I gathered. a. Imbalance of data b. Distribution was not normal
4.	Data Normalization Since the data was not normal, I normalized all the features except the target variable which was dichotomous (Values '1' and '0').
5.	Oversampling of Minority class since the data was expensive, I did not want to lose out on data by under sampling the majority class. Instead, I decided to oversample the minority class using SMOTE.
6.	Build Models Since it was a supervised classification problem, I built 5 models to evaluate performance of each of them: a. Logistic Regression b. Linear SVM c. Decision Tree d. Random forest e. Gradient Boost Classifier Since the data was imbalanced, accuracy was not the correct performance metric. Instead I focused on other metrics like precision, recall and ROC-AUC curve.
 
•	Motivation for the Problem Undertaken
Loan default is always the threat to any financial institution and should be predicted in advance based on various features of the applicant. This study aims at applying machine models, including decision tree, logistic regression and random forest to classify applicants with and without loan default from a group of predicting variables, and evaluate their performance. Comparison between using unbalanced training set and balanced training set suggests that balancing the data is the key to improve model performance.


Analytical Problem Framing
•	Mathematical/ Analytical Modeling of the Problem
Since the target variable (label) is dichotomous, our problem can be considered as a classification problem under supervised learning. We decided to use decision tree, logistic regression and random forest in our project. We realized that the target variable is highly unbalanced in the data. Only 12.5% of the target variable is 0 (Default). As we’ve learned, these classifiers will be influenced by highly unbalanced data and be more likely to fail to classify the minority label in the test set. However, in the real-life scenario, these default loan (labelled as 0) will be more harmful to the financial institution. So, we decided to use SMOTE (Synthetic Minority Over-Sampling Technique) to over-sample the minority group in the data and make both labels occupied 50% of the training set. We will evaluate the performance of the classifiers trained with unbalanced data and balanced data.
•	Data Sources and their formats
The Data Set is provided to us In order to improve the selection of customers for the credit, the client wants some predictions that could help them in further investment and improvement in selection of customers.
Below are the attributes provided in dataset.
Variable	Definition	Comment
label	Flag indicating whether the user paid back the credit amount within 5 days of issuing the loan{1:success, 0:failure}	 
msisdn	mobile number of user	 
aon	age on cellular network in days	 
daily_decr30	Daily amount spent from main account, averaged over last 30 days (in Indonesian Rupiah)	 
daily_decr90	Daily amount spent from main account, averaged over last 90 days (in Indonesian Rupiah)	 
rental30	Average main account balance over last 30 days	Unsure of given definition
rental90	Average main account balance over last 90 days	Unsure of given definition
last_rech_date_ma	Number of days till last recharge of main account	 
last_rech_date_da	Number of days till last recharge of data account	 
last_rech_amt_ma	Amount of last recharge of main account (in Indonesian Rupiah)	 
cnt_ma_rech30	Number of times main account got recharged in last 30 days	 
fr_ma_rech30	Frequency of main account recharged in last 30 days	Unsure of given definition
sumamnt_ma_rech30	Total amount of recharge in main account over last 30 days (in Indonesian Rupiah)	 
medianamnt_ma_rech30	Median of amount of recharges done in main account over last 30 days at user level (in Indonesian Rupiah)	 
medianmarechprebal30	Median of main account balance just before recharge in last 30 days at user level (in Indonesian Rupiah)	 
cnt_ma_rech90	Number of times main account got recharged in last 90 days	 
fr_ma_rech90	Frequency of main account recharged in last 90 days	Unsure of given definition
sumamnt_ma_rech90	Total amount of recharge in main account over last 90 days (in Indonasian Rupiah)	 
medianamnt_ma_rech90	Median of amount of recharges done in main account over last 90 days at user level (in Indonasian Rupiah)	 
medianmarechprebal90	Median of main account balance just before recharge in last 90 days at user level (in Indonasian Rupiah)	 
cnt_da_rech30	Number of times data account got recharged in last 30 days	 
fr_da_rech30	Frequency of data account recharged in last 30 days	 
cnt_da_rech90	Number of times data account got recharged in last 90 days	 
fr_da_rech90	Frequency of data account recharged in last 90 days	 
cnt_loans30	Number of loans taken by user in last 30 days	 
amnt_loans30	Total amount of loans taken by user in last 30 days	 
maxamnt_loans30	maximum amount of loan taken by the user in last 30 days	There are only two options: 5 & 10 Rs., for which the user needs to pay back 6 & 12 Rs. respectively
medianamnt_loans30	Median of amounts of loan taken by the user in last 30 days	 
cnt_loans90	Number of loans taken by user in last 90 days	 
amnt_loans90	Total amount of loans taken by user in last 90 days	 
maxamnt_loans90	maximum amount of loan taken by the user in last 90 days	 
medianamnt_loans90	Median of amounts of loan taken by the user in last 90 days	 
payback30	Average payback time in days over last 30 days	 
payback90	Average payback time in days over last 90 days	 
pcircle	telecom circle	 
pdate	date	 

•	Data Preprocessing Done
Although the dataset contains 209593 observations and 37 variables, it doesn’t mean that we can directly feed the dataset into the machine learning models. The model will have a low performance or even return errors if the data hasn’t been pre-processed. That’s the reason why we need to do some observations and pre-processing with the data set so that we can optimize the performance of our models. There are some negative values in each variable or observation can be a problem to our model. We removed the records with negative values. Too few unique data in the variable can be a problem so I removed single unique valued variables. We will transform this variable to a dichotomous variable, for example, we will change the majority value in that variable to one label and all the other values to another label
•	Hardware and Software Requirements and Tools Used
1.	This notebook should be run under python with all necessary Data Science packages, Anaconda environment is recommended.
2.	Training the model will occupy a lot of memory (about 3 GB or 2915MB), please turn off other programs if necessary.
3.	Please use conda install -c glemaitre imbalanced-learn in conda prompt to install SMOTE package, or follow instructions on http://contrib.scikit-learn.org/imbalanced-learn/stable/install.html

Model/s Development and Evaluation 

•	Identification of possible problem-solving approaches (methods)
Since the target variable (label) is dichotomous, our problem can be considered as a classification problem under supervised learning. We decided to use decision tree, logistic regression and random forest in our project. We realized that the target variable is highly unbalanced in the data. Only 12.5% of the target variable is 0 (Default). As we’ve learned, these classifiers will be influenced by highly unbalanced data and be more likely to fail to classify the minority label in the test set. However, in the real-life scenario, these default loan (labelled as 0) will be more harmful to the financial institution. So, we decided to use SMOTE (Synthetic Minority Over-Sampling Technique) to over-sample the minority group in the data and make both labels occupied 50% of the training set. We will evaluate the performance of the classifiers trained with unbalanced data and balanced data.  
•	Testing of Identified Approaches (Algorithms)
a.	Logistic Regression
b.	Linear SVM
c.	Decision Tree
d.	Random forest
e.	Gradient Boost Classifier 
•	Run and Evaluate selected models
We use three functions to automate the model training, testing and evaluation process.
  
We only need to call the higher-level function (train_eval_model) and pass in the training set and test set and all the models in dictionary format and we can get the report of the models in the format of pandas data frame, which is easy to visualize with matplotlib

•	Key Metrics for success in solving problem under consideration
 
We used four metrics to evaluate the model performance as shown in Table 3. In the real-life scenario, the financial institutions are more aware of the risk of misclassifying a bad loan applicant to a good one because these misclassifications will be more harmful to them comparing with losing some good applicants. Thus, we decide to mainly focus on recall score because it reflects the model’s ability to find all the positive samples, which means the default applicant in our scenario.










•	Visualizations
	On visualizing Label  vs Median of amounts of loan taken by the user in last 30 days, we notice that high number of loans taken people are mostly non-defaulters
  
	On visualizing Label  vs maximum amount of loan taken by the user in last 90 days, we notice that there is very low impact on default and non default.
 
	On visualizing Label  vs Number of loans taken by user in last 90 days, we notice that there is almost 50%  people are defaulters who are taking low count of loans in last 90 days
 
	On visualizing data, there were two important insights I gathered. a. Imbalance of data b. Distribution was not normal, for this in further we will do normalization of data
	Since the data was not normal, I normalized all the features except the target variable which was dichotomous (Values '1' and '0').
 
•	Interpretation of the Results
•	Oversampling of Minority class since the data was expensive; I did not want to lose out on data by under sampling the majority class. Instead, I decided to oversample the minority class using SMOTE
•	Since it was a supervised classification problem, I built 5 models to evaluate performance of each of them: a. Logistic Regression b. Linear SVM c. Decision Tree d. Random forest e. Gradient Boost Classifier Since the data was imbalanced, accuracy was not the correct performance metric. Instead I focused on other metrics like precision, recall and ROC-AUC curve.
While doing logistic regression, below is the before and after OverSampling counts	
 
	Logistic Regression Scores as below
 
	Decision Tree Precision, recall and f1-score are below 
 
	Random Forest Precision, recall and f1-score are below 
 
	Gradient Boost Classifier Precision, recall and f1-score are below 
 

VALIDATION CURVE
	From the above 4 models, we can identify that "RandomForestClassifier" is the best model with AUC score = 89%
	To fine tune the model values by using Validation curve 
	Random state = 1
	Max_depth = 15
	n_estimators = 500
	min_samples_split = 2
	min_samples_leaf = 1
 
 

CONCLUSION 

•	Learning Outcomes of the Study in respect of Data Science
This project focused on building different machine learning models and evaluating the performance of random forest, decision tree and logistic regression with unbalanced train data and balanced train data. We found that after oversampling the minority class by Synthetic Minority Over-Sampling Technique (SMOTE) in the training set, the recall score improved for every model, especially logistic regression. The recall score indicates that the logistic regression can classify 70% of the applicants that will be default. However, we can’t find a random forest model that works better than logistic regression.
•	Limitations of this work and Scope for Future Work
1.	This is an imbalanced data; if it is the balanced data then we can work more efficiently.
2.	We also think that the value of features in the dataset makes the decision boundary more like a linear boundary, which suits better to logistic regression instead of tree models. Besides the high recall score with balanced data, logistic regression also has other benefits such as easy to interpret by viewing the parameters, which is the reason why we would recommend the financial institutions to use logistic regression for default prediction
