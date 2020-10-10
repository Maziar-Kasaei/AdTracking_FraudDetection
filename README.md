# AdTracking_Fraud
 Kaggle competetion: TalkingData AdTracking Fraud Detection Challenge

## Summary
The complete data can be found in the original competetion page (https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/53886). Here a subset of the data is being used only, and it is being provided in the train_sample folder. The training has been done chunk-by-chunk to fit into small laptop memory. Models tested were Random Forest, Neural Networks, XGBoost and SVM. However, the code (in Python) only reflects the XGBoost model.

1. Explanatory Data Analysis

Training data has 184,903,890 rows and test data has 18,790,469 instances. So, we are dealing with a big data problem. Each row of training data has the following features
-	ip: ip address of click.
-	app: app id for marketing.
-	device: device type id of user mobile phone (e.g., iphone 6 plus, iphone 7, etc.)
-	os: os version id of user mobile phone
-	channel: channel id of mobile ad publisher
-	click_time: timestamp of click (UTC)
-	attributed_time: if user download the app for after clicking an ad, this is the time of the app download
-	is_attributed: the target that is to be predicted, indicating the app was downloaded

Test data is like training data except that it excludes variable attributed_time and uses click_id variable to index clicks. Table 1 and Table 2 show the first 4 rows of train and test data respectively.


