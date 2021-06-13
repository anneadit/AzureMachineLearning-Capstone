
# Azure Machine Learning Capstone Project

In this capstone project, I use the Titanic passenger dataset and train models using both the Auto ML and Hyperdrive APIs. After training, I deploy the best performing Auto ML model as a webservice on an Azure Container Instance and consume it. This project demonstrates my ability to use an external dataset in your workspace, train a model using the different tools available in the AzureML framework as well as your ability to deploy the model as a web service.

![image](https://user-images.githubusercontent.com/38438203/121793910-b5955c00-cbd1-11eb-8762-e8505fb976a2.png)

Fig 1: Project Overview

## Project Set Up and Installation
I used my own pay-as-you-go Azure subscription for this project. I have selected the cheapest compute instances and computer clusters for this project in order to minimize costs and also because the computation is not very intensive.

## Dataset

### Overview

The Titanic dataset contains real information of 887 Titanic passengers. This dataset is commonly used to test out classification algorithms and practice ML workflows. The dataset contains the following features:

survived - Survival (0 = No; 1 = Yes)
class - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
name - Name
sex - Sex
age - Age
sibsp - Number of Siblings/Spouses Aboard
parch - Number of Parents/Children Aboard
ticket - Ticket Number
fare - Passenger Fare
cabin - Cabin
embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat - Lifeboat (if survived)
body - Body number (if did not survive and body was recovered)


I downloaded the dataset from https://data.world/nrippner/titanic-disaster-dataset


### Task

This will be an ML classification task with the aim of predicting whether a particular passenger survives given information from the following attributes:

class - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
sex - Sex
age - Age
sibsp - Number of Siblings/Spouses Aboard
parch - Number of Parents/Children Aboard
fare - Passenger Fare
cabin - Cabin
embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

The target column for this task is the survived column (0 = No; 1 = Yes))

Please note that I have removed Names and Ticket Numbers because in my opinion they are not relevant factors in deciding whether a person will survive. I have also removed Lifeboat and Body information as they can be used as a proxy for the target variable.

### Access

I have downloaded the dataset as a .csv file from https://data.world/nrippner/titanic-disaster-dataset and then registered it as 'titanic-prediction' dataset in a workspace blobstore  

## Automated ML
I will be running AutoML with the following settings and configuration.

automl_settings = {
    "experiment_timeout_minutes": 20,
    "max_concurrent_iterations": 1,
    "primary_metric" : 'AUC_weighted'
    }
    
automl_config = AutoMLConfig(compute_target=compute_target,
                             task = "classification",
                             training_data=dataset,
                             label_column_name="survived",   
                             path = project_folder,
                             enable_early_stopping= True,
                             featurization= 'auto',
                             debug_log = "automl_errors.log",
                             **automl_settings
                            )
                            
We will be setting the experiment to be timed out at 20 minutes to cut down on runtime and cost. Since the maximum number of nodes for the compute cluster is 1, the maximum number of concurrent iterations is set to 1. The primary metric is set to a weighted AUC since this is a classification problem with imbalanced classes and I want to assess the tradeoff between the true positive and false positive rates. While the primary metric is weighted AUC, I will still be keeping an eye on the accuracy metric. The target column is called 'survived'. Early stopping is enabled for time and cost efficiency. Featurization is set to auto which is default.

### Results

The top performing model was the Voting Ensemble model using XGBoost with a weighted AUC of 0.87 and an accuracy of 0.81. The parameters of the model are as follows:
"param_kwargs": {
        "booster": "gbtree",
        "colsample_bytree": 1,
        "eta": 0.3,
        "gamma": 0,
        "grow_policy": "lossguide",
        "max_bin": 63,
        "max_depth": 10,
        "max_leaves": 127,
        "n_estimators": 10,
        "objective": "reg:logistic",
        "reg_alpha": 0,
        "reg_lambda": 1.25,
        "subsample": 0.6,
        "tree_method": "hist"
    }
    
I could have maybe achieved a better performing model if I had let the experiment run longer or if I had enabled Deep Learning.

![image](https://user-images.githubusercontent.com/38438203/121794675-56871580-cbd8-11eb-9979-6e938cc84b95.png)

Fig 2: Results from the RunDetails widget

![image](https://user-images.githubusercontent.com/38438203/121794692-728ab700-cbd8-11eb-9339-8cbbbfe222a3.png)

Fig 3: List of all the models from the RunDetails Widget

## Hyperparameter Tuning

Since this is a classification problem, we will be training a Logistic Regression classifier. he reason we used a random parameter sampler is beacuse it can help identify the best hyperparameters in shorter time than an exhastive grid search. Random sampling also searches more of the hyperparameter space that a grid search if the grid search is poorly defined.

Bandit policy is an early termination policy based on slack factor and evaluation interval. Bandit ends runs when the primary metric isn't within the specified slack factor of the most succesful run. The default bandit policy numbers from Microsoft's documentation were used.

I chose the following list of choices for the random sampler because these numbers cover a wide enough variation in magnitude and I also don't want the training to take too long.

"--C": choice(0.01,0.05,0.1,0.5,1),
"--max_iter":choice(30,50,100)


### Results

The best performing model had an accuracy of 0.79 and used 'Regularization Strength:': 0.5, 'Max iterations:': 100 as the hyperparameter values. I could have maybe achieved a better performing model if I expanded my list of choices for the the two hyperparameters or maybe used an ensemble classification model instead of a simple Logistic Regression model. 

![image](https://user-images.githubusercontent.com/38438203/121794989-dada9800-cbda-11eb-8adf-eb09d3802937.png)

Fig 4: RunDetails list of models for Hyperdrive

![image](https://user-images.githubusercontent.com/38438203/121795030-fcd41a80-cbda-11eb-830a-41f6a9c3fc95.png)

Fig 5: Best performing Hyperdrive model

## Model Deployment

Since the best AutoML model had a higher accuracy than the best Hyperdrive model (0.81 vs 0.79), I decided to depoly the best AutoML model as a webservice on an Azure Container Instance with Authentication and Application Insights enabled.

![image](https://user-images.githubusercontent.com/38438203/121795117-98658b00-cbdb-11eb-97fd-6490c0c26776.png)

Fig 6: Deploying the best AutoML Model

After deploying the model, I consumed the endpoint by sending a couple of test instances and retrieving the results.

![image](https://user-images.githubusercontent.com/38438203/121795161-26417600-cbdc-11eb-8fb5-b48f835c3ecc.png)

Fig 7: Data sent for deployed endpoint consumption

![image](https://user-images.githubusercontent.com/38438203/121795173-483af880-cbdc-11eb-973c-d17792acb6c3.png)

Fig 8: Successful retrieval of test data results

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response
