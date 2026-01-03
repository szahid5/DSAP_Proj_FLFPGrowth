DSAP - Capstone project proposal 2025

Title: Identifying the Key Policy Drivers of Female Labor Force Participation (FLFP) Growth: A
Machine Learning Approach

(Initial proposal)
For my updated capstone project, I want to explore which economic, educational, and legal factors help
predict changes in Female Labor Force Participation (FLFP) across developing countries. Data from the
World Bank, UNESCO, the Women, Business and the Law (WBL) database, and Our World in Data will
be used to build a panel dataset from 1990 to 2020 in five-year intervals.
As the dependent variable, I can use the percentage change in FLFP over each period. For predictors,
female mean years of schooling, fertility rate, GDP per capita, and legal rights indicators from WBL can
be used. (Originally, I planned on using the secondary school enrollment, but due to missing data, I can
use the mean years of schooling). I can ten test four models in Python using scikit-learn: Linear
Regression, Ridge/LASSO, Random Forest, and XGBoost. K-fold cross-validation, using R² and RMSE
can help us check the model performance.
To understand which factors matter most, I’ll use SHAP values and permutation importance. The goal is
to combine prediction and interpretability, so we can see which policy areas are most strongly linked to
FLFP growth. This project will give me hands-on experience with the modeling, evaluation, and
interpretation techniques we’ve learned throughout the course. 

(After TA's feedback) 
Following the proposal feedback, I implemented Option 1 (Temporal Validation) for the model evaluation. The dataset was split by time periods, training on the 1991–2006 intervals to predict FLFP growth in the 2011–2016 periods. This approach was chosen to ensure the model captures the temporal structure of policy impacts and demonstrates true predictive power over time, rather than treating country-periods as independent cross-sections. 

Additionally, as suggested, I treated Ridge Regression as the primary linear regularization model and incorporated Gradient Boosting techniques within the final model selection pipeline. This allowed for a more robust comparison between linear interpretability and non-linear predictive performance.