import pandas as pd
import numpy as np
#from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
#from sklearn.model_selection import train_test_split



#print(df.describe())
def cleaning_loan(filename, train = True):
    df = pd.read_csv(filename)
    le = LabelEncoder()
    '''
    Cleaning Gender column
    '''
    df['Gender'].fillna('Male', inplace = True)
    dummy_gender = pd.get_dummies(df['Gender'], prefix = 'Gender',drop_first = True)
    df = pd.concat([df, dummy_gender], axis = 1)
    #df.drop(['Gender'],axis=1,inplace = True)
    
    '''
    Cleaning Married column
    '''
    df['Married'].fillna('Yes', inplace = True)
    dummy_married = pd.get_dummies(df['Married'], prefix = 'Married',drop_first = True)
    df = pd.concat([df, dummy_married], axis = 1)
    #df.drop(['Married'],axis=1,inplace = True)
    
    
    '''
    Cleaning Dependents column
    '''
    #Visualize correlation between Married and Dependents
    #temp_1 = pd.crosstab(df['Married_Yes'],df['Dependents'])
    #temp_1.plot(kind = 'bar',stacked = True,color=['red','blue','green','yellow'])
    
    df['Dependents'].fillna('0', inplace = True)
    
    df['Dependents_coded'] = le.fit_transform(df['Dependents'])
    
    
    '''
    Cleaning Education column
    '''
    df['Education_coded'] = le.fit_transform(df['Education'])
    
    
    '''
    Cleaning Self_Employed column
    '''
    df['Self_Employed'].fillna('No',inplace = True)
    df['Self_Employed_coded'] = le.fit_transform(df['Self_Employed'])
    
    '''
    Combining ApplicantIncome and CoapplicantIncome as TotalIncome
    '''
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['TotalIncome_log'] = np.log(df['TotalIncome'])
    
    
    '''
    Cleaning LoanAmount column
    '''
    loan_amounts_medians = df.pivot_table(values='LoanAmount', index='Self_Employed' ,
                           columns='Education', aggfunc=np.median)
    
    def loan_amount(x):
     return loan_amounts_medians.loc[x['Self_Employed'],x['Education']]
    
    df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(loan_amount, axis=1), 
                            inplace=True)
    
    '''
    Cleaning Loan_Amount_Term column
    '''
    df['Loan_Amount_Term'].fillna(360.0, inplace = True)
    
    
    '''
    Cleaning Credit_History column
    '''
    df['Credit_History'].fillna(1.0, inplace = True)
    
    
    '''
    Cleaning Property_Area  column
    '''
    dummy_Property_Area = pd.get_dummies(df['Property_Area'], prefix = 'Property',drop_first = True)
    df = pd.concat([df, dummy_Property_Area], axis = 1)
    

    '''
    Droping columns
    '''
    df = df.drop(['Gender','Married','Dependents','Education',
                  'Self_Employed','ApplicantIncome','CoapplicantIncome','TotalIncome',
                  'Property_Area'], axis=1)
    
    if train:
    
        X = df.drop(['Loan_Status','Loan_ID'],axis = 1)
        y = df['Loan_Status'].values
        
        return df, X, y
    else:
        X = df
        results = pd.DataFrame(df['Loan_ID'])
        
        return df, X, results

#test_file = 'loan_prediction_train.csv'
#df, X, y = cleaning_loan(test_file)