import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

symptomModel = "sModel.pkl"
clinicalModel = "cModel.pkl"

data = pd.read_csv("pcos_cleaned.csv")
print("data loaded")

if not os.path.exists(symptomModel):

    sym_cols=[' Age (yrs)','Weight (Kg)','Height(Cm) ','Cycle(R/I)','Cycle length(days)',
          'Pregnant(Y/N)','No. of abortions','Weight gain(Y/N)','hair growth(Y/N)',
          'Skin darkening (Y/N)','Hair loss(Y/N)','Pimples(Y/N)','Fast food (Y/N)','Reg.Exercise(Y/N)',]
    x_sym=data[sym_cols]
    y_sym_res=data['PCOS (Y/N)']

    x_train,x_test,y_train,y_test=train_test_split(x_sym,y_sym_res,random_state=42,stratify=y_sym_res)

    sym_pipeline = Pipeline([
        ('scaler',StandardScaler()),
        ('model',LogisticRegression(class_weight='balanced', random_state=42 , max_iter=1000))
    ])

    sym_pipeline.fit(x_train,y_train)

    joblib.dump(sym_pipeline,symptomModel)

    print("sym_model trained")
    
if not os.path.exists(clinicalModel):
    
    x=data.drop(columns=['PCOS (Y/N)'])
    y=data['PCOS (Y/N)']

    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,stratify=y)

    c_pipeline = Pipeline([
        ("scaler",StandardScaler()),
        ("model",LogisticRegression(class_weight='balanced', random_state=42 , max_iter=1000))
    ])

    c_pipeline.fit(x_train,y_train)

    joblib.dump(c_pipeline,clinicalModel)

    print("c_model trained")