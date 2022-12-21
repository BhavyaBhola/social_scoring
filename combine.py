from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


class model_housing(BaseModel):

    CODE_GENDER: float
    CNT_CHILDREN: float
    AMT_INCOME_TOTAL: float
    NAME_EDUCATION_TYPE: float
    NAME_FAMILY_STATUS: float
    NAME_HOUSING_TYPE: float


class model_finance(BaseModel):
    Age: float
    Annual_Income: float
    Monthly_Inhand_Salary: float
    Amount_invested_monthly: float
    Num_Bank_Accounts: float


model1 = pickle.load(open(
    r"hr.pkl", 'rb'))


@app.post('/predict_housing')
def survivalprediction(input_parameters: model_housing):

    input_data = input_parameters.json()
    input_dict = json.loads(input_data)

    CODE_GENDER = input_dict['CODE_GENDER']
    CNT_CHILDREN = input_dict['CNT_CHILDREN']
    AMT_INCOME_TOTAL = input_dict['AMT_INCOME_TOTAL']
    NAME_EDUCATION_TYPE = input_dict['NAME_EDUCATION_TYPE']
    NAME_FAMILY_STATUS = input_dict['NAME_FAMILY_STATUS']
    NAME_HOUSING_TYPE = input_dict['NAME_HOUSING_TYPE']

    input_list = [CODE_GENDER, CNT_CHILDREN, AMT_INCOME_TOTAL,
                  NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS, NAME_HOUSING_TYPE]

    prediction_proba = model1.predict_proba([input_list])
    prediction = model1.predict([input_list])

    if prediction[0] == 0:
        return f"{str(prediction_proba[0][0])}"
    else:
        return f"{str(prediction_proba[0][0])}"


model2 = pickle.load(open(
    r"score_model.pkl", 'rb'))


@app.post('/predict_finance')
def survivalprediction(input_parameters: model_finance):

    input_data = input_parameters.json()
    input_dict = json.loads(input_data)

    Age = input_dict['Age']
    Annual_Income = input_dict['Annual_Income']
    Monthly_Inhand_Salary = input_dict['Monthly_Inhand_Salary']
    Amount_invested_monthly = input_dict['Amount_invested_monthly']
    Num_Bank_Accounts = input_dict['Num_Bank_Accounts']

    input_list = [Age, Annual_Income, Monthly_Inhand_Salary,
                  Amount_invested_monthly, Num_Bank_Accounts]

    prediction_proba = model2.predict_proba([input_list])
    prediction = model2.predict([input_list])

    if prediction[0] == 0:
        return f"{str(3.33 * prediction_proba[0][0])}"
    elif prediction[0] == 1:
        return f"{str(3.33 + (3.33 * prediction_proba[0][1]))}"
    else:
        return f"{str(6.66 + (3.33 * prediction_proba[0][2]))}"
