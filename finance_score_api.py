from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


class model_input(BaseModel):

    Age: float
    Annual_Income: float
    Monthly_Inhand_Salary: float
    Amount_invested_monthly: float
    Num_Bank_Accounts: float


model = pickle.load(open(r"score_model.pkl", 'rb'))


@app.post('/predict_finance')
def survivalprediction(input_parameters: model_input):

    input_data = input_parameters.json()
    input_dict = json.loads(input_data)

    Age = input_dict['Age']
    Annual_Income = input_dict['Annual_Income']
    Monthly_Inhand_Salary = input_dict['Monthly_Inhand_Salary']
    Amount_invested_monthly = input_dict['Amount_invested_monthly']
    Num_Bank_Accounts = input_dict['Num_Bank_Accounts']

    input_list = [Age, Annual_Income, Monthly_Inhand_Salary,
                  Amount_invested_monthly, Num_Bank_Accounts]

    prediction_proba = model.predict_proba([input_list])
    prediction = model.predict([input_list])

    if prediction[0] == 0:
        return f"{str(3.33 * prediction_proba[0][0])}"
    elif prediction[0] == 1:
        return f"{str(3.33 + (3.33 * prediction_proba[0][1]))}"
    else:
        return f"{str(6.66 + (3.33 * prediction_proba[0][2]))}"
