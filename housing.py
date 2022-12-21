from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json
from fastapi.middleware.cors import CORSMiddleware  

app = FastAPI()




class model_input(BaseModel):

    CODE_GENDER:float
    CNT_CHILDREN:float
    AMT_INCOME_TOTAL:float
    NAME_EDUCATION_TYPE:float
    NAME_FAMILY_STATUS:float
    NAME_HOUSING_TYPE:float



model = pickle.load(open(r"C:\Users\91976\Desktop\programming\AI and Ml\projects\survival predicton\housing risk\hr.pkl" , 'rb'))

@app.post('/predict_housing')
def survivalprediction(input_parameters:model_input):

    input_data = input_parameters.json()
    input_dict = json.loads(input_data)
    
    CODE_GENDER =  input_dict['CODE_GENDER']
    CNT_CHILDREN =  input_dict['CNT_CHILDREN']
    AMT_INCOME_TOTAL =  input_dict['AMT_INCOME_TOTAL']
    NAME_EDUCATION_TYPE =  input_dict['NAME_EDUCATION_TYPE']
    NAME_FAMILY_STATUS = input_dict['NAME_FAMILY_STATUS']
    NAME_HOUSING_TYPE =  input_dict['NAME_HOUSING_TYPE']
    

    input_list = [CODE_GENDER , CNT_CHILDREN , AMT_INCOME_TOTAL , NAME_EDUCATION_TYPE , NAME_FAMILY_STATUS , NAME_HOUSING_TYPE]

    prediction_proba = model.predict_proba([input_list])
    prediction = model.predict([input_list])

    if prediction[0]==0:
        return f"{str(prediction_proba[0][0])}"  
    else:
        return f"{str(prediction_proba[0][0])}"     

