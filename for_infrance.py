from src.inferance import predict

sample = {
    "Age":50,
    "Sex":"Male",
    "Ethnicity":"Asian",
    "BMI":30,
    "Alcohol_Consumption":"Moderate",
    "Smoking_Status":"Never",
    "Family_History_of_Diabetes":1
}

print(predict(sample))