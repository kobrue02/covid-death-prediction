import pandas as pd
from sklearn.preprocessing import RobustScaler


def process(df):

    # get rid of all boolean rows with cells that miss data (if they contain 99 or 97 it means data is missing)
    df = df[(df.PNEUMONIA == 1) | (df.PNEUMONIA == 2)]
    df = df[(df.DIABETES == 1) | (df.DIABETES == 2)]
    df = df[(df.COPD == 1) | (df.COPD == 2)]
    df = df[(df.ASTHMA == 1) | (df.ASTHMA == 2)]
    df = df[(df.INMSUPR == 1) | (df.INMSUPR == 2)]
    df = df[(df.HIPERTENSION == 1) | (df.HIPERTENSION == 2)]
    df = df[(df.OTHER_DISEASE == 1) | (df.OTHER_DISEASE == 2)]
    df = df[(df.CARDIOVASCULAR == 1) | (df.CARDIOVASCULAR == 2)]
    df = df[(df.OBESITY == 1) | (df.OBESITY == 2)]
    df = df[(df.RENAL_CHRONIC == 1) | (df.RENAL_CHRONIC == 2)]
    df = df[(df.TOBACCO == 1) | (df.TOBACCO == 2)]

    # If we have "9999-99-99" values that means this patient is alive.
    df["DEATH"] = [2 if each == "9999-99-99" else 1 for each in df.DATE_DIED]

    # Converting process according to inference above
    df.PREGNANT = df.PREGNANT.replace(97, 2)

    # Getting rid of the missing values
    df = df[(df.PREGNANT == 1) | (df.PREGNANT == 2)]

    df.drop(columns=["INTUBED", "ICU", "DATE_DIED"], inplace=True)

    # these columns have very little correlation with death so we drop them
    irrelevant = ["SEX", "PREGNANT", "COPD", "ASTHMA", "INMSUPR", "OTHER_DISEASE"]
    df.drop(columns=irrelevant, inplace=True)

    df = pd.get_dummies(df, columns=["MEDICAL_UNIT", "CLASIFFICATION_FINAL"], drop_first=True)

    # scaling numeric values
    scaler = RobustScaler()
    df.AGE = scaler.fit_transform(df.AGE.values.reshape(-1, 1))

    return df