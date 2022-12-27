from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from imblearn.under_sampling import RandomUnderSampler
from data_preprocessing import process
from model import model

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# read covid data and pre process it
data = pd.read_csv("Covid Data.csv")
df = process(data)

# resampling the data so death and no death is even
rus = RandomUnderSampler(random_state=0)
x, y = rus.fit_resample(df.drop(columns="DEATH"), y=df["DEATH"])

# 80% for training and 20% for testing
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)
model.fit(train_x, train_y)

y_pred = model.predict(test_x)

report = classification_report(test_y, y_pred)
print(report)


