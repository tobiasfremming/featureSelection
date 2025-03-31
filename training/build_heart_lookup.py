import sys
sys.path.insert(1, './Modules')

from Modules import DataManager
from Modules import Regression
from Modules import GenericModel
import pandas as pd
import time
from sklearn.metrics import accuracy_score
import numpy as np

LAMBDA = 0.3

df = pd.read_csv("../data_sets/heart.csv")
data = DataManager.Data(df)

data.make_categorical("HeartDisease")
data.normalise_numeric()

train_data, test_data = data.train_test_split(split_ratio=0.8)

train_x, train_y = train_data.input_output_split("HeartDisease")
test_x, test_y = test_data.input_output_split("HeartDisease")

results = {bin(0)[2:].zfill(len(train_x.columns)): 1}
penalties = {bin(0)[2:].zfill(len(train_x.columns)): 0}
totals = {bin(0)[2:].zfill(len(train_x.columns)): 1}

for i in range(1, 2**len(train_x.columns)):
    stime = time.time()
    bitmask = [bool(int(b)) for b in bin(i)[2:].zfill(len(train_x.columns))]
    masked_train_x = DataManager.Data(train_x.get_data().loc[:, bitmask])
    masked_test_x = DataManager.Data(test_x.get_data().loc[:, bitmask])

    masked_train_x = masked_train_x.encode_categoricals()
    masked_test_x = masked_test_x.encode_categoricals()

    model = Regression.BinaryRegression()
    model.set_err_function(accuracy_score)
    model.add_training_data(masked_train_x, train_y)
    model.add_testing_data(masked_test_x, test_y)
    model.fit()

    score = 1 - model.evaluate_testing_performance()
    
    results[bin(i)[2:].zfill(len(train_x.columns))] = score

    penalty = np.sum(np.abs(list(model.get_coefficients().values())))
    penalty = len(train_x.columns)
    
    penalties[bin(i)[2:].zfill(len(train_x.columns))] = penalty
    
    totals[bin(i)[2:].zfill(len(train_x.columns))] = LAMBDA*penalty + score

    print(f"Model {i}: Misclass: {score} | Eval time: {time.time() - stime} seconds")

pd.to_pickle(totals, "../luts/heart.pickle")
pd.DataFrame(totals.items()).to_csv("../luts/heart.csv", index=False, header=False)
pd.to_pickle(results, "../luts/heart_err.pickle")
pd.DataFrame(results.items()).to_csv("../luts/heart_err.csv", index=False, header=False)
pd.to_pickle(penalties, "../luts/heart_pen.pickle")
pd.DataFrame(penalties.items()).to_csv("../luts/heart_pen.csv", index=False, header=False)