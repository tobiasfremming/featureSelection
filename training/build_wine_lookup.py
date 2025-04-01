import sys
sys.path.insert(1, './Modules')

from Modules import DataManager
from Modules import Regression
from Modules import GenericModel
import pandas as pd
import time
import numpy as np
from sklearn.metrics import accuracy_score

LAMBDA = 0.3

df = pd.read_csv("../data_sets/wine.data")
data = DataManager.Data(df)

data.make_categorical("Class")
data.normalise_numeric()

data_x, data_y = data.input_output_split("Class")

results = {bin(0)[2:].zfill(len(data_x.columns)): 1}
penalties = {bin(0)[2:].zfill(len(data_x.columns)): 0}
totals = {bin(0)[2:].zfill(len(data_x.columns)): 1}

for i in range(1, 2**len(data_x.columns)):
    stime = time.time()
    bitmask = [bool(int(b)) for b in bin(i)[2:].zfill(len(data_x.columns))]
    masked_x = DataManager.Data(data_x.get_data().loc[:, bitmask])

    masked_all = masked_x.encode_categoricals()
    masked_all.add_cols(data_y.df)

    masked_all.make_numerical("Class")

    train_data, test_data = masked_all.train_test_split(0.8)

    train_x, train_y = train_data.input_output_split("Class")
    test_x, test_y = test_data.input_output_split("Class")

    model = Regression.BinaryRegression()
    model.set_err_function(accuracy_score)
    model.add_training_data(train_x, train_y)
    model.add_testing_data(test_x, test_y)
    model.fit()

    score = 1 - model.evaluate_testing_performance()
    results[bin(i)[2:].zfill(len(data_x.columns))] = score

    penalty = np.sum(np.abs(list(model.get_coefficients().values())))
    penalty = sum(bitmask)
    
    penalties[bin(i)[2:].zfill(len(data_x.columns))] = penalty
    
    totals[bin(i)[2:].zfill(len(data_x.columns))] = LAMBDA*penalty + score

    print(f"Model {i}: Misclass: {score} | Eval time: {time.time() - stime} seconds")

pd.to_pickle(totals, "../luts/wine.pickle")
pd.DataFrame(totals.items()).to_csv("../luts/wine.csv", index=False, header=False)
pd.to_pickle(results, "../luts/wine_err.pickle")
pd.DataFrame(results.items()).to_csv("../luts/wine_err.csv", index=False, header=False)
pd.to_pickle(penalties, "../luts/wine_pen.pickle")
pd.DataFrame(penalties.items()).to_csv("../luts/wine_pen.csv", index=False, header=False)