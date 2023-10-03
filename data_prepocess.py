import pandas as pd
import numpy as np 
# from config import TRAIN_DATA_PATH





def data_preprocess(path:str):
    raw_data = pd.read_csv(path,encoding="utf-8")
    data = raw_data[["review", "polarity"]]
    data = data.dropna()
    data = data.sample(frac=0.01)

    X = np.array(data['review'])
    y = np.array(data['polarity'])

    return X,y



# if __name__ == "__main__":
#     X,y = data_preprocess(TRAIN_DATA_PATH)

#     print(y)