import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


def load_data(data_name, seed):
    if data_name=="naval":
        print("maintenance of naval propulsion plants dataset")

        filename='../data/naval/data.txt'
        data = np.loadtxt(filename)
        print(data.shape)

        print("Are there some nan values:" , np.isnan(data).any())
        

        data_features = data[:, 0:-2]    #The 16 attributes.
        data_target = data[:, -2]    #Note there are two columns to predict.

        X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, train_size=0.90, test_size=0.10, random_state=seed)

        #df=pd.DataFrame(X_train)
        #df2=pd.DataFrame(y_train)
        #print(df)
        #train_stats = df.describe()
        #print(train_stats)
        #train_stats2 = df2.describe()
        #print(train_stats2)
    elif data_name=="robot":
        print("forward kinematics of an 8 link robot arm")

        filename='../data/kin8nm/data.csv'
        data = pd.read_csv(filename)
        print("Are there some nan values:" , np.isnan(data).any())

        feature_names = data.iloc[:, 0:-1].columns    #The 8 attributes.
        target = data.iloc[:, -1:].columns

        df_features= data[feature_names]
        df_target = data[target]
        
        data_target=np.array(df_target)
        data_target=data_target.reshape((data_target.shape[0], ))
        data_features=np.array(df_features)
    

        X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, train_size=0.90, test_size=0.10, random_state=seed)
        print(X_train.shape)
        print(y_train.shape)

    elif data_name=="power":

        filename='../data/power/Folds5x2_pp.xlsx'
        data = pd.read_excel(filename, comment='#')
        #print(data)

        feature_names = data.iloc[:, 0:-1].columns   #The 4 attributes.
        target = data.iloc[:, -1:].columns

        df_features= data[feature_names]
        df_target = data[target]

        
        data_target=np.array(df_target)
        data_target=data_target.reshape((data_target.shape[0], ))
        data_features=np.array(df_features)
        #print(data_features)


        X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, train_size=0.90, test_size=0.10, random_state=seed)
        print(X_train.shape)
        print(y_train.shape)

    elif data_name=="wine":
        print("red wine quality dataset")

        filename='../data/wine/winequality-red.csv'
        data = pd.read_csv(filename, delimiter=';')
        print("Are there some nan values:" , np.isnan(data).any())
        print(data.shape)
        feature_names = data.iloc[:, 0:-1].columns    #The 11 attributes.
        target = data.iloc[:, -1:].columns

        df_features= data[feature_names]
        df_target = data[target]
        
        data_target=np.array(df_target)
        data_target=data_target.reshape((data_target.shape[0], ))
        data_features=np.array(df_features)

        print(data_features)

   
        X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, train_size=0.90, test_size=0.10, random_state=seed)
        print(X_train.shape)
        print(y_train.shape)
    
    elif data_name=="protein":

        print("Physicochemical Properties of Protein Tertiary Structure dataset")

        filename='../data/protein/CASP.csv'
        data = pd.read_csv(filename)
        print("Are there some nan values:" , np.isnan(data).any())
        #print(data.shape)

        feature_names = data.iloc[:, 1:].columns    #The 8 attributes.
        target = data.iloc[:, 0:1].columns

        df_features= data[feature_names]
        df_target = data[target]
        
        data_target=np.array(df_target)
        data_target=data_target.reshape((data_target.shape[0], ))
        data_features=np.array(df_features)


        X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, train_size=0.90, test_size=0.10, random_state=seed)
        print(X_train.shape)
        print(y_train.shape)

    if data_name=="year":
        print("release year of a song from audio featurest dataset")

        filename='../data/year/YearPredictionMSD.txt'
        data = np.loadtxt(filename, delimiter=',')
        #print(data[:, :4])

        print("Are there some nan values:" , np.isnan(data).any())
        

        data_features = data[:, 1:]    #The 90 attributes.
        data_target = data[:, 0]    #Note there are two columns to predict.

        X_train=np.array(data_features[:463715, :])
        print(X_train.shape)

        y_train=np.array(data_target[:463715])
        print(y_train.shape)
     
        X_test=np.array(data_features[463715:, :])
        print(X_test.shape)
        y_test=np.array(data_target[463715:])
        print(y_test.shape)
        
    else:
        pass

    return X_train, X_test, y_train, y_test


def main():
    load_data('year', 1) 

if __name__ == '__main__':
    main()