import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from pylab import rcParams
from seaborn import sns
import sklearn
from sklearn.cluster import DBSCAN
from collections import  Counter
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest






class DataPreprocessing():
    '''
    Implements standard data preprocessing steps like Feature scaling, Missing values handling,
    perform One-Hot Encoding for categorical values.
    '''
    
    def __init__(self):
        pass

        
    @staticmethod
    def missing_val_info(df):
        """Prints missing value information about dataframe""" 
        pd.options.display.max_columns = len(df)
        print("\nShape of the dataframe : \n {}" .format(df.shape))
        print("===================================================")
        print("\nInfo about the dataframe : \n {}" .format(df.info()))
        print("===================================================")
        #print("\nNum of non missing values for each column : \n \n {}" .format(df.count()))
        #print("===================================================")
        df_missing = df.isna()
        df_missing = pd.DataFrame(df_missing)
        print("\nTrue indicates, values is missing in that particular position  : \n ")
        print(df_missing.head(5))
        print("===================================================")
        #print("\nCheck if the data type is Boolean for every column : \n")
        #df_missin_dt_check = df_missing.dtypes
        #print(df_missin_dt_check)
        #print("===================================================")
        df_num_missing_val = df_missing.sum()
        print("\nMissing values per column : \n")
        print(df_num_missing_val)
        print("===================================================")
        print("\nPercentage of missing values per column : \n")
        df_missing_pcent = df.isna().mean().round(4) * 100
        print(df_missing_pcent)
        print("===================================================")
        pcent_missing = df.dropna().shape[0]/ np.float(df.shape[0])
        pcent_missing = (1 - round(pcent_missing, 3))*100
        print('Percentage of data with missing values : ', round(pcent_missing, 4))
        
        
    
    @staticmethod
    def obj_ohe(df):
        """checks if the column has 'object' data type. It treats it as categorical and one-hot-encodes it."""
        newDf = df.select_dtypes(include = ['object'])
        for col in newDf.columns:
            if len(newDf.groupby([col]).size()) > 2:
                newDf = pd.get_dummies(newDf, prefix = [col], columns = [col])
            obj_ohe_df  = pd.get_dummies(newDf, drop_first = True)
            return obj_ohe_df
    
    
    @staticmethod    
    def treat_float_missing_val(df):
        """ Checks if the column has 'float' values and imputes the missing values with 'mean' value."""
        newDf = df.select_dtypes(include = ['float'])
        columns = newDf.columns
        imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
        imp_df = imp.fit_transform(newDf)
        imp_df = pd.DataFrame(imp_df, columns = columns)
        return imp_df
    
    @staticmethod
    def treat_num_missing_val(df):
        """ Checks if the column has 'int' values and imputes the missing values with 'most frequent' value. """
        newDf = df.select_dtypes(include = ['int'])
        columns = newDf.columns
        imp = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
        imp_df = imp.fit_transform(newDf)
        imp_df = pd.DataFrame(imp_df, columns = columns)
        return imp_df
    
    
    @staticmethod
    def scaled_df_minmax(df):
        """Takes in input a dataframe and returns the same with values scaled in range [0,1] """
        for col in df.columns:
            scaled_df[col] = (df[col] - df[col].min())/ (df[col].max() - df[col].min())
        return scaled_df
    
    
    @staticmethod
    def scaled_df_stdscalar(df):
        """ Takes in input a dataframe and returns the same with values scaled such that every columns has a mean of 0 and SD of 1 """
        scaler = StandardScaler()
        scaled_df = scaler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled_df, columns = df.columns)
        return scaled_df
    
    
    @staticmethod
    def imputed_df(df):
        """ Concatenate all the dataframes returned by other methods after missing value imputation """
        obj_ohe_df = DataPreprocessing.obj_ohe(df)
        int_miss_df = DataPreprocessing.treat_num_missing_val(df)
        flt_miss_df = DataPreprocessing.treat_float_missing_val(df)
        imputed_df = pd.concat([obj_ohe_df, int_miss_df, flt_miss_df], axis = 1)
        return imputed_df
    




class AnomalyControlChart:
    '''
    Implements Upper control Limit (UCL) and Lower
    Control Limit (LCL) in a statistical process control
    UCL = mean + 3 * s.d   , LCL = mean - 3 * s.d   
    '''

    def __init__(self):
        pass
    
    
    @staticmethod
    def timer(start,end):
        """Returns the total time taken to run a method """
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Time taken to run this function for this sequence : {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

    @staticmethod
    def lcl_ucl(df):
        """ Return lower & upper control limits. """
        df_mean = np.mean(df)
        df_std = np.std(df)
        lcl = df_mean - 3 * df_std
        ucl = df_mean + 3 * df_std
        return  lcl, ucl
        
        
    @staticmethod
    def get_lcl_ucl(df):
        """ Prints the value of lower and upper and lower control limit. """
        start = time.time()
        lcl, ucl = AnomalyControlChart.lcl_ucl(df)
        print("Centre :{} \nLower control Limit :{} \nUpper Control Limit :{}".format(np.mean(df),lcl, ucl))
        end = time.time()
        AnomalyControlChart.timer(start,end)        
    
    @staticmethod
    def vpoints(df):
        """ Returns two lists with points that violates lcl and ucl respectively. """
        lcl, ucl = AnomalyControlChart.lcl_ucl(df)
        lcl_vpnts = filter((lambda x : x < lcl), df)  
        ucl_vpnts = filter((lambda x : x > ucl), df)
        return  list(lcl_vpnts), list(ucl_vpnts) 
    
    
    @staticmethod
    def get_vpoints(df):
        """ Prints the values of data points that are violating lcl and ucl """
        start = time.time()
        lcl_violating_points, ucl_violating_points = AnomalyControlChart.vpoints(df)
        print("Points Violating LCL : {} \nPoints Violating UCL : {}".format(lcl_violating_points, ucl_violating_points))
        end = time.time()
        AnomalyControlChart.timer(start,end)


    @staticmethod
    def get_controlchart_plot(x_values, y_values):
        """ Plot a control chart for y_values corresponding to x_values with three straight lines 
        representing Centre, UCL and LCL. For example : x_values = Time, y_values = esales per day"""
        figure(num = None, figsize = (8,6), dpi = 80, facecolor = 'w', edgecolor = 'k')
        y_values_std = np.std(y_values)
        y_values_mean = [np.mean(y_values)]*len(y_values)
        y_values_lcl = [np.mean(y_values) - 3*y_values_std]*len(y_values)
        y_values_ucl = [np.mean(y_values) + 3*y_values_std]*len(y_values)
        plt.plot(x_values, y_values, label = 'signal', linewidth = 1)
        plt.plot(x_values, y_values_mean, linewidth = 2, label  = "Central Line")
        plt.plot(x_values, y_values_lcl, linewidth  = 2, label = "Lower Control Line")
        plt.plot(x_values, y_values_ucl, linewidth  =2, label = "Upper Control Line")
        plt.xlabel = "x_values"
        plt.ylabel = "y_values"
        plt.legend(loc='best')
        plt.show()






class AnomalyIf :
    '''
    Implements Isolation forest algorithm to detect anomalies 
    '''
    
    def __init__(self):
        pass
    
    
    @staticmethod
     def IForest(df):
        clf = IsolationForest(n_estimators = 300, contamination = 0.01, behaviour = "new")
        clf.fit(df)
        anomaly_score = clf.decision_function(df)
        anomalies = clf.predict(df)
        print("\n Decision function for Isolation forest")
        print(anomaly_score)
        print("\n The label -1 suggests that the point is anomolous and 1 suggests that the point is normal")
        print(anomalies)
        anomolus_dp = data[np.where(outlier == -1, True, False)]
        print("\nThe data has {} anomolus points and the data points are the following for  : ".format(len(anomolus_dp)))
        print(anomolus_dp)
        
        
    
    
    
    
    
    
    
            

    
        
        
        
    
        
    
        

        
    
    
    
    
    
    
    
    
    
    
    
    
