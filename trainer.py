import pandas as pd
import numpy as np
import time 
import matplotlib.pyplot as plt
import ast
import csv

from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

#Sti til data og omdannelse til pandas Dataframe
trail = "1"

gesture1 = "G1"
file1 = f'../EMG data/{gesture1}/{gesture1}_features_Tr{trail}.csv'
file1 = pd.read_csv(file1)

gesture2 = "G2"
# trail2 = "1"
file2 = f'../EMG data/{gesture2}/{gesture2}_features_Tr{trail}.csv'
file2 = pd.read_csv(file2)

gesture3 = "G9"
# trail3 = "1"
file3 = f'../EMG data/{gesture3}/{gesture3}_features_Tr{trail}.csv'
file3 = pd.read_csv(file3)

gesture4 = "G13"
# trail4 = "1"
file4 = f'../EMG data/{gesture4}/{gesture4}_features_Tr{trail}.csv'
file4 = pd.read_csv(file4)

gesture5 = "G22"
# trail5 = "1"
file5 = f'../EMG data/{gesture5}/{gesture5}_features_Tr{trail}.csv'
file5 = pd.read_csv(file5)

train_files = [file1,file2,file3,file4,file5]

trail = "2"

gesture1 = "G1"
test_file1 = f'../EMG data/{gesture1}/{gesture1}_features_Tr{trail}.csv'
test_file1 = pd.read_csv(test_file1)

gesture2 = "G2"
# trail2 = "1"
test_file2 = f'../EMG data/{gesture2}/{gesture2}_features_Tr{trail}.csv'
test_file2 = pd.read_csv(test_file2)

gesture3 = "G9"
# trail3 = "1"
test_file3 = f'../EMG data/{gesture3}/{gesture3}_features_Tr{trail}.csv'
test_file3 = pd.read_csv(test_file3)

gesture4 = "G13"
# trail4 = "1"
test_file4 = f'../EMG data/{gesture4}/{gesture4}_features_Tr{trail}.csv'
test_file4 = pd.read_csv(test_file4)

gesture5 = "G22"
# trail5 = "1"
test_file5 = f'../EMG data/{gesture5}/{gesture5}_features_Tr{trail}.csv'
test_file5 = pd.read_csv(test_file5)


test_files = [test_file1, test_file2, test_file3, test_file4, test_file5]

n_features = 4
n_segments = 33
n_channels = 8


def load_features(train_files, test_files, is_channel_feature=True, verbose=True):
    ################
    #Læsning af features. 
    #Features bliver gemt i feature_list, som har en shape på 8*33*4
    ################
    files_feature_list_2d = np.empty((0,5))
    
    for file in train_files:
        #Denne variabel bliver kun brugt til at determinere længden af en celle i loopet herunder
        df_window_list_len = len(file.iloc[:,0])
        feature_list = []
    
        for j in range(len(file.columns)): #Looper over kolonnerne i read
    
            col_feature = []
        
            for i in range(df_window_list_len): #Looper over rækkerne
        
        
                # Brug ast.literal_eval til at konvertere stringen tilbage til en liste
                feature = ast.literal_eval(file.iloc[i, j])
                               
                col_feature.append(feature)     
        
            feature_list.append(col_feature)
        
        feature_list_3D = np.array(feature_list)
       
            
        #Dette giver en liste på denne form
        #[feat1,feat2,feat3,feat4]
        #Med 264 (8*33) rækker
        feature_list_2d = feature_list_3D.reshape(-1, n_features)
        # print(feature_list_2d)
        # Laver en liste med hver channel i [0,1..,7]
        #Der bliver brugt en list comprehension: https://stackoverflow.com/questions/6475314/python-for-in-loop-preceded-by-a-variable
        channels_list = np.array([l for l in range(n_channels)]).flatten()

        #Listen bliver gentaget n_segmetns n_segments gange og samme elementer
        #Sættes ved siden af hinanden [0*33,1*33...] ish..
        channels_list = np.repeat(channels_list, n_segments)
        
        #TODO: Find ud af hvad dette betyder
        channels_list_reshaped = channels_list[:feature_list_2d.shape[0]].reshape(-1, 1)
        
        #Channel featuren blev tilføjet til en liste
        feature_list_2d_with_channel = np.column_stack((channels_list_reshaped, feature_list_2d))
        # print(feature_list_2d_with_channel)

        #feature_list_2d_with_channel bliver tilføjet til et nyt np array der holder alle features fra alle filerne
        files_feature_list_2d = np.append(files_feature_list_2d, feature_list_2d_with_channel, axis=0)

    if verbose==True:
        
        #Dette giver Pandas lov til at dataframes må fylde hele deres dimensioner i konsollen
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)

        print(f"Dataframe af features*segments med channels som feature\n{pd.DataFrame(feature_list_2d_with_channel)}")
        print(f"Dataframe af samlet features*segments med channels som feature\n{pd.DataFrame(files_feature_list_2d)}")
    #Resetter 
        
        
    if is_channel_feature==True:
        return feature_list_2d_with_channel    
       
    if is_channel_feature == False:
        feature_list_3D = np.transpose(feature_list_3D, (1, 0, 2))
        return feature_list_3D
        
    else: 
            print("Forkert parameter var givet")
    
# def get_labels():
    
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')

def prepare_data(train_files, test_files, is_channel_feature=True, verbose=True):
    
    
        
    return load_features(train_files, test_files, is_channel_feature, verbose), 
    
    
        
prepare_data(train_files, None, verbose=False)
    
'''
x_train = load_features()

y_train = load_labels()'''