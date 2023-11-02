import pandas as pd
import numpy as np
import time 
sleep = time.sleep
import matplotlib.pyplot as plt
import ast
import csv

from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
quad_discriminant = QuadraticDiscriminantAnalysis()
from sklearn.ensemble import GradientBoostingClassifier
gradient_boosting = GradientBoostingClassifier()
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier()
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
from sklearn.neighbors import KNeighborsClassifier
neighbors_clsf = KNeighborsClassifier()
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
svm = SVC()
from sklearn.tree import DecisionTreeClassifier
dec_tree_clsf = DecisionTreeClassifier()
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression()
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)

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


def load_features(files, one_hot_encoded_label, is_channel_feature=True, verbose=True):
    ################
    #Læsning af features. 
    #Features bliver gemt i feature_list, som har en shape på 8*33*4
    ################

    
    if is_channel_feature==True:

        files_feature_list_labels = []
        files_feature_list_2d = np.empty((0,5))
        feature_list = np.empty((0,4))

        
        for file_nr,file in enumerate(files):
            #Denne variabel bliver kun brugt til at determinere længden af en celle i loopet herunder
            df_window_list_len = len(file.iloc[:,0])
            
            
        
            for j in range(len(file.columns)): #Looper over kolonnerne i read
        
                col_feature = np.empty((0,4))
            
                for i in range(df_window_list_len): #Looper over rækkerne
            
            
                    # Brug ast.literal_eval til at konvertere stringen tilbage til en liste
                    feature = ast.literal_eval(file.iloc[i, j])
                    col_feature = np.vstack((col_feature, feature)) 
                       
                    files_feature_list_labels = np.append(files_feature_list_labels, file_nr)
                    
                #col_feature kobles vertikalt på feature list for hver kolonne, for hver fil i files
                feature_list = np.vstack((feature_list, col_feature))
      
            
        #Dette giver en liste på denne form
        #[feat1,feat2,feat3,feat4]
        #Med 264 (8*33) rækker
        # feature_list_2d = feature_list_3D.reshape(-1, n_features)
        # print(feature_list_2d)
        # Laver en liste med hver channel i [0,1..,7]
        #Der bliver brugt en list comprehension: https://stackoverflow.com/questions/6475314/python-for-in-loop-preceded-by-a-variable
        channels_list = np.array([l for l in range(n_channels)]).flatten()
        
        #Listen bliver gentaget n_segmetns* n_files gange og samme elementer - 
        #sættes ved siden af hinanden [0,0,0 *33*5... 1,1,1 *33*5] ish..
        channels_list = np.repeat(channels_list, n_segments*len(files))
        
        #Dette reshapper channels list sådan det er klar til at sættes sammen med feature_list og hvert segment får en feature mere
        channels_list_reshaped = channels_list[:feature_list.shape[0]].reshape(-1, 1)
        
        #Channel featuren blev tilføjet til en liste
        feature_list_2d_with_channel = np.column_stack((channels_list_reshaped, feature_list))
        # print(feature_list_2d_with_channel)
        
        #feature_list_2d_with_channel bliver tilføjet til et nyt np array der holder alle features fra alle filerne
        files_feature_list_2d = np.append(files_feature_list_2d, feature_list_2d_with_channel, axis=0)
        
        
            

        if verbose==True:
            
            #Dette giver Pandas lov til at dataframes må fylde hele deres dimensioner i konsollen
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)

            print(f"Dataframe af features*segments med channels som feature\n{pd.DataFrame(feature_list_2d_with_channel)}")
            # print(f"Dataframe af samlet features*segments med channels som feature\n{pd.DataFrame(files_feature_list_2d)}")
            print(files_feature_list_labels)
         
        return feature_list_2d_with_channel, files_feature_list_labels
    

    if is_channel_feature == False: 
        #Hvis channel ikke er en feature skal dataen opstilles i dette format: (n_features*channels)*segments altså
        #den første række bliver features for hver channel dvs. der bliver n_features*n_channels = 32 kolonner

        #Denne variabel initialiseres på denne måde da den ellers ikke kan blive appendet til, fordi den ikke har korrekte mængde af kolonner
        #np.empty((0,32)) laver et tomt np array med 0 rækker og 32 kolonner
        files_feature_list = np.empty((0,32))
        files_feature_list_labels = []

        for file_nr,file in enumerate(files): #Looper over filerne

            for feature_list_str in file.values: #Looper over rækkerne

                feature_list_row = []

                for feature_list in feature_list_str: #Looper over elementer i csv filen altså "[feat1,...],[feat1...]...[]"

                    #Elementet konverteres fra en string tilbage til en liste. Feature_list holder derfor 4 elementer [f1,f2,f3,f4]
                    feature_list = ast.literal_eval(feature_list)
                    
                    #feature_list tilføjes til feature_list_row. Dvs feature_list_row får 32 elementer
                    feature_list_row = np.append(feature_list_row, feature_list)

                #feature_list_row tilføjes til et nyt array på den vertikale akse
                files_feature_list = np.vstack((files_feature_list, feature_list_row))
                
                files_feature_list_labels = np.append(files_feature_list_labels, file_nr)
            
        if verbose==True:
            print(files_feature_list.shape)
            # print(pd.DataFrame(files_feature_list))
            print(files_feature_list_labels)

        if one_hot_encoded_label == True:

            files_feature_list_labels = np.array(files_feature_list_labels).reshape(-1, 1)
            onehot_files_feature_list_labels = encoder.fit_transform(files_feature_list_labels)
            
            return files_feature_list, onehot_files_feature_list_labels
        else:
            return files_feature_list, files_feature_list_labels

        
    if is_channel_feature == "3D" or is_channel_feature == "3d":
        return feature_list_3D
    else: 
            print("Forkert parameter var givet")
    
# def get_labels():
    
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')

def prepare_data(train_files, test_files, one_hot_encoded_label, is_channel_feature=True, verbose=True):
            
    x_train, y_train = load_features(train_files, one_hot_encoded_label, is_channel_feature, verbose)

    x_test, y_test = load_features(test_files, one_hot_encoded_label, is_channel_feature, verbose)

    return x_train,y_train,x_test,y_test


    
lda = LinearDiscriminantAnalysis()
x_train, y_train, x_test, y_test = prepare_data(train_files, test_files, one_hot_encoded_label=False, is_channel_feature=True, verbose=False)

lda.fit(x_train, y_train)

score = lda.score(x_test, y_test)
print(f'LDA Accuracy: {score}')

y_pred = lda.predict(x_test)

gaussian.fit(x_train,y_train)
score = gaussian.score(x_test, y_test)
print(f'Gaussian Accuracy: {score}')

random_forest.fit(x_train,y_train)
score = random_forest.score(x_test, y_test)
print(f'Random forest Accuracy: {score}')

dec_tree_clsf.fit(x_train,y_train)
score = dec_tree_clsf.score(x_test, y_test)
print(f'Decision Tree Accuracy: {score}')

quad_discriminant.fit(x_train,y_train)
score = quad_discriminant.score(x_test, y_test)
print(f'Quadratic Discriminant Accuracy: {score}')

gradient_boosting.fit(x_train,y_train)
score = gradient_boosting.score(x_test, y_test)
print(f'Gradient Boosting Accuracy: {score}')

neighbors_clsf.fit(x_train,y_train)
score = neighbors_clsf.score(x_test, y_test)
print(f'Nearest Neighbors Accuracy: {score}')

gaussian.fit(x_train,y_train)
score = gaussian.score(x_test, y_test)
print(f'Gaussian Accuracy: {score}')


lda = LinearDiscriminantAnalysis()
x_train, y_train, x_test, y_test = prepare_data(train_files, test_files, one_hot_encoded_label=False, is_channel_feature=False, verbose=False)

lda.fit(x_train, y_train)

score = lda.score(x_test, y_test)
print(f'LDA Accuracy: {score}')

y_pred = lda.predict(x_test)

gaussian.fit(x_train,y_train)
score = gaussian.score(x_test, y_test)
print(f'Gaussian Accuracy: {score}')

random_forest.fit(x_train,y_train)
score = random_forest.score(x_test, y_test)
print(f'Random forest Accuracy: {score}')

dec_tree_clsf.fit(x_train,y_train)
score = dec_tree_clsf.score(x_test, y_test)
print(f'Decision Tree Accuracy: {score}')

quad_discriminant.fit(x_train,y_train)
score = quad_discriminant.score(x_test, y_test)
print(f'Quadratic Discriminant Accuracy: {score}')

gradient_boosting.fit(x_train,y_train)
score = gradient_boosting.score(x_test, y_test)
print(f'Gradient Boosting Accuracy: {score}')

neighbors_clsf.fit(x_train,y_train)
score = neighbors_clsf.score(x_test, y_test)
print(f'Nearest Neighbors Accuracy: {score}')

gaussian.fit(x_train,y_train)
score = gaussian.score(x_test, y_test)
print(f'Gaussian Accuracy: {score}')