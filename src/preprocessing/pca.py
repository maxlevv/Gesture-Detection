import pandas as pd
import numpy as np
from pathlib import Path
from feature_scaling import StandardScaler
import matplotlib.pyplot as plt
from preprocessing.preprocessing_functions import LabelsMandatory, LabelsOptional



def get_df(preproc_folder_path: Path):
    df = None
    for preproc_csv_file_path in preproc_folder_path.glob('**/*_preproc.csv'):
        #print('using', preproc_csv_file_path)
        next_df = pd.read_csv(preproc_csv_file_path, sep=' *,', engine='python')
        if df is None:
            df = next_df
        else:
            df = pd.concat([df, next_df], axis=0)

    df = df.drop(LabelsOptional.get_column_names(), axis=1)
    df = df.iloc[: , 1:]

    return df


class PCA():

    def fit(self, df: pd.DataFrame, keep_percentage:float = 100):
        self.columns = df.columns.values
        X = df.to_numpy()
        self.initial_data = X[:, 1:]

        self.keep_percentage = keep_percentage

        self.cov_matrix = np.cov(self.initial_data, rowvar=False)
        print(self.cov_matrix.shape)
        eig_values, eig_vectors = np.linalg.eig(self.cov_matrix)    # already normalized eigenvectors
        sort_index = np.argsort(eig_values)
        self.eig_values_sorted = eig_values[sort_index[::-1]]
        self.eig_vectors_sorted = eig_vectors[:, sort_index[::-1]]
        self.pca_components = self.initial_data @ self.eig_vectors_sorted

        self.percentage = (self.eig_values_sorted / self.eig_values_sorted.sum() * 100).round(4)
        self.cum_percentage = np.cumsum(self.percentage)

        if self.keep_percentage == 100:
            self.index = len(self.cum_percentage) - 1
        elif self.keep_percentage < 100:
            self.index = np.argmax(self.cum_percentage >= self.keep_percentage)
        elif self.keep_percentage < 0:
            raise Exception('percentage cannot be negative')
        else:
            raise Exception('percentage cannot be more than 100')


    def transform(self, df: pd.DataFrame):
        X = df.to_numpy()
        X = X[:, 1:]
        if self.keep_percentage == 100:
            return X @ self.eig_vectors_sorted
        else:
            feature_vector = self.eig_vectors_sorted[:, :self.index+1]
            return X @ feature_vector


    def scree_plot(self):
        #labels = ['PC' + str(x) for x in range(1, len(self.eig_values_sorted) + 1)]
        plt.bar(x=range(1, len(self.eig_values_sorted) + 1), height=self.percentage)  #, tick_label=labels)
        plt.xlabel('Principal Components')
        plt.ylabel('Percentage')
        plt.title('PCA Scree Plot')
        plt.show()


    def print_contributing_parameters(self, component:int = 1):
        loading_scores = pd.Series(self.eig_vectors_sorted[component - 1], index=self.columns)
        #print(loading_scores)
        #print(np.linalg.norm(loading_scores.to_numpy()))
        sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
        best_features = sorted_loading_scores[:50].index.values
        print(loading_scores[best_features])

        #plt.table(best_features)
        #plt.show()


def generate_pca_dataset(preproc_folder_path: Path, scaler: StandardScaler = None,
                         select_mandatory_label: bool = True, pca = None, keep_percentage:float = 100):

    df = None
    for preproc_csv_file_path in preproc_folder_path.glob('**/*_preproc.csv'):

        if 'nina' in str(preproc_csv_file_path):
            print(f'{preproc_csv_file_path} continue !')
            continue
        print('using', preproc_csv_file_path)

        next_df = pd.read_csv(preproc_csv_file_path, sep=' *,', engine='python')
        if df is None:
            df = next_df
        else:
            df = pd.concat([df, next_df], axis=0)

    if select_mandatory_label == True:
        Labels = LabelsMandatory
        y = df[Labels.get_column_names()].to_numpy()
    else:
        Labels = LabelsOptional
        y = df[Labels.get_column_names()].to_numpy()

    df = df.drop(LabelsOptional.get_column_names(), axis=1)

    if scaler == None:
        # standardize columns for training data
        new_scaler = StandardScaler()
        new_scaler.fit(df)
        X_df = new_scaler.transform(df)
        if pca == None:
            new_pca = PCA()
            new_pca.fit(X_df, keep_percentage=keep_percentage)
            X = new_pca.transform(X_df)
            return X, y, new_scaler, new_pca
        else: raise Exception('scale data first')

    else:
        # validation/test data
        X_df = scaler.transform(df)
        if pca == None: raise Exception('pass a fitted pca object to method')
        else:
            X = pca.transform(X_df)
            return X, y




if __name__ == '__main__':

    folder_path_train = Path(r'C:\Users\Max\PycharmProjects\ml_dev_repo\data\preprocessed_frames\new_window=10,cumsum=all\train')
    folder_path_val = Path(r'C:\Users\Max\PycharmProjects\ml_dev_repo\data\preprocessed_frames\new_window=10,cumsum=all\validation')
    df_train = get_df(folder_path_train)
    #df_val = get_df(folder_path_val)

    scaler = StandardScaler()
    scaler.fit(df_train)
    df_train = scaler.transform(df_train)
   # df_val = scaler.transform(df_val)

    pca = PCA()
    pca.fit(df_train, keep_percentage=99)

    pca.print_contributing_parameters(component=2)

    #X_train_pca = pca.transform(df_train)
    #X_val_pca = pca.transform(df_val)

    pca.scree_plot()

