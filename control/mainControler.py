import os, sys
import pandas as pd
import codecs
from constants.constants import *
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Perceptron, LogisticRegression, LinearRegression, RANSACRegressor, ElasticNet
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, BaggingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error


class MainControler:
    """機械学習処理の制御クラス"""

    def __init__(self):
        self.df_train = None
        self.df_test = None

    def set_train_dataframe(self, path):
        """もらったパスからトレーニングDataFrameメンバ変数作成"""

        self.df_train = pd.read_csv(path)

    def set_test_dataframe(self, path):
        """もらったパスからテストDataFrameメンバ変数作成"""

        self.df_test = pd.read_csv(path)

    def get_train_dataframe(self):
        """トレーニングDataFrameメンバ変数取得"""

        return self.df_train

    def get_test_dataframe(self):
        """テストDataFrameメンバ変数取得"""

        return self.df_test

    def delete_train_dataframe(self):
        """トレーニングデータの削除"""
        self.df_train = None

    def delete_test_dataframe(self):
        """テストデータの削除"""
        self.df_test = None

    def export_params(self, file_name, param_dict):
        """パラメータなど書き出し"""

        """同名ファイルが存在している場合は書き出し中止"""
        if os.path.exists((str(file_name) + '.txt')):
            print('既に存在しています')
            return

        """ファイル名が無い場合は連番書き出し"""
        if '' == file_name:
            file_name = 1
            is_ok = False
            while(not is_ok):
                if os.path.exists((str(file_name) + '.txt')):
                    file_name += 1
                else:
                    is_ok = True

        """書き出し"""
        file_name = str(file_name) + '.txt'
        f = codecs.open(file_name, 'w', 'utf-8')

        for key, value in param_dict.items():
            o_str = str(key) + ' ' + str(value) + '\n'
            f.write(o_str)

        f.close()


class machine_learning:
    """機械学習処理のベースクラス"""

    def __init__(self):
        self.df_train_X = None
        self.df_train_Y = None
        self.test_ID = None
        self.df_test = None

    def set_df_train(self, df):
        """トレーニングデータ設定"""

        self.df_train_X = df.iloc[:, 2:]
        self.df_train_Y = df.iloc[:, 1]

    def set_df_test(self, df):
        """テストデータ設定"""

        self.test_ID = df.iloc[:, 0]
        self.df_test = df.iloc[:, 1:]

    def set_params(self, params):
        pass

    def standardize_datas(self):
        """データ標準化"""

        sc = StandardScaler()
        self.df_train_X = pd.DataFrame(sc.fit_transform(self.df_train_X))
        if self.df_test is not None:
            self.df_test = pd.DataFrame(sc.fit_transform(self.df_test))

    def compress_datas(self, method, threshold):
        """データ圧縮"""

        if COMBO_ITEM_SELECT_FEATURES == method:
            # 特徴量選択によりデータ圧縮
            self.df_train_X = pd.DataFrame(get_import_features(threshold, self.df_train_X, self.df_train_Y))

        elif COMBO_ITEM_PCA == method:
            # PCAによりデータ圧縮
            pca = PCA()
            self.df_train_X = pd.DataFrame(pca.fit_transform(self.df_train_X))

        elif COMBO_ITEM_LDA == method:
            # LDAによりデータ圧縮
            lda = LDA()
            self.df_train_X = pd.DataFrame(lda.fit_transform(self.df_train_X, self.df_train_Y))

        elif COMBO_ITEM_KERNEL_PCA == method:
            # カーネルPCAによりデータ圧縮
            kpca = KernelPCA(kernel='rbf')
            self.df_train_X = pd.DataFrame(kpca.fit_transform(self.df_train_X))

        print('テスト出力', os.path.basename(__file__), sys._getframe().f_code.co_name)
        print(self.df_train_X.head())

def get_import_features(threshold, X, y):
    """重要な特徴量取得"""

    labels = X.columns
    forest = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    forest.fit(X, y)
    importances = forest.feature_importances_   # 特徴量の重要度を抽出
    indices = np.argsort(importances)[::-1]     # 重要度の降順で特徴量のインデックスを抽出
    for f in range(X.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30,
                                labels[indices[f]],
                                importances[indices[f]]))
    print()

    sfm = SelectFromModel(forest, threshold=threshold, prefit=True)
    return_X = sfm.transform(X)

    return return_X

class Classifier(machine_learning):
    """分類に特化した機械学習処理クラス"""

    def __init__(self):
        super().__init__()


class Predictor(machine_learning):
    """予測に特化した機械学習処理クラス"""

    def __init__(self):
        super().__init__()