import os, sys, re
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
        self.params = {}

    def set_df_train(self, df):
        """トレーニングデータ設定"""

        self.df_train_X = df.iloc[:, 2:]
        self.df_train_Y = df.iloc[:, 1]

    def set_df_test(self, df):
        """テストデータ設定"""

        self.test_ID = df.iloc[:, 0]
        self.df_test = df.iloc[:, 1:]

    def set_params(self, params):
        """パラメータ設定"""

        """分析手法はそのまま保存し、ハイパーパラメータは不要入力値を除去しリスト化して保存"""
        for key, value in params.items():

            """分析手法"""
            if (key == PARAM_ANALYSIS) or (key == PARAM_BAGADA):
                self.params[key] = value
                print(value)
                continue

            """ハイパーパラメータ（penaltyとkernelはコンボで文字列で指定される）"""

            # 半角スペースと全角文字はあらかじめ除去
            hyper_param = value.replace(' ', '')
            hyper_param = re.sub('[^\x01-\x7E]','',hyper_param)

            if (PARAM_PENALTY == key) or (PARAM_KERNEL == key):
                hyper_param = hyper_param.split(',')
                self.params[key] = hyper_param
                print(hyper_param)
                continue

            """ガンマパラメータはfloatも文字列も取りうるので別処理"""
            if (PARAM_GAMMA == key) and ('auto' in hyper_param):
                hyper_param = ['auto']
            else:
                # 半角数字、コンマ、ピリオドのみ抽出
                hyper_param = re.sub('[^0-9,.]', '', hyper_param)
                hyper_param = hyper_param.split(',')
                # 空のアイテムを削除
                hyper_param = [item for item in hyper_param if item != '']
                # 1アイテムに2以上のピリオドが入っている場合は無効とし削除
                hyper_param = [item for item in hyper_param if item.count('.')<=1]
                """パラメータがint型なら一度floatにすることで値を丸める"""
                hyper_param = list(map(float, hyper_param))
                if self._is_int_param(key):
                    hyper_param = list(map(int, hyper_param))

            print(hyper_param)
            self.params[key] = hyper_param

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
            self.df_train_X = pd.DataFrame(self.get_import_features(threshold, self.df_train_X, self.df_train_Y))

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

        # print(self.df_train_X.head())

    def get_import_features(self, threshold, X, y):
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

    def run_predict(self):
        """予測実行"""

        if self.df_test is None:
            return

    def _is_int_param(self, key):
        """パラメータがint型であるか判定"""

        if (PARAM_NEIGHBORS == key) or (PARAM_MAXDEPTH == key) or (PARAM_MAXFEATURES == key)\
           or (PARAM_CLS_NESTIMATORS == key) or (PARAM_PRD_NESTIMATORS == key) or (PARAM_BATCHSIZE == key)\
           or (PARAM_NHIDDEN == key) or (PARAM_NUNIT == key) or (PARAM_BA_NESTIMATOR == key) or (PARAM_BA_MAX_FEATURES == key):
            return True
        else:
            return False


class Classifier(machine_learning):
    """分類に特化した機械学習処理クラス"""

    def __init__(self):
        super().__init__()

    def run_learning(self):
        """学習実行"""

        """分析手法別に学習実施"""
        if COMBO_ITEM_PERCEPTRON == self.params[PARAM_ANALYSIS]:
            pass
        elif COMBO_ITEM_ROGISTICREGRESSION == self.params[PARAM_ANALYSIS]:
            pass
        elif COMBO_ITEM_SVM == self.params[PARAM_ANALYSIS]:
            pass
        elif COMBO_ITEM_RANDOMFOREST == self.params[PARAM_ANALYSIS]:
            pass
        elif COMBO_ITEM_KNEIGHBORS == self.params[PARAM_ANALYSIS]:
            pass
        else:
            print('該当分析手法なし')



class Predictor(machine_learning):
    """予測に特化した機械学習処理クラス"""

    def __init__(self):
        super().__init__()

    def run_learning(self):
        """学習実行"""

        """分析手法別に学習実施"""
        if COMBO_ITEM_LINEARREGRESSION == self.params[PARAM_ANALYSIS]:
            pass
        elif COMBO_ITEM_ELASTICNET == self.params[PARAM_ANALYSIS]:
            pass
        elif COMBO_ITEM_RANDOMFOREST == self.params[PARAM_ANALYSIS]:
            pass
        elif COMBO_ITEM_EXTRATREE == self.params[PARAM_ANALYSIS]:
            pass
        elif COMBO_ITEM_DEEPLEARNING == self.params[PARAM_ANALYSIS]:
            pass
        else:
            print('該当分析手法なし')
