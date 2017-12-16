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
from sklearn.model_selection import train_test_split


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
        self.do_analysis_gridsearch = False
        self.do_bagada_gridsearch = False

    def set_df_train(self, df):
        """トレーニングデータ設定"""

        self.df_train_X = df.iloc[:, 2:]
        self.df_train_Y = df.iloc[:, 1]
        self.predicted_label = df.columns[1]

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
                continue


            # 半角スペースと全角文字はあらかじめ除去
            hyper_param = value.replace(' ', '')
            hyper_param = re.sub('[^\x01-\x7E]','',hyper_param)

            """ハイパーパラメータ（penaltyとkernelはコンボで文字列で指定される）"""
            if (PARAM_PENALTY == key) or (PARAM_KERNEL == key):
                hyper_param = hyper_param.split(',')
                self.params[key] = hyper_param

                """ハイパーパラメータが2つ以上の場合はグリッドサーチ実行フラグ有効化"""
                if len(hyper_param) >= 2:
                    self.do_analysis_gridsearch = True

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

            self.params[key] = hyper_param

            """分析・バグアダに応じてハイパーパラメータが2つ以上の場合はグリッドサーチ実行フラグ有効化"""
            if (key == PARAM_BA_NESTIMATOR) or (key == PARAM_BA_MAXSAMPLES)\
                or (key == PARAM_BA_MAX_FEATURES) or (key == PARAM_BA_LEARNINGRATE):
                if len(hyper_param) >= 2:
                    self.do_bagada_gridsearch = True
            else:
                if len(hyper_param) >= 2:
                    self.do_analysis_gridsearch = True

    def standardize_datas(self):
        """データ標準化"""

        # 列名退避
        columns = list(self.df_train_X.columns)

        sc = StandardScaler()
        self.df_train_X = pd.DataFrame(sc.fit_transform(self.df_train_X), columns=columns)
        if self.df_test is not None:
            self.df_test = pd.DataFrame(sc.fit_transform(self.df_test), columns=columns)

    def compress_datas(self, method, threshold):
        """データ圧縮"""

        if COMBO_ITEM_SELECT_FEATURES == method:
            # 特徴量選択によりデータ圧縮
            self.df_train_X = pd.DataFrame(self.get_import_features(threshold, self.df_train_X, self.df_train_Y))

        elif COMBO_ITEM_PCA == method:
            # PCAによりデータ圧縮
            pca = PCA()
            self.df_train_X = pd.DataFrame(pca.fit_transform(self.df_train_X))
            self.df_test = pd.DataFrame(pca.transform(self.df_test))

        elif COMBO_ITEM_LDA == method:
            # LDAによりデータ圧縮
            lda = LDA()
            self.df_train_X = pd.DataFrame(lda.fit_transform(self.df_train_X, self.df_train_Y))
            self.df_test = pd.DataFrame(lda.transform(self.df_test))

        elif COMBO_ITEM_KERNEL_PCA == method:
            # カーネルPCAによりデータ圧縮
            kpca = KernelPCA(kernel='rbf')
            self.df_train_X = pd.DataFrame(kpca.fit_transform(self.df_train_X))
            self.df_test = pd.DataFrame(kpca.transform(self.df_test))

    def get_import_features(self, threshold, X, y):
        """重要な特徴量取得"""

        # 列削除用に列名リスト確保
        list_for_delete_column = []

        labels = X.columns
        forest = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
        forest.fit(X, y)
        importances = forest.feature_importances_   # 特徴量の重要度を抽出
        indices = np.argsort(importances)[::-1]     # 重要度の降順で特徴量のインデックスを抽出
        for f in range(X.shape[1]):
            label = labels[indices[f]]
            importance = importances[indices[f]]
            print("%2d) %-*s %f" % (f + 1, 30, label, importance))

            list_for_delete_column.append(label)
        print()

        sfm = SelectFromModel(forest, threshold=threshold, prefit=True)
        return_X = sfm.transform(X)

        """特徴量全削除の場合は渡って来た引数をそのまま返す"""
        if 0 == return_X.shape[1]:
            return self.df_train_X

        """削除列数からテストデータの列削除実行"""
        if self.df_test is not None:
            delete_number = X.shape[1] - return_X.shape[1]
            for i in range(delete_number):
                delete_column = list_for_delete_column[-(i+1)]
                del self.df_test[delete_column]

        return return_X

    def run_predict(self, estimator):
        """予測実行"""

        predicted = None

        """テストデータが存在する場合は予測結果をcsv出力"""
        if self.df_test is not None:
            predicted = pd.DataFrame(estimator.predict(self.df_test), columns=[self.predicted_label])
            self.process_csv(predicted)

        return predicted

    def _is_int_param(self, key):
        """パラメータがint型であるか判定"""

        if (PARAM_NEIGHBORS == key) or (PARAM_MAXDEPTH == key) or (PARAM_MAXFEATURES == key)\
           or (PARAM_CLS_NESTIMATORS == key) or (PARAM_PRD_NESTIMATORS == key) or (PARAM_BATCHSIZE == key)\
           or (PARAM_NHIDDEN == key) or (PARAM_NUNIT == key) or (PARAM_BA_NESTIMATOR == key):
            return True
        else:
            return False

    def make_method_param_grid(self):
        """分析グリッドサーチ用グリッド作成"""

        param_grid = []

        # パーセプトロン
        if self.params[PARAM_ANALYSIS] == COMBO_ITEM_PERCEPTRON:
            param_grid = [
                {'eta0': self.params[PARAM_ETA0], 'penalty': self.params[PARAM_PENALTY]}
            ]

        # ロジスティック回帰
        elif self.params[PARAM_ANALYSIS] == COMBO_ITEM_ROGISTICREGRESSION:
            param_grid = [
                {'penalty': self.params[PARAM_PENALTY]}
            ]

        # SVM
        elif self.params[PARAM_ANALYSIS] == COMBO_ITEM_SVM:
            if COMBO_ITEM_RBF in self.params[PARAM_KERNEL]:
                param_grid.append({'kernel': ['rbf'], 'gamma': self.params[PARAM_GAMMA], 'C': self.params[PARAM_C]})
            if COMBO_ITEM_LINEAR in self.params[PARAM_KERNEL]:
                param_grid.append({'kernel': ['linear'], 'C': self.params[PARAM_C]})

        # 分類ランダムフォレスト
        elif self.params[PARAM_ANALYSIS] == COMBO_ITEM_RANDOMFOREST_CLS:
            param_grid = [
                {'n_estimators': self.params[PARAM_CLS_NESTIMATORS]}
            ]

        # k近傍法
        elif self.params[PARAM_ANALYSIS] == COMBO_ITEM_KNEIGHBORS:
            param_grid = [
                {'n_neighbors': self.params[PARAM_NEIGHBORS]}
            ]

        # 線形回帰
        elif self.params[PARAM_ANALYSIS] == COMBO_ITEM_LINEARREGRESSION:
            param_grid = []

        # ElasticNet
        elif self.params[PARAM_ANALYSIS] == COMBO_ITEM_ELASTICNET:
            param_grid = [
                {'alpha': self.params[PARAM_ALPHA], 'l1_ratio': self.params[PARAM_L1RATIO]}
            ]

        # 回帰ランダムフォレスト
        elif self.params[PARAM_ANALYSIS] == COMBO_ITEM_RANDOMFOREST_PRD:
            param_grid = [
                {'max_features': self.params[PARAM_MAXFEATURES], 'max_depth': self.params[PARAM_MAXDEPTH]}
            ]

        # エクストラツリー
        elif self.params[PARAM_ANALYSIS] == COMBO_ITEM_EXTRATREE:
            param_grid = [
                {'n_estimators': self.params[PARAM_PRD_NESTIMATORS],
                 'max_features': self.params[PARAM_MAXFEATURES], 'max_depth': self.params[PARAM_MAXDEPTH]}
            ]

        return param_grid

    def make_bagada_param_grid(self):
        """バギング/アダブーストグリッドサーチ用グリッド作成"""

        param_grid = []

        # バギング
        if self.params[PARAM_BAGADA] == COMBO_ITEM_BAGGING:
            param_grid = [
                {'n_estimators': self.params[PARAM_BA_NESTIMATOR], 'max_samples': self.params[PARAM_BA_MAXSAMPLES],
                 'max_features': self.params[PARAM_BA_MAX_FEATURES]}
            ]

        # アダブースト
        if self.params[PARAM_BAGADA] == COMBO_ITEM_ADABOOST:
            param_grid = [
                {'n_estimators': self.params[PARAM_BA_NESTIMATOR], 'learning_rate': self.params[PARAM_BA_LEARNINGRATE]}
            ]

        return param_grid

    def get_grid_search_estimator(self, estimator, X_train, y_train, param_grid, scoring='accuracy'):
        """グリッドサーチを実行し、実行後の推定器を返す"""

        gs = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring=scoring, cv=10, n_jobs=-1)
        gs.fit(X_train, y_train)

        print(gs.best_params_)

        return gs.best_estimator_

    def get_classifer_result(self, estimator, predicted=None):
        """分類結果返却"""

        train_score = estimator.score(self.X_train, self.y_train)
        test_score = estimator.score(self.X_test, self.y_test)

        difference= None
        if predicted is not None:
            difference = abs(self.df_train_Y.mean() - int(predicted.mean()))

        return train_score, test_score, difference

    def process_csv(self, predicted):
        """csv出力処理"""

        """「predict_data.csv」を削除してから出力"""
        if os.path.exists('predict_data.csv'):
            os.remove('predict_data.csv')

        df_output = pd.concat([self.test_ID, predicted], axis=1)
        df_output.to_csv('predict_data.csv', index=False)

    def make_grid_search_estimator(self, estimator):
        """グリッドサーチ用パラメータを作成し実行"""

        param_grid = self.make_method_param_grid()
        estimator = self.get_grid_search_estimator(estimator, self.X_train, self.y_train, param_grid)

        return estimator

    def make_bag_ada_estimator(self, estimator):
        """バギング/アダブースト推定器作成"""

        if COMBO_ITEM_BAGGING == self.params[PARAM_BAGADA]:
            """バギング/アダブーストグリッドサーチ実行フラグに応じて推定器作成"""
            if True == self.do_bagada_gridsearch:
                estimator = BaggingClassifier(base_estimator=estimator, random_state=0, n_jobs=-1)
                bagging_param_grid = self.make_bagada_param_grid()
                estimator = self.get_grid_search_estimator(estimator, self.X_train, self.y_train, bagging_param_grid)

            else:
                estimator = BaggingClassifier(base_estimator=estimator,
                                              n_estimators=self.params[PARAM_BA_NESTIMATOR][0],
                                              max_samples=self.params[PARAM_BA_MAXSAMPLES][0],
                                              max_features=self.params[PARAM_BA_MAX_FEATURES][0],
                                              random_state=0)
                estimator.fit(self.X_train, self.y_train)

        elif COMBO_ITEM_ADABOOST == self.params[PARAM_BAGADA]:
            """バギング/アダブーストグリッドサーチ実行フラグに応じて推定器作成"""
            if True == self.do_bagada_gridsearch:
                estimator = AdaBoostClassifier(base_estimator=estimator, random_state=0)
                bagging_param_grid = self.make_bagada_param_grid()
                estimator = self.get_grid_search_estimator(estimator, self.X_train, self.y_train, bagging_param_grid)

            else:
                estimator = AdaBoostClassifier(base_estimator=estimator,
                                               n_estimators=self.params[PARAM_BA_NESTIMATOR][0],
                                               learning_rate=self.params[PARAM_BA_LEARNINGRATE][0],
                                               random_state=0)
                estimator.fit(self.X_train, self.y_train)

        return estimator

    def Perceptron_(self):
        """パーセプトロン実行"""

        estimator = None

        """分析グリッドサーチ実行フラグに応じて推定器作成"""
        if True == self.do_analysis_gridsearch:
            estimator = Perceptron(random_state=0, shuffle=True)
            estimator = self.make_grid_search_estimator(estimator)

        else:
            estimator = Perceptron(eta0=self.params[PARAM_ETA0][0], penalty=self.params[PARAM_PENALTY][0],
                                   random_state=0, shuffle=True, n_jobs=-1)
            estimator.fit(self.X_train, self.y_train)

        """バギング/アダブースト推定器作成"""
        estimator = self.make_bag_ada_estimator(estimator)

        return estimator

    def LogisticRegression_(self):
        """ロジスティック回帰実行"""

        estimator = None

        """分析グリッドサーチ実行フラグに応じて推定器作成"""
        if True == self.do_analysis_gridsearch:
            estimator = LogisticRegression(random_state=0)
            estimator = self.make_grid_search_estimator(estimator)

        else:
            estimator = LogisticRegression(penalty=self.params[PARAM_PENALTY][0], random_state=0, n_jobs=-1)
            estimator.fit(self.X_train, self.y_train)

        """バギング/アダブースト推定器作成"""
        estimator = self.make_bag_ada_estimator(estimator)

        return estimator

    def SVM_(self):
        """SVM実行"""

        estimator = None

        """分析グリッドサーチ実行フラグに応じて推定器作成"""
        if True == self.do_analysis_gridsearch:
            estimator = SVC(random_state=0)
            estimator = self.make_grid_search_estimator(estimator)

        else:
            estimator = SVC(kernel=self.params[PARAM_KERNEL][0], gamma=self.params[PARAM_GAMMA][0],
                            C=self.params[PARAM_C][0], random_state=0)
            estimator.fit(self.X_train, self.y_train)

        """バギング/アダブースト推定器作成"""
        estimator = self.make_bag_ada_estimator(estimator)

        return estimator

    def RandomForest_(self):
        """ランダムフォレスト実行"""

        estimator = None

        """分析グリッドサーチ実行フラグに応じて推定器作成"""
        if True == self.do_analysis_gridsearch:
            estimator = RandomForestClassifier(random_state=0)
            estimator = self.make_grid_search_estimator(estimator)

        else:
            estimator = RandomForestClassifier(n_estimators=self.params[PARAM_CLS_NESTIMATORS][0], random_state=0)
            estimator.fit(self.X_train, self.y_train)

        """バギング/アダブースト推定器作成"""
        estimator = self.make_bag_ada_estimator(estimator)

        return estimator

    def KNeighbors_(self):
        """k近傍法実行"""

        estimator = None

        """分析グリッドサーチ実行フラグに応じて推定器作成"""
        if True == self.do_analysis_gridsearch:
            estimator = KNeighborsClassifier()
            estimator = self.make_grid_search_estimator(estimator)

        else:
            estimator = KNeighborsClassifier(n_neighbors=self.params[PARAM_NEIGHBORS][0])
            estimator.fit(self.X_train, self.y_train)

        """バギング/アダブースト推定器作成"""
        estimator = self.make_bag_ada_estimator(estimator)

        return estimator


class Classifier(machine_learning):
    """分類に特化した機械学習処理クラス"""

    def __init__(self):
        super().__init__()

    def run_learning(self):
        """学習実行"""

        """訓練データを訓練用とテスト用に分割"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df_train_X, self.df_train_Y, test_size=TEST_SIZE, random_state=0)

        """分析手法別に学習実施"""
        estimator = None
        if COMBO_ITEM_PERCEPTRON == self.params[PARAM_ANALYSIS]:
            estimator = super().Perceptron_()
        elif COMBO_ITEM_ROGISTICREGRESSION == self.params[PARAM_ANALYSIS]:
            estimator = super().LogisticRegression_()
        elif COMBO_ITEM_SVM == self.params[PARAM_ANALYSIS]:
            estimator = super().SVM_()
        elif COMBO_ITEM_RANDOMFOREST_CLS == self.params[PARAM_ANALYSIS]:
            estimator = super().RandomForest_()
        elif COMBO_ITEM_KNEIGHBORS == self.params[PARAM_ANALYSIS]:
            estimator = super().KNeighbors_()
        else:
            print('該当分析手法なし')

        return estimator


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
        elif COMBO_ITEM_RANDOMFOREST_PRD == self.params[PARAM_ANALYSIS]:
            pass
        elif COMBO_ITEM_EXTRATREE == self.params[PARAM_ANALYSIS]:
            pass
        elif COMBO_ITEM_DEEPLEARNING == self.params[PARAM_ANALYSIS]:
            pass
        else:
            print('該当分析手法なし')
