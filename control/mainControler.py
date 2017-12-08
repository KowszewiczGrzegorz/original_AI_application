import os
import pandas as pd


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
        f = open(file_name, 'w')

        for key, value in param_dict.items():
            o_str = str(key) + ' ' + str(value) + '\n'
            f.write(o_str)

        f.close()


class machine_learning:
    """機械学習処理のベースクラス"""

    def set_df_train(self, df):
        """トレーニングデータ設定"""

        self.df_train_X = df.iloc[:, 2:]
        self.df_train_Y = df.iloc[:, 1]

    def set_df_test(self, df):
        """テストデータ設定"""

        self.test_ID = df.iloc[:, 0]
        self.df_test = df.iloc[:, 1:]

    def standardize_datas(self):
        """データ標準化"""

        self.df_train_X = self.standardize(self.df_train_X)
        self.df_test = self.standardize(self.df_train_X)

    def standardize(X):
        """変数の標準化"""

        """もらったデータの標準化(平均=0, 標準偏差=1)を行う"""
        sc = StandardScaler()
        return_X = pd.DataFrame(sc.fit_transform(X))

        return return_X


class Classifier(machine_learning):
    """分類に特化した機械学習処理クラス"""

    def __init__(self):
        super().__init__()


class Predictor(machine_learning):
    """予測に特化した機械学習処理クラス"""

    def __init__(self):
        super().__init__()