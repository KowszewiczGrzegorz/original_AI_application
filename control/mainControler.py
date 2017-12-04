from lib import *


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

        self.df_train_X = standardize_datas(self.df_train_X)
        self.df_test = standardize_datas(self.df_train_X)


class Classifier(machine_learning):
    """分類に特化した機械学習処理クラス"""

    def __init__(self):
        super().__init__()


class Predictor(machine_learning):
    """予測に特化した機械学習処理クラス"""

    def __init__(self):
        super().__init__()