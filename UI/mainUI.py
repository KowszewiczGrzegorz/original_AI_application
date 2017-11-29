from constants import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from mainControler import MainControler


class Main(QWidget):
    """UIのメインクラス"""

    def __init__(self):
        super(Main, self).__init__(None)
        self.__initialize()

    def __initialize(self):
        """初期化"""

        """ウィンドウの基本設定"""
        self.setGeometry(300, 300, 480, 300)
        self.setWindowTitle(WINDOW_TITLE_MAIN)
        self.setWindowIcon(QIcon(APPLICATION_ICON_PATH))
        self.setStyleSheet(WINDOW_APPLICATION)

        """ボタンウィジェット定義"""
        button_selecting_traincsv = QPushButton(BUTTON_SELECTING_TRAINCSV, self)
        button_selecting_testcsv = QPushButton(BUTTON_SELECTING_TESTCSV, self)

        button_selecting_traincsv.setStyleSheet(BUTTON_STYLE_SELECT_DATA)
        button_selecting_testcsv.setStyleSheet(BUTTON_STYLE_SELECT_DATA)

        button_selecting_traincsv.clicked.connect(self.__select_csv)
        button_selecting_testcsv.clicked.connect(self.__select_csv)

        """ラベルウィジェット定義"""
        label_displaying_selectfile = QLabel(LABEL_DISPLAYING_SELECTFILE, self)
        self.label_displaying_traincsv = QLabel(NOT_SELECTING, self)
        self.label_displaying_testcsv = QLabel(NOT_SELECTING, self)
        label_displaying_selectmethod = QLabel(LABEL_DISPLAYING_SELECTMETHOD, self)
        self.label_displaying_notselecting = QLabel(LABEL_DISPLAYING_NOTSELECT, self)

        label_displaying_selectfile.setStyleSheet(LABEL_STYLE_BASIC_MSG)
        self.label_displaying_traincsv.setStyleSheet(LABEL_STYLE_SELECT_DATA)
        self.label_displaying_testcsv.setStyleSheet(LABEL_STYLE_SELECT_DATA)
        self.label_displaying_notselecting.setStyleSheet(LABEL_STYLE_NOT_SELECT_DATA)
        label_displaying_selectmethod.setStyleSheet(LABEL_STYLE_BASIC_MSG)


        """コンボボックスウィジェット定義"""
        self.combo_selecting_cls_or_prd = QComboBox(self)
        self.combo_selecting_cls_or_prd.addItem(COMBO_ITEM_NOTSELECT)
        self.combo_selecting_cls_or_prd.addItem(COMBO_ITEM_CLASSIFIER)
        self.combo_selecting_cls_or_prd.addItem(COMBO_ITEM_PREDICTOR)
        self.combo_selecting_cls_or_prd.activated[str].connect(self.__show_AI_UI)
        self.combo_selecting_cls_or_prd.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.combo_selecting_cls_or_prd.setStyleSheet(COMBO_STYLE_SELECT_METHOD)

        """レイアウト設定"""
        hbox1 = QHBoxLayout()
        hbox1.addWidget(button_selecting_traincsv)
        hbox1.addWidget(self.label_displaying_traincsv)
        hbox1.addStretch()

        hbox2 = QHBoxLayout()
        hbox2.addWidget(button_selecting_testcsv)
        hbox2.addWidget(self.label_displaying_testcsv)
        hbox2.addStretch()

        hbox3 = QHBoxLayout()
        hbox3.addWidget(self.combo_selecting_cls_or_prd)
        hbox3.addWidget(self.label_displaying_notselecting)

        vbox = QVBoxLayout()
        vbox.addWidget(label_displaying_selectfile)
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addSpacing(SPACE_BETWEEN_DATA_AND_METHOD)
        vbox.addWidget(label_displaying_selectmethod)
        vbox.addLayout(hbox3)
        vbox.addStretch()

        self.setLayout(vbox)

        """主制御クラスインスタンス化"""
        self.main_controler = MainControler()

    def __select_csv(self):
        """csvファイル選択"""

        """送り主特定"""
        sender_name = self.sender().text()

        """ファイル名取得後ラベルに表示&データ取得"""
        file_path = QFileDialog.getOpenFileName(self, FILE_SELECT_DIALOG, CSV_DIRECTORY_PATH)
        # ファイルが選択されなかった場合は処理スルー
        if not file_path[0]:
            return
        # csvファイルが選択されなかった場合はその旨を表示(適宜データ削除)
        elif -1 == file_path[0].rfind(EXTENSION_NAME_CSV):
            # trainボタン押下の場合
            if -1 == sender_name.find(BUTTON_SELECTING_TESTCSV):
                self.label_displaying_traincsv.setText(ERROR_MSG_NOTCSV)
                self.label_displaying_notselecting.setText(LABEL_DISPLAYING_NOTSELECT)
                self.main_controler.delete_train_dataframe()
                self.combo_selecting_cls_or_prd.setCurrentIndex(INDEX_NOT_SELECTING)
            # testボタン押下の場合
            else:
                self.label_displaying_testcsv.setText(ERROR_MSG_NOTCSV)
                self.main_controler.delete_test_dataframe()
        # 正常系
        else:
            # trainボタン押下の場合
            if -1 == sender_name.find(BUTTON_SELECTING_TESTCSV):
                self.label_displaying_traincsv.setText(file_path[0])
                self.main_controler.set_train_dataframe(file_path[0])
                self.label_displaying_notselecting.setText(SUCCESSFULLY_SELECTED)

            # testボタン押下の場合
            else:
                self.label_displaying_testcsv.setText(file_path[0])
                self.main_controler.set_test_dataframe(file_path[0])

    def __show_AI_UI(self, text):
        """コンボボックスの選択値により分類または予測のUI表示"""

        """トレーニングデータが未選択の場合は分類/予測を選択させない"""
        train_df = self.main_controler.get_train_dataframe()
        if train_df is not None:
            if COMBO_ITEM_CLASSIFIER == text:
                classifierUI = ClassifierUI()
                classifierUI.exec_()
            elif COMBO_ITEM_PREDICTOR == text:
                predictorUI = PredictorUI()
                predictorUI.exec_()
            # プルダウンが「未選択」のままの場合はスルー
            else:
                pass
        else:
            self.combo_selecting_cls_or_prd.setCurrentIndex(0)


class machine_learning_UI(QDialog):
    """機械学習系UI親クラス"""

    def __init__(self):
        super().__init__()
        self.__initialize()

    def __initialize(self):
        """初期化"""

        """ウィンドウの基本設定"""
        self.setGeometry(320, 320, 480, 300)
        self.setStyleSheet(WINDOW_APPLICATION)

        """ラベルウィジェット定義"""
        label_displaying_use_std = QLabel(LABEL_DISPLAYING_USE_STD, self)
        label_displaying_compress_method = QLabel(LABEL_DISPLAYING_COMPRESS_METHOD, self)
        self.label_displaying_threshold = QLabel(LABEL_DISPLAYING_THRESHOLD, self)

        label_displaying_use_std.setStyleSheet(LABEL_STYLE_BASIC_MSG)
        label_displaying_compress_method.setStyleSheet(LABEL_STYLE_BASIC_MSG)
        # self.label_displaying_threshold.setStyleSheet(LABEL_STYLE_THRESHOLD)

        self.label_displaying_threshold.setEnabled(False)

        """チェックボックスウィジェット定義"""
        chk_selecting_std_to_train = QCheckBox(BUTTON_SELECTING_TRAINCSV, self)
        chk_selecting_std_to_test = QCheckBox(BUTTON_SELECTING_TESTCSV, self)

        chk_selecting_std_to_train.setStyleSheet(CHK_SELECTING_STD)
        chk_selecting_std_to_test.setStyleSheet(CHK_SELECTING_STD)

        """コンボボックスウィジェット定義"""
        combo_selecting_data_compress_method = QComboBox(self)
        combo_selecting_data_compress_method.addItem(COMBO_ITEM_METHOD_NOTSELECT)
        combo_selecting_data_compress_method.addItem(COMBO_ITEM_SELECT_FEATURES)
        combo_selecting_data_compress_method.addItem(COMBO_ITEM_PCA)
        combo_selecting_data_compress_method.addItem(COMBO_ITEM_LDA)
        combo_selecting_data_compress_method.addItem(COMBO_ITEM_KERNEL_PCA)
        combo_selecting_data_compress_method.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        combo_selecting_data_compress_method.setStyleSheet(COMBO_STYLE_SELECT_COMPRESS)
        combo_selecting_data_compress_method.activated[str].connect(self.__on_selecting_compress_method)

        """ラインエディットウィジェット定義"""
        self.ledit_input_threshold = QLineEdit(self)

        self.ledit_input_threshold.setFixedWidth(30)

        # self.ledit_input_threshold.setStyleSheet(LEDIT_STYLE_THRESHOLD)

        self.ledit_input_threshold.setEnabled(False)

        """レイアウト設定"""
        hbox1 = QHBoxLayout()
        hbox1.addWidget(chk_selecting_std_to_train)
        hbox1.addWidget(chk_selecting_std_to_test)
        hbox1.addStretch()

        hbox2 = QHBoxLayout()
        hbox2.addWidget(combo_selecting_data_compress_method)
        hbox2.addSpacing(SPACE_BETWEEN_COMPRESS_AND_THRESHOLD)
        hbox2.addWidget(self.label_displaying_threshold)
        hbox2.addWidget(self.ledit_input_threshold)
        hbox2.addStretch()

        vbox = QVBoxLayout()
        vbox.addWidget(label_displaying_use_std)
        vbox.addLayout(hbox1)
        vbox.addSpacing(SPACE_BETWEEN_DATA_AND_METHOD)
        vbox.addWidget(label_displaying_compress_method)
        vbox.addLayout(hbox2)
        vbox.addStretch()

        self.setLayout(vbox)

    def __on_selecting_compress_method(self, method):
        """データ圧縮方法プルダウン選択時"""

        """特徴量選択が選ばれた場合は閾値入力ウィジェット有効化"""
        if COMBO_ITEM_SELECT_FEATURES == method:
            self.label_displaying_threshold.setEnabled(True)
            self.ledit_input_threshold.setEnabled(True)
        else:
            self.label_displaying_threshold.setEnabled(False)
            self.ledit_input_threshold.setEnabled(False)


class ClassifierUI(machine_learning_UI):
    """分類UIクラス"""

    def __init__(self):
        super().__init__()
        self.__initialize()

    def __initialize(self):
        """初期化"""

        """ウィンドウの基本設定"""
        self.setWindowTitle(WINDOW_TITLE_CLASSIFIER)


class PredictorUI(machine_learning_UI):
    """予測UIクラス"""

    def __init__(self):
        super().__init__()
        self.__initialize()

    def __initialize(self):
        """初期化"""

        """ウィンドウの基本設定"""
        self.setWindowTitle(WINDOW_TITLE_PREDICTOR)