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


class ClassifierUI(machine_learning_UI):
    """分類UIクラス"""

    def __init__(self):
        super().__init__()
        self.__initialize()

    def __initialize(self):
        """初期化"""

        self.setGeometry(320, 320, 480, 300)
        self.setWindowTitle(WINDOW_TITLE_CLASSIFIER)


class PredictorUI(machine_learning_UI):
    """予測UIクラス"""

    def __init__(self):
        super().__init__()
        self.__initialize()

    def __initialize(self):
        """初期化"""

        self.setGeometry(320, 320, 480, 300)
        self.setWindowTitle(WINDOW_TITLE_PREDICTOR)