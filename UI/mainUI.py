from constants import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from mainControler import MainControler


class Main(QWidget):
    """UIのメインクラス"""

    def __init__(self):
        super(Main, self).__init__(None)
        self._initialize()

    def _initialize(self):
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

        button_selecting_traincsv.clicked.connect(self._on_press_csv_button)
        button_selecting_testcsv.clicked.connect(self._on_press_csv_button)

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
        self.combo_selecting_cls_or_prd.activated[str].connect(self._on_select_method_combo)
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
        vbox.addSpacing(SPACE_BETWEEN_PARTS)
        vbox.addWidget(label_displaying_selectmethod)
        vbox.addLayout(hbox3)
        vbox.addStretch()

        self.setLayout(vbox)

        """主制御クラスインスタンス化"""
        self.main_controler = MainControler()

    def _on_press_csv_button(self):
        """csvファイル選択ボタン押下時"""

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

    def _on_select_method_combo(self, text):
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

    def _on_check_std_chkbutton(self):
        """データ標準化チェックボックス押下時"""

        """送り主特定"""
        sender_name = self.sender().text()

        pass

    def _on_select_compress_combo(self, method):
        """データ圧縮方法プルダウン選択時"""

        """特徴量選択が選ばれた場合は閾値入力ウィジェット有効化"""
        if COMBO_ITEM_SELECT_FEATURES == method:
            self.label_displaying_threshold.setEnabled(True)
            self.ledit_input_threshold.setEnabled(True)
        else:
            self.label_displaying_threshold.setEnabled(False)
            self.ledit_input_threshold.setEnabled(False)

    def _on_input_threshold(self, value):
        """閾値入力時"""

        pass

    def _on_input_params(self, value):
        """パラメータ入力時"""

        """送り主特定"""
        sender_name = self.sender().accessibleName()
        print('name:', sender_name)
        print('value:', value)

        pass

    def _valid_param_wiget_by_method(self, label_wigets, input_wigets, ids):
        """分析手法によってパラメータ系ウィジェットの有効化/無効化"""

        """2つのウィジェットを平行的に取り出すと同時にインデックスも取り出し処理"""
        for (i, label_wiget), input_wiget in zip(enumerate(label_wigets), input_wigets):
            """idがTrueの場合有効化"""
            if ids[i]:
                label_wiget.setStyleSheet(LABEL_STYLE_PARAM_VALID)
                input_wiget.setEnabled(True)
            else:
                label_wiget.setStyleSheet(LABEL_STYLE_PARAM_INVALID)
                input_wiget.setEnabled(False)

    def _make_common_part(self):
        """分類と予測で共通部分の作成"""

        """ウィンドウの基本設定"""
        self.setGeometry(320, 320, 480, 300)
        self.setStyleSheet(WINDOW_APPLICATION)

        """ラベルウィジェット定義"""
        label_displaying_use_std = QLabel(LABEL_DISPLAYING_USE_STD, self)
        PARAM_Compress_method = QLabel(PARAM_COMPRESS_METHOD, self)
        self.label_displaying_threshold = QLabel(LABEL_DISPLAYING_THRESHOLD, self)

        label_displaying_use_std.setStyleSheet(LABEL_STYLE_BASIC_MSG)
        PARAM_Compress_method.setStyleSheet(LABEL_STYLE_BASIC_MSG)
        # self.label_displaying_threshold.setStyleSheet(LABEL_STYLE_THRESHOLD)

        self.label_displaying_threshold.setEnabled(False)

        """チェックボックスウィジェット定義"""
        chk_selecting_std_to_train = QCheckBox(BUTTON_SELECTING_TRAINCSV, self)
        chk_selecting_std_to_test = QCheckBox(BUTTON_SELECTING_TESTCSV, self)

        chk_selecting_std_to_train.setStyleSheet(CHK_SELECTING_STD)
        chk_selecting_std_to_test.setStyleSheet(CHK_SELECTING_STD)

        chk_selecting_std_to_train.stateChanged.connect(self._on_check_std_chkbutton)
        chk_selecting_std_to_test.stateChanged.connect(self._on_check_std_chkbutton)

        """コンボボックスウィジェット定義"""
        combo_selecting_data_compress_method = QComboBox(self)
        combo_selecting_data_compress_method.addItem(COMBO_ITEM_METHOD_NOTSELECT)
        combo_selecting_data_compress_method.addItem(COMBO_ITEM_SELECT_FEATURES)
        combo_selecting_data_compress_method.addItem(COMBO_ITEM_PCA)
        combo_selecting_data_compress_method.addItem(COMBO_ITEM_LDA)
        combo_selecting_data_compress_method.addItem(COMBO_ITEM_KERNEL_PCA)
        combo_selecting_data_compress_method.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        combo_selecting_data_compress_method.setStyleSheet(COMBO_STYLE_SELECT_COMPRESS)
        combo_selecting_data_compress_method.activated[str].connect(self._on_select_compress_combo)

        """ラインエディットウィジェット定義"""
        self.ledit_input_threshold = QLineEdit(self)
        self.ledit_input_threshold.setFixedWidth(LEDIT_WIDTH_THRESHOLD)
        self.ledit_input_threshold.setEnabled(False)
        self.ledit_input_threshold.textChanged[str].connect(self._on_input_threshold)

        """レイアウト設定"""
        hbox1 = QHBoxLayout()
        hbox1.addWidget(chk_selecting_std_to_train)
        hbox1.addWidget(chk_selecting_std_to_test)
        hbox1.addStretch()

        hbox2 = QHBoxLayout()
        hbox2.addWidget(combo_selecting_data_compress_method)
        hbox2.addSpacing(SPACE_BETWEEN_PARTS)
        hbox2.addWidget(self.label_displaying_threshold)
        hbox2.addWidget(self.ledit_input_threshold)
        hbox2.addStretch()

        vbox = QVBoxLayout()
        vbox.addWidget(label_displaying_use_std)
        vbox.addLayout(hbox1)
        vbox.addSpacing(SPACE_BETWEEN_PARTS)
        vbox.addWidget(PARAM_Compress_method)
        vbox.addLayout(hbox2)
        vbox.addSpacing(SPACE_BETWEEN_PARTS)

        return vbox


class ClassifierUI(machine_learning_UI):
    """分類UIクラス"""

    def __init__(self):
        super().__init__()
        self._initialize()

    def _initialize(self):
        """初期化"""

        """ウィンドウの基本設定"""
        self.setWindowTitle(WINDOW_TITLE_CLASSIFIER)

        """ラベルウィジェット定義"""
        self.label_displaying_classifier = QLabel(LABEL_DISPLAYING_CLASSIFIER, self)
        self.label_displaying_param_penalty = QLabel(PARAM_PENALTY, self)
        self.label_displaying_param_kernel = QLabel(PARAM_KERNEL, self)
        self.label_displaying_param_eta0 = QLabel(PARAM_ETA0, self)
        self.label_displaying_param_C = QLabel(PARAM_C, self)
        self.label_displaying_param_gamma = QLabel(PARAM_GAMMA, self)
        self.label_displaying_param_neighbors = QLabel(PARAM_NEIGHBORS, self)
        self.label_displaying_param_nestimators = QLabel(PARAM_NESTIMATORS, self)

        self.label_displaying_classifier.setStyleSheet(LABEL_STYLE_BASIC_MSG)
        self.label_displaying_param_penalty.setStyleSheet(LABEL_STYLE_PARAM_VALID)
        self.label_displaying_param_kernel.setStyleSheet(LABEL_STYLE_PARAM_INVALID)
        self.label_displaying_param_eta0.setStyleSheet(LABEL_STYLE_PARAM_VALID)
        self.label_displaying_param_C.setStyleSheet(LABEL_STYLE_PARAM_INVALID)
        self.label_displaying_param_gamma.setStyleSheet(LABEL_STYLE_PARAM_INVALID)
        self.label_displaying_param_neighbors.setStyleSheet(LABEL_STYLE_PARAM_INVALID)
        self.label_displaying_param_nestimators.setStyleSheet(LABEL_STYLE_PARAM_INVALID)

        """コンボボックスウィジェット定義"""
        self.combo_selecting_analysis_method = QComboBox(self)
        self.combo_selecting_analysis_method.addItem(COMBO_ITEM_PERCEPTRON)
        self.combo_selecting_analysis_method.addItem(COMBO_ITEM_ROGISTICREGRESSION)
        self.combo_selecting_analysis_method.addItem(COMBO_ITEM_SVM)
        self.combo_selecting_analysis_method.addItem(COMBO_ITEM_RANDOMFOREST)
        self.combo_selecting_analysis_method.addItem(COMBO_ITEM_KNEIGHBORS)
        self.combo_selecting_analysis_method.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.combo_selecting_analysis_method.setStyleSheet(COMBO_STYLE_SELECT_CLASSIFIER)
        self.combo_selecting_analysis_method.activated[str].connect(self._on_select_analysis_method)

        self.combo_selecting_penalty = QComboBox(self)
        self.combo_selecting_penalty.addItem(COMBO_ITEM_L1)
        self.combo_selecting_penalty.addItem(COMBO_ITEM_L2)
        self.combo_selecting_penalty.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.combo_selecting_penalty.setStyleSheet(COMBO_STYLE_SELECT_PARAMS)
        self.combo_selecting_penalty.activated[str].connect(super()._on_input_params)
        self.combo_selecting_penalty.setEnabled(True)
        self.combo_selecting_penalty.setAccessibleName(PARAM_PENALTY)

        self.combo_selecting_kernel = QComboBox(self)
        self.combo_selecting_kernel.addItem(COMBO_ITEM_RBF)
        self.combo_selecting_kernel.addItem(COMBO_ITEM_LINEAR)
        self.combo_selecting_kernel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.combo_selecting_kernel.setStyleSheet(COMBO_STYLE_SELECT_PARAMS)
        self.combo_selecting_kernel.activated[str].connect(super()._on_input_params)
        self.combo_selecting_kernel.setEnabled(False)
        self.combo_selecting_kernel.setAccessibleName(PARAM_KERNEL)

        """ラインエディットウィジェット定義"""
        self.ledit_param_eta0 = QLineEdit(self)
        self.ledit_param_C = QLineEdit(self)
        self.ledit_param_gamma = QLineEdit(self)
        self.ledit_param_neighbors = QLineEdit(self)
        self.ledit_param_nestimators = QLineEdit(self)

        self.ledit_param_eta0.setFixedWidth(LEDIT_WIDTH_PARAMS)
        self.ledit_param_C.setFixedWidth(LEDIT_WIDTH_PARAMS)
        self.ledit_param_gamma.setFixedWidth(LEDIT_WIDTH_PARAMS)
        self.ledit_param_neighbors.setFixedWidth(LEDIT_WIDTH_PARAMS)
        self.ledit_param_nestimators.setFixedWidth(LEDIT_WIDTH_PARAMS)

        self.ledit_param_eta0.setEnabled(True)
        self.ledit_param_C.setEnabled(False)
        self.ledit_param_gamma.setEnabled(False)
        self.ledit_param_neighbors.setEnabled(False)
        self.ledit_param_nestimators.setEnabled(False)
        
        self.ledit_param_eta0.textChanged[str].connect(super()._on_input_params)
        self.ledit_param_C.textChanged[str].connect(super()._on_input_params)
        self.ledit_param_gamma.textChanged[str].connect(super()._on_input_params)
        self.ledit_param_neighbors.textChanged[str].connect(super()._on_input_params)
        self.ledit_param_nestimators.textChanged[str].connect(super()._on_input_params)
        
        self.ledit_param_eta0.setAccessibleName(PARAM_ETA0)
        self.ledit_param_C.setAccessibleName(PARAM_C)
        self.ledit_param_gamma.setAccessibleName(PARAM_GAMMA)
        self.ledit_param_neighbors.setAccessibleName(PARAM_NEIGHBORS)
        self.ledit_param_nestimators.setAccessibleName(PARAM_NESTIMATORS)

        """共通部分作成"""
        vbox = super()._make_common_part()

        """レイアウト設定"""
        grid = QGridLayout()
        grid.addWidget(self.label_displaying_param_penalty, 0, 0)
        grid.addWidget(self.combo_selecting_penalty, 0, 1)
        grid.addWidget(self.label_displaying_param_kernel, 0, 2)
        grid.addWidget(self.combo_selecting_kernel, 0, 3)
        grid.addWidget(self.label_displaying_param_eta0, 0, 4)
        grid.addWidget(self.ledit_param_eta0, 0, 5)
        grid.addWidget(self.label_displaying_param_C, 1, 0)
        grid.addWidget(self.ledit_param_C, 1, 1)
        grid.addWidget(self.label_displaying_param_gamma, 1, 2)
        grid.addWidget(self.ledit_param_gamma, 1, 3)
        grid.addWidget(self.label_displaying_param_neighbors, 1, 4)
        grid.addWidget(self.ledit_param_neighbors, 1, 5)
        grid.addWidget(self.label_displaying_param_nestimators, 2, 0)
        grid.addWidget(self.ledit_param_nestimators, 2, 1)

        vbox.addWidget(self.label_displaying_classifier)
        vbox.addWidget(self.combo_selecting_analysis_method)
        vbox.addLayout(grid)

        vbox.addStretch()

        self.setLayout(vbox)

        """パラメータ系ウィジェットを扱いやすいようタプル化/辞書登録"""
        self.param_label_wigets = (self.label_displaying_param_penalty,
                                   self.label_displaying_param_kernel,
                                   self.label_displaying_param_eta0,
                                   self.label_displaying_param_C,
                                   self.label_displaying_param_gamma,
                                   self.label_displaying_param_neighbors,
                                   self.label_displaying_param_nestimators)

        self.param_input_wigets = (self.combo_selecting_penalty,
                                   self.combo_selecting_kernel,
                                   self.ledit_param_eta0,
                                   self.ledit_param_C,
                                   self.ledit_param_gamma,
                                   self.ledit_param_neighbors,
                                   self.ledit_param_nestimators)

        self.param_dictionary = {PARAM_PENALTY: 0,
                                 PARAM_KERNEL: 1,
                                 PARAM_ETA0: 2,
                                 PARAM_C: 3,
                                 PARAM_GAMMA: 4,
                                 PARAM_NEIGHBORS: 5,
                                 PARAM_NESTIMATORS: 6}

    def _on_select_analysis_method(self, method):
        """分析手法選択時"""

        """パラメータ入力解禁/禁止IDを作成しメソッド呼び出し"""
        valid_ids = [False for i in range(len(self.param_dictionary))]

        if COMBO_ITEM_PERCEPTRON == method:
            valid_ids[self.param_dictionary[PARAM_PENALTY]] = True
            valid_ids[self.param_dictionary[PARAM_ETA0]] = True

        elif COMBO_ITEM_ROGISTICREGRESSION == method:
            valid_ids[self.param_dictionary[PARAM_PENALTY]] = True

        elif COMBO_ITEM_SVM == method:
            valid_ids[self.param_dictionary[PARAM_KERNEL]] = True
            valid_ids[self.param_dictionary[PARAM_GAMMA]] = True
            valid_ids[self.param_dictionary[PARAM_C]] = True

        elif COMBO_ITEM_RANDOMFOREST == method:
            valid_ids[self.param_dictionary[PARAM_NESTIMATORS]] = True

        elif COMBO_ITEM_KNEIGHBORS == method:
            valid_ids[self.param_dictionary[PARAM_NEIGHBORS]] = True

        super()._valid_param_wiget_by_method(self.param_label_wigets,
                                             self.param_input_wigets,
                                             valid_ids)


class PredictorUI(machine_learning_UI):
    """予測UIクラス"""

    def __init__(self):
        super().__init__()
        self._initialize()

    def _initialize(self):
        """初期化"""

        """ウィンドウの基本設定"""
        self.setWindowTitle(WINDOW_TITLE_PREDICTOR)

        """ラベルウィジェット定義"""
        label_displaying_predictor_method = QLabel(LABEL_DISPLAYING_PREDICTOR, self)

        label_displaying_predictor_method.setStyleSheet(LABEL_STYLE_BASIC_MSG)

        """コンボボックスウィジェット定義"""
        self.combo_selecting_analysis_method = QComboBox(self)
        self.combo_selecting_analysis_method.addItem(COMBO_ITEM_LINEARREGRESSION)
        self.combo_selecting_analysis_method.addItem(COMBO_ITEM_ELASTICNET)
        self.combo_selecting_analysis_method.addItem(COMBO_ITEM_RANDOMFOREST)
        self.combo_selecting_analysis_method.addItem(COMBO_ITEM_EXTRATREE)
        self.combo_selecting_analysis_method.addItem(COMBO_ITEM_DEEPLEARNING)
        self.combo_selecting_analysis_method.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.combo_selecting_analysis_method.setStyleSheet(COMBO_STYLE_SELECT_PREDICTOR)
        self.combo_selecting_analysis_method.activated[str].connect(self._on_select_analysis_method)

        """共通部分作成"""
        vbox = super()._make_common_part()

        """レイアウト設定"""
        vbox.addWidget(label_displaying_predictor_method)
        vbox.addWidget(self.combo_selecting_analysis_method)
        vbox.addStretch()

        self.setLayout(vbox)

    def _on_select_analysis_method(self):
        """分析手法選択時"""

        pass