from constants.constants import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from control.mainControler import MainControler, Classifier, Predictor


class Main(QWidget):
    """UIのメインクラス"""

    def __init__(self):
        super(Main, self).__init__(None)

        """主制御クラスインスタンス化"""
        self.main_controler = MainControler()

        self._initialize()

    def _initialize(self):
        """初期化"""

        """ウィンドウの基本設定"""
        self.setGeometry(300, 300, 0, 0)
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
        test_df = self.main_controler.get_test_dataframe()

        if train_df is None:
            self.combo_selecting_cls_or_prd.setCurrentIndex(0)
            return

        """分類または予測ダイアログ表示"""
        if COMBO_ITEM_CLASSIFIER == text:
            classifierUI = ClassifierUI()
            classifierUI.set_df_train(train_df)

            if test_df is not None:
                classifierUI.set_df_test(test_df)

            classifierUI.exec_()

        elif COMBO_ITEM_PREDICTOR == text:
            predictorUI = PredictorUI()
            predictorUI.set_df_train(train_df)

            if test_df is not None:
                predictorUI.set_df_test(test_df)

            predictorUI.exec_()
        # プルダウンが「未選択」のままの場合はスルー
        else:
            pass


class machine_learning_UI(QDialog):
    """機械学習系UI親クラス"""

    def __init__(self):
        super().__init__()

        self.df_train = None
        self.df_test = None
        self.do_std = False

    def _on_check_std_chkbutton(self, state):
        """データ標準化チェックボックス押下時"""

        """標準化フラグON/OFF"""
        if Qt.Checked == state:
            self.do_std = True
        else:
            self.do_std = False

    def _on_select_compress_combo(self, method):
        """データ圧縮方法プルダウン選択時"""

        """特徴量選択が選ばれた場合は閾値入力ウィジェット有効化"""
        if COMBO_ITEM_SELECT_FEATURES == method:
            self.label_displaying_threshold.setEnabled(True)
            self.ledit_input_threshold.setEnabled(True)
            self.label_displaying_threshold.setStyleSheet(LABEL_STYLE_PARAM_VALID)
            self.ledit_input_threshold.setStyleSheet(INPUT_STYLE_PARAMS_VALID)
        else:
            self.label_displaying_threshold.setEnabled(False)
            self.ledit_input_threshold.setEnabled(False)
            self.label_displaying_threshold.setStyleSheet(LABEL_STYLE_PARAM_INVALID)
            self.ledit_input_threshold.setStyleSheet(INPUT_STYLE_PARAMS_INVALID)

    def _on_input_threshold(self, value):
        """閾値入力時"""

        pass

    def _on_input_params(self, value):
        """パラメータ入力時"""

        """送り主特定"""
        sender_name = self.sender().accessibleName()
        print('name:', sender_name)
        print('value:', value)

    def _on_select_bagada_combo(self, method):
        """バギング/アダブースト設定コンボ選択時"""

        """パラメータ入力解禁/禁止IDを作成しメソッド呼び出し"""
        valid_ids = [False for i in range(len(self.bagada_param_dictionary))]

        if COMBO_ITEM_BAGGING == method:
            valid_ids[self.bagada_param_dictionary[PARAM_NESTIMATORS]] = True
            valid_ids[self.bagada_param_dictionary[PARAM_MAXSAMPLES]] = True
            valid_ids[self.bagada_param_dictionary[PARAM_MAXFEATURES]] = True

        elif COMBO_ITEM_ADABOOST == method:
            valid_ids[self.bagada_param_dictionary[PARAM_NESTIMATORS]] = True
            valid_ids[self.bagada_param_dictionary[PARAM_LEARNINGRATE]] = True

        self._valid_param_wiget_by_method(self.bagada_param_label_wigets,
                                         self.bagada_param_input_wigets,
                                         valid_ids)

    def _on_input_bag_ada(self, value):
        """バギング/アダブーストパラメータ入力時"""

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
                input_wiget.setStyleSheet(INPUT_STYLE_PARAMS_VALID)
            else:
                label_wiget.setStyleSheet(LABEL_STYLE_PARAM_INVALID)
                input_wiget.setEnabled(False)
                input_wiget.setStyleSheet(INPUT_STYLE_PARAMS_INVALID)

    def _make_std_and_compress_part(self):
        """データ標準化・圧縮部の作成"""

        """ウィンドウの基本設定"""
        self.setGeometry(320, 320, 0, 0)
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
        self.chk_selecting_std = QCheckBox(CHK_SELECTING_STD, self)
        self.chk_selecting_std.setStyleSheet(CHK_SELECTING_STD)
        self.chk_selecting_std.stateChanged.connect(self._on_check_std_chkbutton)

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
        self.ledit_input_threshold.setStyleSheet(INPUT_STYLE_PARAMS_INVALID)

        """レイアウト設定"""
        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.chk_selecting_std)
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

    def _make_bag_and_ada_part(self, vbox):
        """バギング・アダブースト部の作成"""

        """ラベルウィジェット定義"""
        label_displaying_bag_ada = QLabel(LABEL_DISPLAYING_BAG_ADA, self)
        self.label_displaying_bagada_nestimator = QLabel(PARAM_NESTIMATORS, self)
        self.label_displaying_bagada_maxsamples = QLabel(PARAM_MAXSAMPLES, self)
        self.label_displaying_bagada_maxfeatures = QLabel(PARAM_MAXFEATURES, self)
        self.label_displaying_bagada_learningrate= QLabel(PARAM_LEARNINGRATE, self)

        label_displaying_bag_ada.setStyleSheet(LABEL_STYLE_BASIC_MSG)
        self.label_displaying_bagada_nestimator.setStyleSheet(LABEL_STYLE_PARAM_INVALID)
        self.label_displaying_bagada_maxsamples.setStyleSheet(LABEL_STYLE_PARAM_INVALID)
        self.label_displaying_bagada_maxfeatures.setStyleSheet(LABEL_STYLE_PARAM_INVALID)
        self.label_displaying_bagada_learningrate.setStyleSheet(LABEL_STYLE_PARAM_INVALID)

        """コンボボックスウィジェット定義"""
        combo_selecting_bagging_or_adaboost = QComboBox(self)
        combo_selecting_bagging_or_adaboost.addItem(COMBO_ITEM_NOTSELECT)
        combo_selecting_bagging_or_adaboost.addItem(COMBO_ITEM_BAGGING)
        combo_selecting_bagging_or_adaboost.addItem(COMBO_ITEM_ADABOOST)
        combo_selecting_bagging_or_adaboost.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        combo_selecting_bagging_or_adaboost.setStyleSheet(COMBO_STYLE_SELECT_COMPRESS)
        combo_selecting_bagging_or_adaboost.activated[str].connect(self._on_select_bagada_combo)

        """ラインエディットウィジェット定義"""
        self.ledit_input_bagada_nestimator = QLineEdit(self)
        self.ledit_input_bagada_maxsamples = QLineEdit(self)
        self.ledit_input_bagada_maxfeatures = QLineEdit(self)
        self.ledit_input_bagada_learningrate = QLineEdit(self)

        self.ledit_input_bagada_nestimator.setEnabled(False)
        self.ledit_input_bagada_maxsamples.setEnabled(False)
        self.ledit_input_bagada_maxfeatures.setEnabled(False)
        self.ledit_input_bagada_learningrate.setEnabled(False)

        self.ledit_input_bagada_nestimator.textChanged[str].connect(self._on_input_bag_ada)
        self.ledit_input_bagada_maxsamples.textChanged[str].connect(self._on_input_bag_ada)
        self.ledit_input_bagada_maxfeatures.textChanged[str].connect(self._on_input_bag_ada)
        self.ledit_input_bagada_learningrate.textChanged[str].connect(self._on_input_bag_ada)

        self.ledit_input_bagada_nestimator.setStyleSheet(INPUT_STYLE_PARAMS_INVALID)
        self.ledit_input_bagada_maxsamples.setStyleSheet(INPUT_STYLE_PARAMS_INVALID)
        self.ledit_input_bagada_maxfeatures.setStyleSheet(INPUT_STYLE_PARAMS_INVALID)
        self.ledit_input_bagada_learningrate.setStyleSheet(INPUT_STYLE_PARAMS_INVALID)

        self.ledit_input_bagada_nestimator.setAccessibleName(PARAM_NESTIMATORS)
        self.ledit_input_bagada_maxsamples.setAccessibleName(PARAM_MAXSAMPLES)
        self.ledit_input_bagada_maxfeatures.setAccessibleName(PARAM_MAXFEATURES)
        self.ledit_input_bagada_learningrate.setAccessibleName(PARAM_LEARNINGRATE)

        """レイアウト設定"""
        vbox.addSpacing(SPACE_BETWEEN_PARTS)
        vbox.addWidget(label_displaying_bag_ada)
        vbox.addWidget(combo_selecting_bagging_or_adaboost)

        grid = QGridLayout()
        grid.addWidget(self.label_displaying_bagada_nestimator, 0, 0)
        grid.addWidget(self.ledit_input_bagada_nestimator, 0, 1)
        grid.addWidget(self.label_displaying_bagada_maxsamples, 0, 2)
        grid.addWidget(self.ledit_input_bagada_maxsamples, 0, 3)
        grid.addWidget(self.label_displaying_bagada_maxfeatures, 1, 0)
        grid.addWidget(self.ledit_input_bagada_maxfeatures, 1, 1)
        grid.addWidget(self.label_displaying_bagada_learningrate, 1, 2)
        grid.addWidget(self.ledit_input_bagada_learningrate, 1, 3)

        vbox.addLayout(grid)

        """バギング/アダブーストパラメータ系ウィジェットを扱いやすいようタプル化/辞書登録"""
        self.bagada_param_label_wigets = (self.label_displaying_bagada_nestimator,
                                          self.label_displaying_bagada_maxsamples,
                                          self.label_displaying_bagada_maxfeatures,
                                          self.label_displaying_bagada_learningrate)

        self.bagada_param_input_wigets = (self.ledit_input_bagada_nestimator,
                                          self.ledit_input_bagada_maxsamples,
                                          self.ledit_input_bagada_maxfeatures,
                                          self.ledit_input_bagada_learningrate)

        self.bagada_param_dictionary = {PARAM_NESTIMATORS: 0,
                                        PARAM_MAXSAMPLES: 1,
                                        PARAM_MAXFEATURES: 2,
                                        PARAM_LEARNINGRATE: 3}

        return vbox

    def _make_running_machine_learning_part(self, vbox, button):
        """学習・予測実行部作成"""

        """ラベルウィジェット定義"""
        label_displaying_running = QLabel(LABEL_DISPLAYING_RUNNING, self)
        label_displaying_running.setStyleSheet(LABEL_STYLE_BASIC_MSG)

        """ボタンウィジェット定義"""
        button.setStyleSheet(BUTTON_STYLE_RUNNING_MACHINE_LEARNING)

        """レイアウト設定"""
        vbox.addSpacing(SPACE_BETWEEN_PARTS)
        vbox.addWidget(label_displaying_running)
        vbox.addWidget(button)

        return vbox

    def set_df_train(self, df):
        """トレーニングデータ設定"""

        self.df_train = df

    def set_df_test(self, df):
        """テストデータ設定"""

        self.df_test = df

    def set_datas(self, method_object):
        """分類/予測オブジェクトにデータ設定"""

        method_object.set_df_train(self.df_train)
        if self.df_test is not None:
            method_object.set_df_test(self.df_test)

    def standardize_datas(self, method_object):
        """データの標準化"""

        """標準化チェック時のみ標準化メソッド呼び出し"""
        if self.do_std:
            method_object.standardize_datas()


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
        self.label_displaying_classifier_method = QLabel(LABEL_DISPLAYING_CLASSIFIER, self)
        self.label_displaying_param_penalty = QLabel(PARAM_PENALTY, self)
        self.label_displaying_param_kernel = QLabel(PARAM_KERNEL, self)
        self.label_displaying_param_eta0 = QLabel(PARAM_ETA0, self)
        self.label_displaying_param_C = QLabel(PARAM_C, self)
        self.label_displaying_param_gamma = QLabel(PARAM_GAMMA, self)
        self.label_displaying_param_neighbors = QLabel(PARAM_NEIGHBORS, self)
        self.label_displaying_param_nestimators = QLabel(PARAM_NESTIMATORS, self)

        self.label_displaying_classifier_method.setStyleSheet(LABEL_STYLE_BASIC_MSG)
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
        self.combo_selecting_penalty.setStyleSheet(INPUT_STYLE_PARAMS_VALID)
        self.combo_selecting_penalty.activated[str].connect(super()._on_input_params)
        self.combo_selecting_penalty.setEnabled(True)
        self.combo_selecting_penalty.setAccessibleName(PARAM_PENALTY)

        self.combo_selecting_kernel = QComboBox(self)
        self.combo_selecting_kernel.addItem(COMBO_ITEM_RBF)
        self.combo_selecting_kernel.addItem(COMBO_ITEM_LINEAR)
        self.combo_selecting_kernel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.combo_selecting_kernel.setStyleSheet(INPUT_STYLE_PARAMS_INVALID)
        self.combo_selecting_kernel.activated[str].connect(super()._on_input_params)
        self.combo_selecting_kernel.setEnabled(False)
        self.combo_selecting_kernel.setAccessibleName(PARAM_KERNEL)

        """ラインエディットウィジェット定義"""
        self.ledit_param_eta0 = QLineEdit(self)
        self.ledit_param_C = QLineEdit(self)
        self.ledit_param_gamma = QLineEdit(self)
        self.ledit_param_neighbors = QLineEdit(self)
        self.ledit_param_nestimators = QLineEdit(self)

        self.ledit_param_eta0.setStyleSheet(INPUT_STYLE_PARAMS_VALID)
        self.ledit_param_C.setStyleSheet(INPUT_STYLE_PARAMS_INVALID)
        self.ledit_param_gamma.setStyleSheet(INPUT_STYLE_PARAMS_INVALID)
        self.ledit_param_neighbors.setStyleSheet(INPUT_STYLE_PARAMS_INVALID)
        self.ledit_param_nestimators.setStyleSheet(INPUT_STYLE_PARAMS_INVALID)

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

        """ボタンウィジェット定義"""
        button_running_machine_learning = QPushButton(BUTTON_RUNNING_MACHINE_LEARNING, self)
        button_running_machine_learning.clicked.connect(self._on_clicked_running_button)

        """共通部分作成"""
        vbox = super()._make_std_and_compress_part()

        """レイアウト設定"""
        grid = QGridLayout()
        grid.addWidget(self.label_displaying_param_penalty, 0, 0)
        grid.addWidget(self.combo_selecting_penalty, 0, 1)
        grid.addWidget(self.label_displaying_param_kernel, 0, 2)
        grid.addWidget(self.combo_selecting_kernel, 0, 3)
        grid.addWidget(self.label_displaying_param_eta0, 1, 0)
        grid.addWidget(self.ledit_param_eta0, 1, 1)
        grid.addWidget(self.label_displaying_param_C, 1, 2)
        grid.addWidget(self.ledit_param_C, 1, 3)
        grid.addWidget(self.label_displaying_param_gamma, 1, 4)
        grid.addWidget(self.ledit_param_gamma, 1, 5)
        grid.addWidget(self.label_displaying_param_neighbors, 2, 0)
        grid.addWidget(self.ledit_param_neighbors, 2, 1)
        grid.addWidget(self.label_displaying_param_nestimators, 2,2)
        grid.addWidget(self.ledit_param_nestimators, 2, 3)

        vbox.addWidget(self.label_displaying_classifier_method)
        vbox.addWidget(self.combo_selecting_analysis_method)
        vbox.addLayout(grid)
        
        vbox = super()._make_bag_and_ada_part(vbox)

        vbox = super()._make_running_machine_learning_part(vbox, button_running_machine_learning)

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

    def _on_clicked_running_button(self):
        """学習実行ボタン押下時"""

        """分類オブジェクト作成"""
        classifier = Classifier()

        """データ設定"""
        super().set_datas(classifier)

        """データ標準化"""
        super().standardize_datas(classifier)


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
        self.label_displaying_predictor_method = QLabel(LABEL_DISPLAYING_PREDICTOR, self)
        self.label_displaying_param_alpha = QLabel(PARAM_ALPHA, self)
        self.label_displaying_param_l1ratio = QLabel(PARAM_L1RATIO, self)
        self.label_displaying_param_maxfeatures = QLabel(PARAM_MAXFEATURES, self)
        self.label_displaying_param_maxdepth = QLabel(PARAM_MAXDEPTH, self)
        self.label_displaying_param_nestimators = QLabel(PARAM_NESTIMATORS, self)
        self.label_displaying_param_batchsize = QLabel(PARAM_BATCHSIZE, self)
        self.label_displaying_param_nhidden = QLabel(PARAM_NHIDDEN, self)
        self.label_displaying_param_nunit = QLabel(PARAM_NUNIT, self)
        self.label_displaying_param_keepdrop = QLabel(PARAM_KEEPDROP, self)

        self.label_displaying_predictor_method.setStyleSheet(LABEL_STYLE_BASIC_MSG)
        self.label_displaying_param_alpha.setStyleSheet(LABEL_STYLE_PARAM_INVALID)
        self.label_displaying_param_l1ratio.setStyleSheet(LABEL_STYLE_PARAM_INVALID)
        self.label_displaying_param_maxfeatures.setStyleSheet(LABEL_STYLE_PARAM_INVALID)
        self.label_displaying_param_maxdepth.setStyleSheet(LABEL_STYLE_PARAM_INVALID)
        self.label_displaying_param_nestimators.setStyleSheet(LABEL_STYLE_PARAM_INVALID)
        self.label_displaying_param_batchsize.setStyleSheet(LABEL_STYLE_PARAM_INVALID)
        self.label_displaying_param_nhidden.setStyleSheet(LABEL_STYLE_PARAM_INVALID)
        self.label_displaying_param_nunit.setStyleSheet(LABEL_STYLE_PARAM_INVALID)
        self.label_displaying_param_keepdrop.setStyleSheet(LABEL_STYLE_PARAM_INVALID)

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

        """ラインエディットウィジェット定義"""
        self.ledit_param_alpha = QLineEdit(self)
        self.ledit_param_l1ratio = QLineEdit(self)
        self.ledit_param_maxfeatures = QLineEdit(self)
        self.ledit_param_maxdepth = QLineEdit(self)
        self.ledit_param_nestimators = QLineEdit(self)
        self.ledit_param_batchsize = QLineEdit(self)
        self.ledit_param_nhidden = QLineEdit(self)
        self.ledit_param_nunit = QLineEdit(self)
        self.ledit_param_keepdroop = QLineEdit(self)

        self.ledit_param_alpha.setStyleSheet(INPUT_STYLE_PARAMS_INVALID)
        self.ledit_param_l1ratio.setStyleSheet(INPUT_STYLE_PARAMS_INVALID)
        self.ledit_param_maxfeatures.setStyleSheet(INPUT_STYLE_PARAMS_INVALID)
        self.ledit_param_maxdepth.setStyleSheet(INPUT_STYLE_PARAMS_INVALID)
        self.ledit_param_nestimators.setStyleSheet(INPUT_STYLE_PARAMS_INVALID)
        self.ledit_param_batchsize.setStyleSheet(INPUT_STYLE_PARAMS_INVALID)
        self.ledit_param_nhidden.setStyleSheet(INPUT_STYLE_PARAMS_INVALID)
        self.ledit_param_nunit.setStyleSheet(INPUT_STYLE_PARAMS_INVALID)
        self.ledit_param_keepdroop.setStyleSheet(INPUT_STYLE_PARAMS_INVALID)

        self.ledit_param_alpha.setEnabled(False)
        self.ledit_param_l1ratio.setEnabled(False)
        self.ledit_param_maxfeatures.setEnabled(False)
        self.ledit_param_maxdepth.setEnabled(False)
        self.ledit_param_nestimators.setEnabled(False)
        self.ledit_param_batchsize.setEnabled(False)
        self.ledit_param_nhidden.setEnabled(False)
        self.ledit_param_nunit.setEnabled(False)
        self.ledit_param_keepdroop.setEnabled(False)

        self.ledit_param_alpha.textChanged[str].connect(super()._on_input_params)
        self.ledit_param_l1ratio.textChanged[str].connect(super()._on_input_params)
        self.ledit_param_maxfeatures.textChanged[str].connect(super()._on_input_params)
        self.ledit_param_maxdepth.textChanged[str].connect(super()._on_input_params)
        self.ledit_param_nestimators.textChanged[str].connect(super()._on_input_params)
        self.ledit_param_batchsize.textChanged[str].connect(super()._on_input_params)
        self.ledit_param_nhidden.textChanged[str].connect(super()._on_input_params)
        self.ledit_param_nunit.textChanged[str].connect(super()._on_input_params)
        self.ledit_param_keepdroop.textChanged[str].connect(super()._on_input_params)

        self.ledit_param_alpha.setAccessibleName(PARAM_ALPHA)
        self.ledit_param_l1ratio.setAccessibleName(PARAM_L1RATIO)
        self.ledit_param_maxfeatures.setAccessibleName(PARAM_MAXFEATURES)
        self.ledit_param_maxdepth.setAccessibleName(PARAM_MAXDEPTH)
        self.ledit_param_nestimators.setAccessibleName(PARAM_NESTIMATORS)
        self.ledit_param_batchsize.setAccessibleName(PARAM_BATCHSIZE)
        self.ledit_param_nhidden.setAccessibleName(PARAM_NHIDDEN)
        self.ledit_param_nunit.setAccessibleName(PARAM_NUNIT)
        self.ledit_param_keepdroop.setAccessibleName(PARAM_KEEPDROP)

        """ボタンウィジェット定義"""
        button_running_machine_learning = QPushButton(BUTTON_RUNNING_MACHINE_LEARNING, self)
        button_running_machine_learning.clicked.connect(self._on_clicked_running_button)

        """共通部分作成"""
        vbox = super()._make_std_and_compress_part()

        """レイアウト設定"""
        grid = QGridLayout()
        grid.addWidget(self.label_displaying_param_alpha, 0, 0)
        grid.addWidget(self.ledit_param_alpha, 0, 1)
        grid.addWidget(self.label_displaying_param_l1ratio, 0, 2)
        grid.addWidget(self.ledit_param_l1ratio, 0, 3)
        grid.addWidget(self.label_displaying_param_maxfeatures, 0, 4)
        grid.addWidget(self.ledit_param_maxfeatures, 0, 5)
        grid.addWidget(self.label_displaying_param_maxdepth, 1, 0)
        grid.addWidget(self.ledit_param_maxdepth, 1, 1)
        grid.addWidget(self.label_displaying_param_nestimators, 1, 2)
        grid.addWidget(self.ledit_param_nestimators, 1, 3)
        grid.addWidget(self.label_displaying_param_batchsize, 1, 4)
        grid.addWidget(self.ledit_param_batchsize, 1, 5)
        grid.addWidget(self.label_displaying_param_nhidden, 2, 0)
        grid.addWidget(self.ledit_param_nhidden, 2, 1)
        grid.addWidget(self.label_displaying_param_nunit, 2, 2)
        grid.addWidget(self.ledit_param_nunit, 2, 3)
        grid.addWidget(self.label_displaying_param_keepdrop, 2, 4)
        grid.addWidget(self.ledit_param_keepdroop, 2, 5)

        vbox.addWidget(self.label_displaying_predictor_method)
        vbox.addWidget(self.combo_selecting_analysis_method)
        vbox.addLayout(grid)

        vbox = super()._make_bag_and_ada_part(vbox)

        vbox = super()._make_running_machine_learning_part(vbox, button_running_machine_learning)

        self.setLayout(vbox)

        """パラメータ系ウィジェットを扱いやすいようタプル化/辞書登録"""
        self.param_label_wigets = (self.label_displaying_param_alpha,
                                   self.label_displaying_param_l1ratio,
                                   self.label_displaying_param_maxfeatures,
                                   self.label_displaying_param_maxdepth,
                                   self.label_displaying_param_nestimators,
                                   self.label_displaying_param_batchsize,
                                   self.label_displaying_param_nhidden,
                                   self.label_displaying_param_nunit,
                                   self.label_displaying_param_keepdrop)

        self.param_input_wigets = (self.ledit_param_alpha,
                                   self.ledit_param_l1ratio,
                                   self.ledit_param_maxfeatures,
                                   self.ledit_param_maxdepth,
                                   self.ledit_param_nestimators,
                                   self.ledit_param_batchsize,
                                   self.ledit_param_nhidden,
                                   self.ledit_param_nunit,
                                   self.ledit_param_keepdroop)

        self.param_dictionary = {PARAM_ALPHA: 0,
                                 PARAM_L1RATIO: 1,
                                 PARAM_MAXFEATURES: 2,
                                 PARAM_MAXDEPTH: 3,
                                 PARAM_NESTIMATORS: 4,
                                 PARAM_BATCHSIZE: 5,
                                 PARAM_NHIDDEN: 6,
                                 PARAM_NUNIT: 7,
                                 PARAM_KEEPDROP: 8}

    def _on_select_analysis_method(self, method):
        """分析手法選択時"""

        """パラメータ入力解禁/禁止IDを作成しメソッド呼び出し"""
        valid_ids = [False for i in range(len(self.param_dictionary))]

        if COMBO_ITEM_LINEARREGRESSION == method:
            pass

        elif COMBO_ITEM_ELASTICNET == method:
            valid_ids[self.param_dictionary[PARAM_ALPHA]] = True
            valid_ids[self.param_dictionary[PARAM_L1RATIO]] = True

        elif COMBO_ITEM_RANDOMFOREST == method:
            valid_ids[self.param_dictionary[PARAM_MAXFEATURES]] = True
            valid_ids[self.param_dictionary[PARAM_MAXDEPTH]] = True

        elif COMBO_ITEM_EXTRATREE == method:
            valid_ids[self.param_dictionary[PARAM_MAXFEATURES]] = True
            valid_ids[self.param_dictionary[PARAM_MAXDEPTH]] = True
            valid_ids[self.param_dictionary[PARAM_NESTIMATORS]] = True

        elif COMBO_ITEM_DEEPLEARNING == method:
            valid_ids[self.param_dictionary[PARAM_BATCHSIZE]] = True
            valid_ids[self.param_dictionary[PARAM_NHIDDEN]] = True
            valid_ids[self.param_dictionary[PARAM_NUNIT]] = True
            valid_ids[self.param_dictionary[PARAM_KEEPDROP]] = True

        super()._valid_param_wiget_by_method(self.param_label_wigets,
                                             self.param_input_wigets,
                                             valid_ids)

    def _on_clicked_running_button(self):
        """学習ボタン押下時"""

        print("predictor run")