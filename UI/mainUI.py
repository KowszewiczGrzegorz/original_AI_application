import re
from constants.constants import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from control.mainControler import MainControler, Classifier, Predictor
from collections import OrderedDict


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

        self.main_controler = MainControler()

        self.df_train = None
        self.df_test = None
        self.do_std = True
        self.compress_method = COMBO_ITEM_METHOD_NOTSELECT
        self.threshold = DEFAULT_THRESHOLD
        self.analysis_method = COMBO_ITEM_PERCEPTRON
        self.bagada_method = COMBO_ITEM_NOTSELECT
        self.export_file_name = ''

        self.param_penalty = COMBO_ITEM_L1
        self.param_kernel = COMBO_ITEM_RBF
        self.param_eta0 = str(DEFAULT_ETA0)
        self.param_C = str(DEFAULT_C)
        self.param_gamma = str(DEFAULT_GAMMA)
        self.param_neighbors = str(DEFAULT_NEIGHBORS)
        self.param_nestimators = str(DEFAULT_NESTIMATORS)
        self.param_batchsize = str(DEFAULT_BATCHSIZE)
        self.param_nhidden = str(DEFAULT_NHIDDEN)
        self.param_nunit = str(DEFAULT_NUNIT)
        self.param_keepdrop = str(DEFAULT_KEEPDROP)
        self.classifier_param_dict = OrderedDict()
        self.classifier_param_dict[PARAM_PENALTY] = self.param_penalty
        self.classifier_param_dict[PARAM_KERNEL] = self.param_kernel
        self.classifier_param_dict[PARAM_ETA0] = self.param_eta0
        self.classifier_param_dict[PARAM_C] = self.param_C
        self.classifier_param_dict[PARAM_GAMMA] = self.param_gamma
        self.classifier_param_dict[PARAM_NEIGHBORS] = self.param_neighbors
        self.classifier_param_dict[PARAM_CLS_NESTIMATORS] = self.param_nestimators
        self.classifier_param_dict[PARAM_BATCHSIZE] = self.param_batchsize
        self.classifier_param_dict[PARAM_NHIDDEN] = self.param_nhidden
        self.classifier_param_dict[PARAM_NUNIT] = self.param_nunit
        self.classifier_param_dict[PARAM_KEEPDROP] = self.param_keepdrop
        self.classifier_param_dict_for_export = OrderedDict()

        self.param_alpha = str(DEFAULT_ALPHA)
        self.param_l1ratio = str(DEFAULT_L1RATIO)
        self.param_maxfeatures = str(DEFAULT_MAXFEATURES)
        self.param_maxdepth = str(DEFAULT_MAXDEPTH)
        self.param_nestimators = str(DEFAULT_NESTIMATORS)
        self.predictor_param_dict = OrderedDict()
        self.predictor_param_dict[PARAM_ALPHA] = self.param_alpha
        self.predictor_param_dict[PARAM_L1RATIO] = self.param_l1ratio
        self.predictor_param_dict[PARAM_MAXFEATURES] = self.param_maxfeatures
        self.predictor_param_dict[PARAM_MAXDEPTH] = self.param_maxdepth
        self.predictor_param_dict[PARAM_PRD_NESTIMATORS] = self.param_nestimators
        self.predictor_param_dict[PARAM_BATCHSIZE] = self.param_batchsize
        self.predictor_param_dict[PARAM_NHIDDEN] = self.param_nhidden
        self.predictor_param_dict[PARAM_NUNIT] = self.param_nunit
        self.predictor_param_dict[PARAM_KEEPDROP] = self.param_keepdrop
        self.predictor_param_dict_for_export = OrderedDict()

        self.param_bagada_nestimator = str(DEFAULT_BA_NESTIMATOR)
        self.param_bagada_maxsamples = str(DEFAULT_BA_MAXSAMPLES)
        self.param_bagada_maxfeatures = str(DEFAULT_BA_MAXFEATURES)
        self.param_bagada_learningrate = str(DEFAULT_BA_LEARNINGRATE)
        self.bagada_param_dict = OrderedDict()
        self.bagada_param_dict[PARAM_BA_NESTIMATOR] = self.param_bagada_nestimator
        self.bagada_param_dict[PARAM_BA_MAXSAMPLES] = self.param_bagada_maxsamples
        self.bagada_param_dict[PARAM_BA_MAX_FEATURES] = self.param_bagada_maxfeatures
        self.bagada_param_dict[PARAM_BA_LEARNINGRATE] = self.param_bagada_learningrate
        self.bagada_param_dict_for_export = OrderedDict()

    def _on_check_std_chkbutton(self, state):
        """データ標準化チェックボックス押下時"""

        """標準化フラグON/OFF"""
        if Qt.Checked == state:
            self.do_std = True
        else:
            self.do_std = False

    def _on_select_compress_combo(self, method):
        """データ圧縮方法プルダウン選択時"""

        """選択圧縮方法格納"""
        self.compress_method = method

        """特徴量選択が選ばれた場合は閾値入力ウィジェット有効化"""
        if COMBO_ITEM_SELECT_FEATURES == self.compress_method:
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

        self.threshold = float(value)

    def _on_input_params(self, value):
        """パラメータ入力時"""

        """送り主特定し然るべきパラメータ更新"""
        sender_name = self.sender().accessibleName()

        if sender_name == PARAM_PENALTY:
            self.param_penalty = value
            self.classifier_param_dict_for_export[PARAM_PENALTY] = value
        elif sender_name == PARAM_KERNEL:
            self.param_kernel = value
            self.classifier_param_dict_for_export[PARAM_KERNEL] = value
        elif sender_name == PARAM_ETA0:
            self.param_eta0 = value
            self.classifier_param_dict_for_export[PARAM_ETA0] = value
        elif sender_name == PARAM_C:
            self.param_C = value
            self.classifier_param_dict_for_export[PARAM_C] = value
        elif sender_name == PARAM_GAMMA:
            self.param_gamma = value
            self.classifier_param_dict_for_export[PARAM_GAMMA] = value
        elif sender_name == PARAM_NEIGHBORS:
            self.param_neighbors = value
            self.classifier_param_dict_for_export[PARAM_NEIGHBORS] = value
        elif sender_name == PARAM_CLS_NESTIMATORS:
            self.param_nestimators = value
            self.classifier_param_dict_for_export[PARAM_CLS_NESTIMATORS] = value

        elif sender_name == PARAM_ALPHA:
            self.param_alpha = value
            self.predictor_param_dict_for_export[PARAM_ALPHA] = value
        elif sender_name == PARAM_L1RATIO:
            self.param_l1ratio = value
            self.predictor_param_dict_for_export[PARAM_L1RATIO] = value
        elif sender_name == PARAM_MAXDEPTH:
            self.param_maxdepth = value
            self.predictor_param_dict_for_export[PARAM_MAXDEPTH] = value
        elif sender_name == PARAM_MAXFEATURES:
            self.param_maxfeatures = value
            self.predictor_param_dict_for_export[PARAM_MAXFEATURES] = value
        elif sender_name == PARAM_PRD_NESTIMATORS:
            self.param_nestimators = value
            self.predictor_param_dict_for_export[PARAM_PRD_NESTIMATORS] = value
        elif sender_name == PARAM_BATCHSIZE:
            self.param_batchsize = value
            self.predictor_param_dict_for_export[PARAM_BATCHSIZE] = value
        elif sender_name == PARAM_NHIDDEN:
            self.param_nhidden = value
            self.predictor_param_dict_for_export[PARAM_NHIDDEN] = value
        elif sender_name == PARAM_NUNIT:
            self.param_nunit = value
            self.predictor_param_dict_for_export[PARAM_NUNIT] = value
        elif sender_name == PARAM_KEEPDROP:
            self.param_keepdrop = value
            self.predictor_param_dict_for_export[PARAM_KEEPDROP] = value

        elif sender_name == PARAM_BA_NESTIMATOR:
            self.param_bagada_nestimator = value
            self.bagada_param_dict_for_export[PARAM_BA_NESTIMATOR] = value
        elif sender_name == PARAM_BA_MAXSAMPLES:
            self.param_bagada_maxsamples = value
            self.bagada_param_dict_for_export[PARAM_BA_MAXSAMPLES] = value
        elif sender_name == PARAM_BA_MAX_FEATURES:
            self.param_bagada_maxfeatures = value
            self.bagada_param_dict_for_export[PARAM_BA_MAX_FEATURES] = value
        elif sender_name == PARAM_BA_LEARNINGRATE:
            self.param_bagada_learningrate = value
            self.bagada_param_dict_for_export[PARAM_BA_LEARNINGRATE] = value

    def _on_select_bagada_combo(self, method):
        """バギング/アダブースト設定コンボ選択時"""

        self.bagada_method = method

        """パラメータ入力解禁/禁止IDを作成しメソッド呼び出し"""
        valid_ids = [False for i in range(len(self.bagada_param_dictionary))]

        if COMBO_ITEM_BAGGING == method:
            valid_ids[self.bagada_param_dictionary[PARAM_BA_NESTIMATOR]] = True
            valid_ids[self.bagada_param_dictionary[PARAM_BA_MAXSAMPLES]] = True
            valid_ids[self.bagada_param_dictionary[PARAM_BA_MAX_FEATURES]] = True

        elif COMBO_ITEM_ADABOOST == method:
            valid_ids[self.bagada_param_dictionary[PARAM_BA_NESTIMATOR]] = True
            valid_ids[self.bagada_param_dictionary[PARAM_BA_LEARNINGRATE]] = True

        self._valid_param_wiget_by_method(self.bagada_param_label_wigets,
                                          self.bagada_param_input_wigets,
                                          valid_ids)

        """パラメータ書き出し用辞書作成"""
        self._make_export_param_dict(valid_ids, self.bagada_param_dict.keys(),
                                     self.bagada_param_dict.values(), self.bagada_param_dict_for_export)

    def _on_clicked_param_save_button(self):
        """パラメータ保存ボタン押下時"""

        self.main_controler = MainControler()

        """送り主特定"""
        sender_name = self.sender().accessibleName()

        """書き出しパラメータ作成"""
        export_dict = {}
        export_dict[PARAM_CLASSFIER_OR_PREDICTOR] = sender_name
        export_dict[PARAM_STD] = self.do_std
        export_dict[PARAM_COMPRESS] = self.compress_method
        if COMBO_ITEM_SELECT_FEATURES == self.compress_method:
            export_dict[PARAM_THRESHOLD] = self.threshold
        export_dict[PARAM_ANALYSIS] = self.analysis_method
        export_dict = self._make_dict(export_dict, self.classifier_param_dict_for_export)
        export_dict = self._make_dict(export_dict, self.predictor_param_dict_for_export)
        export_dict[PARAM_BAGADA] = self.bagada_method
        export_dict = self._make_dict(export_dict, self.bagada_param_dict_for_export)

        self.main_controler.export_params(self.export_file_name, export_dict)

    def _on_input_file_name(self, name):
        """書き出しファイル名入力時"""

        self.export_file_name = name

    def _valid_param_wiget_by_method(self, label_wigets, input_wigets, ids):
        """分析手法によってパラメータ系ウィジェットの有効化/無効化"""

        """2つのウィジェットを平行的に取り出すと同時にインデックスも取り出し処理"""
        for (i, label_wiget), input_wiget in zip(enumerate(label_wigets), input_wigets):
            """idがTrueの場合有効化"""
            if ids[i]:
                label_wiget.setStyleSheet(LABEL_STYLE_PARAM_VALID)
                input_wiget.setEnabled(True)
                if isinstance(input_wiget, QComboBox):
                    input_wiget.setStyleSheet(COMBO_INPUT_STYLE_PARAMS_VALID)
                else:
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

        self.label_displaying_threshold.setEnabled(False)

        """チェックボックスウィジェット定義"""
        self.chk_selecting_std = QCheckBox(CHK_SELECTING_STD, self)
        self.chk_selecting_std.setStyleSheet(CHK_SELECTING_STD)
        self.chk_selecting_std.setChecked(True)
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
        self.ledit_input_threshold.setText(str(DEFAULT_THRESHOLD))
        self.ledit_input_threshold.setValidator(QDoubleValidator())

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
        combo_selecting_bagging_or_adaboost.currentIndexChanged[str].connect(self._on_select_bagada_combo)

        """ラインエディットウィジェット定義"""
        self.ledit_input_bagada_nestimator = self._make_param_ledit(INPUT_STYLE_PARAMS_INVALID, PARAM_BA_NESTIMATOR, DEFAULT_BA_NESTIMATOR)
        self.ledit_input_bagada_maxsamples = self._make_param_ledit(INPUT_STYLE_PARAMS_INVALID, PARAM_BA_MAXSAMPLES, DEFAULT_BA_MAXSAMPLES)
        self.ledit_input_bagada_maxfeatures = self._make_param_ledit(INPUT_STYLE_PARAMS_INVALID, PARAM_BA_MAX_FEATURES, DEFAULT_BA_MAXFEATURES)
        self.ledit_input_bagada_learningrate = self._make_param_ledit(INPUT_STYLE_PARAMS_INVALID, PARAM_BA_LEARNINGRATE, DEFAULT_BA_LEARNINGRATE)

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

        self.bagada_param_dictionary = {PARAM_BA_NESTIMATOR: 0,
                                        PARAM_BA_MAXSAMPLES: 1,
                                        PARAM_BA_MAX_FEATURES: 2,
                                        PARAM_BA_LEARNINGRATE: 3}

        """初期化処理のためシグナル発行（インデックスを0に設定するため1度1にしている）"""
        combo_selecting_bagging_or_adaboost.setCurrentIndex(1)
        combo_selecting_bagging_or_adaboost.setCurrentIndex(0)

        return vbox

    def _make_save_params_part(self, vbox, button, name):
        """パラメータ保存部作成"""

        """ラベルウィジェット定義"""
        label_displaying_save = QLabel(LABEL_DISPLAYING_SAVE)
        label_displaying_filename = QLabel(LABEL_DISPLAYING_FILENAME)
        label_displaying_save.setStyleSheet(LABEL_STYLE_BASIC_MSG)
        label_displaying_filename.setStyleSheet(LABEL_STYLE_SAVE_FILE)

        """ボタンウィジェット定義"""
        button.setStyleSheet(BUTTON_STYLE_SAVE_PARAMS)
        button.setAccessibleName(name)

        """ラインエディットウィジェット定義"""
        ledit = QLineEdit(self)
        ledit.setStyleSheet(INPUT_STYLE_PARAMS_VALID)
        ledit.textChanged[str].connect(self._on_input_file_name)
        ledit.setAccessibleName(PARAM_FILENAME)


        """レイアウト設定"""
        hbox = QHBoxLayout()
        hbox.addWidget(label_displaying_filename)
        hbox.addWidget(ledit)
        hbox.addWidget(button)

        vbox.addSpacing(SPACE_BETWEEN_PARTS)
        vbox.addWidget(label_displaying_save)
        vbox.addLayout(hbox)

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

    def _make_param_ledit(self, style, name, default):
        """パラメータラインエディット作成"""

        ledit = QLineEdit(self)
        ledit.setStyleSheet(style)
        ledit.setEnabled(True)
        ledit.textChanged[str].connect(self._on_input_params)
        ledit.setAccessibleName(name)
        ledit.setText(str(default))

        return ledit

    def _make_param_combo(self, item_list, style, name, is_editable=False):
        """パラメータコンボボックス作成"""

        combo = QComboBox(self)
        for item in item_list:
            combo.addItem(item)
        combo.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        combo.setStyleSheet(style)
        combo.currentTextChanged[str].connect(self._on_input_params)
        combo.setEnabled(True)
        combo.setAccessibleName(name)
        if is_editable:
            combo.setEditable(True)

        return combo

    def _make_label_wiget(self, text, base_wiget, style):
        """ラベルウィジェット作成"""

        label = QLabel(text, base_wiget)
        label.setStyleSheet(style)

        return label

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

    def set_params(self, method_object):
        """パラメータ設定"""

        """パラメータをリスト化し設定関数呼び出し"""
        param_dict = OrderedDict()
        param_dict[PARAM_ANALYSIS] = self.analysis_method
        param_dict = self._make_dict(param_dict, self.classifier_param_dict_for_export)
        param_dict = self._make_dict(param_dict, self.predictor_param_dict_for_export)
        param_dict[PARAM_BAGADA] = self.bagada_method
        param_dict = self._make_dict(param_dict, self.bagada_param_dict_for_export)

        method_object.set_params(param_dict)

    def standardize_datas(self, method_object):
        """データの標準化"""

        """標準化チェック時のみ標準化メソッド呼び出し"""
        if self.do_std:
            method_object.standardize_datas()

    def compress_datas(self, method_object):
        """データ圧縮"""

        """データ圧縮メソッド呼び出し"""
        method_object.compress_datas(self.compress_method, self.threshold)

    def _make_dict(self, making_dict, used_dict):
        """辞書を使って辞書作成"""

        for (key, value) in used_dict.items():
            making_dict[key] = value

        return making_dict

    def _make_export_param_dict(self, ids, keys, values, export_dict):
        """パラメータ書き出し用辞書作成"""

        for id, key, value in zip(ids, keys, values):
            if id:
                export_dict[key] = value
            else:
                if key in export_dict:
                    del export_dict[key]


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
        self.label_displaying_classifier_method = super()._make_label_wiget(LABEL_DISPLAYING_CLASSIFIER, self, LABEL_STYLE_BASIC_MSG)
        self.label_displaying_param_penalty = super()._make_label_wiget(PARAM_PENALTY, self, LABEL_STYLE_PARAM_VALID)
        self.label_displaying_param_kernel = super()._make_label_wiget(PARAM_KERNEL, self, LABEL_STYLE_PARAM_INVALID)
        self.label_displaying_param_eta0 = super()._make_label_wiget(PARAM_ETA0, self, LABEL_STYLE_PARAM_VALID)
        self.label_displaying_param_C = super()._make_label_wiget(PARAM_C, self, LABEL_STYLE_PARAM_INVALID)
        self.label_displaying_param_gamma = super()._make_label_wiget(PARAM_GAMMA, self, LABEL_STYLE_PARAM_INVALID)
        self.label_displaying_param_neighbors = super()._make_label_wiget(PARAM_NEIGHBORS, self, LABEL_STYLE_PARAM_INVALID)
        self.label_displaying_param_nestimators = super()._make_label_wiget(PARAM_CLS_NESTIMATORS, self, LABEL_STYLE_PARAM_INVALID)
        self.label_displaying_param_batchsize = super()._make_label_wiget(PARAM_BATCHSIZE, self, LABEL_STYLE_PARAM_INVALID)
        self.label_displaying_param_nhidden = super()._make_label_wiget(PARAM_NHIDDEN, self, LABEL_STYLE_PARAM_INVALID)
        self.label_displaying_param_nunit = super()._make_label_wiget(PARAM_NUNIT, self, LABEL_STYLE_PARAM_INVALID)
        self.label_displaying_param_keepdrop = super()._make_label_wiget(PARAM_KEEPDROP, self, LABEL_STYLE_PARAM_INVALID)

        """コンボボックスウィジェット定義"""
        self.combo_selecting_analysis_method = QComboBox(self)
        self.combo_selecting_analysis_method.addItem(COMBO_ITEM_PERCEPTRON)
        self.combo_selecting_analysis_method.addItem(COMBO_ITEM_ROGISTICREGRESSION)
        self.combo_selecting_analysis_method.addItem(COMBO_ITEM_SVM)
        self.combo_selecting_analysis_method.addItem(COMBO_ITEM_RANDOMFOREST_CLS)
        self.combo_selecting_analysis_method.addItem(COMBO_ITEM_KNEIGHBORS)
        self.combo_selecting_analysis_method.addItem(COMBO_ITEM_DEEPLEARNING_CLS)
        self.combo_selecting_analysis_method.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.combo_selecting_analysis_method.setStyleSheet(COMBO_STYLE_SELECT_CLASSIFIER)
        self.combo_selecting_analysis_method.currentIndexChanged[str].connect(self._on_select_analysis_method)

        self.combo_selecting_penalty = super()._make_param_combo([COMBO_ITEM_L1, COMBO_ITEM_L2],
                                                                 COMBO_INPUT_STYLE_PARAMS_VALID, PARAM_PENALTY, True)
        self.combo_selecting_kernel = super()._make_param_combo([COMBO_ITEM_RBF, COMBO_ITEM_LINEAR],
                                                                COMBO_INPUT_STYLE_PARAMS_VALID, PARAM_KERNEL, True)

        """ラインエディットウィジェット定義"""
        self.ledit_param_eta0 = super()._make_param_ledit(INPUT_STYLE_PARAMS_VALID, PARAM_ETA0, DEFAULT_ETA0)
        self.ledit_param_C = super()._make_param_ledit(INPUT_STYLE_PARAMS_INVALID, PARAM_C, DEFAULT_C)
        self.ledit_param_gamma = super()._make_param_ledit(INPUT_STYLE_PARAMS_INVALID, PARAM_GAMMA, DEFAULT_GAMMA)
        self.ledit_param_neighbors = super()._make_param_ledit(INPUT_STYLE_PARAMS_INVALID, PARAM_NEIGHBORS, DEFAULT_NEIGHBORS)
        self.ledit_param_nestimators = super()._make_param_ledit(INPUT_STYLE_PARAMS_INVALID, PARAM_CLS_NESTIMATORS, DEFAULT_NESTIMATORS)
        self.ledit_param_batchsize = super()._make_param_ledit(INPUT_STYLE_PARAMS_INVALID, PARAM_BATCHSIZE, DEFAULT_BATCHSIZE)
        self.ledit_param_nhidden = super()._make_param_ledit(INPUT_STYLE_PARAMS_INVALID, PARAM_NHIDDEN, DEFAULT_NHIDDEN)
        self.ledit_param_nunit = super()._make_param_ledit(INPUT_STYLE_PARAMS_INVALID, PARAM_NUNIT, DEFAULT_NUNIT)
        self.ledit_param_keepdroop = super()._make_param_ledit(INPUT_STYLE_PARAMS_INVALID, PARAM_KEEPDROP, DEFAULT_KEEPDROP)

        """ボタンウィジェット定義"""
        self.button_running_machine_learning = QPushButton(BUTTON_RUNNING_MACHINE_LEARNING, self)
        self.button_saving_params = QPushButton(BUTTON_SAVING_PARAMS, self)

        self.button_running_machine_learning.clicked.connect(self._on_clicked_running_button)
        self.button_saving_params.clicked.connect(super()._on_clicked_param_save_button)

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
        grid.addWidget(self.label_displaying_param_gamma, 2, 0)
        grid.addWidget(self.ledit_param_gamma, 2, 1)
        grid.addWidget(self.label_displaying_param_neighbors, 2, 2)
        grid.addWidget(self.ledit_param_neighbors, 2, 3)
        grid.addWidget(self.label_displaying_param_nestimators, 3, 0)
        grid.addWidget(self.ledit_param_nestimators, 3, 1)
        grid.addWidget(self.label_displaying_param_batchsize, 3, 2)
        grid.addWidget(self.ledit_param_batchsize, 3,3)
        grid.addWidget(self.label_displaying_param_nhidden, 4, 0)
        grid.addWidget(self.ledit_param_nhidden, 4, 1)
        grid.addWidget(self.label_displaying_param_nunit, 4, 2)
        grid.addWidget(self.ledit_param_nunit, 4, 3)
        grid.addWidget(self.label_displaying_param_keepdrop, 5, 0)
        grid.addWidget(self.ledit_param_keepdroop, 5, 1)

        vbox.addWidget(self.label_displaying_classifier_method)
        vbox.addWidget(self.combo_selecting_analysis_method)
        vbox.addLayout(grid)
        
        vbox = super()._make_bag_and_ada_part(vbox)
        vbox = super()._make_save_params_part(vbox, self.button_saving_params, SAVE_BUTTON_CLASSIFIER)
        vbox = super()._make_running_machine_learning_part(vbox, self.button_running_machine_learning)

        self.setLayout(vbox)

        """パラメータ系ウィジェットを扱いやすいようタプル化/辞書登録"""
        self.param_label_wigets = (self.label_displaying_param_penalty,
                                   self.label_displaying_param_kernel,
                                   self.label_displaying_param_eta0,
                                   self.label_displaying_param_C,
                                   self.label_displaying_param_gamma,
                                   self.label_displaying_param_neighbors,
                                   self.label_displaying_param_nestimators,
                                   self.label_displaying_param_batchsize,
                                   self.label_displaying_param_nhidden,
                                   self.label_displaying_param_nunit,
                                   self.label_displaying_param_keepdrop)

        self.param_input_wigets = (self.combo_selecting_penalty,
                                   self.combo_selecting_kernel,
                                   self.ledit_param_eta0,
                                   self.ledit_param_C,
                                   self.ledit_param_gamma,
                                   self.ledit_param_neighbors,
                                   self.ledit_param_nestimators,
                                   self.ledit_param_batchsize,
                                   self.ledit_param_nhidden,
                                   self.ledit_param_nunit,
                                   self.ledit_param_keepdroop,)

        self.param_dictionary = {PARAM_PENALTY: 0,
                                 PARAM_KERNEL: 1,
                                 PARAM_ETA0: 2,
                                 PARAM_C: 3,
                                 PARAM_GAMMA: 4,
                                 PARAM_NEIGHBORS: 5,
                                 PARAM_CLS_NESTIMATORS: 6,
                                 PARAM_BATCHSIZE: 7,
                                 PARAM_NHIDDEN: 8,
                                 PARAM_NUNIT: 9,
                                 PARAM_KEEPDROP: 10
                                 }

        """初期化処理のためシグナル発行（インデックスを0に設定するため1度1にしている）"""
        self.combo_selecting_analysis_method.setCurrentIndex(1)
        self.combo_selecting_analysis_method.setCurrentIndex(0)

    def _on_select_analysis_method(self, method):
        """分析手法選択時"""

        self.analysis_method = method


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

        elif COMBO_ITEM_RANDOMFOREST_CLS == method:
            valid_ids[self.param_dictionary[PARAM_CLS_NESTIMATORS]] = True

        elif COMBO_ITEM_KNEIGHBORS == method:
            valid_ids[self.param_dictionary[PARAM_CLS_NESTIMATORS]] = True

        elif COMBO_ITEM_DEEPLEARNING_CLS == method:
            valid_ids[self.param_dictionary[PARAM_BATCHSIZE]] = True
            valid_ids[self.param_dictionary[PARAM_NHIDDEN]] = True
            valid_ids[self.param_dictionary[PARAM_NUNIT]] = True
            valid_ids[self.param_dictionary[PARAM_KEEPDROP]] = True

        super()._valid_param_wiget_by_method(self.param_label_wigets,
                                             self.param_input_wigets,
                                             valid_ids)

        """パラメータ書き出し用辞書作成"""
        self._make_export_param_dict(valid_ids, self.classifier_param_dict.keys(),
                                     self.classifier_param_dict.values(), self.classifier_param_dict_for_export)

    def _on_clicked_running_button(self):
        """学習実行ボタン押下時"""

        """分類オブジェクト作成"""
        classifier = Classifier()

        """データ設定"""
        super().set_datas(classifier)

        """データ標準化"""
        super().standardize_datas(classifier)

        """データ圧縮"""
        super().compress_datas(classifier)

        """パラメータ設定"""
        super().set_params(classifier)

        """学習実行"""
        estimator = classifier.run_learning()

        """予測実行"""
        predicted = None
        if self.df_test is not None:
            predicted = classifier.run_predict(estimator)

        """結果取得"""
        train_score, test_score, difference, params = classifier.get_classifer_result(estimator, predicted)
        train_score = round(train_score, NUMBER_OF_DECIMAL_DIGIT)
        test_score = round(test_score, NUMBER_OF_DECIMAL_DIGIT)
        if difference is not None:
            difference = round(difference, NUMBER_OF_DECIMAL_DIGIT)

        """結果出力"""
        result_shower = ClsResultShowerUI(self, train_score, test_score, difference, params)
        result_shower.show()


class PredictorUI(machine_learning_UI):
    """予測UIクラス"""

    def __init__(self):
        super().__init__()
        self._initialize()

    def _initialize(self):
        """初期化"""

        """ウィンドウの基本設定"""
        self.setWindowTitle(WINDOW_TITLE_PREDICTOR)
        self.setModal(False)

        """ラベルウィジェット定義"""
        self.label_displaying_predictor_method = super()._make_label_wiget(LABEL_DISPLAYING_PREDICTOR, self, LABEL_STYLE_BASIC_MSG)
        self.label_displaying_param_alpha = super()._make_label_wiget(PARAM_ALPHA, self, LABEL_STYLE_PARAM_INVALID)
        self.label_displaying_param_l1ratio = super()._make_label_wiget(PARAM_L1RATIO, self, LABEL_STYLE_PARAM_INVALID)
        self.label_displaying_param_maxfeatures = super()._make_label_wiget(PARAM_MAXFEATURES, self, LABEL_STYLE_PARAM_INVALID)
        self.label_displaying_param_maxdepth = super()._make_label_wiget(PARAM_MAXDEPTH, self, LABEL_STYLE_PARAM_INVALID)
        self.label_displaying_param_nestimators = super()._make_label_wiget(PARAM_PRD_NESTIMATORS, self, LABEL_STYLE_PARAM_INVALID)
        self.label_displaying_param_batchsize = super()._make_label_wiget(PARAM_BATCHSIZE, self, LABEL_STYLE_PARAM_INVALID)
        self.label_displaying_param_nhidden = super()._make_label_wiget(PARAM_NHIDDEN, self, LABEL_STYLE_PARAM_INVALID)
        self.label_displaying_param_nunit = super()._make_label_wiget(PARAM_NUNIT, self, LABEL_STYLE_PARAM_INVALID)
        self.label_displaying_param_keepdrop = super()._make_label_wiget(PARAM_KEEPDROP, self, LABEL_STYLE_PARAM_INVALID)

        """コンボボックスウィジェット定義"""
        self.combo_selecting_analysis_method = QComboBox(self)
        self.combo_selecting_analysis_method.addItem(COMBO_ITEM_LINEARREGRESSION)
        self.combo_selecting_analysis_method.addItem(COMBO_ITEM_ELASTICNET)
        self.combo_selecting_analysis_method.addItem(COMBO_ITEM_RANDOMFOREST_PRD)
        self.combo_selecting_analysis_method.addItem(COMBO_ITEM_EXTRATREE)
        self.combo_selecting_analysis_method.addItem(COMBO_ITEM_DEEPLEARNING_PRD)
        self.combo_selecting_analysis_method.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.combo_selecting_analysis_method.setStyleSheet(COMBO_STYLE_SELECT_PREDICTOR)
        self.combo_selecting_analysis_method.currentIndexChanged[str].connect(self._on_select_analysis_method)

        """ラインエディットウィジェット定義"""
        self.ledit_param_alpha = super()._make_param_ledit(INPUT_STYLE_PARAMS_INVALID, PARAM_ALPHA, DEFAULT_ALPHA)
        self.ledit_param_l1ratio = super()._make_param_ledit(INPUT_STYLE_PARAMS_INVALID, PARAM_L1RATIO, DEFAULT_L1RATIO)
        self.ledit_param_maxfeatures = super()._make_param_ledit(INPUT_STYLE_PARAMS_INVALID, PARAM_MAXFEATURES, DEFAULT_MAXFEATURES)
        self.ledit_param_maxdepth = super()._make_param_ledit(INPUT_STYLE_PARAMS_INVALID, PARAM_MAXDEPTH, DEFAULT_MAXDEPTH)
        self.ledit_param_nestimators = super()._make_param_ledit(INPUT_STYLE_PARAMS_INVALID, PARAM_PRD_NESTIMATORS, DEFAULT_NESTIMATORS)
        self.ledit_param_batchsize = super()._make_param_ledit(INPUT_STYLE_PARAMS_INVALID, PARAM_BATCHSIZE, DEFAULT_BATCHSIZE)
        self.ledit_param_nhidden = super()._make_param_ledit(INPUT_STYLE_PARAMS_INVALID, PARAM_NHIDDEN, DEFAULT_NHIDDEN)
        self.ledit_param_nunit = super()._make_param_ledit(INPUT_STYLE_PARAMS_INVALID, PARAM_NUNIT, DEFAULT_NUNIT)
        self.ledit_param_keepdroop = super()._make_param_ledit(INPUT_STYLE_PARAMS_INVALID, PARAM_KEEPDROP, DEFAULT_KEEPDROP)

        """ボタンウィジェット定義"""
        self.button_running_machine_learning = QPushButton(BUTTON_RUNNING_MACHINE_LEARNING, self)
        self.button_saving_params = QPushButton(BUTTON_SAVING_PARAMS, self)

        self.button_running_machine_learning.clicked.connect(self._on_clicked_running_button)
        self.button_saving_params.clicked.connect(super()._on_clicked_param_save_button)

        """共通部分作成"""
        vbox = super()._make_std_and_compress_part()

        """レイアウト設定"""
        grid = QGridLayout()
        grid.addWidget(self.label_displaying_param_alpha, 0, 0)
        grid.addWidget(self.ledit_param_alpha, 0, 1)
        grid.addWidget(self.label_displaying_param_l1ratio, 0, 2)
        grid.addWidget(self.ledit_param_l1ratio, 0, 3)
        grid.addWidget(self.label_displaying_param_maxfeatures, 1, 0)
        grid.addWidget(self.ledit_param_maxfeatures, 1, 1)
        grid.addWidget(self.label_displaying_param_maxdepth, 1, 2)
        grid.addWidget(self.ledit_param_maxdepth, 1, 3)
        grid.addWidget(self.label_displaying_param_nestimators, 2, 0)
        grid.addWidget(self.ledit_param_nestimators, 2, 1)
        grid.addWidget(self.label_displaying_param_batchsize, 2, 2)
        grid.addWidget(self.ledit_param_batchsize, 2, 3)
        grid.addWidget(self.label_displaying_param_nhidden, 3, 0)
        grid.addWidget(self.ledit_param_nhidden, 3, 1)
        grid.addWidget(self.label_displaying_param_nunit, 3, 2)
        grid.addWidget(self.ledit_param_nunit, 3, 3)
        grid.addWidget(self.label_displaying_param_keepdrop, 4, 0)
        grid.addWidget(self.ledit_param_keepdroop, 4, 1)

        vbox.addWidget(self.label_displaying_predictor_method)
        vbox.addWidget(self.combo_selecting_analysis_method)
        vbox.addLayout(grid)

        vbox = super()._make_bag_and_ada_part(vbox)
        vbox = super()._make_save_params_part(vbox, self.button_saving_params, SAVE_BUTTON_PREDICTOR)
        vbox = super()._make_running_machine_learning_part(vbox, self.button_running_machine_learning)

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
                                 PARAM_PRD_NESTIMATORS: 4,
                                 PARAM_BATCHSIZE: 5,
                                 PARAM_NHIDDEN: 6,
                                 PARAM_NUNIT: 7,
                                 PARAM_KEEPDROP: 8}

        """初期化処理のためシグナル発行（インデックスを0に設定するため1度1にしている）"""
        self.combo_selecting_analysis_method.setCurrentIndex(1)
        self.combo_selecting_analysis_method.setCurrentIndex(0)

    def _on_select_analysis_method(self, method):
        """分析手法選択時"""

        self.analysis_method = method

        """パラメータ入力解禁/禁止IDを作成しメソッド呼び出し"""
        valid_ids = [False for i in range(len(self.param_dictionary))]

        if COMBO_ITEM_LINEARREGRESSION == method:
            pass

        elif COMBO_ITEM_ELASTICNET == method:
            valid_ids[self.param_dictionary[PARAM_ALPHA]] = True
            valid_ids[self.param_dictionary[PARAM_L1RATIO]] = True

        elif COMBO_ITEM_RANDOMFOREST_PRD == method:
            valid_ids[self.param_dictionary[PARAM_MAXFEATURES]] = True
            valid_ids[self.param_dictionary[PARAM_MAXDEPTH]] = True

        elif COMBO_ITEM_EXTRATREE == method:
            valid_ids[self.param_dictionary[PARAM_MAXFEATURES]] = True
            valid_ids[self.param_dictionary[PARAM_MAXDEPTH]] = True
            valid_ids[self.param_dictionary[PARAM_PRD_NESTIMATORS]] = True

        elif COMBO_ITEM_DEEPLEARNING_PRD == method:
            valid_ids[self.param_dictionary[PARAM_BATCHSIZE]] = True
            valid_ids[self.param_dictionary[PARAM_NHIDDEN]] = True
            valid_ids[self.param_dictionary[PARAM_NUNIT]] = True
            valid_ids[self.param_dictionary[PARAM_KEEPDROP]] = True

        super()._valid_param_wiget_by_method(self.param_label_wigets,
                                             self.param_input_wigets,
                                             valid_ids)

        """パラメータ書き出し用辞書作成"""
        self._make_export_param_dict(valid_ids, self.predictor_param_dict.keys(),
                                     self.predictor_param_dict.values(), self.predictor_param_dict_for_export)

    def _on_clicked_running_button(self):
        """学習ボタン押下時"""

        """分類オブジェクト作成"""
        predictor = Predictor()

        """データ設定"""
        super().set_datas(predictor)

        """データ標準化"""
        super().standardize_datas(predictor)

        """データ圧縮"""
        super().compress_datas(predictor)

        """パラメータ設定"""
        super().set_params(predictor)

        """学習実行"""
        estimator = predictor.run_learning()

        """予測実行"""
        predicted = None
        if self.df_test is not None:
            predicted = predictor.run_predict(estimator)

        """結果取得"""
        mean_squared_errors, r2_scores, difference, params = predictor.get_predictor_result(estimator, predicted)
        round(mean_squared_errors[TRAIN], NUMBER_OF_DECIMAL_DIGIT)
        round(mean_squared_errors[TEST], NUMBER_OF_DECIMAL_DIGIT)
        round(r2_scores[TRAIN], NUMBER_OF_DECIMAL_DIGIT)
        round(r2_scores[TEST], NUMBER_OF_DECIMAL_DIGIT)
        if difference is not None:
            difference = round(difference, NUMBER_OF_DECIMAL_DIGIT)

        """結果出力"""
        result_shower = PrdResultShowerUI(self, mean_squared_errors, r2_scores, difference, params)
        result_shower.show()


class ResultShowerUI(QDialog):
    """結果出力親クラス"""

    def __init__(self, parent, params, train_score=None, test_score=None,
                 mean_squared_errors=None, r2_scores=None, difference=None):
        super().__init__(parent)

        """ウィンドウの基本設定"""
        self.setGeometry(340, 340, 0, 0)
        self.setStyleSheet(WINDOW_APPLICATION)
        self.setWindowTitle(WINDOW_TITLE_RESULT_CLASSIFIER)

        self.train_score = train_score
        self.test_score = test_score
        self.mean_squared_errors = mean_squared_errors
        self.r2_scores = r2_scores
        self.difference = difference
        self.params = params

    def _make_result_label_part(self, vbox):
        """結果出力ラベル部作成"""

        """ラベルウィジェット定義"""
        self.label_displaying_result = QLabel(LABEL_DISPLAYING_RESULTLABEL, self)
        self.label_displaying_result.setStyleSheet(LABEL_STYLE_BASIC_MSG)

        """レイアウト設定"""
        vbox.addWidget(self.label_displaying_result)

        return vbox

    def _make_use_param_part(self, vbox):
        """使用パラメータ表示部作成"""

        """ラベルウィジェット定義"""
        param_labels = []
        for (key, value) in self.params.items():
            """パラメータ名"""
            text = key + ': '
            label_widget = QLabel(text, self)
            label_widget.setStyleSheet(LABEL_STYLE_SCORELABEL)
            param_labels.append(label_widget)

            """パラメータ値"""
            text = re.sub('[\]\[\']','',str(value))
            label_widget = QLabel(text, self)
            label_widget.setStyleSheet(LABEL_STYLE_SCORE)
            param_labels.append(label_widget)

        """レイアウト設定"""
        grid = QGridLayout()
        row_number = 0
        column_number = 0
        is_method_value = False
        for label_widget in param_labels:
            """分析手法またはバグアダ手法の場合は1列で表示"""
            param_analysis_text = PARAM_ANALYSIS + ": "
            param_bagada_text = PARAM_BAGADA + ": "
            if param_analysis_text == label_widget.text() or param_bagada_text == label_widget.text():
                column_number = 0
                row_number += 1
                grid.addWidget(label_widget, row_number, column_number)
                column_number += 1
                is_method_value = True
                continue
            if is_method_value:
                hbox = QHBoxLayout()
                hbox.addStretch()
                hbox.addWidget(label_widget)
                grid.addLayout(hbox, row_number, column_number)
                row_number += 1
                column_number = 0
                is_method_value = False
                continue

            """パラメータの場合は2列で表示"""
            if column_number % 2 == 0:
                grid.addWidget(label_widget, row_number, column_number)
            else:
                hbox = QHBoxLayout()
                hbox.addStretch()
                hbox.addWidget(label_widget)
                grid.addLayout(hbox, row_number, column_number)

            column_number += 1
            if column_number >= N_COLUMN_IN_PARAM_RESULT:
                row_number += 1
                column_number = 0

        vbox.addLayout(grid)
        vbox.addSpacing(SPACE_BETWEEN_PARAMS)

        return vbox


class ClsResultShowerUI(ResultShowerUI):
    """分類結果出力クラス"""

    def __init__(self, parenct, train_score, test_score, difference, params):
        super().__init__(parenct, train_score=train_score,
                         test_score=test_score, difference=difference, params=params)
        self._initialize()

    def _initialize(self):
        """初期化"""

        vbox = QVBoxLayout()

        """結果出力ラベル部作成"""
        vbox = super()._make_result_label_part(vbox)

        """使用パラメータ表示部作成"""
        vbox = super()._make_use_param_part(vbox)

        """結果スコア表示部作成"""
        """train_scoreとtest_score部"""
        self.label_displaying_trainscore = QLabel(str(self.train_score), self)
        self.label_displaying_testscore = QLabel(str(self.test_score), self)
        self.label_displaying_trainscorelabel = QLabel(LABEL_DISPLAYING_TRAINSCORE, self)
        self.label_displaying_testscorelabel = QLabel(LABEL_DISPLAYING_TESTSCORE, self)
        self.label_displaying_trainscore.setStyleSheet(LABEL_STYLE_SCORE)
        self.label_displaying_testscore.setStyleSheet(LABEL_STYLE_SCORE)
        self.label_displaying_trainscorelabel.setStyleSheet(LABEL_STYLE_SCORELABEL)
        self.label_displaying_testscorelabel.setStyleSheet(LABEL_STYLE_SCORELABEL)

        hbox1 = QHBoxLayout()
        hbox2 = QHBoxLayout()
        hbox1.addStretch()
        hbox1.addWidget(self.label_displaying_trainscore)
        hbox2.addStretch()
        hbox2.addWidget(self.label_displaying_testscore)
        grid = QGridLayout()
        grid.addWidget(self.label_displaying_trainscorelabel, 0, 0)
        grid.addLayout(hbox1, 0, 1)
        grid.addWidget(self.label_displaying_testscorelabel, 0, 2)
        grid.addLayout(hbox2, 0, 3)

        """difference部"""
        if self.difference is not None:
            self.label_dsplaying_difference = QLabel(str(self.difference), self)
            self.label_dsplaying_differencelabel = QLabel(LABEL_DISPLAYING_DIFFERENCE, self)
            self.label_dsplaying_difference.setStyleSheet(LABEL_STYLE_SCORE)
            self.label_dsplaying_differencelabel.setStyleSheet(LABEL_STYLE_SCORELABEL)

            grid.addWidget(self.label_dsplaying_differencelabel, 1, 0)
            grid.addWidget(self.label_dsplaying_difference, 1, 1)

        vbox.addLayout(grid)
        self.setLayout(vbox)


class PrdResultShowerUI(ResultShowerUI):
    """予測結果出力クラス"""

    def __init__(self, parent, mean_squared_errors, r2_scores, difference, params):
        super().__init__(parent, mean_squared_errors=mean_squared_errors,
                         r2_scores=r2_scores, difference=difference, params=params)
        self._initialize()

    def _initialize(self):
        """初期化"""

        vbox = QVBoxLayout()

        """結果出力ラベル部作成"""
        vbox = super()._make_result_label_part(vbox)

        """使用パラメータ表示部作成"""
        vbox = super()._make_use_param_part(vbox)

        """結果スコア表示部作成"""
        """mean_squared_errorsとr2_scores部"""
        self.label_displaying_mse_train = QLabel(str(self.mean_squared_errors[TRAIN]), self)
        self.label_displaying_mse_test = QLabel(str(self.mean_squared_errors[TEST]), self)
        self.label_displaying_mselabel_train = QLabel(LABEL_DISPLAYING_TRAINMSE, self)
        self.label_displaying_mselabel_test = QLabel(LABEL_DISPLAYING_TESTMSE, self)
        self.label_displaying_r2_train = QLabel(str(self.r2_scores[TRAIN]), self)
        self.label_displaying_r2_test = QLabel(str(self.r2_scores[TEST]), self)
        self.label_displaying_r2label_train = QLabel(LABEL_DISPLAYING_TRAINR2, self)
        self.label_displaying_r2label_test = QLabel(LABEL_DISPLAYING_TESTR2, self)

        self.label_displaying_mse_train.setStyleSheet(LABEL_STYLE_SCORE)
        self.label_displaying_mse_test.setStyleSheet(LABEL_STYLE_SCORE)
        self.label_displaying_mselabel_train.setStyleSheet(LABEL_STYLE_SCORELABEL)
        self.label_displaying_mselabel_test.setStyleSheet(LABEL_STYLE_SCORELABEL)
        self.label_displaying_r2_train.setStyleSheet(LABEL_STYLE_SCORE)
        self.label_displaying_r2_test.setStyleSheet(LABEL_STYLE_SCORE)
        self.label_displaying_r2label_train.setStyleSheet(LABEL_STYLE_SCORELABEL)
        self.label_displaying_r2label_test.setStyleSheet(LABEL_STYLE_SCORELABEL)

        hbox1 = QHBoxLayout()
        hbox2 = QHBoxLayout()
        hbox3 = QHBoxLayout()
        hbox4 = QHBoxLayout()
        hbox1.addStretch()
        hbox1.addWidget(self.label_displaying_mse_train)
        hbox2.addStretch()
        hbox2.addWidget(self.label_displaying_mse_test)
        hbox3.addStretch()
        hbox3.addWidget(self.label_displaying_r2_train)
        hbox4.addStretch()
        hbox4.addWidget(self.label_displaying_r2_test)
        grid = QGridLayout()
        grid.addWidget(self.label_displaying_mselabel_train, 0, 0)
        grid.addLayout(hbox1, 0, 1)
        grid.addWidget(self.label_displaying_mselabel_test, 0, 2)
        grid.addLayout(hbox2, 0, 3)
        grid.addWidget(self.label_displaying_r2label_train, 1, 0)
        grid.addLayout(hbox3, 1, 1)
        grid.addWidget(self.label_displaying_r2label_test, 1, 2)
        grid.addLayout(hbox4, 1, 3)

        """difference部"""
        if self.difference is not None:
            self.label_dsplaying_difference = QLabel(str(self.difference), self)
            self.label_dsplaying_differencelabel = QLabel(LABEL_DISPLAYING_DIFFERENCE, self)
            self.label_dsplaying_difference.setStyleSheet(LABEL_STYLE_SCORE)
            self.label_dsplaying_differencelabel.setStyleSheet(LABEL_STYLE_SCORELABEL)

            grid.addWidget(self.label_dsplaying_differencelabel, 2, 0)
            grid.addWidget(self.label_dsplaying_difference, 2, 1)

        vbox.addLayout(grid)
        self.setLayout(vbox)


