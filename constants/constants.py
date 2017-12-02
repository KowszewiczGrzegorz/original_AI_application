"""定数定義ファイル"""

"""テキスト"""
WINDOW_TITLE_MAIN = 'AI Application'
BUTTON_SELECTING_TRAINCSV = 'train'
BUTTON_SELECTING_TESTCSV = 'test'
LABEL_DISPLAYING_SELECTFILE = 'トレーニングデータまたは両方を指定'
NOT_SELECTING = '未選択'
LABEL_DISPLAYING_SELECTMETHOD = '分類または予測を選択'
LABEL_DISPLAYING_NOTSELECT = 'トレーニングデータ未選択'
COMBO_ITEM_NOTSELECT = '未選択'
COMBO_ITEM_CLASSIFIER = '分類'
COMBO_ITEM_PREDICTOR = '予測'
FILE_SELECT_DIALOG = 'ファイル選択'
EXTENSION_NAME_CSV = '.csv'
ERROR_MSG_NOTCSV = 'csvファイルを選択してください'
SUCCESSFULLY_SELECTED = ''
WINDOW_TITLE_CLASSIFIER = 'classifier'
WINDOW_TITLE_PREDICTOR = 'predictor'
LABEL_DISPLAYING_USE_STD = '説明変数の標準化'
PARAM_COMPRESS_METHOD = 'データ圧縮方法選択'
COMBO_ITEM_METHOD_NOTSELECT = '不使用'
COMBO_ITEM_SELECT_FEATURES = '特徴量選択'
COMBO_ITEM_PCA = 'PCA'
COMBO_ITEM_LDA = 'LDA'
COMBO_ITEM_KERNEL_PCA = 'カーネルPCA'
LABEL_DISPLAYING_THRESHOLD = '閾値'
LABEL_DISPLAYING_CLASSIFIER = '分類手法選択'
LABEL_DISPLAYING_PREDICTOR = '予測手法選択'
COMBO_ITEM_LINEARREGRESSION = '線形回帰'
COMBO_ITEM_ELASTICNET = 'Elastic回帰'
COMBO_ITEM_RANDOMFOREST = 'ランダムフォレスト'
COMBO_ITEM_EXTRATREE = 'エクストラツリー'
COMBO_ITEM_DEEPLEARNING = 'ディープラーニング'
COMBO_ITEM_PERCEPTRON = 'パーセプトロン'
COMBO_ITEM_ROGISTICREGRESSION = 'ロジスティック回帰'
COMBO_ITEM_SVM = 'SVM'
COMBO_ITEM_KNEIGHBORS = 'k近傍法'
PARAM_PENALTY = 'penalty'
PARAM_KERNEL = 'kernel'
PARAM_ETA0 = 'eta0'
PARAM_C = 'C'
PARAM_GAMMA = 'gamma'
PARAM_TREE = 'n_trees'
PARAM_NEIGHBORS = 'n_neighbors'
PARAM_ALPHA = 'alpha'
PARAM_L1RATIO = 'r1_ratio'
PARAM_MAXDEPTH = 'n_max_depth'
PARAM_MAXFEATURES = 'n_max_features'
PARAM_NESTIMATORS = 'n_estimators'
COMBO_ITEM_L1 = 'L1'
COMBO_ITEM_L2 = 'L2'
COMBO_ITEM_RBF = 'RBF'
COMBO_ITEM_LINEAR = 'linear'

"""数値"""
SPACE_BETWEEN_PARTS = 15
INDEX_NOT_SELECTING = 0
LEDIT_WIDTH_THRESHOLD = 30
LEDIT_WIDTH_PARAMS = 30
WIDTH_PARAM_LABEL = 100
SPACE_BETWEEN_PARAMS = 15

"""パス"""
# なぜか相対パスが使えない
APPLICATION_ICON_PATH = 'D:/PycharmProjects/original_application/UI/images/icon_ai.png'
CSV_DIRECTORY_PATH = './../csv_files'

"""スタイル"""
WINDOW_APPLICATION = \
    "QWidget {" \
    "background-color:#ffffff;" \
    "}"

BUTTON_STYLE_SELECT_DATA = \
    'QPushButton:!pressed {'\
    'background-color: white; color: #aaff00;' \
    'font-family: Arial; font-weight: bold; font-size:14px;' \
    'border-width: 1px; border-color: #aaff00; border-radius: 2px; border-style: solid;' \
    'width: 50px;' \
    '}' \
    'QPushButton:pressed {'\
    'background-color: #aaff00; color: white;' \
    'font-family: Arial; font-weight: bold; font-size:14px;' \
    'border-width: 0px; border-radius: 2px;' \
    'width: 50px;' \
    '}'

LABEL_STYLE_SELECT_DATA = \
    'QLabel {'\
    'background-color: white; color: #4a8d00;' \
    'font-family: Arial; font-size:14px;' \
    '}'

COMBO_STYLE_SELECT_METHOD = \
    'QComboBox {'\
    'color: #aaff00;' \
    'font-weight: bold;' \
    'border-width: 1px; border-color: #aaff00; border-radius: 2px; border-style: solid;' \
    'width: 39px;' \
    'padding-left:3px; padding-right:-14px; padding-top:2px; padding-bottom: 2px;' \
    '}' \
    'QComboBox:drop-down {'\
    'border-width: 0px;' \
    '}' \
    'QFrame {' \
    'color: #aaff00;' \
    'border: 1px solid #aaff00; border-radius: 2px;' \
    '}'

LABEL_STYLE_NOT_SELECT_DATA = \
    'QLabel {'\
    'background-color: white; color: #ff0000;' \
    'font-family: Arial; font-size:14px;' \
    '}'

LABEL_STYLE_BASIC_MSG = \
    'QLabel {'\
    'color: #683400;' \
    'font-size:14px; font-family:HGP創英角ﾎﾟｯﾌﾟ体;' \
    'border-bottom: 1px; border-color: rgb(74, 141, 0, 100); border-style: solid;' \
    'padding-bottom: 1px' \
    '}'

CHK_SELECTING_STD = \
    'QCheckBox:indicator {' \
    'color: #683400;' \
    'font-size:14px; font-family:HGP創英角ﾎﾟｯﾌﾟ体;' \
    '}' \
    'QLabel {' \
    'color: #ff0000;' \
    '}'

COMBO_STYLE_SELECT_COMPRESS = \
    'QComboBox {'\
    'color: #aaff00;' \
    'font-weight: bold;' \
    'border-width: 1px; border-color: #aaff00; border-radius: 2px; border-style: solid;' \
    'width: 69px;' \
    'padding-left:3px; padding-right:-14px; padding-top:2px; padding-bottom: 2px;' \
    '}' \
    'QComboBox:drop-down {'\
    'border-width: 0px;' \
    '}' \
    'QFrame {' \
    'color: #aaff00;' \
    'border: 1px solid #aaff00; border-radius: 2px;' \
    '}'

COMBO_STYLE_SELECT_CLASSIFIER = \
    'QComboBox {' \
    'color: #aaff00;' \
    'font-weight: bold;' \
    'border-width: 1px; border-color: #aaff00; border-radius: 2px; border-style: solid;' \
    'width: 92px;' \
    'padding-left:3px; padding-right:-14px; padding-top:2px; padding-bottom: 2px;' \
    '}' \
    'QComboBox:drop-down {'\
    'border-width: 0px;' \
    '}' \
    'QFrame {' \
    'color: #aaff00;' \
    'border: 1px solid #aaff00; border-radius: 2px;' \
    '}'

COMBO_STYLE_SELECT_PREDICTOR = \
    'QComboBox {'\
    'color: #aaff00;' \
    'font-weight: bold;' \
    'border-width: 1px; border-color: #aaff00; border-radius: 2px; border-style: solid;' \
    'width: 92px;' \
    'padding-left:3px; padding-right:-14px; padding-top:2px; padding-bottom: 2px;' \
    '}' \
    'QComboBox:drop-down {'\
    'border-width: 0px;' \
    '}' \
    'QFrame {' \
    'color: #aaff00;' \
    'border: 1px solid #aaff00; border-radius: 2px;' \
    '}'

LABEL_STYLE_THRESHOLD = \
    'QLabel {'\
    'color: #683400;' \
    '}'

LEDIT_STYLE_THRESHOLD = \
    'QLineEdit {'\
    'color: #eeeeee;' \
    '}'

LABEL_STYLE_PARAM_VALID = \
    'QLabel {'\
    'background-color: white; color: #4a8d00;' \
    'font-family: Arial;' \
    '}'

LABEL_STYLE_PARAM_INVALID = \
    'QLabel {'\
    'background-color: white; color: gray;' \
    'width: 90px;' \
    'font-family: Arial;' \
    '}'

COMBO_STYLE_SELECT_PARAMS = \
    'QComboBox {'\
    'color: gray;' \
    'font-weight: bold;' \
    'border-width: 1px; border-color: gray; border-radius: 2px; border-style: solid;' \
    'width: 69px;' \
    'padding-left:3px; padding-right:-14px; padding-top:2px; padding-bottom: 2px;' \
    '}' \
    'QComboBox:drop-down {'\
    'border-width: 0px;' \
    '}' \
    'QFrame {' \
    'color: gray;' \
    'border: 1px solid gray; border-radius: 2px;' \
    '}'