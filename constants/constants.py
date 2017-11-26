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

"""数値"""
SPACE_BETWEEN_DATA_AND_METHOD = 15
INDEX_NOT_SELECTING = 0

"""パス"""
# なぜか相対パスが使えない
APPLICATION_ICON_PATH = 'D:/PycharmProjects/original_application/UI/images/icon_ai.png'
CSV_DIRECTORY_PATH = './../csv_files'

"""スタイル"""
WINDOW_APPLICATION = "QWidget {" \
                     "background-color:#ffffff;" \
                     "}"

BUTTON_STYLE_SELECT_DATA = 'QPushButton:!pressed {'\
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

LABEL_STYLE_SELECT_DATA = 'QLabel {'\
                          'background-color: white; color: #4a8d00;' \
                          'font-family: Arial; font-size:14px;' \
                          '}'

COMBO_STYLE_SELECT_METHOD = 'QComboBox {'\
                            'color: #aaff00;' \
                            'font-family: Arial; font-weight: bold; font-size:14px;' \
                            'border-width: 1px; border-color: #aaff00; border-radius: 2px; border-style: solid;' \
                            'width: 45px;' \
                            'padding-left:3px; padding-right:-14px;' \
                            '}' \
                            'QComboBox:drop-down {'\
                            'border-width: 0px;' \
                            '}' \
                            'QFrame {' \
                            'color: #aaff00;' \
                            'border: 1px solid #aaff00; border-radius: 2px;' \
                            '}'

LABEL_STYLE_NOT_SELECT_DATA = 'QLabel {'\
                              'background-color: white; color: #ff7700;' \
                              'font-family: Arial; font-size:14px;' \
                              '}'

LABEL_STYLE_BASIC_MSG = 'QLabel {'\
                        'background-color: white; color: black;' \
                        'font-family: Arial; font-size:14px;' \
                        'border-bottom: 1px; border-color: rgb(74, 141, 0, 100); border-style: solid;' \
                        '}'