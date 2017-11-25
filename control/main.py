from PyQt5.QtWidgets import QApplication
import sys
from UI.mainUI import Main


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui_main = Main()
    ui_main.show()
    sys.exit(app.exec_())