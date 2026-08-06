import sys
from PyQt6.QtWidgets import QApplication
from ui import App

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") # Better cross-platform look
    window = App()
    window.show()
    sys.exit(app.exec())
