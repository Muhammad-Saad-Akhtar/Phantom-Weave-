import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QMessageBox
from PyQt6.QtGui import QFont, QPixmap, QIcon

ICON_PATH = "C:\\Users\\HP\\Desktop\\Others\\Phantom-Weave-\\Icon1.webp"

class PhantomWeaveMenu(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Phantom Weave Menu")
        self.setGeometry(100, 100, 400, 300)
        self.setStyleSheet("background-color: black;")
        self.setWindowIcon(QIcon(ICON_PATH))  # Set the window icon

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Title label
        title_font = QFont("Arial", 16, QFont.Weight.Bold)
        title_label = QPushButton("PHANTOM WEAVE")
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: white; background-color: black; border: none;")
        title_label.setEnabled(False)
        layout.addWidget(title_label)

        # Buttons
        btn_invisibility = QPushButton("Invisibility Cloak")
        btn_invisibility.setFont(QFont("Arial", 12))
        btn_invisibility.setStyleSheet("background-color: gray; color: white;")
        btn_invisibility.clicked.connect(self.launch_invisibility)
        layout.addWidget(btn_invisibility)

        btn_background = QPushButton("Virtual Background")
        btn_background.setFont(QFont("Arial", 12))
        btn_background.setStyleSheet("background-color: gray; color: white;")
        btn_background.clicked.connect(self.launch_background)
        layout.addWidget(btn_background)

        btn_exit = QPushButton("Exit")
        btn_exit.setFont(QFont("Arial", 12))
        btn_exit.setStyleSheet("background-color: red; color: white;")
        btn_exit.clicked.connect(self.exit_app)
        layout.addWidget(btn_exit)

    def launch_invisibility(self):
        self.show_message("Launching", "Launching Invisibility Cloak...")
        os.system('python "c:\\Users\\HP\\Desktop\\Others\\Phantom-Weave-\\invisibility.py"')

    def launch_background(self):
        self.show_message("Launching", "Launching Virtual Background...")
        os.system('python "c:\\Users\\HP\\Desktop\\Others\\Phantom-Weave-\\background.py"')

    def exit_app(self):
        reply = QMessageBox.question(self, "Exit", "Are you sure you want to exit?", 
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.close()

    def show_message(self, title, text):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        msg_box.setIconPixmap(QPixmap(ICON_PATH))  # Use the specified icon
        msg_box.exec()

if __name__ == "__main__":
    app = QApplication([])
    window = PhantomWeaveMenu()
    window.show()
    app.exec()
