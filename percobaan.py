from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPixmap, QImage
import sys, cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from scipy.stats import entropy


def extract_texture_features(gray_img):
    # Pastikan citra grayscale uint8
    if gray_img.dtype != np.uint8:
        gray_img = (gray_img * 255).astype(np.uint8)

    # Histogram intensitas pixel
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()

    mean_val = np.mean(gray_img)
    variance = np.var(gray_img)
    std_dev = np.std(gray_img)
    ent = entropy(hist, base=2)
    energy = np.sum(hist ** 2)
    contrast = std_dev  # pakai std dev sebagai kontras sederhana

    return mean_val, variance, ent, energy, contrast


class DagingClassifier(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("CitWorks.ui", self)
        self.btnLoadImage.clicked.connect(self.load_image)
        self.btnDetect.clicked.connect(self.detect_daging)

        self.model = load_model("model_daging.h5")
        self.labels = ['busuk', 'segar', 'setengahSegar']
        self.loaded_image = None

    def load_image(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Pilih Gambar", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            img = cv2.imread(file_path)
            self.loaded_image = img
            self.show_image(img)

    def show_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        q_img = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(self.lblOriginalImage.width(), self.lblOriginalImage.height())
        self.lblOriginalImage.setPixmap(pixmap)

    def detect_daging(self):
        if self.loaded_image is None:
            return

        img = self.loaded_image

        # Ekstraksi fitur HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        avg_h = np.mean(h)
        avg_s = np.mean(s)
        avg_v = np.mean(v)

        # Ekstraksi fitur tekstur dengan metode alternatif
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kontras = np.std(gray)
        mean_val, variance, ent, energy, contrast = extract_texture_features(gray)

        # Tampilkan ke GUI (kamu bisa tambahkan fitur lain jika mau)
        self.result_H.setText(f"{avg_h:.2f}")
        self.result_S.setText(f"{avg_s:.2f}")
        self.result_V.setText(f"{avg_v:.2f}")
        self.result_kontras.setText(f"{kontras:.2f}")
        self.result_tekstur.setText(f"{contrast:.2f}")  # pakai std dev sebagai tekstur

        # Klasifikasi CNN
        img_resized = cv2.resize(img, (150, 150))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # Convert ke RGB
        img_array = img_to_array(img_rgb) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = self.model.predict(img_array)
        label = self.labels[np.argmax(prediction)]

        self.result_kondisi.setText(f"{label}")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = DagingClassifier()
    window.show()
    sys.exit(app.exec_())