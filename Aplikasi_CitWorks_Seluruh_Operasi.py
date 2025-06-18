import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.uic import loadUi
from PyQt5 import QtGui
from PyQt5 import QtCore

class AplikasiCitworks(QMainWindow):
    def __init__(self):
        super(AplikasiCitworks, self).__init__()
        loadUi("CitWorks.ui", self)

        # Inisialisasi
        self.image = None
        self.image2 = None
        self.processed_image = None
        self.history = []
        self.history_index = -1

        # Setup action triggers
        self.actionOpenImage.triggered.connect(self.open_image)
        self.actionSaveImage.triggered.connect(self.save_image)
        self.actionSave_as_txt.triggered.connect(self.save_pixel_data_txt)
        self.actionSave_as_xlsx.triggered.connect(self.save_pixel_data_xlsx)
        self.actionExit.triggered.connect(self.clear_images)

        # Setup menuEdit actions
        self.actionUndo.triggered.connect(self.undo_action)
        self.actionRedo.triggered.connect(self.redo_action)

        # Setup buttons
        self.btnLoadImage.clicked.connect(self.open_image)
        self.btnResetImage.clicked.connect(self.reset_processed_image)

        # Setup labels
        self.lblOriginalImage.setText("No Image Loaded")
        self.lblProcessedImage.setText("No Processed Image")

        # Menghubungkan menu ke fungsi

        # Action Operasi Titik
        self.actionContrast_Stretching.triggered.connect(self.contrast_stretching)
        self.actionNegative_Image.triggered.connect(self.negative_image)
        self.actionBiner_Image.triggered.connect(self.biner_image)
        self.actionGray_Histogram.triggered.connect(self.gray_histogram)
        self.actionRGB_Histogram.triggered.connect(self.rgb_histogram)
        self.actionEqual_Histogram.triggered.connect(self.equal_histogram)
        self.actionGrayscale.triggered.connect(self.convert_grayscale)
        self.Slider_Brightness.valueChanged.connect(self.adjust_brightness)
        self.Slider_SimpleContrast.valueChanged.connect(self.adjust_contrast)

        # Action Operasi Geometri
        self.actionRotasimin45.triggered.connect(lambda: self.apply_rotation(-45))
        self.actionRotasi45.triggered.connect(lambda: self.apply_rotation(45))
        self.actionRotasimin90.triggered.connect(lambda: self.apply_rotation(-90))
        self.actionRotasi90.triggered.connect(lambda: self.apply_rotation(90))
        self.actionRotasi_180.triggered.connect(lambda: self.apply_rotation(180))
        self.actionZoomIn.triggered.connect(self.apply_zoom_in)
        self.actionZoomOut.triggered.connect(self.apply_zoom_out)
        self.actionFlip_Horizontal.triggered.connect(lambda: self.apply_flipping(1))
        self.actionFlip_Vertical.triggered.connect(lambda: self.apply_flipping(0))
        self.actionTranslasi.triggered.connect(self.apply_translation)
        self.actionCrop_Image.triggered.connect(self.apply_crop)
        self.actionShearing.triggered.connect(self.apply_shearing)

        #Action Operasi Aritmatika
        self.actionAddition.triggered.connect(self.addition)
        self.actionSubtraction.triggered.connect(self.subtraction)
        self.actionMultiplication.triggered.connect(self.multiplication)
        self.actionDivision.triggered.connect(self.division)
        self.actionBoolean_AND.triggered.connect(lambda: self.boolean_operation("AND"))
        self.actionBoolean_OR.triggered.connect(lambda: self.boolean_operation("OR"))
        self.actionBoolean_XOR.triggered.connect(lambda: self.boolean_operation("XOR"))

        # Action Operasi Spasial
        self.actionLPF.triggered.connect(self.apply_lpf)
        self.actionHPF.triggered.connect(self.apply_hpf)
        self.actionMean_Filter.triggered.connect(self.apply_mean_filter)
        self.actionGaussian.triggered.connect(self.apply_gaussian)
        self.actionMedian_Filter.triggered.connect(self.apply_median_filter)
        self.actionMax_Filter.triggered.connect(self.apply_max_filter)
        self.actionMin_Filter.triggered.connect(self.apply_min_filter)
        self.actionEdge_Enhancement.triggered.connect(self.apply_edge_enhancement)
        self.actionSharpening_Standar.triggered.connect(self.apply_sharpening_standar)
        self.actionStrong_Sharpening.triggered.connect(self.apply_strong_sharpening)
        self.actionLaplace_3x3.triggered.connect(self.apply_laplace_3x3)
        self.actionLaplace_5x5_2.triggered.connect(self.apply_laplace_5x5)
        self.actionNegative_Edge_Detection.triggered.connect(self.apply_negative_edge_detection)

        #Action Operasi Edge Detection
        self.actionSobel.triggered.connect(lambda: self.apply_edge_detection('sobel'))
        self.actionPrewitt.triggered.connect(lambda: self.apply_edge_detection('prewitt'))
        self.actionRobert_Cross.triggered.connect(lambda: self.apply_edge_detection('roberts'))
        self.actionCanny.triggered.connect(lambda: self.apply_edge_detection('canny'))

        # Action Operasi Morfologi
        self.actionErosion.triggered.connect(self.apply_erosion)
        self.actionDilation.triggered.connect(self.apply_dilation)
        self.actionOpening.triggered.connect(self.apply_opening)
        self.actionClosing.triggered.connect(self.apply_closing)
        self.actionThinning.triggered.connect(self.apply_thinning)

        # Action Operasi Ranah Frequensi
        self.actionDFT.triggered.connect(self.apply_dft)
        self.actionIDFT.triggered.connect(self.apply_idft)
        self.actionLow_Pass_Filter.triggered.connect(self.apply_low_pass)
        self.actionHigh_Pass_Filter.triggered.connect(self.apply_high_pass)
        self.actionBand_Pass_Filter.triggered.connect(self.apply_band_pass)



    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")

        if file_path:
            if self.image is None:  # Kalau gambar pertama belum ada, load ke self.image
                self.image = cv2.imread(file_path)
                self.processed_image = self.image.copy()
                self.display_image(self.image, self.lblOriginalImage)
                self.save_history()
                QMessageBox.information(self, "Success", "First image loaded!")
            else:  # Kalau sudah ada gambar pertama, load gambar kedua
                self.image2 = cv2.imread(file_path)
                if self.image2 is None:
                    QMessageBox.warning(self, "Error", "Failed to load second image!")
                else:
                    QMessageBox.information(self, "Success", "Second image loaded!")

    def display_image(self, img, label):
        img_to_show = img.copy()

        # Konversi ke RGB jika gambar masih dalam BGR
        if len(img_to_show.shape) == 3:
            img_to_show = cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB)

        # Ambil ukuran label
        label_width = label.width()
        label_height = label.height()

        # Resize gambar agar pas dengan label tanpa mengubah aspek rasio
        img_to_show = cv2.resize(img_to_show, (label_width, label_height), interpolation=cv2.INTER_AREA)

        # Konversi ke QImage
        h, w, ch = img_to_show.shape
        bytes_per_line = ch * w
        q_img = QtGui.QImage(img_to_show.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

        # Set gambar ke label
        label.setPixmap(QtGui.QPixmap.fromImage(q_img))
        label.setScaledContents(True)  # Pastikan gambar menyesuaikan ukuran label
        label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)  # Memastikan gambar ada di tengah

    def save_image(self):
        if self.processed_image is None:
            QMessageBox.warning(self, "Warning", "No processed image to save!")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Images (*.png *.xpm *.jpg *.bmp)")
        if filename:
            cv2.imwrite(filename, self.processed_image)

    def save_pixel_data_txt(self):
        if self.image is None or self.processed_image is None:
            QMessageBox.warning(self, "Warning", "No image data to save!")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save Pixel Data", "", "Text Files (*.txt)")
        if filename:
            with open(filename, 'w') as file:
                file.write("Original Image Pixels:\n")
                file.write(str(self.image.tolist()))
                file.write("\n\nProcessed Image Pixels:\n")
                file.write(str(self.processed_image.tolist()))

    def save_pixel_data_xlsx(self):
        if self.image is None or self.processed_image is None:
            QMessageBox.warning(self, "Warning", "No image data to save!")
            return

        filename, _ = QFileDialog.getSaveFileName(self, "Save Pixel Data", "", "Excel Files (*.xlsx)")
        if filename:
            try:
                # Cek apakah gambar grayscale atau berwarna
                if len(self.image.shape) == 2:  # Grayscale
                    df_original = pd.DataFrame(self.image)  # Tidak perlu reshape
                else:  # Berwarna
                    df_original = pd.DataFrame(self.image.reshape(-1, self.image.shape[2]))

                if len(self.processed_image.shape) == 2:  # Grayscale
                    df_processed = pd.DataFrame(self.processed_image)  # Tidak perlu reshape
                else:  # Berwarna
                    df_processed = pd.DataFrame(self.processed_image.reshape(-1, self.processed_image.shape[2]))

                # Simpan ke Excel
                with pd.ExcelWriter(filename) as writer:
                    df_original.to_excel(writer, sheet_name="Original Image")
                    df_processed.to_excel(writer, sheet_name="Processed Image")

                QMessageBox.information(self, "Success", "Pixel data saved successfully!")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save file: {str(e)}")

    def clear_images(self):
        self.lblOriginalImage.clear()
        self.lblProcessedImage.clear()
        self.image = None
        self.processed_image = None

    def save_history(self):
        if self.processed_image is not None:
            self.history = self.history[:self.history_index + 1]
            self.history.append(self.processed_image.copy())
            self.history_index += 1

    def undo_action(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.processed_image = self.history[self.history_index].copy()
            self.display_image(self.processed_image, self.lblProcessedImage)

    def redo_action(self):
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.processed_image = self.history[self.history_index].copy()
            self.display_image(self.processed_image, self.lblProcessedImage)

    def reset_processed_image(self):
        self.processed_image = None
        self.lblProcessedImage.clear()
        self.lblProcessedImage.setText("No Processed Image")

    # Fungsi Operasi Titik
    def contrast_stretching(self):
        if self.image is None:
            return
        min_val = np.min(self.image)
        max_val = np.max(self.image)
        self.processed_image = ((self.image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        self.display_image(self.processed_image, self.lblProcessedImage)
        self.save_history()

    def negative_image(self):
        if self.image is None:
            return
        self.processed_image = 255 - self.image
        self.display_image(self.processed_image, self.lblProcessedImage)
        self.save_history()

    def biner_image(self):
        if self.image is None:
            return

        # Konversi ke grayscale jika gambar masih RGB/BGR
        if len(self.image.shape) == 3:
            img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = self.image.copy()

        # Thresholding (Binerisasi)
        _, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

        # Ubah gambar biner ke format RGB agar kompatibel dengan display_image()
        img_binary_rgb = cv2.cvtColor(img_binary, cv2.COLOR_GRAY2RGB)

        # Tampilkan hasil
        self.display_image(img_binary_rgb, self.lblProcessedImage)

        # Simpan hasil agar bisa dipakai lagi
        self.processed_image = img_binary.copy()
        self.save_history()

    def gray_histogram(self):
        if self.processed_image is None:  # Menyesuaikan dengan gambar hasil proses
            return
        gray = self.processed_image  # Ambil gambar yang telah diproses
        plt.figure(figsize=(6, 4))
        plt.hist(gray.ravel(), bins=256, range=[0, 256], color='black', alpha=0.75)
        plt.title("Histogram Grayscale")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    def rgb_histogram(self):
        if self.processed_image is None:  # Menyesuaikan dengan gambar hasil proses
            return
        color = ('b', 'g', 'r')
        plt.figure(figsize=(6, 4))
        for i, col in enumerate(color):
            hist = cv2.calcHist([self.processed_image], [i], None, [256], [0, 256])
            plt.plot(hist, color=col, linewidth=1.5)
        plt.title("Histogram RGB")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    def equal_histogram(self):
        if self.image is None:
            return

        # Konversi gambar ke grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Lakukan histogram equalization
        equalized = cv2.equalizeHist(gray)

        # Ubah ke format RGB agar bisa ditampilkan dengan display_image()
        equalized_rgb = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)

        # Tampilkan hasil
        self.display_image(equalized_rgb, self.lblProcessedImage)

        # Simpan hasil
        self.processed_image = equalized.copy()
        self.save_history()

        # **Menampilkan Histogram Equalization**
        plt.figure(figsize=(6, 4))
        plt.hist(equalized.ravel(), bins=256, range=[0, 256], color='red', alpha=0.75)
        plt.title("Histogram Equalized")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    def convert_grayscale(self):
        """Mengonversi citra ke grayscale dengan rumus manual."""
        if self.image is None:
            return
        H, W = self.image.shape[:2]
        gray = np.zeros((H, W), dtype=np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(
                    0.299 * self.image[i, j, 2] +
                    0.587 * self.image[i, j, 1] +
                    0.114 * self.image[i, j, 0],
                    0, 255).astype(np.uint8)
        self.processed_image = gray
        self.display_image(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), self.lblProcessedImage)
        self.save_history()

    def adjust_brightness(self, value):
        if self.image is None:
            return
        self.processed_image = cv2.convertScaleAbs(self.image, alpha=1, beta=value)
        self.display_image(self.processed_image, self.lblProcessedImage)

    def adjust_contrast(self, value):
        if self.image is None:
            return
        alpha = 1 + (value / 100.0)
        self.processed_image = cv2.convertScaleAbs(self.image, alpha=alpha, beta=0)
        self.display_image(self.processed_image, self.lblProcessedImage)


    # Fungsi Operasi Geometri
    def apply_translation(self):
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please open an image first!")
            return

        tx, ty = 50, 30
        rows, cols = self.image.shape[:2]
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        translated_image = cv2.warpAffine(self.image, translation_matrix, (cols, rows))

        self.processed_image = translated_image
        self.display_image(self.processed_image, self.lblProcessedImage)
        self.save_history()

    def apply_rotation(self, angle):
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please open an image first!")
            return

        rows, cols = self.image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated_image = cv2.warpAffine(self.image, rotation_matrix, (cols, rows))

        self.processed_image = rotated_image
        self.display_image(self.processed_image, self.lblProcessedImage)
        self.save_history()

    def apply_zoom_in(self):
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please open an image first!")
            return

        scale = 1.5  # Zoom in 1.5x
        height, width = self.image.shape[:2]
        new_width = int(width * scale)
        new_height = int(height * scale)
        zoomed_in_image = cv2.resize(self.image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        self.processed_image = zoomed_in_image
        cv2.imshow("Zoom In", self.processed_image)  # Tampilkan di window baru
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.save_history()

    def apply_zoom_out(self):
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please open an image first!")
            return

        scale = 0.5  # Zoom out 0.5x
        height, width = self.image.shape[:2]
        new_width = int(width * scale)
        new_height = int(height * scale)
        zoomed_out_image = cv2.resize(self.image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        self.processed_image = zoomed_out_image
        cv2.imshow("Zoom Out", self.processed_image)  # Tampilkan di window baru
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.save_history()

    def apply_flipping(self, flip_code):
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please open an image first!")
            return

        flipped_image = cv2.flip(self.image, flip_code)
        self.processed_image = flipped_image
        self.display_image(self.processed_image, self.lblProcessedImage)
        self.save_history()

    def apply_crop(self):
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please open an image first!")
            return

        x, y, w, h = 50, 50, 200, 200
        cropped_image = self.image[y:y + h, x:x + w]
        self.processed_image = cropped_image
        self.display_image(self.processed_image, self.lblProcessedImage)
        self.save_history()

    def apply_shearing(self):
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please open an image first!")
            return

        shear_factor = 0.2  # Faktor shear tetap
        rows, cols = self.image.shape[:2]
        shear_matrix = np.float32([[1, shear_factor, 0], [shear_factor, 1, 0]])
        sheared_image = cv2.warpAffine(self.image, shear_matrix, (cols, rows))

        self.processed_image = sheared_image
        self.display_image(self.processed_image, self.lblProcessedImage)
        self.save_history()

    #Fungsi Operasi Aritmatika
    def addition(self):
        if self.image is None or self.processed_image is None:
            QMessageBox.warning(self, "Warning", "Load two images first!")
            return
        result = cv2.add(self.image, self.processed_image)
        self.processed_image = result
        self.display_image(self.processed_image, self.lblProcessedImage)
        self.save_history()

    def subtraction(self):
        if self.image is None or self.processed_image is None:
            QMessageBox.warning(self, "Warning", "Load two images first!")
            return
        result = cv2.subtract(self.image, self.processed_image)
        self.processed_image = result
        self.display_image(self.processed_image, self.lblProcessedImage)
        self.save_history()

    def multiplication(self):
        if self.image is None or self.processed_image is None:
            QMessageBox.warning(self, "Warning", "Load two images first!")
            return
        result = np.multiply(self.image, self.processed_image)
        result = np.clip(result, 0, 255).astype(np.uint8)
        self.processed_image = result
        self.display_image(self.processed_image, self.lblProcessedImage)
        self.save_history()

    def division(self):
        if self.image is None or self.processed_image is None:
            QMessageBox.warning(self, "Warning", "Load two images first!")
            return
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.true_divide(self.image, self.processed_image)
            result = np.nan_to_num(result) * 255
            result = np.clip(result, 0, 255).astype(np.uint8)
        self.processed_image = result
        self.display_image(self.processed_image, self.lblProcessedImage)
        self.save_history()

    def boolean_operation(self, operation):
        if self.image is None or self.image2 is None:
            QMessageBox.warning(self, "Error", "Please load two images first!")
            return

        # Konversi ke grayscale
        img1_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)

        # Resize img2 agar ukurannya sama dengan img1
        img2_gray = cv2.resize(img2_gray, (img1_gray.shape[1], img1_gray.shape[0]))

        # Operasi Boolean
        if operation == "AND":
            result = cv2.bitwise_and(img1_gray, img2_gray)
        elif operation == "OR":
            result = cv2.bitwise_or(img1_gray, img2_gray)
        elif operation == "XOR":
            result = cv2.bitwise_xor(img1_gray, img2_gray)
        else:
            QMessageBox.warning(self, "Error", "Invalid operation!")
            return

        # Tampilkan hasilnya
        self.processed_image = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        self.display_image(self.processed_image, self.lblProcessedImage)

    # Fungsi Operasi Spasial
    def apply_lpf(self):
        """
        Fungsi untuk menerapkan Low Pass Filter (LPF) pada processed_image.
        LPF di sini menggunakan kernel rata-rata 5x5.
        """
        if self.processed_image is None:
            QMessageBox.warning(self, "Warning", "No image to process!")
            return

        # Kernel LPF (average filter)
        kernel = np.ones((5, 5), np.float32) / 25.0
        filtered = cv2.filter2D(self.processed_image, -1, kernel)
        self.processed_image = filtered
        self.display_image(self.processed_image, self.lblProcessedImage)
        self.save_history()

    def apply_hpf(self):
        """
        Fungsi untuk menerapkan High Pass Filter (HPF) pada processed_image.
        HPF di sini menggunakan kernel sharpen 3x3.
        """
        if self.processed_image is None:
            QMessageBox.warning(self, "Warning", "No image to process!")
            return

        # Kernel HPF (sharpening filter)
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        filtered = cv2.filter2D(self.processed_image, -1, kernel)
        self.processed_image = filtered
        self.display_image(self.processed_image, self.lblProcessedImage)
        self.save_history()

    # Fungsi untuk smoothing

    def apply_mean_filter(self):
        """
        Fungsi untuk menerapkan Mean Filter menggunakan fungsi cv2.blur dengan kernel 5x5.
        """
        if self.processed_image is None:
            QMessageBox.warning(self, "Warning", "No image to process!")
            return
        filtered = cv2.blur(self.processed_image, (5, 5))
        self.processed_image = filtered
        self.display_image(self.processed_image, self.lblProcessedImage)
        self.save_history()

    def apply_gaussian(self):
        """
        Fungsi untuk menerapkan Gaussian Filter menggunakan cv2.GaussianBlur dengan kernel 5x5.
        """
        if self.processed_image is None:
            QMessageBox.warning(self, "Warning", "No image to process!")
            return
        filtered = cv2.GaussianBlur(self.processed_image, (5, 5), 0)
        self.processed_image = filtered
        self.display_image(self.processed_image, self.lblProcessedImage)
        self.save_history()

    def apply_median_filter(self):
        """
        Fungsi untuk menerapkan Median Filter menggunakan cv2.medianBlur dengan ukuran kernel 5.
        """
        if self.processed_image is None:
            QMessageBox.warning(self, "Warning", "No image to process!")
            return
        filtered = cv2.medianBlur(self.processed_image, 5)
        self.processed_image = filtered
        self.display_image(self.processed_image, self.lblProcessedImage)
        self.save_history()

    def apply_max_filter(self):
        """
        Fungsi untuk menerapkan Max Filter menggunakan cv2.dilate dengan kernel 3x3.
        Max Filter (dilasi) menonjolkan nilai maksimum di area sekitarnya.
        """
        if self.processed_image is None:
            QMessageBox.warning(self, "Warning", "No image to process!")
            return
        kernel = np.ones((3, 3), np.uint8)
        filtered = cv2.dilate(self.processed_image, kernel, iterations=1)
        self.processed_image = filtered
        self.display_image(self.processed_image, self.lblProcessedImage)
        self.save_history()

    def apply_min_filter(self):
        """
        Fungsi untuk menerapkan Min Filter menggunakan cv2.erode dengan kernel 3x3.
        Min Filter (erosi) menonjolkan nilai minimum di area sekitarnya.
        """
        if self.processed_image is None:
            QMessageBox.warning(self, "Warning", "No image to process!")
            return
        kernel = np.ones((3, 3), np.uint8)
        filtered = cv2.erode(self.processed_image, kernel, iterations=1)
        self.processed_image = filtered
        self.display_image(self.processed_image, self.lblProcessedImage)
        self.save_history()

    def apply_edge_enhancement(self):
        """
        Menerapkan filter untuk meningkatkan kontras tepi dengan kernel edge enhancement.
        """
        if self.processed_image is None:
            QMessageBox.warning(self, "Warning", "No image to process!")
            return

        # Kernel untuk edge enhancement (variasi dari kernel sharpening)
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        filtered = cv2.filter2D(self.processed_image, -1, kernel)
        self.processed_image = filtered
        self.display_image(self.processed_image, self.lblProcessedImage)
        self.save_history()

    def apply_sharpening_standar(self):
        """
        Menerapkan filter sharpening standar menggunakan kernel 3x3.
        """
        if self.processed_image is None:
            QMessageBox.warning(self, "Warning", "No image to process!")
            return

        # Kernel sharpening standar
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        filtered = cv2.filter2D(self.processed_image, -1, kernel)
        self.processed_image = filtered
        self.display_image(self.processed_image, self.lblProcessedImage)
        self.save_history()

    def apply_strong_sharpening(self):
        """
        Menerapkan filter sharpening dengan efek lebih kuat.
        """
        if self.processed_image is None:
            QMessageBox.warning(self, "Warning", "No image to process!")
            return

        # Kernel untuk strong sharpening
        kernel = np.array([[-1, -1, -1],
                           [-1, 11, -1],
                           [-1, -1, -1]])
        filtered = cv2.filter2D(self.processed_image, -1, kernel)
        self.processed_image = filtered
        self.display_image(self.processed_image, self.lblProcessedImage)
        self.save_history()

    def apply_laplace_3x3(self):
        """
        Menerapkan filter Laplace dengan kernel 3x3 untuk mendeteksi tepi.
        """
        if self.processed_image is None:
            QMessageBox.warning(self, "Warning", "No image to process!")
            return

        gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        laplace = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        laplace = np.uint8(np.abs(laplace))  # Konversi ke skala absolut
        self.processed_image = cv2.cvtColor(laplace, cv2.COLOR_GRAY2BGR)

        self.display_image(self.processed_image, self.lblProcessedImage)
        self.save_history()

    def apply_laplace_5x5(self):
        """
        Menerapkan filter Laplace dengan kernel 5x5 untuk deteksi tepi yang lebih halus.
        """
        if self.processed_image is None:
            QMessageBox.warning(self, "Warning", "No image to process!")
            return

        gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        laplace = cv2.Laplacian(gray, cv2.CV_64F, ksize=5)
        laplace = np.uint8(np.abs(laplace))  # Konversi ke skala absolut
        self.processed_image = cv2.cvtColor(laplace, cv2.COLOR_GRAY2BGR)

        self.display_image(self.processed_image, self.lblProcessedImage)
        self.save_history()

    def apply_negative_edge_detection(self):
        """
        Menerapkan deteksi tepi negatif dengan Laplacian dan inversi warna.
        """
        if self.processed_image is None:
            QMessageBox.warning(self, "Warning", "No image to process!")
            return

        gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        laplace = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        laplace = np.uint8(np.abs(laplace))  # Konversi ke skala absolut
        negative = cv2.bitwise_not(laplace)  # Inversi warna
        self.processed_image = cv2.cvtColor(negative, cv2.COLOR_GRAY2BGR)

        self.display_image(self.processed_image, self.lblProcessedImage)
        self.save_history()

    #Fungsi Operasi Edge Detection
    # Fungsi Operasi Edge Detection
    def apply_edge_detection(self, method):
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please open an image first!")
            return

        try:
            original_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            if method == 'sobel':
                sobel_x = cv2.Sobel(original_gray, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(original_gray, cv2.CV_64F, 0, 1, ksize=3)
                sobel_x = cv2.convertScaleAbs(sobel_x)
                sobel_y = cv2.convertScaleAbs(sobel_y)
                result = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

            elif method == 'prewitt':
                kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
                kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
                prewitt_x = cv2.filter2D(original_gray, -1, kernel_x)
                prewitt_y = cv2.filter2D(original_gray, -1, kernel_y)
                prewitt_x = cv2.convertScaleAbs(prewitt_x)
                prewitt_y = cv2.convertScaleAbs(prewitt_y)
                result = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)

            elif method == 'roberts':
                kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
                kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
                roberts_x = cv2.filter2D(original_gray, cv2.CV_64F, kernel_x)
                roberts_y = cv2.filter2D(original_gray, cv2.CV_64F, kernel_y)
                result = np.sqrt(roberts_x ** 2 + roberts_y ** 2)
                result = np.clip(result, 0, 255).astype(np.uint8)

            elif method == 'canny':
                result = cv2.Canny(original_gray, 100, 200)

            else:
                QMessageBox.warning(self, "Warning", "Unknown edge detection method selected!")
                return

            # Konversi hasil edge (grayscale) ke 3 channel agar bisa ditampilkan
            result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

            self.processed_image = result_rgb
            self.display_image(self.processed_image, self.lblProcessedImage)
            self.save_history()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error processing image: {str(e)}")

    #Fungsi Operasi Morfologi
    def apply_erosion(self):
        self.morphology_operation(cv2.erode, "Erosion")

    def apply_dilation(self):
        self.morphology_operation(cv2.dilate, "Dilation")

    def apply_opening(self):
        self.morphology_operation(lambda img, kernel: cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel), "Opening")

    def apply_closing(self):
        self.morphology_operation(lambda img, kernel: cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel), "Closing")

    def apply_thinning(self):
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please open an image first!")
            return
        try:
            # 1. Konversi ke grayscale
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            # 2. Thresholding agar menjadi citra biner (0 dan 255)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            # 3. Lakukan thinning
            thinned = cv2.ximgproc.thinning(binary)

            # 4. Konversi ke 3 channel agar kompatibel dengan display_image
            thinned_rgb = cv2.cvtColor(thinned, cv2.COLOR_GRAY2BGR)

            # 5. Tampilkan hasil
            self.processed_image = thinned_rgb
            self.display_image(self.processed_image, self.lblProcessedImage)
            self.save_history()

        except AttributeError:
            QMessageBox.critical(self, "Error",
                                 "Fitur thinning tidak tersedia. Pastikan opencv-contrib-python sudah terinstall!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Terjadi kesalahan: {str(e)}")

    def morphology_operation(self, operation, name):
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please open an image first!")
            return
        kernel = np.ones((3, 3), np.uint8)
        self.processed_image = operation(self.image, kernel)
        self.display_image(self.processed_image, self.lblProcessedImage)
        self.save_history()

    # Action Operasi Ranah Frequensi
    # DFT (Magnitude Spectrum)
    def apply_dft(self):
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please open an image first!")
            return
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
        magnitude_log = 20 * np.log(magnitude + 1)
        magnitude_norm = cv2.normalize(magnitude_log, None, 0, 255, cv2.NORM_MINMAX)
        self.processed_image = cv2.cvtColor(np.uint8(magnitude_norm), cv2.COLOR_GRAY2BGR)
        self.display_image(self.processed_image, self.lblProcessedImage)
        self.save_history()

    # IDFT
    def apply_idft(self):
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please open an image first!")
            return
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        f_ishift = np.fft.ifftshift(dft_shift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        img_back_norm = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        self.processed_image = cv2.cvtColor(np.uint8(img_back_norm), cv2.COLOR_GRAY2BGR)
        self.display_image(self.processed_image, self.lblProcessedImage)
        self.save_history()

    # Low Pass Filter
    def apply_low_pass(self):
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please open an image first!")
            return
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        mask = np.zeros((rows, cols, 2), np.uint8)
        mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        img_back_norm = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        self.processed_image = cv2.cvtColor(np.uint8(img_back_norm), cv2.COLOR_GRAY2BGR)
        self.display_image(self.processed_image, self.lblProcessedImage)
        self.save_history()

    # High Pass Filter
    def apply_high_pass(self):
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please open an image first!")
            return
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        mask = np.ones((rows, cols, 2), np.uint8)
        mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        img_back_norm = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        self.processed_image = cv2.cvtColor(np.uint8(img_back_norm), cv2.COLOR_GRAY2BGR)
        self.display_image(self.processed_image, self.lblProcessedImage)
        self.save_history()

    # Band Pass Filter
    def apply_band_pass(self):
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please open an image first!")
            return
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        mask = np.zeros((rows, cols, 2), np.uint8)
        mask[crow - 50:crow + 50, ccol - 50:ccol + 50] = 1
        mask[crow - 20:crow + 20, ccol - 20:ccol + 20] = 0
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        img_back_norm = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        self.processed_image = cv2.cvtColor(np.uint8(img_back_norm), cv2.COLOR_GRAY2BGR)
        self.display_image(self.processed_image, self.lblProcessedImage)
        self.save_history()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AplikasiCitworks()
    window.show()
    sys.exit(app.exec_())