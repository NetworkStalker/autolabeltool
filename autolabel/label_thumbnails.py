"""Decode thumbnails off the GUI thread, returning detached QImages."""
from PySide6.QtCore import QObject, QRunnable, Signal
from PySide6.QtGui import QImage
from label_project import open_image


class ThumbnailSignals(QObject):
    ready = Signal(str, int, QImage)


class ThumbnailTask(QRunnable):
    def __init__(self, path, generation, signals):
        super().__init__()
        self.path, self.generation, self.signals = path, generation, signals

    def run(self):
        result = QImage()
        try:
            image = open_image(self.path)
            image.thumbnail((140, 76))
            width, height = image.size
            result = QImage(image.tobytes(), width, height, width * 3,
                            QImage.Format.Format_RGB888).copy()
        except (OSError, ValueError):
            pass
        self.signals.ready.emit(self.path, self.generation, result)
