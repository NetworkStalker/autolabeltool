"""Render the main window with a disposable demo project for layout checks."""
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image, ImageDraw
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase
from label_project import Annotation, Project
from label_tool import STYLE, Window


def main():
    app = QApplication([])
    for filename in ("segoeui.ttf", "segoeuib.ttf"):
        QFontDatabase.addApplicationFont(str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / filename))
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    destination = Path(__file__).parent / "ui-preview"
    destination.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        image = Image.new("RGB", (1200, 800), "#a9bcc6")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 420, 1200, 800), fill="#6e7775")
        for x, width, height in [(50, 240, 260), (330, 300, 330), (700, 200, 220), (940, 220, 300)]:
            draw.rectangle((x, 420-height, x+width, 420), fill="#ced1c5")
            for wx in range(x+20, x+width-20, 55):
                for wy in range(440-height, 390, 65):
                    draw.rectangle((wx, wy, wx+27, wy+40), fill="#758a8e")
        for x, y, color in [(170, 550, "#e0b35d"), (730, 650, "#b7ccd5")]:
            draw.rounded_rectangle((x, y, x+230, y+95), radius=18, fill=color)
            draw.polygon([(x+45, y), (x+80, y-60), (x+160, y-60), (x+200, y)], fill=color)
            draw.ellipse((x+30, y+65, x+80, y+115), fill="#29343c")
            draw.ellipse((x+160, y+65, x+210, y+115), fill="#29343c")
        paths = []
        for index in range(6):
            path = root / f"street_{index+1:03}.jpg"
            image.save(path)
            paths.append(path)
        project = Project(root / "Street dataset.dlabel", create=True)
        car = project.add_class("Car", "#63d8b5")
        project.add_class("Person", "#e8b86e")
        project.add_class("Bicycle", "#8ca9ff")
        project.import_images(paths)
        project.save(project.images()[0]["id"], [Annotation(car, "box", (160, 480, 410, 675), "car1")])
        project.close()
        window = Window()
        window.show()
        app.processEvents()
        window.grab().save(str(destination / "welcome.png"))
        window.open_project(root / "Street dataset.dlabel")
        window.film.pool.waitForDone()
        app.processEvents()
        window.film.pool.waitForDone()
        app.processEvents()
        window.grab().save(str(destination / "workspace.png"))
        window.resize(1100, 740)
        app.processEvents()
        window.grab().save(str(destination / "compact.png"))
        window.close()
        app.processEvents()


if __name__ == "__main__":
    main()
