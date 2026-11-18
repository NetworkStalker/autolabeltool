"""Headless regression checks: run with QT_QPA_PLATFORM=offscreen."""
import tempfile
import time
import unittest
from pathlib import Path

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QListView

from label_project import Annotation, Project
from label_tool import Window


class WindowRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_filters_do_not_resize_or_displace_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            image = root / "image.png"
            Image.new("RGB", (640, 480)).save(image)
            project = Project(root / "filters.dlabel", create=True)
            project.add_class("A very long class name " * 12)
            project.import_images([image])
            project.close()
            window = Window()
            try:
                window.open_project(root / "filters.dlabel")
                window.resize(1100, 760)
                window.show()
                self.app.processEvents()
                size = window.size()
                canvas = window.canvas.geometry()
                gallery = window.gallery.geometry()
                for _ in range(3):
                    window.filters_button.click()
                    self.app.processEvents()
                    self.assertFalse(window.filter_panel.isHidden())
                    screen = window.filters_button.screen().availableGeometry()
                    self.assertTrue(screen.contains(window.filter_panel.frameGeometry()),
                                    f"screen={screen}, popup={window.filter_panel.frameGeometry()}")
                    window.filter_status.setCurrentText("Unlabeled")
                    window.filter_class.setCurrentIndex(1)
                    self.app.processEvents()
                    self.assertEqual(window.size(), size)
                    self.assertEqual(window.canvas.geometry(), canvas)
                    self.assertEqual(window.gallery.geometry(), gallery)
                    window.filter_panel.hide()
                    self.app.processEvents()
                    self.assertFalse(window.filters_button.isChecked())
                    window.reset_filters()
            finally:
                window.close()
                self.app.processEvents()

    def test_thumbnail_image_and_caption_fit_when_scrolling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = Image.new("RGB", (640, 480), "lime")
            source.paste("red", (0, 240, 640, 480))
            paths = []
            for number in range(12):
                path = root / f"image_{number}.png"
                source.save(path)
                paths.append(path)
            project = Project(root / "thumbnails.dlabel", create=True)
            project.add_class("Car")
            project.import_images(paths)
            project.close()
            window = Window()
            try:
                window.open_project(root / "thumbnails.dlabel")
                window.show()
                self.app.processEvents()
                for row in range(12):
                    window.film.data(window.film.index(row), Qt.ItemDataRole.DecorationRole)
                window.film.pool.waitForDone()
                self.app.processEvents()
                for width, height in [(1380, 850), (1100, 740)]:
                    window.resize(width, height)
                    self.app.processEvents()
                    for row in [0, 11, 0]:
                        index = window.film.index(row)
                        window.gallery.setCurrentIndex(index)
                        window.gallery.scrollTo(index)
                        self.app.processEvents()
                        rect = window.gallery.visualRect(index)
                        self.assertTrue(window.gallery.viewport().rect().contains(rect.adjusted(2, 0, -2, 0)),
                                        f"row={row} rect={rect} viewport={window.gallery.viewport().rect()} grid={window.gallery.gridSize()}")
                        shot = window.gallery.viewport().grab().toImage()
                        x, y = rect.x() + 80, rect.y() + 8
                        self.assertEqual(shot.pixelColor(x, y + 2).name(), "#00ff00")
                        self.assertEqual(shot.pixelColor(x, y + 73).name(), "#ff0000")
            finally:
                window.close()
                self.app.processEvents()

    def test_equal_lists_and_expanded_views_share_selection_and_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            image = root / "image.png"
            Image.new("RGB", (640, 480)).save(image)
            project = Project(root / "lists.dlabel", create=True)
            first = project.add_class("Car")
            second = project.add_class("Person")
            project.import_images([image])
            project.save(project.images()[0]["id"], [Annotation(first, "box", (10, 10, 80, 80), "box")])
            project.close()
            window = Window()
            try:
                window.open_project(root / "lists.dlabel")
                window.show()
                self.app.processEvents()
                self.assertLessEqual(abs(window.classes.height() - window.object_list.height()), 2)
                window.expand_list("classes")
                classes = window.list_windows["classes"].findChild(QListView)
                classes.setCurrentIndex(classes.model().index(1, 0))
                self.assertEqual(window.class_id, second)
                window.expand_list("objects")
                objects = window.list_windows["objects"].findChild(QListView)
                objects.setCurrentIndex(objects.model().index(0, 0))
                self.assertEqual(window.selected, "box")
                window.save_object(Annotation(second, "box", (100, 100, 160, 160), "box2"))
                self.assertEqual(objects.model().rowCount(), 2)
                self.assertEqual(objects.currentIndex().data(Qt.ItemDataRole.UserRole), "box2")
                window.list_windows["classes"].close()
                self.assertEqual(window.classes.count(), 2)
                window.expand_list("classes")
                self.assertEqual(classes.model().rowCount(), 2)
            finally:
                window.close()
                self.app.processEvents()

    def test_edit_undo_review_and_background_thumbnails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            image = root / "image.png"
            Image.new("RGB", (640, 480), "red").save(image)
            project = Project(root / "project.dlabel", create=True)
            class_id = project.add_class("Car")
            project.import_images([image])
            project.close()
            window = Window()
            try:
                self.assertFalse(window.context_buttons["Delete"].isEnabled())
                self.assertTrue(window.filter_panel.isHidden())
                self.assertEqual(window.canvas_stack.currentIndex(), 1)
                window.filters_button.click()
                self.assertFalse(window.filter_panel.isHidden())
                window.filters_button.click()
                window.set_tool("keypoint")
                self.assertEqual(window.more_tools.toolTip(), "Keypoint · K")
                self.assertFalse(window.more_tools.icon().isNull())
                window.set_tool("box")
                window.open_project(root / "project.dlabel")
                self.assertEqual(window.canvas_stack.currentIndex(), 0)
                window.save_object(Annotation(class_id, "box", (10, 10, 80, 80), "box"))
                self.assertEqual(len(window.project.objects(window.image["id"])), 1)
                self.assertTrue(window.context_buttons["Delete"].isEnabled())
                self.assertFalse(window.context_buttons["Accept"].isEnabled())
                window.undo(False)
                self.assertEqual(window.objects, [])
                window.undo(True)
                self.assertEqual(len(window.objects), 1)
                window.commit([Annotation(class_id, "box", (10, 10, 80, 80), "predicted", pending=True)])
                self.assertFalse(window.review_bar.isHidden())
                window.review(True)
                self.assertTrue(window.review_bar.isHidden())
                window.toggle_review()
                self.assertTrue(window.review_action.isChecked())
                window.clear_filters()
                window.refresh_images()
                index = window.film.index(0)
                window.film.data(index, Qt.ItemDataRole.DecorationRole)
                deadline = time.monotonic() + 5
                while str(image) not in window.film.cache and time.monotonic() < deadline:
                    self.app.processEvents()
                    time.sleep(0.01)
                self.assertIn(str(image), window.film.cache)
                self.assertFalse(window.film.cache[str(image)].isNull())
                generation = window.film.generation
                window.film.clear_cache()
                window.film.thumbnail_ready("stale", generation, QImage())
                self.assertNotIn("stale", window.film.cache)
                image.unlink()
                window.loaded = False
                window.load_image(window.image)
                self.assertFalse(window.loaded)
                self.assertFalse(window.image_notice.isHidden())
                self.assertEqual(len(window.project.objects(window.image["id"])), 1)
            finally:
                window.close()
                self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
