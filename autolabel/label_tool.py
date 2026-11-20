"""Run with python label_tool.py [project.dlabel]."""
from __future__ import annotations

import argparse
import logging
import sys
from logging.handlers import RotatingFileHandler
from collections import OrderedDict, defaultdict
from dataclasses import replace
from pathlib import Path
from threading import Event

import yaml
from PySide6.QtCore import QAbstractListModel, QEvent, QModelIndex, QRect, QRectF, QSize, Qt, QThread, QThreadPool, QTimer, Signal, QSettings
from PySide6.QtGui import QAction, QColor, QIcon, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QComboBox, QFileDialog, QFrame, QHBoxLayout,
    QInputDialog, QLabel, QLineEdit, QListView, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QSplitter, QToolBar, QVBoxLayout, QWidget,
    QMenu, QToolButton, QStackedWidget, QSizePolicy,
    QStyledItemDelegate, QStyle,
    QDialog,
)

from label_canvas import Canvas, moved
from label_dialogs import ExportDialog, MappingDialog, SourceDialog, TrainDialog, class_editor
from label_export import export_yolo
from label_predict import TEXT_MODEL, approve, batch, inspect_model
from label_project import IMAGE_EXTENSIONS, STATUSES, Project, class_names
from label_train import train_dataset
from label_thumbnails import ThumbnailSignals, ThumbnailTask

LOGGER = logging.getLogger("dlabel")
MACOS = sys.platform == "darwin"
TOGGLE_ANNOTATIONS_KEY = "Ctrl+Shift+H" if MACOS else "Ctrl+H"
REDO_KEY = "Ctrl+Shift+Z" if MACOS else "Ctrl+Y"


def shortcut_label(text):
    return text.replace("Ctrl+", "Cmd+") if MACOS else text


STYLE = """
QWidget { background: #10151c; color: #dde5ef; font-family: "Segoe UI"; font-size: 13px; }
QMenuBar { background: #10151c; color: #8997aa; padding: 3px 10px; }
QMenuBar::item:selected { background: #253142; border-radius: 4px; }
QToolBar { background: #171f2a; padding: 12px; spacing: 8px; border: 0; border-bottom: 1px solid #293444; }
QToolButton, QPushButton, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
    background: #202b39; border: 1px solid #334155; padding: 8px 12px; border-radius: 7px;
}
QPushButton:hover, QToolButton:hover, QComboBox:hover { background: #2b3a4d; border-color: #566980; }
QPushButton:checked, QToolButton:checked { background: #164740; color: #9af2da; border-color: #36b99c; }
QPushButton:disabled, QToolButton:disabled { color: #647183; border-color: #25303e; }
QPushButton[primary="true"] { background: #63d8b5; color: #09251f; border: 1px solid #63d8b5; font-weight: 600; }
QPushButton[primary="true"]:hover { background: #8be8cb; }
QPushButton[primary="true"]:disabled { background: #24483f; color: #79988f; border-color: #24483f; }
QLineEdit { background: #111923; selection-background-color: #24564b; }
QLineEdit:focus { border-color: #63d8b5; }
QListView, QTableWidget { background: #171f2a; border: 1px solid #2b3747; border-radius: 9px; outline: none; }
QListView::item { padding: 8px; border-radius: 5px; margin: 2px; }
QListView::item:hover { background: #233243; }
QListView::item:selected, QTableWidget::item:selected { background: #214b43; color: #b7f5df; }
QListView[dense="true"] { font-size: 12px; }
QListView[dense="true"]::item { padding: 4px 8px; margin: 0; border-radius: 4px; }
QLabel[section="true"] { color: #9cabbc; font-size: 11px; font-weight: 600; padding: 8px 0; }
QFrame#inspector { background: #171f2a; border: 1px solid #2b3747; border-radius: 10px; }
QFrame#inspector QLabel { background: transparent; }
QWidget#inspectorSection { background: transparent; }
QSplitter#inspectorSplit::handle { background: #334155; height: 5px; border-radius: 2px; }
QSplitter#inspectorSplit::handle:hover { background: #63d8b5; }
QFrame#filterPopup { background: #171f2a; border: 1px solid #40536a; border-radius: 9px; }
QLabel#brand { font-size: 20px; font-weight: 700; color: #8ce8cb; padding-right: 16px; }
QLabel#emptyTitle { font-size: 28px; font-weight: 600; color: #ecf4fc; }
QLabel#muted { color: #8c9cb0; background: transparent; }
QMenu { background: #1b2633; border: 1px solid #3c4c61; padding: 6px; }
QMenu::item { padding: 9px 24px 9px 12px; border-radius: 4px; }
QMenu::item:selected { background: #2b4b46; }
QMenu::item:disabled { color: #647183; }
QMenu::separator { height: 1px; background: #334155; margin: 5px; }
QProgressBar { border: 1px solid #334155; background: #111923; border-radius: 5px; text-align: center; }
QProgressBar::chunk { background: #329f85; }
QStatusBar { background: #10151c; color: #8c9cb0; border-top: 1px solid #25303e; }
QSplitter::handle { background: #10151c; height: 8px; }
QScrollBar:horizontal { background: #171f2a; height: 9px; }
QScrollBar::handle:horizontal { background: #3c5064; border-radius: 4px; min-width: 30px; }
QScrollBar:vertical { background: #171f2a; width: 9px; }
QScrollBar::handle:vertical { background: #3c5064; border-radius: 4px; min-height: 30px; }
"""
BADGES = dict(Unlabeled="–", Predicted="P", Labeled="L", Reviewed="R", Null="N")


class ThumbnailDelegate(QStyledItemDelegate):
    """Keep image and caption geometry stable while asynchronous icons arrive."""
    def sizeHint(self, option, index):
        return QSize(160, 76 + option.fontMetrics.height() + 24)

    def paint(self, painter, option, index):
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#214b43" if selected else "#171f2a"))
        painter.drawRoundedRect(QRectF(option.rect.adjusted(2, 2, -2, -2)), 6, 6)
        image_rect = QRect(option.rect.x() + 10, option.rect.y() + 8, 140, 76)
        icon = index.data(Qt.ItemDataRole.DecorationRole)
        if isinstance(icon, QIcon) and not icon.isNull():
            icon.paint(painter, image_rect, Qt.AlignmentFlag.AlignCenter, QIcon.Mode.Normal)
        else:
            painter.setBrush(QColor("#233243"))
            painter.drawRoundedRect(QRectF(image_rect), 4, 4)
            painter.setPen(QColor("#8c9cb0"))
            painter.drawText(image_rect, Qt.AlignmentFlag.AlignCenter, "…")
        caption = QRect(option.rect.x() + 8, image_rect.bottom() + 5,
                        option.rect.width() - 16, option.fontMetrics.height())
        text = option.fontMetrics.elidedText(index.data(Qt.ItemDataRole.DisplayRole) or "",
                                            Qt.TextElideMode.ElideMiddle, caption.width())
        painter.setPen(QColor("#b7f5df" if selected else "#dde5ef"))
        painter.setFont(option.font)
        painter.drawText(caption, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()


class Filmstrip(QAbstractListModel):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.cache = OrderedDict()
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(2)
        self.signals = ThumbnailSignals(self)
        self.signals.ready.connect(self.thumbnail_ready)
        self.pending = set()
        self.generation = 0

    def clear_cache(self):
        self.generation += 1
        self.pool.clear()
        self.pending.clear()
        self.cache.clear()

    def thumbnail_ready(self, path, generation, image):
        if generation != self.generation:
            return
        self.pending.discard(path)
        self.cache[path] = QIcon(QPixmap.fromImage(image)) if not image.isNull() else QIcon()
        if len(self.cache) > 192:
            self.cache.popitem(last=False)
        for row, item in enumerate(self.rows):
            if item["path"] == path:
                index = self.index(row)
                self.dataChanged.emit(index, index, [Qt.ItemDataRole.DecorationRole])
                break

    def set_rows(self, rows):
        self.beginResetModel()
        self.rows = rows
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def data(self, index, role):
        if not index.isValid() or index.row() >= len(self.rows):
            return None
        row = self.rows[index.row()]
        path = row["path"]
        if role == Qt.ItemDataRole.DisplayRole:
            return f"{BADGES[row['status']]}  {Path(path).name}"
        if role == Qt.ItemDataRole.ToolTipRole:
            return f"{row['status']} · {path}"
        if role == Qt.ItemDataRole.DecorationRole:
            if path not in self.cache:
                if path not in self.pending:
                    self.pending.add(path)
                    self.pool.start(ThumbnailTask(path, self.generation, self.signals))
                return QIcon()
            self.cache.move_to_end(path)
            return self.cache[path]
        return None


class Job(QThread):
    result_ready = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int)

    def __init__(self, task, parent):
        super().__init__(parent)
        self.task = task
        self.cancelled = Event()

    def run(self):
        try:
            self.result_ready.emit(self.task(self.cancelled, self.progress.emit))
        except Exception as error:
            LOGGER.exception("Background job failed")
            self.failed.emit(str(error))


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.project = None
        self.image = None
        self.objects = []
        self.class_rows = {}
        self.class_id = None
        self.selected = ""
        self.history = defaultdict(lambda: ([], []))
        self.clipboard = None
        self.preview = {}
        self.job = None
        self.loaded = False
        self.review_mode = False
        self._changing_rows = False
        self._build()
        self.resize(1380, 850)
        self.setWindowTitle("Label Tool · DLabel")
        self.update_status()
        self.statusBar().showMessage("Create or open a project to get started.")

    def guarded(self, callback):
        def call(*_):
            try:
                callback()
            except Exception as error:
                LOGGER.exception("Action failed: %s", getattr(callback, "__name__", "callback"))
                QMessageBox.warning(self, "Label Tool", str(error))
        return call

    def action(self, menu, text, callback, toolbar=None):
        action = QAction(shortcut_label(text), self)
        action.triggered.connect(self.guarded(callback))
        if menu:
            menu.addAction(action)
        if toolbar:
            toolbar.addAction(action)
        return action

    def shortcut(self, key, callback):
        QShortcut(QKeySequence(key), self, activated=self.guarded(
            lambda: callback() if not isinstance(self.focusWidget(), QLineEdit) else None
        ))

    def button(self, layout, text, callback):
        button = QPushButton(text)
        button.clicked.connect(self.guarded(callback))
        layout.addWidget(button)
        if text in {"Apply class", "Hide", "Lock", "Delete", "Accept", "Reject", "Undo joint", "This image"}:
            if not hasattr(self, "context_buttons"):
                self.context_buttons = {}
            self.context_buttons[text] = button
        return button

    def section(self, layout, text):
        label = QLabel(text)
        label.setProperty("section", True)
        layout.addWidget(label)
        return label

    def menu_button(self, layout, text, entries):
        button = QToolButton()
        button.setText(text)
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(button)
        for label, callback in entries:
            if label is None:
                menu.addSeparator()
            else:
                action = self.action(menu, label, callback)
                if label in {"Apply class", "Hide", "Lock", "Delete", "Accept", "Reject", "Undo joint"}:
                    if not hasattr(self, "context_buttons"):
                        self.context_buttons = {}
                    self.context_buttons[label] = action
        button.setMenu(menu)
        layout.addWidget(button)
        return button

    def _build(self):
        bar = self.menuBar()
        file_menu, edit_menu, view, data, model, export, help_menu = (
            bar.addMenu(name) for name in ("File", "Edit", "View", "Dataset", "Model", "Export", "Help")
        )
        toolbar = QToolBar("Actions")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        brand = QLabel("DLabel")
        brand.setObjectName("brand")
        toolbar.addWidget(brand)
        self.action(file_menu, "New project", lambda: self.choose_project(True))
        self.action(file_menu, "Open project", lambda: self.choose_project(False))
        self.menu_button(toolbar, "Project", [
            ("New project…", lambda: self.choose_project(True)),
            ("Open project…", lambda: self.choose_project(False)),
            (None, None), ("Add image folder…", self.add_folder),
            ("Import YOLO dataset…", self.import_yolo), ("Import video…", self.import_video),
            ("Refresh image files", self.refresh_image_files),
        ])
        self.add_images_button = self.button(toolbar, "+ Add images", self.add_images)
        self.action(file_menu, "Add images", self.add_images)
        self.action(file_menu, "Add image folder", self.add_folder)
        self.action(file_menu, "Refresh image files", self.refresh_image_files)
        self.action(file_menu, "Import YOLO dataset", self.import_yolo)
        self.action(file_menu, "Import video frames", self.import_video)
        self.action(edit_menu, "Undo  Ctrl+Z", lambda: self.undo(False))
        self.action(edit_menu, f"Redo  {REDO_KEY}", lambda: self.undo(True))
        self.action(edit_menu, "Copy object  Ctrl+C", self.copy_object)
        self.action(edit_menu, "Paste  Ctrl+V", self.paste)
        self.action(edit_menu, "Copy previous labels  Ctrl+Shift+V", self.paste_previous)
        self.action(edit_menu, "Delete selected  Del", self.delete_object)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        self.auto_label_button = self.button(toolbar, "Auto-label…", self.configure_model)
        self.action(model, "Model Manager / Auto-label", self.configure_model)
        self.action(model, "Your Models…", lambda: self.configure_model(your_models=True))
        self.action(model, "Train Dataset…", self.train_model)
        self.action(data, "Mark background  N", self.mark_null)
        self.action(data, "Approve image  Enter", lambda: self.review(True))
        self.action(data, "Reject predictions  X", lambda: self.review(False))
        self.review_action = self.action(data, "Review mode", self.toggle_review, toolbar)
        self.review_action.setCheckable(True)
        self.action(data, "Assign split to selected images", self.set_split)
        self.export_button = self.button(toolbar, "Export dataset", self.export_dataset)
        self.export_button.setProperty("primary", True)
        self.theme_button = self.button(toolbar, "☀", self.toggle_theme)
        self.theme_button.setToolTip("Switch light / dark mode")
        self.theme_button.setAccessibleName("Switch color theme")
        self.action(export, "Export YOLO", self.export_dataset)
        self.action(view, "Fit image  F", lambda: self.canvas.fit())
        self.action(view, "Actual size  0", lambda: self.canvas.actual_size())
        self.action(view, "Toggle class names  L", lambda: self.toggle_canvas("show_names"))
        self.action(view, f"Toggle annotations  {TOGGLE_ANNOTATIONS_KEY}", lambda: self.toggle_canvas("show_objects"))
        self.action(help_menu, "Keyboard shortcuts", self.show_help)

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(16, 12, 16, 6)
        outer.setSpacing(8)
        self.workflow_hint = QLabel("1  Create or open a project    →    2  Add images and classes    →    3  Label and review    →    4  Export YOLO")
        self.workflow_hint.setWordWrap(True)
        self.workflow_hint.setObjectName("muted")
        self.workflow_hint.setStyleSheet("padding: 2px 0 6px 0; font-size: 12px;")
        outer.addWidget(self.workflow_hint)
        self.image_notice = QLabel()
        self.image_notice.setWordWrap(True)
        self.image_notice.setStyleSheet("padding: 8px; color: #fbbf24;")
        self.image_notice.hide()
        outer.addWidget(self.image_notice)
        split = QSplitter(Qt.Orientation.Vertical)
        outer.addWidget(split, 1)

        workspace = QWidget()
        top = QHBoxLayout(workspace)
        top.setContentsMargins(0, 0, 0, 0)
        tools = QVBoxLayout()
        tools.setSpacing(8)
        self.section(tools, "TOOLS")
        top.addLayout(tools)
        self.tool_buttons = {}
        for name, key, tool in (("Select", "H", "select"), ("Box", "B", "box"),
                                ("Polygon", "G", "polygon")):
            button = self.button(tools, f"{name}  {key}", lambda mode=tool: self.set_tool(mode))
            button.setToolTip(f"{name} tool · {key}")
            button.setCheckable(True)
            self.tool_icon(button, tool)
            self.tool_buttons[tool] = button
        self.more_tools = self.menu_button(tools, "More tools", [
            ("Point  P", lambda: self.set_tool("point")),
            ("Keypoint  K", lambda: self.set_tool("keypoint")),
        ])
        self.tool_icon(self.more_tools, "more")
        self.box_mode = QComboBox()
        self.box_mode.addItems(["Drag box", "Two corners"])
        self.box_mode.currentIndexChanged.connect(self.box_mode_changed)
        tools.addWidget(self.box_mode)
        self.visibility = QComboBox()
        self.visibility.addItems(["Visible", "Occluded", "Missing"])
        self.visibility.currentIndexChanged.connect(lambda index: setattr(self.canvas, "visibility", 2 - index))
        tools.addWidget(self.visibility)
        tools.addStretch()
        fit_button = self.button(tools, "Fit  F", lambda: self.canvas.fit())
        fit_button.setToolTip("Fit image · F")
        self.tool_icon(fit_button, "fit")
        self.canvas = Canvas()
        self.canvas.changed.connect(self.guarded_with_item)
        self.canvas.selected.connect(self.select_object)
        self.canvas.cursor_at.connect(lambda text: self.coordinates.setText(text))
        self.canvas_stack = QStackedWidget()
        self.canvas_stack.addWidget(self.canvas)
        empty = QWidget()
        welcome = QVBoxLayout(empty)
        welcome.setSpacing(16)
        welcome.addStretch()
        title = QLabel("A clear space to label.")
        title.setObjectName("emptyTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome.addWidget(title)
        self.empty_description = QLabel("Create a project, add your images, and start annotating.")
        self.empty_description.setObjectName("muted")
        self.empty_description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_description.setWordWrap(True)
        welcome.addWidget(self.empty_description)
        empty_actions = QHBoxLayout()
        empty_actions.addStretch()
        self.empty_primary = self.button(empty_actions, "Create project", self.empty_start)
        self.empty_primary.setProperty("primary", True)
        self.empty_open = self.button(empty_actions, "Open project…", lambda: self.choose_project(False))
        empty_actions.addStretch()
        welcome.addLayout(empty_actions)
        welcome.addStretch()
        self.canvas_stack.addWidget(empty)
        center = QVBoxLayout()
        center.setContentsMargins(0, 0, 0, 0)
        center.addWidget(self.canvas_stack, 1)
        self.review_bar = QWidget()
        review_row = QHBoxLayout(self.review_bar)
        review_row.setContentsMargins(8, 8, 8, 8)
        review_row.addWidget(QLabel("Predictions ready to review"), 1)
        self.button(review_row, "Reject predictions", lambda: self.review(False))
        approve_button = self.button(review_row, "Approve image", lambda: self.review(True))
        approve_button.setProperty("primary", True)
        self.review_bar.hide()
        center.addWidget(self.review_bar)
        top.addLayout(center, 1)
        self.set_tool("box")

        panel = QFrame()
        panel.setObjectName("inspector")
        panel.setMinimumWidth(240)
        panel.setMaximumWidth(280)
        right = QVBoxLayout(panel)
        right.setContentsMargins(12, 8, 12, 12)
        right.setSpacing(8)
        top.addWidget(panel)
        self.inspector_split = QSplitter(Qt.Orientation.Vertical)
        self.inspector_split.setObjectName("inspectorSplit")
        self.inspector_split.setChildrenCollapsible(False)
        self.inspector_split.setHandleWidth(7)
        right.addWidget(self.inspector_split, 1)
        class_section = QWidget()
        class_section.setObjectName("inspectorSection")
        class_layout = QVBoxLayout(class_section)
        class_layout.setContentsMargins(0, 0, 0, 0)
        class_layout.setSpacing(8)
        class_header = QHBoxLayout()
        self.section(class_header, "CLASSES")
        class_header.addStretch()
        self.button(class_header, "+", self.add_class).setToolTip("Add class")
        self.button(class_header, "↗", lambda: self.expand_list("classes")).setToolTip("Open Classes in a larger window")
        self.menu_button(class_header, "Manage", [
            ("Edit class…", self.edit_class), ("Import classes…", self.import_class_list),
            ("Move up", lambda: self.move_class(-1)), ("Move down", lambda: self.move_class(1)),
            (None, None), ("Merge / delete…", self.remove_class),
        ])
        class_layout.addLayout(class_header)
        self.classes = QListWidget()
        self.classes.setProperty("dense", True)
        self.classes.setMinimumHeight(80)
        self.classes.currentItemChanged.connect(self.select_class)
        self.classes.itemDoubleClicked.connect(lambda _: self.guarded(self.edit_class)())
        class_layout.addWidget(self.classes, 1)
        self.inspector_split.addWidget(class_section)
        object_section = QWidget()
        object_section.setObjectName("inspectorSection")
        object_layout = QVBoxLayout(object_section)
        object_layout.setContentsMargins(0, 0, 0, 0)
        object_layout.setSpacing(8)
        object_header = QHBoxLayout()
        self.object_title = self.section(object_header, "OBJECTS")
        object_header.addStretch()
        self.button(object_header, "↗", lambda: self.expand_list("objects")).setToolTip("Open Objects in a larger window")
        self.menu_button(object_header, "Actions", [
            ("Apply class", self.reclassify), ("Hide", lambda: self.toggle_object("hidden")),
            ("Lock", lambda: self.toggle_object("locked")), (None, None),
            ("Accept", lambda: self.review_object(True)), ("Reject", lambda: self.review_object(False)),
            ("Undo joint", self.remove_joint),
            (None, None), ("Delete", self.delete_object),
        ])
        object_layout.addLayout(object_header)
        self.object_list = QListWidget()
        self.object_list.setProperty("dense", True)
        self.object_list.setMinimumHeight(80)
        self.object_list.currentItemChanged.connect(lambda item, _: self.select_object(
            item.data(Qt.ItemDataRole.UserRole) if item else ""
        ))
        object_layout.addWidget(self.object_list, 1)
        self.inspector_split.addWidget(object_section)
        self.inspector_split.setStretchFactor(0, 1)
        self.inspector_split.setStretchFactor(1, 1)
        self.inspector_split.setSizes([300, 300])
        self.inspector_split.handle(1).setToolTip("Drag to resize Classes and Objects")
        self.model_caption = QLabel("No model configured")
        self.model_caption.setObjectName("muted")
        self.model_caption.setWordWrap(True)
        self.model_caption.setToolTip("Configure Auto-label above, or open the Model menu for training and saved models.")
        right.addWidget(self.model_caption)
        split.addWidget(workspace)

        bottom = QWidget()
        gallery = QVBoxLayout(bottom)
        gallery.setContentsMargins(0, 6, 0, 0)
        row = QHBoxLayout()
        self.section(row, "IMAGES")
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search images…")
        self.search.setClearButtonEnabled(True)
        row.addWidget(self.search, 1)
        self.prev_button = self.button(row, "‹", lambda: self.navigate(-1))
        self.prev_button.setToolTip("Previous image · Page Up")
        self.image_counter = QLabel("0 / 0")
        self.image_counter.setObjectName("muted")
        self.image_counter.setMinimumWidth(60)
        self.image_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(self.image_counter)
        self.next_button = self.button(row, "›", lambda: self.navigate(1))
        self.next_button.setToolTip("Next image · Page Down")
        self.filters_button = self.button(row, "Filters & options", self.toggle_filters)
        self.filters_button.setCheckable(True)
        self.preview_button = self.button(row, "Exit preview", self.clear_preview)
        self.preview_button.hide()
        gallery.addLayout(row)
        # A popup does not change the splitter's minimum height or resize the window.
        self.filter_panel = QFrame(self, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.filter_panel.setObjectName("filterPopup")
        self.filter_panel.setAttribute(Qt.WidgetAttribute.WA_NoMouseReplay)
        self.filter_panel.installEventFilter(self)
        row = QVBoxLayout(self.filter_panel)
        row.setContentsMargins(14, 10, 14, 14)
        row.setSpacing(8)
        self.section(row, "FILTERS & OPTIONS")
        self.filter_status = QComboBox()
        self.filter_status.setToolTip("Filter images by annotation status")
        self.filter_status.addItems(["All statuses", *STATUSES])
        self.filter_status.currentIndexChanged.connect(lambda: self.refresh_images())
        row.addWidget(self.filter_status)
        self.filter_class = QComboBox()
        self.filter_class.setToolTip("Filter images by class")
        self.filter_class.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.filter_class.setMinimumContentsLength(16)
        self.filter_class.addItem("All classes")
        self.filter_class.currentIndexChanged.connect(lambda: self.refresh_images())
        row.addWidget(self.filter_class)
        self.button(row, "Next unlabeled", self.next_unlabeled)
        self.auto_next = self.button(row, "Auto-next box  F8", self.update_status)
        self.auto_next.setCheckable(True)
        self.auto_next.setToolTip("After saving a new box, open the next unlabeled image in this view.")
        self.button(row, "Clear filters", self.reset_filters)
        self.filter_panel.setFixedWidth(300)
        self.filter_panel.hide()
        self.film = Filmstrip()
        self.gallery = QListView()
        self.gallery.setItemDelegate(ThumbnailDelegate(self.gallery))
        self.gallery.setModel(self.film)
        self.gallery.setViewMode(QListView.ViewMode.IconMode)
        self.gallery.setFlow(QListView.Flow.LeftToRight)
        self.gallery.setWrapping(False)
        self.gallery.setResizeMode(QListView.ResizeMode.Adjust)
        self.gallery.setLayoutMode(QListView.LayoutMode.Batched)
        self.gallery.setMovement(QListView.Movement.Static)
        self.gallery.setUniformItemSizes(True)
        self.gallery.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.gallery.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.gallery.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.gallery.setIconSize(QSize(140, 76))
        item_height = 76 + self.gallery.fontMetrics().height() + 24
        self.gallery.setGridSize(QSize(160, item_height))
        scrollbar_height = max(self.gallery.horizontalScrollBar().sizeHint().height(),
                               self.gallery.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent))
        self.gallery.setMinimumHeight(item_height + 2 * self.gallery.frameWidth() + scrollbar_height + 4)
        self.gallery.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.gallery.selectionModel().currentChanged.connect(self.gallery_changed)
        gallery.addWidget(self.gallery)
        split.addWidget(bottom)
        split.setSizes([630, 158])
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        self.search_timer = QTimer(self, interval=180, singleShot=True)
        self.search_timer.timeout.connect(self.refresh_images)
        self.search.textChanged.connect(lambda: self.search_timer.start())

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(185)
        self.cancel = QPushButton("Cancel job")
        self.cancel.clicked.connect(self.cancel_job)
        self.progress.hide()
        self.cancel.hide()
        self.coordinates = QLabel()
        self.statusBar().addPermanentWidget(self.coordinates)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(self.cancel)
        keys = {
            "D": lambda: self.navigate(1), "A": self.previous_or_approve, "Ctrl+Right": self.next_unlabeled,
            "PgUp": lambda: self.navigate(-1), "PgDown": lambda: self.navigate(1),
            "F8": self.auto_next.click, "F1": self.show_help,
            "Ctrl+Z": lambda: self.undo(False), REDO_KEY: lambda: self.undo(True),
            "Ctrl+C": self.copy_object, "Ctrl+V": self.paste, "Ctrl+Shift+V": self.paste_previous,
            "Delete": self.delete_object, "F": lambda: self.canvas.fit(), "0": lambda: self.canvas.actual_size(),
            "L": lambda: self.toggle_canvas("show_names"),
            TOGGLE_ANNOTATIONS_KEY: lambda: self.toggle_canvas("show_objects"),
            "Return": self.enter, "Enter": self.enter, "N": self.mark_null,
            "X": lambda: self.review(False), "R": lambda: self.review(False) if self.review_mode else None,
            "E": lambda: self.set_tool("select"), "Ctrl+R": self.quick_predict,
            "Escape": self.canvas.reset_gesture,
        }
        if MACOS:
            keys["Backspace"] = self.delete_object
        keys.update({key: (lambda tool=tool: self.set_tool(tool)) for key, tool in
                     zip("HBPGK", ("select", "box", "point", "polygon", "keypoint"))})
        for direction, dx, dy in (("Left", -1, 0), ("Right", 1, 0), ("Up", 0, -1), ("Down", 0, 1)):
            keys[direction] = lambda x=dx, y=dy: self.nudge(x, y)
            keys["Shift+" + direction] = lambda x=dx, y=dy: self.nudge(10 * x, 10 * y)
        for key, callback in keys.items():
            self.shortcut(key, callback)
        for number in range(1, 10):
            self.shortcut(str(number), lambda hotkey=number: self.select_hotkey(hotkey))

    def empty_start(self):
        if not self.project:
            self.choose_project(True)
        elif self.search.text() or self.filter_status.currentIndex() or self.filter_class.currentIndex():
            self.reset_filters()
        else:
            self.add_images()

    def tool_icon(self, button, kind):
        from PySide6.QtGui import QPen, QPolygonF
        from PySide6.QtCore import QPointF
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#729eaa"), 2))
        if kind == "select":
            painter.drawPolygon(QPolygonF([QPointF(7, 4), QPointF(24, 17), QPointF(16, 19), QPointF(12, 27)]))
        elif kind == "box":
            painter.drawRect(6, 7, 20, 18)
        elif kind == "polygon":
            painter.drawPolygon(QPolygonF([QPointF(6, 23), QPointF(10, 6), QPointF(25, 10), QPointF(27, 25)]))
        elif kind == "point":
            painter.drawEllipse(12, 12, 8, 8)
        elif kind == "keypoint":
            painter.drawLine(8, 23, 16, 7)
            painter.drawLine(16, 7, 25, 22)
            for x, y in [(8, 23), (16, 7), (25, 22)]:
                painter.drawEllipse(QPointF(x, y), 3, 3)
        elif kind == "fit":
            for x, y, dx, dy in [(6, 6, 7, 7), (26, 6, -7, 7), (6, 26, 7, -7), (26, 26, -7, -7)]:
                painter.drawLine(x, y, x + dx, y)
                painter.drawLine(x, y, x, y + dy)
        else:
            for x in (8, 16, 24):
                painter.drawEllipse(QPointF(x, 16), 2, 2)
        painter.end()
        button.setAccessibleName(button.toolTip() or kind)
        button.setIcon(QIcon(pixmap))
        button.setIconSize(QSize(28, 28))
        button.setText("")
        button.setFixedSize(48, 44)

    def toggle_theme(self):
        self.apply_theme(not getattr(self, "light_mode", False))

    def apply_theme(self, light):
        self.light_mode = light
        style = STYLE
        if light:
            replacements = {"#10151c": "#f3f6fa", "#dde5ef": "#243448", "#171f2a": "#ffffff",
                            "#202b39": "#edf2f7", "#334155": "#c9d3df", "#111923": "#ffffff",
                            "#2b3747": "#ced8e3", "#8997aa": "#52647a", "#8c9cb0": "#586b80",
                            "#9cabbc": "#52647a", "#ecf4fc": "#20334a", "#2b3a4d": "#dbe7ef",
                            "#214b43": "#ceefe4", "#b7f5df": "#155a48", "#8ce8cb": "#16735b"}
            for old, new in replacements.items():
                style = style.replace(old, new)
        QApplication.instance().setStyleSheet(style)
        self.canvas.setBackgroundBrush(QColor("#e1e7ef" if light else "#0e1320"))
        self.theme_button.setText("☾" if light else "☀")
        QSettings("DLabel", "LabelTool").setValue("light_mode", light)

    def expand_list(self, kind):
        if not hasattr(self, "list_windows"):
            self.list_windows = {}
        if kind not in self.list_windows:
            source = self.classes if kind == "classes" else self.object_list
            dialog = QDialog(self, Qt.WindowType.Tool)
            dialog.setModal(False)
            layout = QVBoxLayout(dialog)
            hint = QLabel("Select a class to draw with." if kind == "classes" else
                          "Select an object to edit it on the image.")
            hint.setObjectName("muted")
            hint.setWordWrap(True)
            layout.addWidget(hint)
            listing = QListView()
            listing.setProperty("dense", True)
            listing.setModel(source.model())
            # Both views use the same model and selection, including after edits.
            listing.setSelectionModel(source.selectionModel())
            listing.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            listing.setTextElideMode(Qt.TextElideMode.ElideRight)
            layout.addWidget(listing, 1)
            if kind == "classes":
                listing.doubleClicked.connect(self.guarded(self.edit_class))
            source.selectionModel().currentChanged.connect(
                lambda index, _, view=listing: view.scrollTo(index) if index.isValid() and view.isVisible() else None)
            screen = self.screen().availableGeometry()
            dialog.resize(min(420, screen.width()), min(540, screen.height()))
            self.list_windows[kind] = dialog
        self.update_status()
        dialog = self.list_windows[kind]
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def toggle_filters(self):
        if not self.filters_button.isChecked():
            self.filter_panel.hide()
            return
        self.filter_panel.adjustSize()
        anchor = self.filters_button.mapToGlobal(self.filters_button.rect().bottomRight())
        screen = self.filters_button.screen().availableGeometry()
        width, height = self.filter_panel.width(), self.filter_panel.height()
        x = max(screen.left(), min(anchor.x() - width, screen.right() - width + 1))
        y = anchor.y() + 6
        if y + height > screen.bottom() + 1:
            y = self.filters_button.mapToGlobal(self.filters_button.rect().topLeft()).y() - height - 6
        y = max(screen.top(), min(y, screen.bottom() - height + 1))
        self.filter_panel.move(x, y)
        self.filter_panel.show()
        # Native window frames can offset the requested position on first show.
        frame = self.filter_panel.frameGeometry()
        dx = max(screen.left() - frame.left(), min(0, screen.right() - frame.right()))
        dy = max(screen.top() - frame.top(), min(0, screen.bottom() - frame.bottom()))
        self.filter_panel.move(self.filter_panel.x() + dx, self.filter_panel.y() + dy)
        self.filter_status.setFocus()

    def eventFilter(self, watched, event):
        if watched is getattr(self, "filter_panel", None) and event.type() == QEvent.Type.Hide:
            self.filters_button.setChecked(False)
        return super().eventFilter(watched, event)

    def reset_filters(self):
        self.clear_filters()
        self.refresh_images()

    def show_help(self):
        delete_key = "Delete (⌫)" if MACOS else "Del"
        QMessageBox.information(self, "Keyboard shortcuts", shortcut_label(
            "PgUp / PgDn or A / D: previous / next · Ctrl+→: next unlabeled\n"
            "F8: toggle auto-next after drawing a box · F1: show shortcuts\n"
            "H / B / P / G / K: select / box / point / polygon / keypoint\n"
            f"Shift+click: select in drawing tools · 1–9: class · {delete_key}: delete\n"
            "Arrow keys: move 1 px · Shift+arrows: 10 px\n"
            f"Ctrl+Z: undo · {REDO_KEY}: redo · Ctrl+C/V: copy/paste · Ctrl+Shift+V: previous labels\n"
            "Wheel: zoom · Space+drag or middle button: pan · F: fit · 0: actual size\n"
            "Enter: finish polygon / approve · X: reject · N: null · Ctrl+R: auto-label\n"
            f"L: toggle class names · {TOGGLE_ANNOTATIONS_KEY}: toggle annotations\n"
            "Review mode: A approve, R reject, E edit."))

    def choose_project(self, create):
        if self.job:
            return
        picker = QFileDialog.getSaveFileName if create else QFileDialog.getOpenFileName
        path, _ = picker(self, "New project" if create else "Open project", "", "Label Tool project (*.dlabel)")
        if path:
            if create and not path.lower().endswith(".dlabel"):
                path += ".dlabel"
            self.open_project(path, create)

    def open_project(self, path, create=False):
        project = Project(path, create=create)
        if self.project:
            self.project.close()
        self.project = project
        self.history.clear()
        self.clipboard = None
        self.preview = {}
        self.preview_button.hide()
        self.film.clear_cache()
        self.image = None
        self.review_mode = False
        self.review_action.setChecked(False)
        self.auto_next.setChecked(False)
        self.class_id = None
        self.search.clear()
        self.filter_status.setCurrentIndex(0)
        self.filter_class.setCurrentIndex(0)
        self.setWindowTitle(f"{project.path.stem} · Label Tool")
        self.refresh_classes()
        self.refresh_images()
        self.refresh_model()

    def _ready(self):
        if not self.project:
            self.statusBar().showMessage("Create or open a project first.", 4000)
            return False
        if self.job or self.preview:
            self.statusBar().showMessage("Finish the running job or exit preview first.", 4000)
            return False
        return True

    def editable(self):
        return self.project and self.image and self.loaded and not self.job and not self.preview

    def start_job(self, title, task, callback, cancellable=False, total_steps=None, progress_unit="images"):
        if self.job:
            return
        self.canvas.reset_gesture()
        self.canvas.editable = False
        self.progress.setRange(0, 0)
        self.progress.show()
        self.cancel.setEnabled(True)
        self.cancel.setVisible(cancellable)
        worker = Job(task, self)
        self.job = worker
        self.job_title = title
        self.job_progress = (0, total_steps) if total_steps is not None else None
        self.job_progress_unit = progress_unit
        self.show_job_status()
        self.job_result = None
        self.job_error = None
        worker.result_ready.connect(self.receive_result)
        worker.failed.connect(self.receive_failure)
        worker.progress.connect(self.show_progress)
        worker.finished.connect(lambda: self.finish_job(worker, callback))
        self.update_status()
        worker.start()

    def receive_result(self, result):
        self.job_result = result

    def receive_failure(self, message):
        self.job_error = message

    def show_progress(self, done, total):
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        if self.job_progress is not None:
            self.job_progress = (done, total)
            self.show_job_status()

    def show_job_status(self):
        message = self.job_title
        if self.job.cancelled.is_set():
            message = "Cancelling…" if self.job_progress_unit == "epochs" else "Cancelling after the current image…"
        if self.job_progress is not None:
            done, total = self.job_progress
            message += f"  {done:,}/{total:,} {self.job_progress_unit}"
        self.statusBar().showMessage(message)

    def finish_job(self, worker, callback):
        self.job = None
        self.canvas.editable = bool(self.editable())
        self.cancel.hide()
        self.progress.hide()
        worker.deleteLater()
        if self.job_error:
            self.update_status()
            QMessageBox.warning(self, "Label Tool", self.job_error)
        else:
            self.guarded(lambda: callback(self.job_result))()

    def cancel_job(self):
        if self.job:
            self.job.cancelled.set()
            self.cancel.setEnabled(False)
            self.show_job_status()

    def add_images(self):
        if not self._ready():
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "Add images", "", "Images (" +
            " ".join("*" + suffix for suffix in sorted(IMAGE_EXTENSIONS)) + ")")
        if paths:
            self.import_paths(paths)

    def add_folder(self):
        if not self._ready():
            return
        path = QFileDialog.getExistingDirectory(self, "Add images from folder")
        if path:
            self.import_paths(Path(path))

    def import_yolo(self):
        if not self._ready():
            return
        label_path, _ = QFileDialog.getOpenFileName(self, "Choose YOLO class list or data.yaml", "",
                                                    "Class list (*.txt *.yaml *.yml)")
        if not label_path:
            return
        folder = QFileDialog.getExistingDirectory(self, "Choose dataset or image folder", str(Path(label_path).parent))
        if folder:
            names = class_names(label_path)
            task = "detect"
            shape = None
            if Path(label_path).suffix.lower() in (".yml", ".yaml"):
                data = yaml.safe_load(Path(label_path).read_text(encoding="utf-8-sig"))
                if "kpt_shape" in data:
                    task, shape = "pose", data["kpt_shape"]
                    if not isinstance(shape, list) or len(shape) != 2 or not all(type(v) is int for v in shape) or shape[0] < 1 or shape[1] not in (2, 3):
                        raise ValueError("Invalid kpt_shape in dataset YAML.")
            self.import_paths(Path(folder), yolo_names=names, task=task, keypoint_shape=shape, mark_split=True)

    def import_paths(self, paths, **options):
        project_path = self.project.path

        def task(cancelled, progress):
            sources = sorted(p for p in paths.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS) if isinstance(paths, Path) else paths
            project = Project(project_path)
            try:
                return project.import_images(sources, **options)
            finally:
                project.close()

        def done(result):
            if "yolo_names" in options:
                self.history.clear()
            self.import_done(result)

        self.start_job("Importing images…", task, done)

    def import_done(self, result):
        count, errors = result
        self.refresh_classes()
        self.refresh_images()
        if errors:
            self.report(f"Imported {count} image(s). {len(errors)} error(s).", errors)
        else:
            self.statusBar().showMessage(f"Imported {count} image(s).", 5000)

    def import_video(self):
        if not self._ready():
            return
        from label_video import extract_frames
        path, _ = QFileDialog.getOpenFileName(self, "Choose local video", "", "Video (*.mp4 *.avi *.mov *.mkv)")
        if not path:
            return
        fps, ok = QInputDialog.getDouble(self, "Extract video frames", "Frames per second", 1.0, 0.01, 120, 2)
        if ok:
            project_path = self.project.path
            self.start_job("Extracting video frames…",
                           lambda cancel, progress: extract_frames(project_path, path, fps, cancel, progress),
                           self.import_done, cancellable=True)

    def refresh_classes(self):
        if not self.project:
            return
        rows = self.project.classes()
        self.class_rows = {row["id"]: row for row in rows}
        if self.class_id not in self.class_rows:
            self.class_id = rows[0]["id"] if rows else None
        filter_id = self.filter_class.currentData()
        self.classes.blockSignals(True)
        self.classes.clear()
        for row in rows:
            key = f"[{row['hotkey']}] " if row["hotkey"] else ""
            item = QListWidgetItem(f"{key}{row['name']}   ·   {row['count']}")
            item.setToolTip(item.text())
            item.setForeground(QColor(row["color"]))
            item.setData(Qt.ItemDataRole.UserRole, row["id"])
            self.classes.addItem(item)
            if row["id"] == self.class_id:
                self.classes.setCurrentItem(item)
        self.classes.blockSignals(False)
        self.filter_class.blockSignals(True)
        self.filter_class.clear()
        self.filter_class.addItem("All classes")
        for row in rows:
            self.filter_class.addItem(row["name"], row["id"])
        self.filter_class.setCurrentIndex(max(0, self.filter_class.findData(filter_id)))
        self.filter_class.blockSignals(False)
        self.canvas.active_class = self.class_id

    def select_class(self, item, _):
        if item:
            self.class_id = item.data(Qt.ItemDataRole.UserRole)
            self.canvas.active_class = self.class_id
            self.update_status()

    def select_hotkey(self, hotkey):
        for index, row in enumerate(self.class_rows.values()):
            if row["hotkey"] == hotkey:
                self.classes.setCurrentRow(index)
                break

    def add_class(self):
        if self._ready() and (values := class_editor(self)):
            name, color, _ = values
            self.class_id = self.project.add_class(name, color)
            self.refresh_classes()
            self.refresh_objects()

    def edit_class(self):
        if self._ready() and self.class_id is not None:
            row = self.class_rows[self.class_id]
            if values := class_editor(self, row):
                self.project.edit_class(self.class_id, *values)
                self.refresh_classes()
                self.refresh_objects()

    def import_class_list(self):
        if self._ready():
            path, _ = QFileDialog.getOpenFileName(self, "Import classes", "", "Classes (*.txt *.yaml *.yml)")
            if path:
                self.project.import_classes(class_names(path))
                self.refresh_classes()
                self.refresh_objects()

    def move_class(self, step):
        if self._ready() and self.class_id is not None:
            self.project.move_class(self.class_id, step)
            self.refresh_classes()
            self.refresh_objects()

    def remove_class(self):
        if not self._ready() or self.class_id is None:
            return
        alternatives = [row for row in self.class_rows.values() if row["id"] != self.class_id]
        choices = ["Delete class and its annotations", *[f"Merge into {row['name']}" for row in alternatives]]
        current = self.class_rows[self.class_id]["name"]
        choice, ok = QInputDialog.getItem(self, f"Remove {current}", "Apply to the whole project", choices, 0, False)
        if ok and QMessageBox.question(self, "Apply class change?", f"{current} → {choice}") == QMessageBox.StandardButton.Yes:
            index = choices.index(choice)
            self.project.remove_class(self.class_id, alternatives[index - 1]["id"] if index else None)
            self.history.clear()
            self.clipboard = None
            self.refresh_classes()
            self.refresh_images()

    def refresh_image_files(self):
        if self._ready():
            self.film.clear_cache()
            self.loaded = False
            self.refresh_images()

    def refresh_images(self, target=None):
        if not self.project:
            return
        if target is None and self.image:
            target = self.image["id"]
        status = self.filter_status.currentText() if self.filter_status.currentIndex() else None
        rows = self.project.images(status, self.search.text(), self.filter_class.currentData())
        if self.review_mode:
            scores = self.project.db.execute("""
                SELECT image_id, MIN(json_extract(payload, '$.score')) AS low FROM annotations
                WHERE json_extract(payload, '$.pending') = 1 GROUP BY image_id
            """).fetchall()
            by_id = {row["image_id"]: row["low"] for row in scores}
            rows.sort(key=lambda row: (by_id.get(row["id"]) is None, by_id.get(row["id"]) or 0))
        if self.preview:
            rows = [row for row in self.project.images() if row["id"] in self.preview]
        self._changing_rows = True
        self.film.set_rows(rows)
        index = next((i for i, row in enumerate(rows) if row["id"] == target), 0)
        if rows:
            model_index = self.film.index(index)
            self.gallery.setCurrentIndex(model_index)
            self.gallery.scrollTo(model_index)
        self._changing_rows = False
        self.load_image(rows[index] if rows else None)

    def gallery_changed(self, index, _):
        if not self._changing_rows and index.isValid():
            self.load_image(self.film.rows[index.row()])

    def load_image(self, row):
        same = row is not None and self.image is not None and row["id"] == self.image["id"]
        self.image = self.project.image(row["id"]) if row is not None else None
        if not same or not self.loaded:
            self.image_notice.hide()
            self.selected = ""
            self.loaded = False
            try:
                self.canvas.load(self.image["path"] if self.image else None)
                self.loaded = self.image is not None
            except (OSError, ValueError) as error:
                LOGGER.warning("Could not open image %s", self.image['path'], exc_info=True)
                self.image_notice.setText(f"Could not open image: {self.image['path']}\n"
                                          "Restore the file, then choose File → Refresh image files. "
                                          "Your saved labels are preserved.")
                self.image_notice.show()
        if self.image:
            self.objects = self.preview.get(self.image["id"], self.project.objects(self.image["id"]))
        else:
            self.objects = []
        self.canvas.editable = bool(self.editable())
        self.refresh_objects()

    def navigate(self, step):
        if not self.film.rows:
            return
        row = self.gallery.currentIndex().row()
        self.gallery.setCurrentIndex(self.film.index(max(0, min(row + step, len(self.film.rows) - 1))))
        self.gallery.scrollTo(self.gallery.currentIndex())

    def next_unlabeled(self):
        if not self.project or self.preview:
            return
        current = self.image["id"] if self.image else 0
        unlabeled = self.project.images("Unlabeled")
        if not unlabeled:
            self.statusBar().showMessage("No unlabeled images.", 4000)
            return
        target = next((row["id"] for row in unlabeled if row["id"] > current), unlabeled[0]["id"])
        self.clear_filters()
        self.refresh_images(target)

    def advance_after_box(self):
        index = self.gallery.currentIndex().row()
        rows = self.film.rows
        remaining = rows[index + 1:] + rows[:index]
        target = next((row["id"] for row in remaining if row["status"] == "Unlabeled"), None)
        if target is None:
            self.statusBar().showMessage("Box saved. No unlabeled images left in this view.", 4000)
        else:
            self.refresh_images(target)

    def clear_filters(self):
        self.review_mode = False
        self.review_action.setChecked(False)
        self.search.clear()
        self.filter_status.setCurrentIndex(0)
        self.filter_class.setCurrentIndex(0)

    def selected_images(self):
        return [self.film.rows[index.row()]["id"] for index in self.gallery.selectedIndexes()]

    def set_split(self):
        if not self._ready():
            return
        ids = self.selected_images()
        if not ids:
            return
        choice, ok = QInputDialog.getItem(self, "Assign split", f"{len(ids)} selected image(s)",
                                          ["Auto", "train", "val", "test"], 0, False)
        if ok:
            self.project.set_split(ids, None if choice == "Auto" else choice)
            self.refresh_images()

    def refresh_objects(self):
        self.object_list.blockSignals(True)
        self.object_list.clear()
        self.object_title.setText(f"OBJECTS  ·  {len(self.objects)}")
        if not any(obj.id == self.selected for obj in self.objects):
            self.selected = ""
        for obj in self.objects:
            row = self.class_rows[obj.class_id]
            score = f" · P {obj.score:.2f}" if obj.pending and obj.score is not None else ""
            joints = f" · {len(obj.keypoints)} joints" if obj.keypoints else ""
            flags = (" · hidden" if obj.id in self.canvas.hidden else "") + (" · locked" if obj.id in self.canvas.locked else "")
            item = QListWidgetItem(f"{row['name']} · {obj.kind}{score}{joints}{flags}")
            item.setToolTip(item.text())
            item.setForeground(QColor(row["color"]))
            item.setData(Qt.ItemDataRole.UserRole, obj.id)
            self.object_list.addItem(item)
            if obj.id == self.selected:
                self.object_list.setCurrentItem(item)
        self.object_list.blockSignals(False)
        self.canvas.set_objects(self.objects, self.class_rows, self.selected)
        self.update_status()

    def select_object(self, object_id):
        if self.selected != object_id:
            self.selected = object_id
            self.object_list.blockSignals(True)
            index = next((i for i, obj in enumerate(self.objects) if obj.id == object_id), -1)
            self.object_list.setCurrentRow(index)
            self.object_list.blockSignals(False)
            self.canvas.set_objects(self.objects, self.class_rows, self.selected)

        self.update_status()

    def update_status(self):
        if not hasattr(self, "film"):
            return
        for kind, dialog in getattr(self, "list_windows", {}).items():
            count = self.classes.count() if kind == "classes" else self.object_list.count()
            dialog.setWindowTitle(f"{'Classes' if kind == 'classes' else 'Objects'} · {count}")
        ready = bool(self.project and not self.job and not self.preview)
        self.add_images_button.setEnabled(ready)
        self.auto_label_button.setEnabled(ready and bool(self.film.rows))
        self.export_button.setEnabled(ready and bool(self.class_rows))
        self.review_action.setEnabled(bool(self.project) and not self.job and not self.preview)
        current = self.gallery.currentIndex().row()
        self.prev_button.setEnabled(current > 0)
        self.next_button.setEnabled(0 <= current < len(self.film.rows) - 1)
        self.image_counter.setText(f"{current + 1 if self.image else 0} / {len(self.film.rows)}")
        self.review_bar.setVisible(bool(self.image and any(item.pending for item in self.objects)) and not self.preview)
        self.review_bar.setEnabled(bool(self.editable()))
        self.canvas_stack.setCurrentIndex(0 if self.loaded else 1)
        self.empty_open.setVisible(not self.project)
        filtered = bool(self.search.text() or self.filter_status.currentIndex() or self.filter_class.currentIndex())
        self.empty_primary.setText("Create project" if not self.project else "Clear filters" if filtered else "Add images")
        self.empty_primary.setEnabled(not self.job and not self.preview)
        self.empty_description.setText(
            "Create a project, add your images, and start annotating." if not self.project else
            "No images match this view. Try clearing the filters." if filtered else
            "Add images to your project to start labeling." if not self.image else
            "This image is unavailable. Restore the file and refresh to continue.")
        count = int(self.filter_status.currentIndex() > 0) + int(self.filter_class.currentIndex() > 0)
        self.filters_button.setText(f"Filters & options · {count}" if count else "Filters & options")
        self.auto_next.setText("Auto-next box · On  F8" if self.auto_next.isChecked() else "Auto-next box  F8")
        obj = next((item for item in self.objects if item.id == self.selected), None)
        for name, button in getattr(self, "context_buttons", {}).items():
            enabled = bool(self.editable())
            if name != "This image":
                enabled = enabled and obj is not None
            if name in {"Accept", "Reject"}:
                enabled = enabled and bool(obj and obj.pending)
            if name == "Undo joint":
                enabled = enabled and bool(obj and obj.keypoints)
            button.setEnabled(enabled)
        if hasattr(self, "workflow_hint"):
            if not self.project:
                hint = "Start here: create a new project or open an existing .dlabel file."
            elif not self.film.rows:
                hint = "Add images to begin, or clear the filters to see your images."
            elif not self.class_rows:
                hint = "Add a class on the right, then draw a box with B. Changes save automatically."
            else:
                hint = "Changes save automatically     ·     B  Box     H  Select     1–9  Class     F1  All shortcuts"
            self.workflow_hint.setText(hint)
            self.review_action.setChecked(self.review_mode)
        if self.job:
            return
        if not self.image:
            self.statusBar().showMessage("No images in this view. Add images or change the filters.")
            return
        row = self.image
        cls = self.class_rows.get(self.class_id)
        prefix = "PREVIEW · " if self.preview else "REVIEW · " if self.review_mode else (
            "AUTO-NEXT · " if self.auto_next.isChecked() else ""
        )
        self.statusBar().showMessage(
            f"{prefix}{Path(row['path']).name}  ·  {self.gallery.currentIndex().row() + 1}/{len(self.film.rows)}"
            f"  ·  {row['width']}×{row['height']}  ·  {row['status']}  ·  {row['split'] or 'auto split'}"
            f"  ·  {self.canvas.tool.title()} / {cls['name'] if cls else 'add a class'}"
        )

    def commit(self, objects, status=None):
        if not self.editable():
            return
        image_id = self.image["id"]
        before = (self.objects, self.image["status"])
        self.project.save(image_id, objects, status)
        undo, redo = self.history[image_id]
        undo.append(before)
        del undo[:-100]
        redo.clear()
        self.reload_after_edit()

    def reload_after_edit(self):
        self.image = self.project.image(self.image["id"])
        self.objects = self.project.objects(self.image["id"])
        current = self.gallery.currentIndex()
        if current.isValid() and self.film.rows[current.row()]["id"] == self.image["id"]:
            self.film.rows[current.row()] = self.image
            self.film.dataChanged.emit(current, current)
        self.refresh_classes()
        self.refresh_objects()

    def undo(self, redo):
        if not self.editable():
            return
        undo_stack, redo_stack = self.history[self.image["id"]]
        source, target = (redo_stack, undo_stack) if redo else (undo_stack, redo_stack)
        if source:
            objects, status = source[-1]
            self.project.save(self.image["id"], objects, status)
            source.pop()
            target.append((self.objects, self.image["status"]))
            self.reload_after_edit()

    def guarded_with_item(self, item):
        self.guarded(lambda: self.save_object(item))()

    def save_object(self, item):
        if not self.editable():
            return
        objects = [item if obj.id == item.id else obj for obj in self.objects]
        new_object = not any(obj.id == item.id for obj in objects)
        if new_object:
            objects.append(item)
        self.selected = item.id
        self.commit(objects)
        if new_object and item.kind == "box" and self.auto_next.isChecked() and not self.review_mode:
            self.advance_after_box()

    def current_object(self, unlocked=False):
        if unlocked and self.selected in self.canvas.locked:
            return None
        return next((obj for obj in self.objects if obj.id == self.selected), None)

    def delete_object(self):
        if self.editable() and self.current_object(unlocked=True):
            self.commit([obj for obj in self.objects if obj.id != self.selected])

    def reclassify(self):
        if self.editable() and self.class_id is not None and (obj := self.current_object(unlocked=True)):
            if self.class_id != obj.class_id:
                self.save_object(replace(obj, class_id=self.class_id))

    def remove_joint(self):
        if self.editable() and (obj := self.current_object(unlocked=True)) and obj.keypoints:
            self.save_object(replace(obj, keypoints=obj.keypoints[:-1]))

    def toggle_object(self, attribute):
        if self.current_object():
            values = getattr(self.canvas, attribute)
            values.symmetric_difference_update([self.selected])
            self.refresh_objects()

    def toggle_canvas(self, attribute):
        setattr(self.canvas, attribute, not getattr(self.canvas, attribute))
        self.canvas.viewport().update()

    def copy_object(self):
        if self.image and not self.preview and (obj := self.current_object()):
            self.clipboard = ([obj], self.image["width"], self.image["height"])

    def paste_objects(self, source, width, height):
        if not self.editable():
            return
        sx, sy = self.image["width"] / width, self.image["height"] / height
        copied = [replace(obj.copy(),
                          points=tuple(v * (sy if i % 2 else sx) for i, v in enumerate(obj.points)),
                          keypoints=tuple((x * sx, y * sy, visible) for x, y, visible in obj.keypoints))
                  for obj in source if obj.class_id in self.class_rows]
        if copied:
            self.selected = copied[-1].id
            self.commit(self.objects + copied)

    def paste(self):
        if self.clipboard:
            self.paste_objects(*self.clipboard)

    def paste_previous(self):
        if self.editable():
            previous = self.project.db.execute("SELECT * FROM images WHERE id < ? ORDER BY id DESC LIMIT 1",
                                               (self.image["id"],)).fetchone()
            if previous:
                self.paste_objects(self.project.objects(previous["id"]), previous["width"], previous["height"])

    def nudge(self, x, y):
        if self.editable() and (obj := self.current_object(unlocked=True)):
            changed = moved(obj, x, y, self.image["width"], self.image["height"])
            if changed != obj:
                self.save_object(changed)

    def set_tool(self, tool):
        self.canvas.set_tool(tool)
        for key, button in self.tool_buttons.items():
            button.setChecked(tool == key)
        self.box_mode.setVisible(tool == "box")
        self.visibility.setVisible(tool == "keypoint")
        self.more_tools.setToolTip({"point": "Point · P", "keypoint": "Keypoint · K"}.get(tool, "More tools · Point / Keypoint"))
        self.tool_icon(self.more_tools, tool if tool in ("point", "keypoint") else "more")
        self.update_status()

    def box_mode_changed(self, index):
        self.canvas.two_click = bool(index)
        self.canvas.reset_gesture()

    def enter(self):
        if self.canvas.tool == "polygon" and self.canvas.vertices:
            self.canvas.finish_polygon()
        else:
            self.review(True)

    def mark_null(self):
        if not self.editable():
            return
        if self.objects and QMessageBox.question(self, "Mark null?", "Clear annotations and mark this image as background?") != QMessageBox.StandardButton.Yes:
            return
        if self.image["status"] != "Null":
            self.commit([], "Null")

    def review(self, accepted):
        if not self.editable() or not self.objects:
            return
        updated = approve(self.objects) if accepted else [obj for obj in self.objects if not obj.pending]
        if updated == self.objects and (not accepted or self.image["status"] == "Reviewed"):
            return
        self.commit(updated, "Reviewed" if updated else "Unlabeled")
        if self.review_mode:
            self.refresh_images()

    def review_object(self, accepted):
        if self.editable() and (obj := self.current_object(unlocked=True)) and obj.pending:
            updated = [replace(item, pending=False) if item.id == obj.id else item for item in self.objects
                       if accepted or item.id != obj.id]
            status = "Reviewed" if updated and not any(item.pending for item in updated) else None
            self.commit(updated, status)

    def toggle_review(self):
        if self.project and not self.preview:
            self.review_mode = not self.review_mode
            self.filter_status.setCurrentText("Predicted" if self.review_mode else "All statuses")
            self.refresh_images()
        self.review_action.setChecked(self.review_mode)

    def previous_or_approve(self):
        self.review(True) if self.review_mode else self.navigate(-1)

    def refresh_model(self):
        registered = self.project.models() if self.project else {}
        self.model_caption.setText("No model configured" if not registered else
                                   f"{Path(next(reversed(registered))).name} · {len(registered)} registered")

    def configure_model(self, your_models=False):
        if not self._ready():
            return
        registered = self.project.models()
        dialog = SourceDialog(self, registered, your_models=your_models)
        if not dialog.exec():
            return
        source, device, imgsz = dialog.values()
        if source == TEXT_MODEL:
            if not self.class_rows:
                self.report("Create a project class with + Class before using Auto-label from text.")
                return
            self.map_model(source, None, device, imgsz, registered.get(source, {}))
            return
        self.inspect_for_mapping(source, device, imgsz)

    def inspect_for_mapping(self, source, device, imgsz):
        image_path = self.image["path"] if self.image else None
        defaults = self.project.models().get(source, {})
        self.start_job("Inspecting model…",
                       lambda cancel, progress: inspect_model(source, image_path, device, imgsz),
                       lambda names: self.map_model(source, names, device, imgsz, defaults))

    def train_model(self):
        if not self._ready():
            return
        dataset, _ = QFileDialog.getOpenFileName(
            self, "Choose exported data.yaml", str(self.project.path.parent), "YOLO dataset (*.yaml *.yml)"
        )
        if not dataset:
            return
        dialog = TrainDialog(self)
        if not dialog.exec():
            return
        options = dialog.values()
        output = self.project.path.parent / "training-runs"
        self.start_job(
            "Training YOLO…",
            lambda cancel, progress: train_dataset(dataset, output, cancelled=cancel, progress=progress, **options),
            lambda result: self.training_done(result, options["imgsz"]),
            cancellable=True, total_steps=options["epochs"], progress_unit="epochs",
        )

    def training_done(self, result, imgsz):
        self.update_status()
        if not result or result["cancelled"]:
            message = "Training cancelled."
            if result:
                message += f"\nLog: {result['log']}"
            self.report(message)
            return
        source = str(result["best"].resolve())
        self.project.register_model(source, {"device": "cpu", "imgsz": imgsz})
        self.refresh_model()
        box = QMessageBox(self)
        box.setWindowTitle("Training complete")
        box.setText(f"Model saved to\n{result['best']}\n\nLog: {result['log']}")
        use_model = box.addButton("Use for Auto-label", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Close)
        box.exec()
        if box.clickedButton() == use_model:
            self.inspect_for_mapping(source, "cpu", imgsz)

    def map_model(self, source, names, device, imgsz, defaults):
        self.update_status()
        overrides, ok = QInputDialog.getText(
            self, "Model class names", "Optional: comma-separated names in model ID order (for models without metadata)"
        ) if Path(source).suffix.lower() in (".onnx", ".engine") and len(names) > 100 else ("", True)
        if not ok:
            return
        if overrides.strip():
            names = dict(enumerate(name.strip() for name in overrides.split(",") if name.strip()))
        selected = self.selected_images()
        dialog = MappingDialog(self, names, list(self.class_rows.values()), device, imgsz, defaults, len(selected),
                               active_class=self.class_id, your_model=Path(source).is_absolute())
        if not dialog.exec():
            return
        config = dialog.values()
        if dialog.settings_only:
            self.save_model_settings(source, config)
            self.statusBar().showMessage("Model settings saved.", 4000)
            return
        scope = dialog.scope.currentIndex()
        if scope == 0:
            ids = [self.image["id"]] if self.image else []
        elif scope == 1:
            ids = selected
        else:
            ids = [row["id"] for row in self.project.images("Unlabeled" if scope == 2 else None)]
        if not ids:
            self.report("No images match this scope.")
            return
        preview = dialog.preview.isChecked()
        if not preview:
            self.save_model_settings(source, config)
        self.run_predictions(source, config, ids[:5] if preview else ids, preview)

    def save_model_settings(self, source, config):
        pending_names = list(dict.fromkeys(value for value in config["mapping"].values() if isinstance(value, str)))
        created = dict(zip(pending_names, self.project.import_classes(pending_names)))
        for key in ("mapping", "class_targets"):
            if key in config:
                config[key] = {cid: created.get(value, value) for cid, value in config[key].items()}
        self.project.register_model(source, config)
        self.refresh_classes()
        self.refresh_model()

    def quick_predict(self):
        if not self._ready() or not self.image:
            return
        models = self.project.models()
        if not models:
            self.configure_model()
            return
        source = next(reversed(models))
        config = models[source]
        config["mapping"] = {int(cid): value for cid, value in config.get("mapping", {}).items()
                             if value in self.class_rows}
        if not config["mapping"]:
            if Path(source).is_absolute():
                self.inspect_for_mapping(source, config.get("device", "cpu"), config.get("imgsz", 640))
            else:
                self.configure_model()
            return
        self.run_predictions(source, config, [self.image["id"]], False)

    def run_predictions(self, source, config, ids, preview):
        project_path = self.project.path
        self.start_job("Running auto-label…",
                       lambda cancel, progress: batch(project_path, ids, source, config, preview, cancel, progress),
                       lambda result: self.predictions_done(result, preview, ids),
                       cancellable=True, total_steps=len(ids))

    def predictions_done(self, result, preview, ids):
        if preview:
            self.preview = result["previews"]
            self.preview_button.setVisible(bool(self.preview))
        elif result["changed"]:
            for image_id in ids:
                self.history.pop(image_id, None)
        self.refresh_classes()
        self.refresh_images()
        message = (f"{'Preview' if preview else 'Auto-label'} · {result['changed']} image(s) changed · "
                   f"{result['boxes']} boxes · {result['skipped']} skipped · {len(result['errors'])} errors")
        if result["cancelled"]:
            message += " · Cancelled"
        self.report(message, result["errors"])

    def clear_preview(self):
        self.preview = {}
        self.preview_button.hide()
        self.refresh_images()

    def export_dataset(self):
        if not self._ready():
            return
        dialog = ExportDialog(self)
        if not dialog.exec():
            return
        path, _ = QFileDialog.getSaveFileName(self, "Name a new export folder", str(
            self.project.path.parent / f"{self.project.path.stem}_yolo{dialog.version.currentText()}"
        ), "Dataset folder (*)")
        if not path:
            return
        options, project_path = dialog.values(), self.project.path

        def task(cancel, progress):
            project = Project(project_path)
            try:
                return export_yolo(project, path, progress=progress, **options)
            finally:
                project.close()

        def done(result):
            self.update_status()
            note = "" if result["train"] and result["val"] else "\nTraining needs non-empty train and validation splits."
            self.report(f"Exported to {path}\n" + "\n".join(f"{k}: {v}" for k, v in result.items()) + note)

        self.start_job("Exporting dataset…", task, done)

    def report(self, message, details=()):
        box = QMessageBox(self)
        box.setWindowTitle("Label Tool")
        box.setText(message)
        if details:
            box.setDetailedText("\n".join(details))
        box.exec()

    def closeEvent(self, event):
        if self.job:
            event.ignore()
            self.statusBar().showMessage("Wait for the job to finish before closing.")
            return
        if self.project:
            self.project.close()
        self.film.clear_cache()
        self.film.pool.waitForDone()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", nargs="?")
    args = parser.parse_args()
    app = QApplication([])
    app.setApplicationName("Label Tool")
    log_dir = Path(__file__).resolve().parent / "logs"
    try:
        log_dir.mkdir(exist_ok=True)
        handler = RotatingFileHandler(log_dir / "dlabel.log", maxBytes=2_000_000,
                                      backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        LOGGER.addHandler(handler)
        LOGGER.setLevel(logging.INFO)
    except OSError:
        LOGGER.addHandler(logging.NullHandler())
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = Window()
    window.apply_theme(QSettings("DLabel", "LabelTool").value("light_mode", False, type=bool))
    window.show()
    if args.project:
        window.guarded(lambda: window.open_project(args.project))()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

