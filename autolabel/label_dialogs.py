"""Dialogs for project classes, inference mapping, and export options."""
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from label_export import PROFILES
from label_predict import OFFICIAL, TEXT_MODEL


class Form(QDialog):
    def __init__(self, parent, title):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.layout = QVBoxLayout(self)
        self.form = QFormLayout()
        self.layout.addLayout(self.form)
        self.setMinimumWidth(440)

    def buttons(self, label="OK"):
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(label)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout.addWidget(buttons)
        return buttons

    def choice(self, name, items):
        widget = QComboBox()
        widget.addItems(items)
        self.form.addRow(name, widget)
        return widget

    def number(self, name, value, minimum, maximum, precision=0):
        widget = QDoubleSpinBox() if precision else QSpinBox()
        if precision:
            widget.setDecimals(precision)
            widget.setSingleStep(0.05)
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        self.form.addRow(name, widget)
        return widget

    def check(self, text, checked=False):
        widget = QCheckBox(text)
        widget.setChecked(checked)
        self.form.addRow(widget)
        return widget


def class_editor(parent, row=None):
    dialog = Form(parent, "Edit class" if row else "Add class")
    name = QLineEdit(row["name"] if row else "")
    dialog.form.addRow("Name", name)
    color = [row["color"] if row else None]
    choose = QPushButton(color[0] or "Automatic")
    dialog.form.addRow("Color", choose)

    def pick():
        picked = QColorDialog.getColor(QColor(color[0] or "#ab91ff"), dialog)
        if picked.isValid():
            color[0] = picked.name()
            choose.setText(color[0])

    choose.clicked.connect(pick)
    hotkey = None
    if row:
        hotkey = dialog.number("Shortcut (0 = none)", row["hotkey"] or 0, 0, 9)
    dialog.buttons("Save")
    name.setFocus()
    if dialog.exec():
        return name.text(), color[0], (hotkey.value() or None) if hotkey else None
    return None


class SourceDialog(Form):
    def __init__(self, parent, registered, your_models=False):
        super().__init__(parent, "Model Manager")
        self.registered = registered
        self.mode = self.choice("Source", ["Official YOLO", "Your Models", "Registered model",
                                           "Text prompt (YOLO-World)"])
        self.family = self.choice("Version", list(OFFICIAL))
        self.family.setCurrentText("YOLO11")
        self.size = self.choice("Size", [])
        self.saved = self.choice("Registered", list(registered))
        self.local_models = self.choice("Your Models", [])
        self.local_models.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.local_models.setMinimumContentsLength(28)
        self.local_models.addItem("Choose or add a model…", "")
        for source in reversed(registered):
            path = Path(source)
            if path.is_absolute():
                self.local_models.addItem(f"{path.name} — {path.parent}", source)
        self.path = QLineEdit()
        browse = QPushButton("Add model…")
        browse.clicked.connect(self.browse)
        file_row = QWidget()
        layout = QHBoxLayout(file_row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.path)
        layout.addWidget(browse)
        self.form.addRow("Model file", file_row)
        self.device = QLineEdit("cpu")
        self.form.addRow("Device (cpu, 0, mps)", self.device)
        self.imgsz = self.number("Image size", 640, 32, 4096)
        self.family.currentTextChanged.connect(self.sizes)
        self.mode.currentTextChanged.connect(self.toggle)
        self.saved.currentTextChanged.connect(self.restore)
        self.local_models.currentIndexChanged.connect(self.select_local)
        self.file_row = file_row
        if your_models:
            self.mode.setCurrentIndex(1)
            if self.local_models.count() > 1:
                self.local_models.setCurrentIndex(1)
        self.toggle()
        self.sizes()
        self.buttons("Continue")

    def browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Add your model", "", "YOLO models (*.pt *.onnx *.engine)")
        if path:
            source = str(Path(path).resolve())
            self.local_models.setCurrentIndex(max(0, self.local_models.findData(source)))
            self.path.setText(source)

    def select_local(self):
        source = self.local_models.currentData()
        self.path.setText(source)
        self.restore(source)

    def restore(self, source):
        cfg = self.registered.get(source, {})
        self.device.setText(cfg.get("device", "cpu"))
        self.imgsz.setValue(cfg.get("imgsz", 640))

    def toggle(self):
        mode = self.mode.currentIndex()
        for widget, source_mode in ((self.family, 0), (self.size, 0), (self.file_row, 1),
                                    (self.local_models, 1), (self.saved, 2)):
            self.form.setRowVisible(widget, mode == source_mode)
        if mode == 2:
            self.restore(self.saved.currentText())

    def sizes(self):
        self.size.clear()
        self.size.addItems(list(OFFICIAL[self.family.currentText()][1]))

    def values(self):
        if self.mode.currentIndex() == 0:
            source = OFFICIAL[self.family.currentText()][0].format(self.size.currentText())
        elif self.mode.currentIndex() == 1:
            path = self.path.text().strip()
            source = str(Path(path).resolve()) if path else ""
        elif self.mode.currentIndex() == 2:
            source = self.saved.currentText()
        else:
            source = TEXT_MODEL
        return source, self.device.text().strip(), self.imgsz.value()

    def accept(self):
        if not self.values()[0]:
            QMessageBox.warning(self, "Model Manager", "Choose a model first.")
            return
        super().accept()


class MappingDialog(Form):
    def __init__(self, parent, model_names, classes, device, imgsz, defaults, selected_count,
                 active_class=None, your_model=False):
        title = "Model Options" if your_model else "Auto-label setup"
        super().__init__(parent, "Auto-label from text" if model_names is None else title)
        self.text_mode = model_names is None
        self.your_model = your_model
        self.settings_only = False
        self.device, self.imgsz = device, imgsz
        self.conf = self.number("Confidence", defaults.get("conf", 0.35), 0.01, 1, 2)
        self.iou = self.number("NMS IoU", defaults.get("iou", 0.50), 0.01, 1, 2)
        self.min_box = self.number("Minimum box side (px)", defaults.get("min_box", 2), 2, 9999)
        self.policy = self.choice("Existing annotations", ["Skip", "Replace", "Merge"])
        self.policy.setCurrentText(defaults.get("policy", "Skip"))
        self.merge_iou = self.number("Duplicate IoU", defaults.get("merge_iou", 0.60), 0.01, 1, 2)
        self.scope = self.choice("Run on", ["Current image", f"Selected images ({selected_count})",
                                            "Unlabeled images", "All images"])
        self.preview = self.check("Preview up to 5 images without saving")
        self.rows = {}
        if self.text_mode:
            self.setup_prompts(classes, defaults, active_class)
        elif your_model:
            self.setup_model_options(model_names, classes, defaults)
        else:
            self.setup_mapping(model_names, classes, defaults)
        buttons = self.buttons("Run")
        if your_model:
            save = buttons.addButton("Save settings", QDialogButtonBox.ButtonRole.ActionRole)
            save.clicked.connect(self.save_settings)
        self.resize(640, 650)

    def setup_model_options(self, model_names, classes, defaults):
        self.class_ids = {row["name"].casefold(): row["id"] for row in classes}
        class_names = {row["id"]: row["name"] for row in classes}
        self.class_checks = {}
        options = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter classes")
        options.addWidget(self.search, 1)
        self.class_count = QLabel()
        options.addWidget(self.class_count)
        for label, enabled in (("Select All", True), ("Deselect All", False)):
            button = QPushButton(label)
            button.clicked.connect(lambda _, checked=enabled: self.set_classes_checked(checked))
            options.addWidget(button)
        self.layout.addLayout(options)
        self.table = QTableWidget(len(model_names), 3)
        self.table.setHorizontalHeaderLabels(["Model class", "Enabled", "Label as (editable)"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.layout.addWidget(self.table)
        active = defaults.get("mapping")
        saved = defaults.get("class_targets", active or {})
        for index, (cid, name) in enumerate(sorted(model_names.items())):
            item = QTableWidgetItem(f"{cid} · {name}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(index, 0, item)
            enabled = QCheckBox()
            enabled.setChecked(active is None or str(cid) in active)
            enabled.toggled.connect(self.update_class_count)
            self.table.setCellWidget(index, 1, enabled)
            self.class_checks[cid] = enabled
            choice = QComboBox()
            choice.setEditable(True)
            choice.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
            choice.addItems(list(class_names.values()))
            target = saved.get(str(cid), name)
            choice.setEditText(class_names.get(target, name) if isinstance(target, int) else target)
            enabled.toggled.connect(choice.setEnabled)
            choice.setEnabled(enabled.isChecked())
            self.table.setCellWidget(index, 2, choice)
            self.rows[cid] = choice
        self.search.textChanged.connect(self.filter_rows)
        self.update_class_count()

    def set_classes_checked(self, checked):
        for checkbox in self.class_checks.values():
            checkbox.setChecked(checked)

    def update_class_count(self):
        count = sum(checkbox.isChecked() for checkbox in self.class_checks.values())
        self.class_count.setText(f"{count} selected")

    def setup_mapping(self, model_names, classes, defaults):
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search model classes")
        self.layout.addWidget(self.search)
        self.table = QTableWidget(len(model_names), 2)
        self.table.setHorizontalHeaderLabels(["Model class", "Project class"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.layout.addWidget(self.table)
        by_name = {row["name"].casefold(): row["id"] for row in classes}
        saved = defaults.get("mapping", {})
        for index, (cid, name) in enumerate(sorted(model_names.items())):
            item = QTableWidgetItem(f"{cid} · {name}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(index, 0, item)
            choices = QComboBox()
            choices.addItem("Skip", None)
            choices.addItem(f"Create “{name}”", name)
            for row in classes:
                choices.addItem(row["name"], row["id"])
            suggested = saved.get(str(cid)) if saved else by_name.get(name.casefold())
            if suggested is not None:
                choices.setCurrentIndex(max(0, choices.findData(suggested)))
            self.table.setCellWidget(index, 1, choices)
            self.rows[cid] = choices
        self.search.textChanged.connect(self.filter_rows)

    def setup_prompts(self, classes, defaults, active_class):
        note = QLabel("Enter a short English name for each object to find, such as rainbow. "
                      "Leave a row blank to skip that class. First use downloads model weights.")
        note.setWordWrap(True)
        self.layout.addWidget(note)
        self.table = QTableWidget(len(classes), 2)
        self.table.setHorizontalHeaderLabels(["Project class", "What to find (text prompt)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.layout.addWidget(self.table)
        saved = {
            class_id: defaults["prompts"][int(prompt_id)]
            for prompt_id, class_id in defaults.get("mapping", {}).items()
        } if "prompts" in defaults else {}
        for index, row in enumerate(classes):
            class_id = row["id"]
            item = QTableWidgetItem(row["name"])
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(index, 0, item)
            prompt = QLineEdit(saved.get(class_id, ""))
            if not saved and class_id == active_class and row["name"].isascii():
                prompt.setText(row["name"])
            prompt.setPlaceholderText("e.g. rainbow")
            self.table.setCellWidget(index, 1, prompt)
            self.rows[class_id] = prompt

    def filter_rows(self, query):
        for index in range(self.table.rowCount()):
            self.table.setRowHidden(index, query.casefold() not in self.table.item(index, 0).text().casefold())

    def values(self):
        config = dict(device=self.device, imgsz=self.imgsz, conf=self.conf.value(), iou=self.iou.value(),
                      min_box=self.min_box.value(), policy=self.policy.currentText(), merge_iou=self.merge_iou.value())
        if self.text_mode:
            pairs = [(cid, field.text().strip()) for cid, field in self.rows.items() if field.text().strip()]
            config["prompts"] = [prompt for _, prompt in pairs]
            config["mapping"] = {index: cid for index, (cid, _) in enumerate(pairs)}
        elif self.your_model:
            targets = {cid: choice.currentText().strip() for cid, choice in self.rows.items()}
            config["class_targets"] = {
                cid: self.class_ids.get(name.casefold(), name) for cid, name in targets.items() if name
            }
            config["mapping"] = {cid: config["class_targets"].get(cid) for cid, checkbox in self.class_checks.items()
                                 if checkbox.isChecked()}
        else:
            config["mapping"] = {cid: choice.currentData() for cid, choice in self.rows.items()
                                 if choice.currentData() is not None}
        return config

    def accept(self):
        config = self.values()
        if self.your_model and not self.validate_targets(config):
            return
        if not config["mapping"]:
            message = "Enter at least one text prompt." if self.text_mode else "Map at least one model class to a project class."
            QMessageBox.warning(self, "Auto-label", message)
            return
        prompts = config.get("prompts", ())
        if len({prompt.casefold() for prompt in prompts}) != len(prompts):
            QMessageBox.warning(self, "Auto-label", "Use a different text prompt for each project class.")
            return
        if self.preview.isChecked() and any(isinstance(v, str) for v in config["mapping"].values()):
            QMessageBox.warning(self, "Auto-label", "Preview uses existing classes. Add the destination classes first.")
            return
        super().accept()

    def validate_targets(self, config):
        if None in config["mapping"].values():
            QMessageBox.warning(self, "Model Options", "Enter a label name for each enabled class.")
            return False
        return True

    def save_settings(self):
        if self.validate_targets(self.values()):
            self.settings_only = True
            super().accept()


class TrainDialog(Form):
    def __init__(self, parent):
        super().__init__(parent, "Train Dataset")
        self.weights = QLineEdit("yolo11n.pt")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self.browse_weights)
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.weights)
        layout.addWidget(browse)
        self.form.addRow("Starting weights", row)
        self.epochs = self.number("Epochs", 50, 1, 1000)
        self.imgsz = self.number("Image size", 640, 64, 4096)
        self.imgsz.setSingleStep(32)
        self.device = self.choice("Device", ["CPU", "GPU 0 (CUDA)"])
        self.buttons("Start training")

    def browse_weights(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose starting weights", "", "YOLO weights (*.pt)")
        if path:
            self.weights.setText(path)

    def accept(self):
        weights = self.weights.text().strip()
        if weights != "yolo11n.pt" and not (Path(weights).is_file() and Path(weights).suffix.lower() == ".pt"):
            QMessageBox.warning(self, "Train Dataset", "Choose an existing YOLO .pt file or use yolo11n.pt.")
            return
        super().accept()

    def values(self):
        weights = self.weights.text().strip()
        model = "yolo11n.pt" if weights == "yolo11n.pt" else str(Path(weights).resolve())
        return dict(model=model, epochs=self.epochs.value(), imgsz=self.imgsz.value(),
                    device="cpu" if self.device.currentIndex() == 0 else "0")


class ExportDialog(Form):
    def __init__(self, parent):
        super().__init__(parent, "Export YOLO dataset")
        self.version = self.choice("Version", list(PROFILES))
        self.version.setCurrentText("v11")
        self.task = self.choice("Task", [])
        self.version.currentTextChanged.connect(self.tasks)
        self.tasks()
        self.ratios = [self.number(name, value, 0, 100) for name, value in
                       (("Train weight", 70), ("Validation weight", 20), ("Test weight", 10))]
        self.seed = self.number("Random seed", 42, 0, 2_147_483_647)
        self.points = self.choice("Free points", ["Skip", "Fixed-size box (detect)", "Separate CSV"])
        self.side = self.number("Point box side (px)", 16, 1, 10000)
        self.null = self.check("Include null images", True)
        self.reviewed = self.check("Reviewed images only")
        self.archive = self.check("Also create ZIP")
        note = QLabel("Saved split assignments take priority. Pending predictions are excluded. "
                      "Detect export uses the bounding rectangle for polygons.")
        note.setWordWrap(True)
        self.layout.addWidget(note)
        self.buttons("Choose destination")

    def tasks(self):
        self.task.clear()
        self.task.addItems(PROFILES[self.version.currentText()]["tasks"])

    def values(self):
        return dict(version=self.version.currentText(), task=self.task.currentText(),
                    ratios=[widget.value() for widget in self.ratios], seed=self.seed.value(),
                    point_mode=("skip", "box", "csv")[self.points.currentIndex()], point_size=self.side.value(),
                    include_null=self.null.isChecked(), reviewed_only=self.reviewed.isChecked(),
                    zip_output=self.archive.isChecked())

