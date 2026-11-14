"""Pixel-coordinate editor for boxes, points, polygons, and box keypoints."""
from dataclasses import replace

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView

from label_project import Annotation, open_image


def pixmap_for(path, thumbnail=False):
    image = open_image(path)
    if thumbnail:
        image.thumbnail((140, 76))
    w, h = image.size
    return QPixmap.fromImage(QImage(image.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888).copy())


def moved(item, dx, dy, width, height):
    x1, y1, x2, y2 = item.bounds()
    for x, y, visible in item.keypoints:
        if visible:
            x1, y1, x2, y2 = min(x, x1), min(y, y1), max(x, x2), max(y, y2)
    dx = min(max(dx, -x1), width - x2)
    dy = min(max(dy, -y1), height - y2)
    return replace(item, points=tuple(v + (dy if i % 2 else dx) for i, v in enumerate(item.points)),
                   keypoints=tuple((x + dx, y + dy, v) if v else (0, 0, 0)
                                   for x, y, v in item.keypoints))


class Canvas(QGraphicsView):
    changed = Signal(object)
    selected = Signal(str)
    cursor_at = Signal(str)

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(QColor("#0e1320"))
        self.setMouseTracking(True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(400, 300)
        self.image_rect = QRectF()
        self.items_data = []
        self.classes = {}
        self.active_class = None
        self.selected_id = ""
        self.tool = "box"
        self.two_click = False
        self.show_names = True
        self.show_objects = True
        self.hidden = set()
        self.locked = set()
        self.editable = True
        self.visibility = 2
        self.fitting = True
        self.space = False
        self.pan_at = None
        self.hover = None
        self.reset_gesture()

    def reset_gesture(self):
        self.start = None
        self.preview = None
        self.original = None
        self.handle = None
        self.vertices = []
        self.viewport().update()

    def load(self, path=None):
        self.reset_gesture()
        self.hidden.clear()
        self.locked.clear()
        self.hover = None
        self.scene().clear()
        self.image_rect = QRectF()
        if path:
            pixmap = pixmap_for(path)
            self.scene().addPixmap(pixmap)
            self.image_rect = QRectF(pixmap.rect())
        self.scene().setSceneRect(self.image_rect)
        self.fit()

    def fit(self):
        self.fitting = True
        self.resetTransform()
        if not self.image_rect.isEmpty():
            self.fitInView(self.image_rect, Qt.AspectRatioMode.KeepAspectRatio)

    def actual_size(self):
        self.fitting = False
        self.resetTransform()
        self.centerOn(self.image_rect.center())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.fitting:
            self.fit()

    def wheelEvent(self, event):
        if self.image_rect.isEmpty():
            return
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        if 0.025 <= self.transform().m11() * factor <= 24:
            self.fitting = False
            self.scale(factor, factor)
        event.accept()

    def set_tool(self, tool):
        self.tool = tool
        self.reset_gesture()

    def set_objects(self, objects, classes, selected_id):
        self.items_data = objects
        self.classes = classes
        self.selected_id = selected_id
        self.viewport().update()

    def selected_object(self):
        return next((obj for obj in self.items_data if obj.id == self.selected_id), None)

    def _unit(self):
        return 1 / self.transform().m11()

    def _point(self, event):
        point = self.mapToScene(event.position().toPoint())
        return QPointF(max(0, min(point.x(), self.image_rect.width())),
                       max(0, min(point.y(), self.image_rect.height())))

    @staticmethod
    def handles(item):
        x1, y1, x2, y2 = item.bounds()
        xs, ys = (x1, (x1 + x2) / 2, x2), (y1, (y1 + y2) / 2, y2)
        return {(i, j): QPointF(x, y) for i, x in enumerate(xs) for j, y in enumerate(ys)
                if (i, j) != (1, 1)}

    def _hit(self, point):
        tol = 8 * self._unit()
        selected = self.selected_object() if self.show_objects else None
        if selected and selected.id not in self.hidden and selected.kind == "box":
            for handle, location in self.handles(selected).items():
                if (point - location).manhattanLength() <= tol:
                    return selected, handle
        for item in reversed(self.items_data if self.show_objects else []):
            if item.id in self.hidden:
                continue
            if item.kind == "point":
                if (point - QPointF(*item.points)).manhattanLength() <= tol:
                    return item, None
            elif item.kind == "polygon":
                if self._polygon(item.points).containsPoint(point, Qt.FillRule.OddEvenFill):
                    return item, None
            else:
                x1, y1, x2, y2 = item.bounds()
                if QRectF(x1, y1, x2 - x1, y2 - y1).contains(point):
                    return item, None
        return None, None

    def mousePressEvent(self, event):
        self.setFocus()
        if event.button() == Qt.MouseButton.MiddleButton or (self.space and event.button() == Qt.MouseButton.LeftButton):
            self.pan_at = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if event.button() == Qt.MouseButton.RightButton:
            self.reset_gesture()
            return
        if event.button() != Qt.MouseButton.LeftButton or self.image_rect.isEmpty():
            return
        raw = self.mapToScene(event.position().toPoint())
        if not self.image_rect.contains(raw):
            return
        point = self._point(event)
        if self.tool == "select" or event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            hit, handle = self._hit(point)
            self.selected.emit(hit.id if hit else "")
            if hit and self.editable and hit.id not in self.locked:
                self.start, self.original, self.handle = point, hit, handle
            elif self.tool == "select" and not hit:
                self.pan_at = event.position()
            return
        if not self.editable:
            return
        if self.tool == "keypoint":
            item = self.selected_object()
            if item and item.kind == "box" and item.id not in self.locked:
                joint = (point.x(), point.y(), self.visibility) if self.visibility else (0, 0, 0)
                self.changed.emit(replace(item, keypoints=(*item.keypoints, joint)))
        elif self.active_class is not None:
            if self.tool == "point":
                self.changed.emit(Annotation(self.active_class, "point", (point.x(), point.y())).copy())
            elif self.tool == "polygon":
                if len(self.vertices) >= 3 and (point - self.vertices[0]).manhattanLength() < 10 * self._unit():
                    self.finish_polygon()
                else:
                    self.vertices.append(point)
                    self.viewport().update()
            elif self.tool == "box":
                if self.two_click and self.start is not None:
                    self._box_preview(point)
                    self._commit()
                else:
                    self.start = point

    def _box_preview(self, point):
        x1, x2 = sorted((self.start.x(), point.x()))
        y1, y2 = sorted((self.start.y(), point.y()))
        self.preview = Annotation(self.active_class, "box", (x1, y1, x2, y2))

    def mouseMoveEvent(self, event):
        if self.pan_at is not None:
            delta = event.position() - self.pan_at
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - round(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - round(delta.y()))
            self.pan_at = event.position()
            return
        point = self._point(event)
        self.hover = point
        self.cursor_at.emit(f"x {point.x():.0f} · y {point.y():.0f}")
        if self.start is not None:
            if self.original is None:
                self._box_preview(point)
            elif self.handle is None:
                delta = point - self.start
                self.preview = moved(self.original, delta.x(), delta.y(), self.image_rect.width(), self.image_rect.height())
            else:
                x1, y1, x2, y2 = self.original.bounds()
                i, j = self.handle
                x1 = min(point.x(), x2 - 2) if i == 0 else x1
                x2 = max(point.x(), x1 + 2) if i == 2 else x2
                y1 = min(point.y(), y2 - 2) if j == 0 else y1
                y2 = max(point.y(), y1 + 2) if j == 2 else y2
                self.preview = replace(self.original, points=(
                    max(0, x1), max(0, y1), min(self.image_rect.width(), x2), min(self.image_rect.height(), y2),
                ))
        self.viewport().update()

    def mouseReleaseEvent(self, event):
        if self.pan_at is not None:
            self.pan_at = None
            self.unsetCursor()
        elif event.button() == Qt.MouseButton.LeftButton and self.start is not None:
            if self.original is not None:
                self._commit()
            elif not self.two_click:
                self._box_preview(self._point(event))
                self._commit()

    def _commit(self):
        item, original = self.preview, self.original
        self.reset_gesture()
        if item and item.kind == "box":
            x1, y1, x2, y2 = item.bounds()
            if x2 - x1 < 2 or y2 - y1 < 2:
                return
        if item is not None and item != original:
            self.changed.emit(item if item.id else item.copy())

    def finish_polygon(self):
        if not self.editable or len(set((p.x(), p.y()) for p in self.vertices)) < 3 or self.active_class is None:
            return False
        points = tuple(v for p in self.vertices for v in (p.x(), p.y()))
        vertices = list(zip(points[::2], points[1::2]))
        area = abs(sum(x * yy - y * xx for (x, y), (xx, yy) in zip(vertices, vertices[1:] + vertices[:1]))) / 2
        if area < 1:
            return False
        self.reset_gesture()
        self.changed.emit(Annotation(self.active_class, "polygon", points).copy())
        return True

    def leaveEvent(self, event):
        self.hover = None
        self.viewport().update()
        super().leaveEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.space = True
            event.accept()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.space = False
            event.accept()
        else:
            super().keyReleaseEvent(event)

    def focusOutEvent(self, event):
        self.space = False
        super().focusOutEvent(event)

    @staticmethod
    def _polygon(points):
        return QPolygonF([QPointF(x, y) for x, y in zip(points[::2], points[1::2])])

    def _draw_item(self, painter, item, unit):
        row = self.classes.get(item.class_id)
        color = QColor(row["color"] if row else "#ffffff")
        selected = item.id == self.selected_id
        pen = QPen(color)
        pen.setCosmetic(True)
        pen.setWidth(3 if selected else 2)
        if item.pending:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(34 if item.kind != "point" else 255)
        painter.setBrush(fill)
        x1, y1, x2, y2 = item.bounds()
        if item.kind == "box":
            painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
        elif item.kind == "point":
            painter.drawEllipse(QPointF(x1, y1), 5 * unit, 5 * unit)
        else:
            painter.drawPolygon(self._polygon(item.points))
        for index, (x, y, visibility) in enumerate(item.keypoints, 1):
            if visibility:
                painter.setBrush(color if visibility == 2 else QColor("#161c29"))
                painter.drawEllipse(QPointF(x, y), 4 * unit, 4 * unit)
                self._badge(painter, f"{index}", x + 5 * unit, y, unit, color)
        if selected and item.kind == "box" and item.id not in self.locked:
            painter.setBrush(QColor("#f3efff"))
            for handle in self.handles(item).values():
                painter.drawRect(QRectF(handle.x() - 3 * unit, handle.y() - 3 * unit, 6 * unit, 6 * unit))
        if row and self.show_names:
            suffix = f"  {item.score:.2f}" if item.pending and item.score is not None else ""
            self._badge(painter, row["name"] + suffix, x1, y1 - 19 * unit, unit, color)

    @staticmethod
    def _badge(painter, text, x, y, unit, color):
        painter.save()
        painter.translate(x, max(0, y))
        painter.scale(unit, unit)
        font = painter.font()
        font.setPixelSize(12)
        painter.setFont(font)
        width = painter.fontMetrics().horizontalAdvance(text) + 10
        painter.fillRect(QRectF(0, 0, width, 18), color)
        painter.setPen(QColor("#101421"))
        painter.drawText(QPointF(5, 13), text)
        painter.restore()

    def drawForeground(self, painter, rect):
        if self.image_rect.isEmpty():
            painter.setPen(QColor("#8d9bb6"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Open a project and select an image")
            return
        painter.save()
        unit = self._unit()
        if self.show_objects:
            for item in self.items_data:
                if item.id not in self.hidden and (self.preview is None or self.preview.id != item.id):
                    self._draw_item(painter, item, unit)
        if self.preview:
            self._draw_item(painter, self.preview, unit)
        if self.vertices:
            pen = QPen(QColor("#ab91ff"), 2)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPolyline(QPolygonF(self.vertices + ([self.hover] if self.hover else [])))
            for point in self.vertices:
                painter.drawEllipse(point, 3 * unit, 3 * unit)
        if self.hover is not None and self.tool in ("box", "point", "keypoint", "polygon"):
            x, y = self.hover.x(), self.hover.y()
            pen = QPen(QColor(255, 255, 255, 100))
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawLine(QPointF(x, 0), QPointF(x, self.image_rect.height()))
            painter.drawLine(QPointF(0, y), QPointF(self.image_rect.width(), y))
        painter.restore()

