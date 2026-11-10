"""
Light Board - the on-screen part of Hand Mouse (PySide6).

* a transparent, click-through window over the whole screen where the hand
  draws glowing neon lines (lightboard style), with smart shapes
* a laser pointer for presentation mode
* a small floating toolbar (modes, colours, undo, clear, board, screenshot,
  recording) that works with the real mouse or with the hand
* the camera preview window
* explainer recording: screen + drawings + webcam circle + microphone -> MP4

Started from hand_mouse.py; the hand tracking engine runs in a background thread
and sends events here through a queue.
"""

import datetime
import math
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import wave

if sys.platform.startswith("linux"):
    # OpenCV ships its own Qt plugins and points Qt at them; undo that for PySide6
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    os.environ.pop("QT_QPA_FONTDIR", None)
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")   # Wayland can't keep windows on top

import cv2
import numpy as np
from PySide6.QtCore import QEasingCurve, QPointF, QRectF, Qt, QTimer, QVariantAnimation
from PySide6.QtGui import (QColor, QFont, QGuiApplication, QImage, QPainter, QPainterPath,
                           QPen, QPixmap, QRadialGradient)
from PySide6.QtWidgets import QAbstractButton, QApplication, QHBoxLayout, QWidget

import shapes
from ui_kit import (ACCENT, AMBER, GREEN, NEON, RED, TEXT, TOOLTIP_CSS, ColorDot, Divider, Grip,
                    IconButton, paint_glass, paint_shadow, rounded_path, soft_glow, ui_font)

COLORS = NEON
LASER = QColor("#FF3B4E")
REC_FPS = 20
MAX_REC_WIDTH = 1920


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def _mss_cls(mss):
    return getattr(mss, "MSS", None) or mss.mss


def default_out_dir(arg):
    if arg:
        path = os.path.expanduser(arg)
    else:
        home = os.path.expanduser("~")
        pics = os.path.join(home, "Pictures")
        path = os.path.join(pics if os.path.isdir(pics) else home, "HandMouse")
    os.makedirs(path, exist_ok=True)
    return path


def safe_path(path):
    """OpenCV on Windows can't open non-English paths; use the short 8.3 form there."""
    if sys.platform != "win32" or path.isascii():
        return path
    import ctypes
    folder, name = os.path.split(path)
    buf = ctypes.create_unicode_buffer(1024)
    if ctypes.windll.kernel32.GetShortPathNameW(folder, buf, 1024):
        return os.path.join(buf.value, name)
    return path


def exclude_from_capture(widget):
    """Windows 10 2004+: keep the toolbar / preview out of screenshots and recordings."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.user32.SetWindowDisplayAffinity(int(widget.winId()), 0x11)
    except Exception:
        pass


def lighter(color, amount=0.6):
    c = QColor(color)
    return QColor(int(c.red() + (255 - c.red()) * amount),
                  int(c.green() + (255 - c.green()) * amount),
                  int(c.blue() + (255 - c.blue()) * amount))


GLOW_LAYERS = ((9.0, 7), (6.5, 14), (4.4, 28), (3.0, 56), (2.1, 120), (1.5, 255))


def glow_path(p, path, color, width):
    """Neon tube: soft additive halo, saturated body, near-white hot core."""
    p.setBrush(Qt.NoBrush)
    mode = p.compositionMode()
    p.setCompositionMode(QPainter.CompositionMode_Plus)
    for mult, alpha in GLOW_LAYERS[:-1]:
        c = QColor(color)
        c.setAlpha(alpha)
        p.setPen(QPen(c, width * mult, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawPath(path)
    p.setCompositionMode(mode)
    mult, alpha = GLOW_LAYERS[-1]
    c = QColor(color)
    c.setAlpha(alpha)
    p.setPen(QPen(c, width * mult, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    p.drawPath(path)
    p.setPen(QPen(lighter(color, 0.62), width * 0.45, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    p.drawPath(path)


def smooth_path(points):
    path = QPainterPath()
    if not points:
        return path
    path.moveTo(QPointF(*points[0]))
    if len(points) == 1:
        path.lineTo(QPointF(points[0][0] + 0.1, points[0][1]))
        return path
    for i in range(1, len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        path.quadTo(QPointF(x0, y0), QPointF((x0 + x1) / 2, (y0 + y1) / 2))
    path.lineTo(QPointF(*points[-1]))
    return path


def shape_path(shape):
    path = QPainterPath()
    for line, closed in zip(shape["lines"], shape["closed"]):
        path.moveTo(QPointF(*line[0]))
        for pt in line[1:]:
            path.lineTo(QPointF(*pt))
        if closed:
            path.closeSubpath()
    return path


class Stroke:
    def __init__(self, color, width):
        self.color = QColor(color)
        self.width = width
        self.points = []
        self.shape = None
        self.time = time.monotonic()

    def path(self):
        return shape_path(self.shape) if self.shape else smooth_path(self.points)

    def paint(self, p):
        glow_path(p, self.path(), self.color, self.width)


# --------------------------------------------------------------------------- #
# overlay
# --------------------------------------------------------------------------- #

class Overlay(QWidget):
    def __init__(self, app):
        flags = (Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
                 | Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus)
        super().__init__(None, flags)
        self.app = app
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        if sys.platform == "darwin":
            self.setAttribute(Qt.WA_MacAlwaysShowToolWindow)
        geo = QGuiApplication.primaryScreen().geometry()
        self.setGeometry(geo)

        self.strokes = []
        self.history = []        # undo actions
        self.current = None
        self.cache = None
        self.cursor = None       # (QPointF, pen_down, style)
        self.laser_trail = []
        self.board = False
        self.toast_text, self.toast_until, self.toast_start = "", 0.0, 0.0
        self.toast_color = QColor(ACCENT)
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._anim_tick)
        self.rebuild()

    # -- strokes ---------------------------------------------------------- #
    def rebuild(self):
        dpr = self.devicePixelRatioF()
        img = QImage(max(1, int(self.width() * dpr)), max(1, int(self.height() * dpr)),
                     QImage.Format_ARGB32_Premultiplied)
        img.setDevicePixelRatio(dpr)
        img.fill(Qt.transparent)
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing)
        for s in self.strokes:
            s.paint(p)
        p.end()
        self.cache = img
        self.update()

    def add_to_cache(self, stroke):
        p = QPainter(self.cache)
        p.setRenderHint(QPainter.Antialiasing)
        stroke.paint(p)
        p.end()

    def begin_stroke(self, pt):
        self.current = Stroke(self.app.color, self.app.pen_width)
        self.current.points.append((pt.x(), pt.y()))

    def extend_stroke(self, pt):
        if self.current is None:
            return
        lx, ly = self.current.points[-1]
        if math.hypot(pt.x() - lx, pt.y() - ly) >= 1.5:
            self.current.points.append((pt.x(), pt.y()))

    def end_stroke(self):
        s, self.current = self.current, None
        if s is None:
            return
        if len(s.points) < 2:
            return                       # a tap, not a line
        if self.app.shapes_on:
            s.shape = shapes.recognize(s.points)
            if s.shape is None and self.history and self.history[-1][0] == "add":
                prev = self.history[-1][1]
                if prev.shape and time.monotonic() - prev.time < 5:
                    arrow = shapes.merge_arrow_head(prev.shape, s.points)
                    if arrow:
                        self.history.append(("merge", prev, prev.shape, s))
                        prev.shape = arrow
                        self.rebuild()
                        return
        self.strokes.append(s)
        self.history.append(("add", s))
        self.add_to_cache(s)
        self.update()

    def undo(self):
        if self.current is not None:
            self.current = None
        if not self.history:
            return
        act = self.history[-1]
        if act[0] == "add":
            s = act[1]
            if s.shape is not None:      # first undo: back to what you drew by hand
                s.shape = None
            else:
                self.history.pop()
                if s in self.strokes:
                    self.strokes.remove(s)
        elif act[0] == "merge":
            self.history.pop()
            _, prev, old_shape, v = act
            prev.shape = old_shape
            v.shape = None
            self.strokes.append(v)
            self.history.append(("add", v))
        elif act[0] == "clear":
            self.history.pop()
            self.strokes = act[1]
        self.rebuild()
        self.app.refresh_visibility()

    def clear(self, note=True):
        self.current = None
        if self.strokes:
            self.history.append(("clear", self.strokes))
            self.strokes = []
            self.rebuild()
            if note:
                self.toast("נוקה (בטל מחזיר)")
        self.app.refresh_visibility()

    # -- cursor ------------------------------------------------------------ #
    def set_cursor(self, pt, pen, style):
        self.cursor = (pt, pen, style)
        if style == "laser":
            self.laser_trail.append(pt)
            self.laser_trail = self.laser_trail[-14:]
        else:
            self.laser_trail = []
        self.update()

    def hide_cursor(self):
        self.end_stroke()
        self.cursor = None
        self.laser_trail = []
        self.update()

    def toast(self, text, secs=1.8, color=None):
        self.toast_text = text
        self.toast_color = QColor(color) if color else QColor(self.app.color)
        self.toast_start = time.monotonic()
        self.toast_until = self.toast_start + secs
        self.app.refresh_visibility()
        if not self.anim_timer.isActive():
            self.anim_timer.start(16)
        self.update()

    def _anim_tick(self):
        if not (self.toast_text and time.monotonic() < self.toast_until + 0.05):
            self.anim_timer.stop()
        self.update()

    # -- painting ---------------------------------------------------------- #
    def paint_board(self, p):
        r = QRectF(self.rect())
        g = QRadialGradient(r.center(), max(r.width(), r.height()) * 0.75)
        g.setColorAt(0.0, QColor(10, 14, 28, 200))
        g.setColorAt(1.0, QColor(3, 5, 12, 236))
        p.fillRect(r, g)

    def paint_cursor(self, p):
        pt, pen, style = self.cursor
        if style == "laser":
            n = len(self.laser_trail)
            if n > 1:
                for i in range(1, n):
                    c = QColor(LASER)
                    k = i / n
                    c.setAlpha(int(150 * k * k))
                    p.setPen(QPen(c, 1.5 + 4.5 * k, Qt.SolidLine, Qt.RoundCap))
                    p.drawLine(self.laser_trail[i - 1], self.laser_trail[i])
            soft_glow(p, pt, LASER, 20, core=3.2)
            return
        col = QColor(self.app.color)
        if pen:
            soft_glow(p, pt, col, 16 * self.app.pen_width / 4, core=2.6)
        else:
            halo = QColor(col)
            halo.setAlpha(40)
            p.setPen(QPen(halo, 5))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(pt, 11, 11)
            ring = QColor(col)
            ring.setAlpha(235)
            p.setPen(QPen(ring, 1.6))
            p.drawEllipse(pt, 11, 11)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, 230))
            p.drawEllipse(pt, 1.8, 1.8)

    def paint_toast(self, p):
        now = time.monotonic()
        if not self.toast_text or now > self.toast_until:
            return
        t_in = min(1.0, (now - self.toast_start) / 0.22)
        t_out = min(1.0, max(0.0, (self.toast_until - now) / 0.35))
        a = min(t_in, t_out)
        ease = 1 - (1 - a) ** 3
        p.save()
        p.setOpacity(ease)
        p.setFont(ui_font(11.5, QFont.DemiBold))
        fm = p.fontMetrics()
        w = fm.horizontalAdvance(self.toast_text) + 58
        h = 42
        y = self.height() - 104 + (1 - ease) * 12
        r = QRectF((self.width() - w) / 2, y, w, h)
        paint_shadow(p, r, h / 2, strength=0.9, spread=14, offset=5)
        paint_glass(p, r, h / 2)
        dot = QPointF(r.right() - 22, r.center().y())
        soft_glow(p, dot, self.toast_color, 9, core=None)
        p.setPen(Qt.NoPen)
        p.setBrush(self.toast_color)
        p.drawEllipse(dot, 3.6, 3.6)
        p.setPen(TEXT)
        p.drawText(r.adjusted(18, 0, -36, 0), Qt.AlignCenter, self.toast_text)
        p.restore()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if self.board:
            self.paint_board(p)
        if self.cache is not None:
            p.drawImage(QRectF(self.rect()), self.cache)
        if self.current is not None:
            self.current.paint(p)
        if self.cursor is not None:
            self.paint_cursor(p)
        self.paint_toast(p)
        p.end()


# --------------------------------------------------------------------------- #
# toolbar
# --------------------------------------------------------------------------- #

class ModeSwitch(QWidget):
    """Segmented control with a highlight pill that slides between modes."""

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        self.buttons = {}
        for mode, icon, tip in (("mouse", "pointer", "עכבר"),
                                ("draw", "pen", "ציור על המסך  ·  Ctrl+Alt+D"),
                                ("present", "present", "מצגת  ·  Ctrl+Alt+T")):
            b = IconButton(icon, tip, self, checkable=True, flat_checked=True)
            b.clicked.connect(lambda _=False, m=mode: app.engine.set_mode(m))
            row.addWidget(b)
            self.buttons[mode] = b
        self._x = 0.0
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(260)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.valueChanged.connect(lambda v: (setattr(self, "_x", float(v)), self.update()))
        self.current = None
        self.set_mode(app.engine.mode, animate=False)

    def set_mode(self, mode, animate=True):
        self.current = mode
        for m, b in self.buttons.items():
            b.setChecked(m == mode)
        self.layout().activate()
        target = float(self.buttons[mode].x())
        if animate and self.isVisible():
            self._anim.stop()
            self._anim.setStartValue(self._x)
            self._anim.setEndValue(target)
            self._anim.start()
        else:
            self._x = target
            self.update()

    def showEvent(self, e):
        super().showEvent(e)
        self._x = float(self.buttons[self.current].x())

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        track = QRectF(self.rect()).adjusted(0, 1, 0, -1)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 255, 255, 14))
        p.drawRoundedRect(track, 12, 12)
        b = self.buttons[self.current]
        r = QRectF(self._x + 2, 3, b.width() - 4, self.height() - 6)
        glow = QColor(ACCENT)
        glow.setAlpha(40)
        p.setBrush(glow)
        p.drawRoundedRect(r.adjusted(-1, -1, 1, 1), 11, 11)
        p.setBrush(QColor(ACCENT.red(), ACCENT.green(), ACCENT.blue(), 52))
        p.setPen(QPen(QColor(ACCENT.red(), ACCENT.green(), ACCENT.blue(), 150), 1))
        p.drawRoundedRect(r, 10, 10)
        p.end()


class Toolbar(QWidget):
    PAD = 20          # transparent margin for the shadow

    def __init__(self, app):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
                         | Qt.WindowDoesNotAcceptFocus)
        self.app = app
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        if sys.platform == "darwin":
            self.setAttribute(Qt.WA_MacAlwaysShowToolWindow)
        self._drag = None
        self.hot = None
        self.anchor_center = None

        row = QHBoxLayout(self)
        row.setContentsMargins(self.PAD + 8, self.PAD + 6, self.PAD + 8, self.PAD + 6)
        row.setSpacing(3)

        def btn(icon, tip, slot, checkable=False, accent=None):
            b = IconButton(icon, tip, self, checkable=checkable, accent=accent)
            b.clicked.connect(slot)
            row.addWidget(b)
            return b

        self.grip = Grip(self)
        row.addWidget(self.grip)
        self.modes = ModeSwitch(app, self)
        row.addWidget(self.modes)

        # drawing tools: shown in draw and present modes
        self.draw_group = [Divider(self)]
        row.addWidget(self.draw_group[0])
        self.swatches = []
        for c in COLORS:
            d = ColorDot(c, "צבע  ·  Ctrl+Alt+K", self)
            d.clicked.connect(lambda _=False, col=c: app.set_color(col))
            row.addWidget(d)
            self.swatches.append(d)
            self.draw_group.append(d)
        self.shapes_btn = btn("shapes", "צורות חכמות", app.toggle_shapes, checkable=True)
        self.shapes_btn.setChecked(app.shapes_on)
        self.board_btn = btn("board", "לוח כהה  ·  Ctrl+Alt+B", app.toggle_board, checkable=True)
        self.draw_group += [self.shapes_btn, self.board_btn]

        row.addWidget(Divider(self))
        btn("undo", "ביטול  ·  Ctrl+Alt+Z", app.overlay_undo)
        btn("trash", "ניקוי הציור  ·  Ctrl+Alt+C", app.overlay_clear)
        row.addWidget(Divider(self))
        btn("camera", "צילום מסך  ·  Ctrl+Alt+S", app.screenshot)
        self.rec_btn = btn("record", "הקלטת הסבר  ·  Ctrl+Alt+R", app.toggle_recording, accent=RED)
        self.rec_btn.icon_color = RED
        row.addWidget(Divider(self))
        self.preview_btn = btn("video", "חלון מצלמה  ·  Ctrl+Alt+P", app.toggle_preview, checkable=True)
        self.preview_btn.setChecked(app.engine.show_preview)
        self.pause_btn = btn("pause", "השהיה  ·  Ctrl+Alt+M", app.engine.toggle)
        btn("close", "יציאה  ·  Ctrl+Alt+Q", app.engine.quit)

        self.update_swatches()
        self.apply_mode(app.engine.mode, animate=False)
        geo = QGuiApplication.primaryScreen().availableGeometry()
        self.anchor_center = (geo.x() + geo.width() // 2, geo.y() + 14 - self.PAD)
        self.reposition()

    # -- layout ------------------------------------------------------------- #
    def reposition(self):
        self.adjustSize()
        cx, top = self.anchor_center
        self.move(cx - self.width() // 2, top)

    def apply_mode(self, mode, animate=True):
        show = mode != "mouse"
        for w in self.draw_group:
            w.setVisible(show)
        self.modes.set_mode(mode, animate)
        if self.anchor_center:
            self.reposition()

    def set_mode(self, mode):
        self.apply_mode(mode)

    def update_swatches(self):
        for d, c in zip(self.swatches, COLORS):
            d.setChecked(c == self.app.color)

    def set_recording(self, on, secs=0):
        b = self.rec_btn
        if on:
            b.icon = "stop"
            b.text_label = f"{secs // 60:02d}:{secs % 60:02d}"
            b.setChecked(True)
            if b.width() != 86:
                b.setFixedSize(86, 36)
                self.reposition()
        else:
            b.icon = "record"
            b.text_label = ""
            b.setChecked(False)
            if b.width() != 36:
                b.setFixedSize(36, 36)
                self.reposition()
        b.update()

    def set_paused(self, paused):
        self.pause_btn.icon = "play" if paused else "pause"
        self.pause_btn.setToolTip("המשך  ·  Ctrl+Alt+M" if paused else "השהיה  ·  Ctrl+Alt+M")
        self.pause_btn.update()

    # -- hand interaction ---------------------------------------------------- #
    def button_at(self, global_pt):
        if not self.isVisible() or not self.frameGeometry().contains(global_pt):
            return None
        w = self.childAt(self.mapFromGlobal(global_pt))
        while w is not None and not isinstance(w, QAbstractButton):
            w = w.parentWidget()
        return w

    def set_hot(self, btn):
        if btn is self.hot:
            return
        if self.hot is not None:
            self.hot.set_hot(False)
        if btn is not None:
            btn.set_hot(True)
        self.hot = btn

    # -- look ---------------------------------------------------------------- #
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(self.PAD, self.PAD, -self.PAD, -self.PAD)
        radius = 20
        paint_shadow(p, r, radius, strength=1.0, spread=18, offset=7)
        paint_glass(p, r, radius)
        p.end()

    # drag the toolbar by its background or grip
    def mousePressEvent(self, e):
        self._drag = e.globalPosition().toPoint() - self.pos()
        self.grip.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        if self._drag is not None:
            self.move(e.globalPosition().toPoint() - self._drag)
            self.anchor_center = (self.x() + self.width() // 2, self.y())

    def mouseReleaseEvent(self, _):
        self._drag = None
        self.grip.setCursor(Qt.OpenHandCursor)


# --------------------------------------------------------------------------- #
# preview
# --------------------------------------------------------------------------- #

class Preview(QWidget):
    """Small rounded camera window with a status chip. Drag to move."""
    PAD = 16

    def __init__(self, app):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.app = app
        self.setWindowTitle("Hand Mouse")
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        if sys.platform == "darwin":
            self.setAttribute(Qt.WA_MacAlwaysShowToolWindow)
        self.pix = None
        self._drag = None
        w = app.engine.args.preview_width
        self.img_w, self.img_h = w, int(w * 0.75)
        self.resize(self.img_w + 2 * self.PAD, self.img_h + 2 * self.PAD)
        geo = QGuiApplication.primaryScreen().availableGeometry()
        self.move(geo.right() - self.width() - 8, geo.bottom() - self.height() - 8)

    def show_frame(self, bgr):
        h, w = bgr.shape[:2]
        img = QImage(bgr.data, w, h, bgr.strides[0], QImage.Format_BGR888).copy()
        self.pix = QPixmap.fromImage(img)
        if (w, h) != (self.img_w, self.img_h):
            self.img_w, self.img_h = w, h
            self.resize(w + 2 * self.PAD, h + 2 * self.PAD)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        r = QRectF(self.PAD, self.PAD, self.img_w, self.img_h)
        radius = 18
        paint_shadow(p, r, radius, strength=1.0, spread=14, offset=6)
        p.save()
        p.setClipPath(rounded_path(r, radius))
        if self.pix is not None:
            p.drawPixmap(r, self.pix, QRectF(self.pix.rect()))
        else:
            p.fillRect(r, QColor(12, 14, 22))
        p.restore()
        p.setPen(QPen(QColor(255, 255, 255, 46), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)

        eng = self.app.engine
        names = {"mouse": "עכבר", "draw": "ציור", "present": "מצגת"}
        text = names.get(eng.mode, eng.mode) + ("" if eng.active else "  ·  מושהה")
        p.setFont(ui_font(9.5, QFont.DemiBold))
        fm = p.fontMetrics()
        cw = fm.horizontalAdvance(text) + 34
        chip = QRectF(r.left() + 10, r.top() + 10, cw, 26)
        paint_glass(p, chip, 13)
        dot = QPointF(chip.right() - 13, chip.center().y())
        dc = GREEN if eng.active else AMBER
        soft_glow(p, dot, dc, 8)
        p.setPen(Qt.NoPen)
        p.setBrush(dc)
        p.drawEllipse(dot, 3.2, 3.2)
        p.setPen(TEXT)
        p.drawText(chip.adjusted(10, 0, -22, 0), Qt.AlignCenter, text)
        p.end()

    def mousePressEvent(self, e):
        self._drag = e.globalPosition().toPoint() - self.pos()

    def mouseMoveEvent(self, e):
        if self._drag is not None:
            self.move(e.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, _):
        self._drag = None

    def keyPressEvent(self, e):
        k = e.key()
        eng = self.app.engine
        if k in (Qt.Key_Q, Qt.Key_Escape):
            eng.quit()
        elif k == Qt.Key_Space:
            eng.toggle()
        elif k == Qt.Key_H:
            eng.show_preview = False
        elif k == Qt.Key_D:
            eng.set_mode("mouse" if eng.mode == "draw" else "draw")
        elif k == Qt.Key_T:
            eng.set_mode("mouse" if eng.mode == "present" else "present")


# --------------------------------------------------------------------------- #
# explainer recording
# --------------------------------------------------------------------------- #

class Recorder:
    def __init__(self, engine, out_dir, mic=True):
        self.engine, self.out_dir, self.mic = engine, out_dir, mic
        self.stop_flag = threading.Event()
        self.thread = None
        self.t0 = 0.0
        self.audio_chunks = []
        self.audio_rate = 44100
        self.audio_stream = None
        self.error = None
        self.result = None       # final path once done
        self.status = ""

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self):
        try:
            import mss  # noqa: F401
        except ImportError:
            self.error = "חסרה ספריית mss:  pip install mss"
            return False
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.final = os.path.join(self.out_dir, f"explainer-{stamp}.mp4")
        self.tmpdir = tempfile.mkdtemp(prefix="handmouse-")
        self.video_tmp = os.path.join(self.tmpdir, "video.mp4")
        self.audio_tmp = os.path.join(self.tmpdir, "audio.wav")
        self.stop_flag.clear()
        self.audio_chunks = []
        self.t0 = time.monotonic()
        if self.mic:
            self._start_audio()
        self.thread = threading.Thread(target=self._video_loop, daemon=True)
        self.thread.start()
        return True

    def _start_audio(self):
        try:
            import sounddevice as sd
            info = sd.query_devices(kind="input")
            self.audio_rate = int(info["default_samplerate"]) or 44100

            def cb(indata, frames, t, status):
                self.audio_chunks.append(indata.copy())

            self.audio_stream = sd.InputStream(samplerate=self.audio_rate, channels=1,
                                               dtype="int16", callback=cb)
            self.audio_stream.start()
        except Exception as e:
            print(f"Microphone unavailable, recording without sound ({e})")
            self.audio_stream = None

    def _pip(self, frame):
        with self.engine.frame_lock:
            cam = None if self.engine.latest_raw is None else self.engine.latest_raw.copy()
        if cam is None:
            return
        H, W = frame.shape[:2]
        d = int(H * 0.26)
        ch, cw = cam.shape[:2]
        s = min(ch, cw)
        sq = cam[(ch - s) // 2:(ch + s) // 2, (cw - s) // 2:(cw + s) // 2]
        sq = cv2.resize(sq, (d, d), interpolation=cv2.INTER_AREA)
        x0, y0 = W - d - 28, H - d - 28
        mask = np.zeros((d, d), np.uint8)
        cv2.circle(mask, (d // 2, d // 2), d // 2 - 2, 255, -1, cv2.LINE_AA)
        roi = frame[y0:y0 + d, x0:x0 + d]
        m = (mask.astype(np.float32) / 255.0)[..., None]
        roi[:] = (sq * m + roi * (1 - m)).astype(np.uint8)
        c = (x0 + d // 2, y0 + d // 2)
        cv2.circle(frame, c, d // 2 + 2, (255, 229, 0), 6, cv2.LINE_AA)    # cyan ring (BGR)
        cv2.circle(frame, c, d // 2 + 2, (255, 250, 200), 2, cv2.LINE_AA)

    def _video_loop(self):
        import mss
        writer = None
        try:
            with _mss_cls(mss)() as sct:
                mon = sct.monitors[1]
                w, h = mon["width"], mon["height"]
                scale = min(1.0, MAX_REC_WIDTH / w)
                W, H = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
                writer = cv2.VideoWriter(safe_path(self.video_tmp),
                                         cv2.VideoWriter_fourcc(*"mp4v"), REC_FPS, (W, H))
                if not writer.isOpened():
                    raise RuntimeError("could not open the video writer")
                n = 0
                while not self.stop_flag.is_set():
                    shot = np.asarray(sct.grab(mon))[:, :, :3]
                    frame = cv2.resize(shot, (W, H), interpolation=cv2.INTER_AREA)
                    frame = np.ascontiguousarray(frame)
                    self._pip(frame)
                    target = int((time.monotonic() - self.t0) * REC_FPS) + 1
                    while n < target:            # keep real time even if capture is slow
                        writer.write(frame)
                        n += 1
                    wait = self.t0 + n / REC_FPS - time.monotonic()
                    if wait > 0:
                        time.sleep(wait)
        except Exception as e:
            self.error = f"ההקלטה נעצרה: {e}"
        finally:
            if writer is not None:
                writer.release()

    def stop(self):
        """Stops capture and finishes the file in the background."""
        self.stop_flag.set()
        if self.audio_stream is not None:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except Exception:
                pass
        threading.Thread(target=self._finish, daemon=True).start()

    def _finish(self):
        if self.thread is not None:
            self.thread.join()
        self.status = "working"
        have_audio = bool(self.audio_chunks)
        if have_audio:
            data = np.concatenate(self.audio_chunks)
            with wave.open(self.audio_tmp, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.audio_rate)
                wf.writeframes(data.tobytes())
        ffmpeg = None
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = shutil.which("ffmpeg")
        try:
            if ffmpeg and os.path.exists(self.video_tmp):
                cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", self.video_tmp]
                if have_audio:
                    cmd += ["-i", self.audio_tmp, "-c:a", "aac", "-b:a", "128k", "-shortest"]
                cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", self.final]
                subprocess.run(cmd, check=True, capture_output=True)
            else:
                shutil.move(self.video_tmp, self.final)
                if have_audio:
                    shutil.move(self.audio_tmp, self.final[:-4] + ".wav")
            self.result = self.final
        except Exception as e:
            try:  # keep at least the silent video
                shutil.move(self.video_tmp, self.final)
                self.result = self.final
            except Exception:
                self.error = f"שמירת ההקלטה נכשלה: {e}"
        finally:
            shutil.rmtree(self.tmpdir, ignore_errors=True)
            self.status = "done"


# --------------------------------------------------------------------------- #
# app
# --------------------------------------------------------------------------- #

class LightBoardApp:
    def __init__(self, engine):
        self.engine = engine
        self.events = queue.Queue()
        engine.ui = self.events
        args = engine.args
        self.color = COLORS[0]
        self.pen_width = 4.0 * args.pen_width
        self.shapes_on = not args.no_shapes
        self.out_dir = default_out_dir(args.out_dir)
        self.recorder = None
        self.rec_started = 0.0
        self.preview_hidden_for_rec = False
        self.pen_on_button = False

        self.overlay = Overlay(self)
        self.toolbar = Toolbar(self)
        self.preview = Preview(self)
        self.toolbar.show()
        exclude_from_capture(self.toolbar)
        exclude_from_capture(self.preview)
        self.refresh_visibility()

        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(15)

    # -- actions --------------------------------------------------------- #
    def set_color(self, c):
        self.color = c
        self.toolbar.update_swatches()
        self.overlay.update()

    def next_color(self):
        i = (COLORS.index(self.color) + 1) % len(COLORS)
        self.set_color(COLORS[i])

    def toggle_shapes(self):
        self.shapes_on = not self.shapes_on
        self.toolbar.shapes_btn.setChecked(self.shapes_on)
        self.overlay.toast("צורות חכמות: פועל" if self.shapes_on else "צורות חכמות: כבוי", color=ACCENT)

    def toggle_preview(self):
        self.engine.show_preview = not self.engine.show_preview
        self.toolbar.preview_btn.setChecked(self.engine.show_preview)

    def toggle_board(self):
        self.overlay.board = not self.overlay.board
        self.toolbar.board_btn.setChecked(self.overlay.board)
        self.refresh_visibility()
        self.overlay.update()

    def overlay_undo(self):
        self.overlay.undo()

    def overlay_clear(self):
        self.overlay.clear()

    def refresh_visibility(self):
        """The overlay is shown only when there is something on it."""
        o = self.overlay
        need = (self.engine.mode != "mouse" or bool(o.strokes) or o.board
                or (o.toast_text and time.monotonic() < o.toast_until))
        if need and not o.isVisible():
            o.show()
            self.toolbar.raise_()
            if self.preview.isVisible():
                self.preview.raise_()
        elif not need and o.isVisible():
            o.hide()

    def screenshot(self):
        try:
            import mss
        except ImportError:
            self.overlay.toast("חסרה ספריית mss")
            return
        self.toolbar.hide()
        prev_visible = self.preview.isVisible()
        self.preview.hide()
        cursor, self.overlay.cursor = self.overlay.cursor, None
        self.overlay.update()

        def grab():
            try:
                with _mss_cls(mss)() as sct:
                    shot = np.asarray(sct.grab(sct.monitors[1]))[:, :, :3]
                stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                path = os.path.join(self.out_dir, f"drawing-{stamp}.png")
                ok, buf = cv2.imencode(".png", shot)
                with open(path, "wb") as f:
                    f.write(buf.tobytes())
                print(f"Saved {path}")
                msg = "נשמר בתיקייה HandMouse"
            except Exception as e:
                msg = f"הצילום נכשל: {e}"
            self.toolbar.show()
            if prev_visible:
                self.preview.show()
            self.overlay.cursor = cursor
            self.overlay.toast(msg)

        QTimer.singleShot(180, grab)

    def toggle_recording(self):
        if self.recorder is not None and self.recorder.running:
            self.recorder.stop()
            self.toolbar.set_recording(False)
            self.overlay.toast("שומר את ההקלטה...", 2.5)
            if self.preview_hidden_for_rec:
                self.engine.show_preview = True
                self.preview_hidden_for_rec = False
            return
        if self.recorder is not None and self.recorder.status == "working":
            self.overlay.toast("עוד שומר את ההקלטה הקודמת")
            return
        self.recorder = Recorder(self.engine, self.out_dir, mic=not self.engine.args.no_mic)
        if not self.recorder.start():
            self.overlay.toast(self.recorder.error)
            return
        self.rec_started = time.monotonic()
        self.toolbar.set_recording(True, 0)
        if self.engine.show_preview and sys.platform != "win32":
            self.engine.show_preview = False   # the camera circle is in the video anyway
            self.preview_hidden_for_rec = True

    # -- events from the engine -------------------------------------------- #
    def to_overlay(self, nx, ny):
        return QPointF(nx * self.overlay.width(), ny * self.overlay.height())

    def handle_cursor(self, nx, ny, pen, style):
        o = self.overlay
        pt = self.to_overlay(nx, ny)
        gpt = (o.geometry().topLeft() + pt.toPoint())
        btn = self.toolbar.button_at(gpt) if style == "pen" else None
        self.toolbar.set_hot(btn)
        if pen:
            if o.current is None and not self.pen_on_button:
                if btn is not None:              # "press" a toolbar button with the hand
                    self.pen_on_button = True
                    btn.click()
                else:
                    o.begin_stroke(pt)
            elif o.current is not None:
                o.extend_stroke(pt)
        else:
            self.pen_on_button = False
            if o.current is not None:
                o.end_stroke()
                self.refresh_visibility()
        o.set_cursor(pt, pen, style)

    def handle(self, ev):
        kind = ev[0]
        o = self.overlay
        if kind == "cursor":
            self.handle_cursor(*ev[1:])
        elif kind == "hide":
            o.hide_cursor()
            self.toolbar.set_hot(None)
            self.pen_on_button = False
            self.refresh_visibility()
        elif kind == "undo":
            o.undo()
        elif kind == "clear":
            o.clear()
        elif kind == "color_next":
            self.next_color()
        elif kind == "board":
            self.toggle_board()
        elif kind == "screenshot":
            self.screenshot()
        elif kind == "record":
            self.toggle_recording()
        elif kind == "slide":
            o.clear(note=False)
        elif kind == "mode":
            self.toolbar.set_mode(ev[1])
            o.hide_cursor()
            self.refresh_visibility()
            o.toast({"mouse": "מצב עכבר", "draw": "מצב ציור", "present": "מצב מצגת"}[ev[1]],
                    color=ACCENT)
            self.preview.update()
        elif kind == "active":
            self.toolbar.set_paused(not ev[1])
            o.toast("פעיל" if ev[1] else "מושהה", color=GREEN if ev[1] else AMBER)
            self.preview.update()

    def tick(self):
        for _ in range(200):
            try:
                ev = self.events.get_nowait()
            except queue.Empty:
                break
            try:
                self.handle(ev)
            except Exception as e:
                print(f"UI event {ev[0]} failed: {e}")
        eng = self.engine
        if self.toolbar.preview_btn.isChecked() != eng.show_preview:
            self.toolbar.preview_btn.setChecked(eng.show_preview)
        if eng.show_preview:
            with eng.frame_lock:
                frame = eng.latest_preview
            if frame is not None:
                self.preview.show_frame(frame)
                if not self.preview.isVisible():
                    self.preview.show()
        elif self.preview.isVisible():
            self.preview.hide()
        self.refresh_visibility()
        r = self.recorder
        if r is not None:
            if r.running:
                self.toolbar.set_recording(True, int(time.monotonic() - self.rec_started))
            if r.error:
                self.overlay.toast(r.error, 3)
                self.toolbar.set_recording(False)
                r.error = None
            if r.status == "done":
                r.status = "reported"
                if r.result:
                    print(f"Saved {r.result}")
                    self.overlay.toast("ההקלטה נשמרה בתיקייה HandMouse", 3)
        if not eng.running:
            QApplication.instance().quit()


def run(engine):
    app = QApplication.instance() or QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setFont(ui_font(10))
    app.setStyleSheet(TOOLTIP_CSS)
    engine.preview_style = "modern"
    board = LightBoardApp(engine)
    hk = engine.start_hotkeys()
    worker = threading.Thread(target=engine.run_worker, daemon=True)
    worker.start()
    signal.signal(signal.SIGINT, lambda *_: engine.quit())
    print(f"Screen {engine.sw}x{engine.sh}. Mode: {engine.mode}. "
          f"Saving to {board.out_dir}. Ctrl+Alt+Q quits.")
    code = app.exec()
    engine.running = False
    if board.recorder is not None and board.recorder.running:
        board.recorder.stop()
        while board.recorder.status != "done":
            time.sleep(0.1)
    worker.join(timeout=3)
    if hk is not None:
        hk.stop()
    print("Bye.")
    return code
