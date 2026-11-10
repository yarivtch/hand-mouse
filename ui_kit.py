"""
UI kit for Hand Mouse: palette, fonts, line icons and the small custom widgets
the floating toolbar is built from (glass panel, icon buttons, colour dots).
"""

from PySide6.QtCore import QByteArray, QEasingCurve, QPointF, QRectF, Qt, QVariantAnimation
from PySide6.QtGui import (QColor, QFont, QFontDatabase, QPainter, QPainterPath, QPen,
                           QPixmap, QRadialGradient)
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QAbstractButton, QWidget

# --------------------------------------------------------------------------- #
# palette
# --------------------------------------------------------------------------- #

ACCENT = QColor("#22D3EE")        # cyan
TEXT = QColor(244, 247, 252)
TEXT_DIM = QColor(244, 247, 252, 150)
GLASS_TOP = QColor(30, 34, 48, 214)
GLASS_BOTTOM = QColor(14, 16, 26, 226)
HAIRLINE = QColor(255, 255, 255, 30)
RED = QColor("#FF4D5E")
AMBER = QColor("#FBBF24")
GREEN = QColor("#34D399")

# refined neon set: cyan, violet, pink, lime, amber, white
NEON = ["#22D3EE", "#A78BFA", "#F472B6", "#A3E635", "#FBBF24", "#F8FAFC"]

FONT_CANDIDATES = ["Inter", "Inter Variable", "Segoe UI Variable Text", "Segoe UI Variable Display",
                   "SF Pro Text", ".AppleSystemUIFont", "Helvetica Neue", "Rubik", "Heebo",
                   "Assistant", "Noto Sans Hebrew", "Noto Sans", "Segoe UI", "Arial"]


def ui_font_family():
    have = set(QFontDatabase.families())
    for f in FONT_CANDIDATES:
        if f in have:
            return f
    return QFont().defaultFamily()


def ui_font(size=10, weight=QFont.Medium):
    f = QFont(ui_font_family())
    f.setPointSizeF(size)
    f.setWeight(weight)
    f.setHintingPreference(QFont.PreferNoHinting)
    return f


TOOLTIP_CSS = """
QToolTip { color: #F4F7FC; background-color: rgba(18, 21, 32, 240);
           border: 1px solid rgba(255,255,255,40); border-radius: 8px; padding: 6px 9px; }
"""

# --------------------------------------------------------------------------- #
# icons: 24x24 line icons drawn for this app, stroke = currentColor
# --------------------------------------------------------------------------- #

_S = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
      'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{}</svg>')

ICONS = {
    "pointer": '<path d="M5 3.5l13.5 6.8-5.9 1.9-2.6 6.3z"/><path d="M12.6 12.2l4.6 4.6"/>',
    "pen": '<path d="M15.2 4.3l4.5 4.5L9 19.5l-5 1 1-5z"/><path d="M13.2 6.3l4.5 4.5"/>',
    "present": '<rect x="3" y="4" width="18" height="12" rx="2.5"/><path d="M12 16v4M8.5 20h7"/>'
               '<circle cx="15.5" cy="9" r="1.2" fill="currentColor" stroke="none"/>',
    "shapes": '<circle cx="8" cy="8.5" r="4.5"/><rect x="12" y="12" width="8.5" height="8.5" rx="1.8"/>',
    "undo": '<path d="M9 14L4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/>',
    "trash": '<path d="M4 7h16M10 11v6M14 11v6"/><path d="M6 7l1 12.2a1.8 1.8 0 0 0 1.8 1.8h6.4'
             'a1.8 1.8 0 0 0 1.8-1.8L18 7"/><path d="M9 7V4.8A.8.8 0 0 1 9.8 4h4.4a.8.8 0 0 1 .8.8V7"/>',
    "board": '<rect x="3" y="4" width="18" height="14" rx="2.5"/><path d="M7 9.5c2-1.6 3.6 1.6 5.6 0'
             's3.4-1.4 4.4-.4M7 13.5h6"/>',
    "camera": '<path d="M4 8.5h3l1.8-2.6h6.4L17 8.5h3v10H4z"/><circle cx="12" cy="13.3" r="3.3"/>',
    "record": '<circle cx="12" cy="12" r="6.5" fill="currentColor" stroke="none"/>',
    "stop": '<rect x="7" y="7" width="10" height="10" rx="2.2" fill="currentColor" stroke="none"/>',
    "video": '<rect x="3" y="6.5" width="12.5" height="11" rx="2.5"/><path d="M15.5 11l5.5-3v8l-5.5-3z"/>',
    "pause": '<path d="M9 5.5v13M15 5.5v13"/>',
    "play": '<path d="M7.5 5.2l11 6.8-11 6.8z"/>',
    "close": '<path d="M6.5 6.5l11 11M17.5 6.5l-11 11"/>',
    "grip": ''.join(f'<circle cx="{x}" cy="{y}" r="1.1" fill="currentColor" stroke="none"/>'
                    for x in (9.5, 14.5) for y in (7, 12, 17)),
}

_icon_cache = {}


def icon_pixmap(name, color, size, dpr=1.0):
    key = (name, QColor(color).name(QColor.HexArgb), size, round(dpr, 2))
    pm = _icon_cache.get(key)
    if pm is not None:
        return pm
    c = QColor(color)
    svg = _S.format(ICONS[name]).replace("currentColor", c.name())
    r = QSvgRenderer(QByteArray(svg.encode()))
    pm = QPixmap(int(size * dpr), int(size * dpr))
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setOpacity(c.alphaF())
    r.render(p)
    p.end()
    pm.setDevicePixelRatio(dpr)
    _icon_cache[key] = pm
    return pm


# --------------------------------------------------------------------------- #
# painting helpers
# --------------------------------------------------------------------------- #

def paint_shadow(p, rect, radius, strength=1.0, spread=16, offset=6):
    """Soft drop shadow made of stacked translucent rounded rects."""
    p.setPen(Qt.NoPen)
    steps = 10
    for i in range(steps):
        t = i / steps
        grow = spread * (1 - t)
        a = int(strength * 26 * (t ** 1.6))
        p.setBrush(QColor(0, 0, 0, a))
        r = rect.adjusted(-grow, -grow + offset, grow, grow + offset)
        p.drawRoundedRect(r, radius + grow, radius + grow)


def paint_glass(p, rect, radius):
    """Frosted dark glass: vertical gradient, top sheen, hairline border."""
    from PySide6.QtGui import QLinearGradient
    g = QLinearGradient(rect.topLeft(), rect.bottomLeft())
    g.setColorAt(0.0, GLASS_TOP)
    g.setColorAt(1.0, GLASS_BOTTOM)
    p.setPen(Qt.NoPen)
    p.setBrush(g)
    p.drawRoundedRect(rect, radius, radius)
    sheen = QLinearGradient(rect.topLeft(), QPointF(rect.left(), rect.top() + rect.height() * 0.55))
    sheen.setColorAt(0.0, QColor(255, 255, 255, 22))
    sheen.setColorAt(1.0, QColor(255, 255, 255, 0))
    p.setBrush(sheen)
    p.drawRoundedRect(rect, radius, radius)
    p.setBrush(Qt.NoBrush)
    p.setPen(QPen(HAIRLINE, 1))
    p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)


def soft_glow(p, center, color, radius, core=None):
    g = QRadialGradient(center, radius)
    c = QColor(color)
    stops = ((0.0, 210), (0.18, 150), (0.4, 60), (0.7, 18), (1.0, 0))
    for pos, a in stops:
        cc = QColor(c)
        cc.setAlpha(a)
        g.setColorAt(pos, cc)
    p.setPen(Qt.NoPen)
    p.setBrush(g)
    p.drawEllipse(center, radius, radius)
    if core:
        p.setBrush(QColor(255, 255, 255, 235))
        p.drawEllipse(center, core, core)


# --------------------------------------------------------------------------- #
# widgets
# --------------------------------------------------------------------------- #

class _Animated(QAbstractButton):
    """Button with eased hover / hot (hand) / press animation values."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.NoFocus)
        self.hot = False
        self._hover = 0.0
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(160)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.valueChanged.connect(self._set_hover)

    def _set_hover(self, v):
        self._hover = float(v)
        self.update()

    def _animate_to(self, target):
        self._anim.stop()
        self._anim.setStartValue(self._hover)
        self._anim.setEndValue(float(target))
        self._anim.start()

    def set_hot(self, on):
        self.hot = on
        self._animate_to(1.0 if (on or self.underMouse()) else 0.0)

    def enterEvent(self, e):
        self._animate_to(1.0)
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._animate_to(1.0 if self.hot else 0.0)
        super().leaveEvent(e)


class IconButton(_Animated):
    def __init__(self, icon, tip, parent=None, checkable=False, size=36, accent=None,
                 flat_checked=False):
        super().__init__(parent)
        self.icon = icon
        self.accent = QColor(accent) if accent else ACCENT
        self.flat_checked = flat_checked      # mode buttons: the switch paints the highlight
        self.setCheckable(checkable)
        self.setToolTip(tip)
        self.setFixedSize(size, size)
        self.text_label = ""
        self.icon_color = None

    def sizeHint(self):
        return self.size()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(2, 2, -2, -2)
        checked = self.isChecked()
        if checked and not self.flat_checked:
            bg = QColor(self.accent)
            bg.setAlpha(46)
            p.setPen(Qt.NoPen)
            p.setBrush(bg)
            p.drawRoundedRect(r, 11, 11)
        if self._hover > 0.01:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, int(26 * self._hover)))
            p.drawRoundedRect(r, 11, 11)
        if self.hot:
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(self.accent, 1.4))
            p.drawRoundedRect(r.adjusted(0.7, 0.7, -0.7, -0.7), 11, 11)
        if self.isDown():
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, 22))
            p.drawRoundedRect(r, 11, 11)
        color = QColor(self.accent if checked else (self.icon_color or TEXT))
        if not checked:
            color.setAlpha(int(175 + 80 * self._hover))
        s = 20
        dpr = self.devicePixelRatioF()
        if self.text_label:
            pm = icon_pixmap(self.icon, color, 14, dpr)
            p.drawPixmap(QPointF(10, (self.height() - 14) / 2), pm)
            p.setPen(TEXT)
            p.setFont(ui_font(9.5, QFont.DemiBold))
            p.drawText(QRectF(28, 0, self.width() - 32, self.height()), Qt.AlignVCenter | Qt.AlignLeft,
                       self.text_label)
        else:
            pm = icon_pixmap(self.icon, color, s, dpr)
            p.drawPixmap(QPointF((self.width() - s) / 2, (self.height() - s) / 2), pm)
        p.end()


class ColorDot(_Animated):
    def __init__(self, color, tip, parent=None):
        super().__init__(parent)
        self.color = QColor(color)
        self.setToolTip(tip)
        self.setCheckable(True)
        self.setFixedSize(26, 36)
        self._sel = 0.0
        self._sel_anim = QVariantAnimation(self)
        self._sel_anim.setDuration(220)
        self._sel_anim.setEasingCurve(QEasingCurve.OutBack)
        self._sel_anim.valueChanged.connect(lambda v: (setattr(self, "_sel", float(v)), self.update()))
        self.toggled.connect(self._on_toggle)

    def _on_toggle(self, on):
        self._sel_anim.stop()
        self._sel_anim.setStartValue(self._sel)
        self._sel_anim.setEndValue(1.0 if on else 0.0)
        self._sel_anim.start()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = QPointF(self.width() / 2, self.height() / 2)
        base = 6.0 + 1.2 * self._hover
        if self._sel > 0.01 or self.hot:
            g = QColor(self.color)
            g.setAlpha(int(70 * max(self._sel, 0.6 if self.hot else 0)))
            p.setPen(Qt.NoPen)
            p.setBrush(g)
            p.drawEllipse(c, 11, 11)
        p.setPen(Qt.NoPen)
        p.setBrush(self.color)
        p.drawEllipse(c, base, base)
        if self._sel > 0.01:
            ring = QColor(255, 255, 255, int(235 * min(self._sel, 1.0)))
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(ring, 1.8))
            rr = base + 3.2 * self._sel
            p.drawEllipse(c, rr, rr)
        p.end()


class Divider(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(11, 36)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setPen(QPen(QColor(255, 255, 255, 34), 1))
        p.drawLine(QPointF(5.5, 9), QPointF(5.5, self.height() - 9))
        p.end()


class Grip(QWidget):
    """Drag handle at the start of the toolbar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(22, 36)
        self.setCursor(Qt.OpenHandCursor)
        self.setToolTip("גרירה")

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        pm = icon_pixmap("grip", QColor(255, 255, 255, 110), 18, self.devicePixelRatioF())
        p.drawPixmap(QPointF(2, 9), pm)
        p.end()


def rounded_path(rect, radius):
    path = QPainterPath()
    path.addRoundedRect(rect, radius, radius)
    return path

