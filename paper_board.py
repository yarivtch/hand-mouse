#!/usr/bin/env python3
"""
Paper Board - draw with a real marker on real paper, see it glow on the screen.

Point a camera (a phone used as a webcam works well) down at a white sheet on a
darker desk. The page is found and straightened, the ink is picked out, the
hand and pen are ignored, and the drawing is shown as neon lines over the
whole screen.

Run:  python paper_board.py --camera 1        (see --help for options)

Keys (click the small camera window first):
  Space  freeze / unfreeze the drawing      B  dark board
  R      rotate the page a quarter turn     M  mirror
  K      next colour for black ink          H  hide / show the drawing
  Q/Esc  quit

The first half of this file is plain OpenCV (PaperTracker, render_neon) and can
be used and tested without a screen; the Qt part starts at run().
"""

import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np

# neon colours for black / grey ink, BGR (same set as the hand drawing mode)
NEON_BGR = [(238, 211, 34), (250, 139, 167), (182, 114, 244),
            (53, 230, 163), (36, 191, 251), (252, 250, 248)]

A4 = 2 ** 0.5


# --------------------------------------------------------------------------- #
# camera
# --------------------------------------------------------------------------- #

def open_camera(index, width, height):
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
    for be in backends:
        cap = cv2.VideoCapture(index, be)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
    return None


# --------------------------------------------------------------------------- #
# finding the page
# --------------------------------------------------------------------------- #

def order_corners(pts):
    """Corners as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(1)
    d = pts[:, 1] - pts[:, 0]
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]],
                    dtype=np.float32)


def find_page(frame):
    """The sheet of paper as 4 ordered corners (frame pixels), or None.

    Looks for the largest bright four-sided blob: a white page on a darker desk."""
    h, w = frame.shape[:2]
    scale = 480.0 / w
    small = cv2.resize(frame, (480, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    total = small.shape[0] * small.shape[1]
    if not 0.08 * total < cv2.contourArea(c) < 0.97 * total:
        return None
    hull = cv2.convexHull(c)            # the hull closes the bite a hand takes out of an edge
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in (0.02, 0.03, 0.045, 0.06, 0.08, 0.1):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            quad = approx
            break
        if len(approx) < 4:
            break
    if quad is None or cv2.contourArea(quad) < 0.85 * cv2.contourArea(hull):
        return None
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(mask, quad.reshape(4, 2).astype(np.int32), 255)
    if cv2.mean(gray, mask)[0] < cv2.mean(gray, cv2.bitwise_not(mask))[0] + 25:
        return None                     # not clearly brighter than its surroundings
    return order_corners(quad.reshape(4, 2) / scale)


def quad_area(q):
    return float(cv2.contourArea(np.asarray(q, dtype=np.float32)))


# --------------------------------------------------------------------------- #
# tracker: frame -> straightened page -> steady ink mask
# --------------------------------------------------------------------------- #

class PaperTracker:
    """Keeps a steady picture of the ink on the page.

    conf  float32 (H, W), 0..1: how sure we are each pixel is ink
    color uint8 (H, W, 3): the ink colour seen there (BGR)
    """

    def __init__(self, long_side=1000, rotate=0, threshold=0.78):
        self.long_side = long_side
        self.rotate = rotate % 4          # quarter turns
        self.threshold = threshold        # ink = darker than this fraction of the paper
        self.quad = None                  # smoothed page corners in the camera frame
        self.size = None                  # (W, H) of the straightened page
        self.conf = None
        self.color = None
        self.visible = None
        self.found = False                # page seen in the latest frame
        self.frozen = False
        self._odd = 0                     # frames in a row the page was somewhere else
        self._pending = None

    # -- page lock -------------------------------------------------------- #
    def _page_size(self, q):
        w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
        h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
        ratio = max(w, h) / max(min(w, h), 1.0)
        if abs(ratio - A4) < 0.2:
            ratio = A4                    # A4 / Letter seen at an angle
        ratio = min(max(ratio, 1.0), 2.5)
        long, short = self.long_side, int(round(self.long_side / ratio))
        W, H = (long, short) if w >= h else (short, long)
        if self.rotate % 2:
            W, H = H, W
        return W, H

    def _lock(self, q):
        self.quad = q
        size = self._page_size(q)
        if size != self.size:
            self.size = size
            self.reset()

    def reset(self):
        if self.size is None:
            return
        W, H = self.size
        self.conf = np.zeros((H, W), np.float32)
        self.color = np.zeros((H, W, 3), np.uint8)
        self.visible = np.zeros((H, W), bool)

    def set_rotate(self, k):
        self.rotate = k % 4
        if self.quad is not None:
            self.size = None
            self._lock(self.quad)

    def _update_quad(self, cand):
        self.found = cand is not None
        if cand is None:
            return
        if self.quad is None:
            self._lock(cand)
            return
        diag = np.linalg.norm(self.quad[2] - self.quad[0])
        move = np.linalg.norm(cand - self.quad, axis=1).max()
        same_area = abs(quad_area(cand) / max(quad_area(self.quad), 1.0) - 1) < 0.06
        if move < 0.03 * diag and same_area:
            self.quad = (0.7 * self.quad + 0.3 * cand).astype(np.float32)
            self._odd = 0
            return
        # Different place: a hand over a corner (ignore) or the page really moved
        # (follow once the new place has held still for a moment).
        if self._pending is not None and np.linalg.norm(cand - self._pending, axis=1).max() < 0.03 * diag:
            self._odd += 1
        else:
            self._odd = 1
        self._pending = cand
        if self._odd >= 12:
            self._odd = 0
            self._lock(cand)

    def warp(self, frame):
        W, H = self.size
        src = np.roll(self.quad, -self.rotate, axis=0)
        dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
        return cv2.warpPerspective(frame, cv2.getPerspectiveTransform(src, dst), (W, H))

    # -- ink ---------------------------------------------------------------- #
    def _ink_and_cover(self, page):
        """(ink now, covered): thin dark marks, and the area hidden by hand / pen."""
        W, H = self.size
        k = max(9, self.long_side // 56) | 1                      # at half size, for speed
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        half = cv2.resize(page, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        closed = cv2.morphologyEx(cv2.cvtColor(half, cv2.COLOR_BGR2GRAY), cv2.MORPH_CLOSE, ker)
        # closed = the page with thin lines removed: its own lighting, plus anything big on it
        bg = cv2.resize(cv2.GaussianBlur(closed, (k, k), 0), (W, H)).astype(np.float32)
        ink = gray.astype(np.float32) < self.threshold * np.maximum(bg, 1.0)
        b = max(6, self.long_side // 80)                          # ignore the very edge of the page
        ink[:b], ink[-b:], ink[:, :b], ink[:, -b:] = False, False, False, False

        level = float(np.percentile(closed, 75))                  # brightness of bare paper
        cover = closed < 0.7 * level                              # big dark things: pen, sleeve
        sat = cv2.cvtColor(half, cv2.COLOR_BGR2HSV)[:, :, 1]
        cover |= cv2.morphologyEx(sat, cv2.MORPH_OPEN, ker) > 55  # big coloured things: skin
        cover = cv2.dilate(cover.astype(np.uint8), ker)
        cover = cv2.resize(cover, (W, H), interpolation=cv2.INTER_NEAREST) > 0
        return ink, cover

    def process(self, frame):
        """Feed one camera frame. Returns True when the visible drawing changed."""
        self._update_quad(find_page(frame))
        if self.quad is None or self.frozen:
            return False
        page = self.warp(frame)
        ink, cover = self._ink_and_cover(page)
        free = ~cover
        # Ink must hold still for a few frames to count, so a moving pen tip never does.
        self.conf[free] += 0.3 * (ink[free].astype(np.float32) - self.conf[free])
        sel = ink & free
        fresh = sel & (self.conf < 0.35)
        self.color[fresh] = page[fresh]
        old = sel & ~fresh
        self.color[old] = (0.8 * self.color[old] + 0.2 * page[old]).astype(np.uint8)

        vis = np.where(self.visible, self.conf > 0.4, self.conf > 0.6)   # hysteresis
        changed = int(np.count_nonzero(vis != self.visible)) > 12
        if changed:
            self.visible = vis
        return changed


# --------------------------------------------------------------------------- #
# neon rendering
# --------------------------------------------------------------------------- #

def render_neon(tracker, out_w, out_h, default_bgr=NEON_BGR[0]):
    """The drawing as a premultiplied BGRA image (out_h, out_w, 4) with a neon glow."""
    m = tracker.visible.astype(np.float32)
    m = cv2.GaussianBlur(m, (3, 3), 0)
    m = cv2.resize(m, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    # coloured markers keep their hue, pushed to full neon; black / grey ink gets the default
    col = cv2.resize(tracker.color, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    hsv = cv2.cvtColor(col, cv2.COLOR_BGR2HSV)
    coloured = (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 40)
    hsv[:, :, 1], hsv[:, :, 2] = 215, 255
    neon = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)
    neon[~coloured] = default_bgr

    body = neon * m[..., None]
    small = cv2.resize(body, (max(1, out_w // 2), max(1, out_h // 2)), interpolation=cv2.INTER_AREA)
    glow = cv2.GaussianBlur(small, (0, 0), 3) * 0.9 + cv2.GaussianBlur(small, (0, 0), 10) * 1.2
    glow = cv2.resize(glow, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    core = cv2.GaussianBlur(cv2.erode(m, np.ones((3, 3), np.uint8)), (3, 3), 0)   # hot white centre
    rgb = np.clip(glow + body * 0.9 + core[..., None] * 150, 0, 255)

    alpha = np.maximum(rgb.max(2) / 255.0, m)
    out = np.empty((out_h, out_w, 4), np.uint8)
    out[:, :, :3] = np.minimum(rgb, alpha[..., None] * 255).astype(np.uint8)
    out[:, :, 3] = (alpha * 255).astype(np.uint8)
    return out


def fit_size(page_w, page_h, max_w, max_h):
    s = min(max_w / page_w, max_h / page_h)
    return max(1, int(page_w * s)), max(1, int(page_h * s))


# --------------------------------------------------------------------------- #
# engine thread
# --------------------------------------------------------------------------- #

class Engine:
    def __init__(self, args):
        self.args = args
        self.tracker = PaperTracker(rotate=args.rotate, threshold=args.threshold)
        self.color_i = args.color % len(NEON_BGR)
        self.mirror = args.mirror
        self.running = True
        self.redraw = False               # set by the UI to force a new render
        self.turn = False                 # set by the UI: rotate a quarter turn (done in loop())
        self.error = None
        self.lock = threading.Lock()
        self.preview = None               # small BGR frame with the page outlined
        self.image = None                 # latest BGRA drawing
        self.image_id = 0

    def _preview(self, frame):
        t = self.tracker
        if t.quad is not None:
            colour = (120, 220, 80) if t.found else (60, 170, 250)
            cv2.polylines(frame, [t.quad.astype(np.int32)], True, colour, 3, cv2.LINE_AA)
            top = np.roll(t.quad, -t.rotate, axis=0)[:2].mean(0)         # marks the top of the page
            cv2.circle(frame, tuple(int(v) for v in top), 9, colour, -1, cv2.LINE_AA)
        pw = self.args.preview_width
        h, w = frame.shape[:2]
        return cv2.resize(frame, (pw, int(h * pw / w)), interpolation=cv2.INTER_AREA)

    def loop(self):
        cap = open_camera(self.args.camera, 1280, 720)
        if cap is None:
            self.error = "Could not open the camera. Try --camera 1 (or 0, 2), and check camera permissions."
            self.running = False
            return
        t = self.tracker
        last_render = 0.0
        dirty = False
        try:
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                if self.mirror:
                    frame = cv2.flip(frame, 1)
                if self.turn:              # here, not in the UI thread: it resizes the arrays
                    self.turn = False
                    t.set_rotate(t.rotate + 1)
                    dirty = True
                size_before = t.size
                dirty |= t.process(frame)
                dirty |= t.size != size_before
                if self.redraw:
                    self.redraw, dirty = False, True
                now = time.monotonic()
                if dirty and t.size is not None and now - last_render > 0.06:
                    img = render_neon(t, *t.size, NEON_BGR[self.color_i])
                    with self.lock:
                        self.image = img
                        self.image_id += 1
                    dirty, last_render = False, now
                small = self._preview(frame)
                with self.lock:
                    self.preview = small
        except Exception as e:  # tell the UI instead of dying silently
            self.error = f"Stopped: {e}"
        finally:
            self.running = False
            cap.release()


# --------------------------------------------------------------------------- #
# Qt: overlay + small camera window
# --------------------------------------------------------------------------- #

def run(args):
    if sys.platform.startswith("linux"):
        # OpenCV ships its own Qt plugins and points Qt at them; undo that for PySide6
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
        os.environ.pop("QT_QPA_FONTDIR", None)
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    from PySide6.QtCore import QRectF, Qt, QTimer
    from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPixmap, QRadialGradient
    from PySide6.QtWidgets import QApplication, QWidget

    class Overlay(QWidget):
        """Transparent, click-through window over the whole screen."""

        def __init__(self):
            super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
                             | Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus)
            self.setAttribute(Qt.WA_TranslucentBackground)
            self.setAttribute(Qt.WA_TransparentForMouseEvents)
            self.setAttribute(Qt.WA_ShowWithoutActivating)
            if sys.platform == "darwin":
                self.setAttribute(Qt.WA_MacAlwaysShowToolWindow)
            self.setGeometry(QGuiApplication.primaryScreen().geometry())
            self.img = None
            self.board = args.board
            self.show_ink = True

        def set_image(self, bgra):
            h, w = bgra.shape[:2]
            self.img = QImage(bgra.data, w, h, bgra.strides[0],
                              QImage.Format_ARGB32_Premultiplied).copy()
            self.update()

        def paintEvent(self, _):
            p = QPainter(self)
            r = QRectF(self.rect())
            if self.board:
                g = QRadialGradient(r.center(), max(r.width(), r.height()) * 0.75)
                g.setColorAt(0.0, QColor(10, 14, 28, 200))
                g.setColorAt(1.0, QColor(3, 5, 12, 236))
                p.fillRect(r, g)
            if self.img is not None and self.show_ink:
                p.setRenderHint(QPainter.SmoothPixmapTransform)
                w, h = fit_size(self.img.width(), self.img.height(), r.width(), r.height())
                p.drawImage(QRectF((r.width() - w) / 2, (r.height() - h) / 2, w, h), self.img)
            p.end()

    class Preview(QWidget):
        """Camera view with the page outlined; also takes the keys."""

        def __init__(self, engine, overlay):
            super().__init__(None, Qt.WindowStaysOnTopHint)
            self.engine, self.overlay = engine, overlay
            self.setWindowTitle("Paper Board")
            self.pix = None
            w = args.preview_width
            self.resize(w, int(w * 9 / 16))
            geo = QGuiApplication.primaryScreen().availableGeometry()
            self.move(geo.right() - self.width() - 24, geo.bottom() - self.height() - 60)

        def show_frame(self, bgr):
            h, w = bgr.shape[:2]
            self.pix = QPixmap.fromImage(
                QImage(bgr.data, w, h, bgr.strides[0], QImage.Format_BGR888).copy())
            if (w, h) != (self.width(), self.height()):
                self.resize(w, h)
            self.update()

        def paintEvent(self, _):
            p = QPainter(self)
            if self.pix is not None:
                p.drawPixmap(0, 0, self.pix)
            t = self.engine.tracker
            if t.frozen:
                text, colour = "FROZEN  (Space)", QColor("#FBBF24")
            elif t.quad is None:
                text, colour = "No page yet: white sheet on a darker desk", QColor("#FF4D5E")
            elif not t.found:
                text, colour = "Page hidden, holding", QColor("#FBBF24")
            else:
                text, colour = "Page found", QColor("#34D399")
            p.fillRect(0, 0, self.width(), 24, QColor(0, 0, 0, 170))
            p.setPen(colour)
            p.drawText(8, 17, text)
            p.end()

        def keyPressEvent(self, e):
            k, eng, t = e.key(), self.engine, self.engine.tracker
            if k in (Qt.Key_Q, Qt.Key_Escape):
                eng.running = False
            elif k == Qt.Key_Space:
                t.frozen = not t.frozen
            elif k == Qt.Key_B:
                self.overlay.board = not self.overlay.board
                self.overlay.update()
            elif k == Qt.Key_H:
                self.overlay.show_ink = not self.overlay.show_ink
                self.overlay.update()
            elif k == Qt.Key_R:
                eng.turn = True
            elif k == Qt.Key_M:
                eng.mirror = not eng.mirror
            elif k == Qt.Key_K:
                eng.color_i = (eng.color_i + 1) % len(NEON_BGR)
                eng.redraw = True

        def closeEvent(self, e):
            self.engine.running = False
            e.accept()

    app = QApplication.instance() or QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    overlay = Overlay()
    engine = Engine(args)
    preview = Preview(engine, overlay)
    overlay.show()
    preview.show()
    preview.activateWindow()
    worker = threading.Thread(target=engine.loop, daemon=True)
    worker.start()
    shown = [0]

    def tick():
        with engine.lock:
            frame, img, img_id = engine.preview, engine.image, engine.image_id
        if frame is not None:
            preview.show_frame(frame)
        if img is not None and img_id != shown[0]:
            shown[0] = img_id
            img = np.ascontiguousarray(img)
            overlay.set_image(img)
        if not engine.running:
            app.quit()

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(30)
    print("Paper Board: put a white sheet on a darker desk under the camera. Q quits.")
    code = app.exec()
    engine.running = False
    worker.join(timeout=3)
    if engine.error:
        print(engine.error)
        return 1
    return code


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Draw on real paper, see it as neon on the screen.")
    p.add_argument("--camera", type=int, default=0,
                   help="camera index; a phone used as a webcam is usually 1 (default 0)")
    p.add_argument("--rotate", type=int, default=0, choices=[0, 1, 2, 3],
                   help="quarter turns, if the drawing shows up sideways (or press R)")
    p.add_argument("--mirror", action="store_true", help="flip the camera image (or press M)")
    p.add_argument("--threshold", type=float, default=0.78,
                   help="how dark a mark must be, 0.5-0.95; higher picks up fainter lines (default 0.78)")
    p.add_argument("--color", type=int, default=0, help="neon colour for black ink, 0-5 (or press K)")
    p.add_argument("--board", action="store_true", help="start with the dark board on")
    p.add_argument("--preview-width", type=int, default=360, help="camera window width (default 360)")
    a = p.parse_args(argv)
    a.threshold = min(max(a.threshold, 0.5), 0.95)
    return a


if __name__ == "__main__":
    sys.exit(run(parse_args()))
