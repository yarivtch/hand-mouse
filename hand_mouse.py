#!/usr/bin/env python3
"""
Hand Mouse - control the computer with your hand through the webcam.

Three modes, switched from the floating toolbar or with hotkeys:

MOUSE (Ctrl+Alt+M pauses)
  Move ............ point with your index finger
  Left click ...... quick pinch (thumb + index tips touch and let go)
  Drag ............ pinch and hold ~0.3 s, move, let go
  Right click ..... pinch thumb + MIDDLE finger
  Scroll .......... index + middle fingers up together, move hand up/down
  Stop ............ fist
  Pause / resume .. open spread palm held still ~1 s

DRAW (Ctrl+Alt+D) - glowing neon lines over the whole screen, lightboard style
  Pinch and move = draw, open hand = pen up, thumb + middle pinch = undo,
  two fingers together = next colour, open palm held ~1 s = clear.
  Smart shapes: rough circles, ellipses, rectangles, triangles, lines and
  arrows snap to clean shapes (undo brings back what you drew).

PRESENT (Ctrl+Alt+T)
  Swipe an open hand left = next slide, right = previous slide.
  Point with the index finger = laser pointer. Pinch = scribble on the slide
  (cleared on the next slide). Fist = rest.

Also: Ctrl+Alt+R start/stop an explainer recording (screen + drawings +
webcam circle + microphone), Ctrl+Alt+S screenshot, Ctrl+Alt+Z undo,
Ctrl+Alt+C clear, Ctrl+Alt+K next colour, Ctrl+Alt+B dark board,
Ctrl+Alt+P show/hide camera preview, Ctrl+Alt+Q quit.

Run:  python hand_mouse.py          (see --help for options)
"""

import argparse
import collections
import math
import os
import queue
import sys
import threading
import time
import urllib.request

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError:  # pragma: no cover
    sys.exit("MediaPipe is missing. Run:  pip install -r requirements.txt")

from pynput.keyboard import Controller as KeyboardController, Key
from pynput.mouse import Button, Controller as MouseController

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "hand_landmarker.task")

# Landmark indices (MediaPipe hand model)
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

# Tuning (pinch distances are relative to hand size, so they work near or far)
PINCH_ON = 0.28          # thumb-tip to finger-tip distance that counts as a pinch
PINCH_OFF = 0.42         # distance at which the pinch is released (hysteresis)
CLICK_HOLD = 0.30        # seconds: shorter pinch = click, longer = drag
PALM_HOLD = 1.0          # seconds the open palm must be held to pause/resume
PALM_STILL = 0.035       # max hand movement (frame fraction / frame) while holding palm
SCROLL_TOGETHER = 0.45   # index and middle tips closer than this (x hand size) = scroll pose
POSE_FRAMES = 3          # frames a pose must persist before it switches mode


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def set_dpi_awareness():
    """On Windows, use real pixels so the cursor lands where expected."""
    if sys.platform != "win32":
        return
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def screen_size():
    """Size of the main screen, in the same units the mouse controller uses."""
    if sys.platform == "win32":
        import ctypes
        u = ctypes.windll.user32
        return u.GetSystemMetrics(0), u.GetSystemMetrics(1)
    if sys.platform == "darwin":
        try:
            import Quartz  # installed together with pynput on macOS
            b = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
            return int(b.size.width), int(b.size.height)
        except Exception:
            pass
    else:
        try:
            from Xlib import display  # installed together with pynput on Linux
            s = display.Display().screen()
            return s.width_in_pixels, s.height_in_pixels
        except Exception:
            pass
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        return 1920, 1080


def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        return
    print("Downloading the hand model (about 8 MB, first run only)...")
    tmp = MODEL_PATH + ".part"
    try:
        try:
            import ssl
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            ctx = None
        with urllib.request.urlopen(MODEL_URL, context=ctx, timeout=60) as r, open(tmp, "wb") as f:
            f.write(r.read())
        os.replace(tmp, MODEL_PATH)
        print("Model saved next to the script.")
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        sys.exit(f"Could not download the model ({e}).\n"
                 f"Download it manually from:\n  {MODEL_URL}\n"
                 f"and save it as:\n  {MODEL_PATH}")


def open_camera(index, width, height):
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
    for be in backends:
        cap = cv2.VideoCapture(index, be)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
    return None


class OneEuro:
    """One Euro filter: smooth when slow, responsive when fast."""

    def __init__(self, min_cutoff=1.0, beta=0.01, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x = self.dx = self.t = None

    def __call__(self, x, t):
        if self.t is None:
            self.x, self.dx, self.t = x, 0.0, t
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * abs(self.dx)
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# --------------------------------------------------------------------------- #
# Gesture analysis
# --------------------------------------------------------------------------- #

class Hand:
    def __init__(self, lm, frame_w, frame_h):
        # Use pixel-like coordinates so distances are not skewed by aspect ratio
        self.p = [(l.x * frame_w, l.y * frame_h) for l in lm]
        self.w, self.h = frame_w, frame_h
        self.size = max(dist(self.p[WRIST], self.p[MIDDLE_MCP]), 1.0)

    def rel(self, a, b):
        return dist(self.p[a], self.p[b]) / self.size

    def extended(self, tip, pip):
        return dist(self.p[WRIST], self.p[tip]) > dist(self.p[WRIST], self.p[pip]) * 1.15

    def fingers(self):
        return (self.extended(INDEX_TIP, INDEX_PIP),
                self.extended(MIDDLE_TIP, MIDDLE_PIP),
                self.extended(RING_TIP, RING_PIP),
                self.extended(PINKY_TIP, PINKY_PIP))

    def thumb_out(self):
        return self.rel(THUMB_TIP, INDEX_MCP) > 0.55

    def is_open_palm(self):
        i, m, r, k = self.fingers()
        spread = self.rel(INDEX_TIP, PINKY_TIP) > 0.85
        return i and m and r and k and self.thumb_out() and spread

    def is_scroll_pose(self):
        i, m, r, k = self.fingers()
        return i and m and not r and not k and self.rel(INDEX_TIP, MIDDLE_TIP) < SCROLL_TOGETHER

    def is_fist(self):
        # Fingertips folded back toward the wrist (a pinch keeps the index tip forward)
        curled = all(dist(self.p[WRIST], self.p[tip]) < dist(self.p[WRIST], self.p[mcp]) * 1.05
                     for tip, mcp in ((INDEX_TIP, INDEX_MCP), (MIDDLE_TIP, MIDDLE_MCP)))
        i, m, r, k = self.fingers()
        return curled and not (r or k)

    def norm(self, idx):
        return self.p[idx][0] / self.w, self.p[idx][1] / self.h

    def anchor(self, mode):
        if mode == "tip":
            return self.norm(INDEX_TIP)
        (ax, ay), (bx, by) = self.norm(THUMB_TIP), self.norm(INDEX_TIP)
        return (ax + bx) / 2, (ay + by) / 2



    def four_up(self):
        i, m, r, k = self.fingers()
        return i and m and r and k


# --------------------------------------------------------------------------- #
# Engine: camera + gestures -> mouse / keyboard / drawing events
# --------------------------------------------------------------------------- #

MODES = ("mouse", "draw", "present")


class HandMouse:
    def __init__(self, args):
        self.args = args
        self.mouse = MouseController()
        self.keyboard = KeyboardController()
        self.sw, self.sh = args.screen or screen_size()
        self.fx = OneEuro(args.min_cutoff, args.beta)
        self.fy = OneEuro(args.min_cutoff, args.beta)

        self.mode = args.mode
        self.preview_style = "classic"   # the overlay app switches this to "modern"
        self.cam_aspect = 4 / 3
        self.active = not args.start_paused
        self.running = True
        self.show_preview = not args.no_preview
        self.preview_open = False

        self.ui = None                 # set by the overlay app; gets drawing events
        self.commands = queue.Queue()  # requests from hotkeys / toolbar, run in the engine thread
        self.frame_lock = threading.Lock()
        self.latest_preview = None     # annotated frame for the preview window
        self.latest_raw = None         # clean mirrored camera frame (for recordings)

        self.left_down = False
        self.pinch = None            # None | "left" | "right"
        self.pinch_t0 = 0.0
        self.right_fired = False
        self.pose_candidate, self.pose_count = None, 0
        self.pose = "move"
        self.scroll_prev_y = None
        self.scroll_acc = 0.0
        self.palm_t0 = None
        self.palm_armed = True
        self.prev_anchor = None
        self.last_pos = None
        self.pen_down = False
        self.swipe_hist = collections.deque()
        self.swipe_until = 0.0
        self.cursor_shown = False

    # -- events to the overlay -------------------------------------------- #
    def emit(self, *event):
        if self.ui is not None:
            self.ui.put(event)

    # -- control (safe to call from any thread) ---------------------------- #
    def toggle(self):
        self.commands.put(("toggle",))

    def set_mode(self, mode):
        self.commands.put(("mode", mode))

    def next_mode(self):
        self.commands.put(("next_mode",))

    def toggle_preview(self):
        self.show_preview = not self.show_preview

    def quit(self):
        self.running = False

    def _run_commands(self):
        while True:
            try:
                cmd = self.commands.get_nowait()
            except queue.Empty:
                return
            if cmd[0] == "toggle":
                self.active = not self.active
                self.release_all()
                print("Active" if self.active else "Paused")
                self.emit("active", self.active)
            elif cmd[0] in ("mode", "next_mode"):
                new = cmd[1] if cmd[0] == "mode" else MODES[(MODES.index(self.mode) + 1) % len(MODES)]
                if new != "mouse" and self.ui is None:
                    print("Draw and presentation modes need PySide6:  pip install PySide6-Essentials")
                    continue
                self.release_all()
                self.mode = new
                print(f"Mode: {new}")
                self.emit("mode", new)

    def release_all(self):
        if self.left_down:
            try:
                self.mouse.release(Button.left)
            except Exception:
                pass
            self.left_down = False
        if self.pen_down:
            self.pen_down = False
        if self.cursor_shown:
            self.emit("hide")
            self.cursor_shown = False
        self.pinch = None
        self.right_fired = False
        self.scroll_prev_y = None
        self.scroll_acc = 0.0
        self.swipe_hist.clear()
        self.fx.reset()
        self.fy.reset()

    def start_hotkeys(self):
        if self.args.no_hotkeys:
            return None
        try:
            from pynput import keyboard
            keys = {
                "<ctrl>+<alt>+m": self.toggle,
                "<ctrl>+<alt>+p": self.toggle_preview,
                "<ctrl>+<alt>+q": self.quit,
                "<ctrl>+<alt>+d": lambda: self.set_mode("mouse" if self.mode == "draw" else "draw"),
                "<ctrl>+<alt>+t": lambda: self.set_mode("mouse" if self.mode == "present" else "present"),
            }
            if self.ui is not None:
                keys.update({
                    "<ctrl>+<alt>+z": lambda: self.emit("undo"),
                    "<ctrl>+<alt>+c": lambda: self.emit("clear"),
                    "<ctrl>+<alt>+s": lambda: self.emit("screenshot"),
                    "<ctrl>+<alt>+r": lambda: self.emit("record"),
                    "<ctrl>+<alt>+k": lambda: self.emit("color_next"),
                    "<ctrl>+<alt>+b": lambda: self.emit("board"),
                })
            hk = keyboard.GlobalHotKeys(keys)
            hk.daemon = True
            hk.start()
            return hk
        except Exception as e:
            print(f"Global hotkeys unavailable ({e}). Use the toolbar, the preview window or the open palm.")
            return None

    # -- mapping ------------------------------------------------------------ #
    def region(self):
        """Part of the camera frame (x and y margins) that maps onto the whole screen.

        Mouse mode stretches it to fill the screen. Draw and present modes keep the
        proportions, so a circle drawn in the air stays a circle on screen."""
        m = self.args.margin
        if self.mode == "mouse":
            return m, m
        hy = 1 - 2 * m
        fx = (self.sw / self.sh) * hy / self.cam_aspect
        if fx > 1:
            fx, hy = 1.0, self.cam_aspect * self.sh / self.sw
        return (1 - fx) / 2, (1 - hy) / 2

    def to_screen(self, nx, ny):
        mx, my = self.region()
        ux = (nx - mx) / (1 - 2 * mx)
        uy = (ny - my) / (1 - 2 * my)
        ux, uy = min(max(ux, 0.0), 1.0), min(max(uy, 0.0), 1.0)
        return ux * (self.sw - 1), uy * (self.sh - 1)

    def filtered(self, nx, ny, t):
        x, y = self.to_screen(nx, ny)
        return self.fx(x, t), self.fy(y, t)

    def move_to(self, nx, ny, t):
        x, y = self.filtered(nx, ny, t)
        pos = (int(round(x)), int(round(y)))
        if pos != self.last_pos:
            try:
                self.mouse.position = pos
            except Exception:
                pass
            self.last_pos = pos

    def show_cursor(self, nx, ny, t, style):
        """Send the drawing / laser cursor to the overlay in screen fractions."""
        x, y = self.filtered(nx, ny, t)
        self.emit("cursor", x / max(self.sw - 1, 1), y / max(self.sh - 1, 1), self.pen_down, style)
        self.cursor_shown = True

    def hide_cursor(self):
        if self.cursor_shown:
            self.emit("hide")
            self.cursor_shown = False

    # -- per-frame logic --------------------------------------------------- #
    def palm_action(self):
        if self.mode == "draw":
            self.emit("clear")
        else:
            self.commands.put(("toggle",))

    def handle_palm(self, hand, t, anchor):
        """Open palm held still for PALM_HOLD seconds: pause/resume, or clear in draw mode."""
        if self.mode == "present":         # palm is used for swiping there
            self.palm_t0 = None
            return 0.0
        moved = 1.0 if self.prev_anchor is None else dist(anchor, self.prev_anchor)
        if hand is not None and hand.is_open_palm() and moved < PALM_STILL:
            if not self.palm_armed:
                return 0.0
            if self.palm_t0 is None:
                self.palm_t0 = t
            progress = (t - self.palm_t0) / PALM_HOLD
            if progress >= 1.0:
                self.palm_action()
                self.palm_t0 = None
                self.palm_armed = False   # must lower the palm before the next one
                return 0.0
            return progress
        if hand is None or not hand.is_open_palm():
            self.palm_armed = True
        self.palm_t0 = None
        return 0.0

    def update_pose(self, hand):
        if hand.is_scroll_pose():
            cand = "scroll"
        elif hand.is_fist():
            cand = "hold"
        else:
            cand = "move"
        if cand == self.pose_candidate:
            self.pose_count += 1
        else:
            self.pose_candidate, self.pose_count = cand, 1
        changed = False
        if self.pose_count >= POSE_FRAMES and cand != self.pose:
            self.pose = cand
            changed = True
            self.scroll_prev_y = None
            self.scroll_acc = 0.0
            if cand == "move":
                self.fx.reset()
                self.fy.reset()
        return changed

    def update_pinch(self, hand, t):
        """Mouse mode: click, drag and right click."""
        d_left = hand.rel(THUMB_TIP, INDEX_TIP)
        d_right = hand.rel(THUMB_TIP, MIDDLE_TIP)
        fist = hand.is_fist()

        if self.pinch is None:
            if fist:
                return
            if d_left < PINCH_ON and d_left <= d_right:
                self.pinch, self.pinch_t0 = "left", t
            elif d_right < PINCH_ON and d_right < d_left:
                self.pinch, self.pinch_t0 = "right", t
                self.right_fired = False
            return

        if self.pinch == "left":
            held = t - self.pinch_t0
            if fist and not self.left_down:            # closing into a fist, not a click
                self.pinch = None
            elif d_left > PINCH_OFF:                  # released
                if self.left_down:
                    self.mouse.release(Button.left)
                    self.left_down = False
                else:
                    self.mouse.click(Button.left)
                self.pinch = None
            elif not self.left_down and held >= CLICK_HOLD:
                self.mouse.press(Button.left)       # start drag
                self.left_down = True
        elif self.pinch == "right":
            if not self.right_fired:
                self.mouse.click(Button.right)
                self.right_fired = True
            if d_right > PINCH_OFF:
                self.pinch = None

    def update_pen(self, hand):
        """Draw / present modes: thumb+index pinch = pen down, thumb+middle = undo."""
        d_left = hand.rel(THUMB_TIP, INDEX_TIP)
        d_right = hand.rel(THUMB_TIP, MIDDLE_TIP)
        if self.pinch is None:
            if hand.is_fist():
                return
            if d_left < PINCH_ON and d_left <= d_right:
                self.pinch = "left"
                self.pen_down = True
            elif d_right < PINCH_ON and d_right < d_left:
                self.pinch = "right"
                self.emit("undo")
        elif self.pinch == "left" and d_left > PINCH_OFF:
            self.pinch = None
            self.pen_down = False
        elif self.pinch == "right" and d_right > PINCH_OFF:
            self.pinch = None

    def step(self, hand, t):
        """Returns (mode label, palm progress)."""
        self._run_commands()
        if hand is None:
            self.release_all()
            self.prev_anchor = None
            self.handle_palm(None, t, None)
            return ("no hand", 0.0)

        anchor = hand.anchor(self.args.anchor)
        progress = self.handle_palm(hand, t, anchor)
        self.prev_anchor = anchor

        if not self.active:
            self.hide_cursor()
            return ("paused", progress)
        if self.mode == "draw":
            return (self.step_draw(hand, t, anchor), progress)
        if self.mode == "present":
            return (self.step_present(hand, t, anchor), progress)
        return (self.step_mouse(hand, t, anchor), progress)

    def step_mouse(self, hand, t, anchor):
        if self.pinch is None:
            self.update_pose(hand)

        if self.pose == "scroll" and self.pinch is None:
            y = (hand.norm(INDEX_TIP)[1] + hand.norm(MIDDLE_TIP)[1]) / 2
            if self.scroll_prev_y is not None:
                dy = (self.scroll_prev_y - y) * self.args.scroll_speed
                if self.args.invert_scroll:
                    dy = -dy
                self.scroll_acc += dy
                steps = int(self.scroll_acc)
                if steps:
                    self.mouse.scroll(0, steps)
                    self.scroll_acc -= steps
            self.scroll_prev_y = y
            return "scroll"

        self.update_pinch(hand, t)

        if self.pinch == "left" and not self.left_down:
            return "click?"          # freeze briefly so clicks land
        if self.pinch == "right":
            return "right click"
        if self.left_down:
            self.move_to(*anchor, t)
            return "drag"
        if self.pose == "hold":
            return "hold"
        self.move_to(*anchor, t)
        return "move"

    def step_draw(self, hand, t, anchor):
        if self.pinch is None and self.update_pose(hand) and self.pose == "scroll":
            self.emit("color_next")
        self.update_pen(hand)
        if self.pose == "hold" and not self.pen_down:
            return "hold"
        if self.pose == "scroll" and not self.pen_down:
            self.hide_cursor()
            return "color"
        self.show_cursor(*anchor, t, "pen")
        return "drawing" if self.pen_down else "pen up"

    def step_present(self, hand, t, anchor):
        # Swipe: open hand moving fast sideways -> next / previous slide
        cx = hand.norm(MIDDLE_MCP)[0]
        if hand.four_up() and not self.pen_down:
            self.swipe_hist.append((t, cx))
            while self.swipe_hist and t - self.swipe_hist[0][0] > 0.45:
                self.swipe_hist.popleft()
            old_t, old_x = self.swipe_hist[0]
            dx = cx - old_x
            if t >= self.swipe_until and t - old_t > 0.08 and abs(dx) > self.args.swipe_distance:
                left = dx < 0
                if self.args.invert_swipe:
                    left = not left
                key = Key.right if left else Key.left     # swipe left = next slide
                self.keyboard.press(key)
                self.keyboard.release(key)
                self.emit("slide", "next" if key == Key.right else "prev")
                self.swipe_until = t + 0.8
                self.swipe_hist.clear()
            self.hide_cursor()
            return "swipe"
        self.swipe_hist.clear()

        if self.pinch is None:
            self.update_pose(hand)
        self.update_pen(hand)
        if self.pen_down:
            self.show_cursor(*anchor, t, "pen")
            return "drawing"
        if self.pose == "hold":
            self.hide_cursor()
            return "rest"
        i, m, r, k = hand.fingers()
        if i and not r and not k:                       # pointing = laser
            self.show_cursor(*hand.norm(INDEX_TIP), t, "laser")
            return "laser"
        self.hide_cursor()
        return "ready"

    # -- frames ------------------------------------------------------------- #
    def draw_modern(self, frame, hand, progress):
        """Clean preview: viewfinder corners, thin skeleton, accent pointer. No text."""
        h, w = frame.shape[:2]
        mx, my = self.region()
        x0, y0, x1, y1 = int(mx * w), int(my * h), int((1 - mx) * w), int((1 - my) * h)
        layer = frame.copy()
        L = 22
        white = (245, 245, 245)
        for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            cv2.line(layer, (cx, cy), (cx + dx * L, cy), white, 2, cv2.LINE_AA)
            cv2.line(layer, (cx, cy), (cx, cy + dy * L), white, 2, cv2.LINE_AA)
        accent = (238, 211, 34)            # #22D3EE in BGR
        if hand is not None:
            for a, b in HAND_CONNECTIONS:
                pa = tuple(int(v) for v in hand.p[a])
                pb = tuple(int(v) for v in hand.p[b])
                cv2.line(layer, pa, pb, (235, 235, 235), 2, cv2.LINE_AA)
            for i in (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP):
                cv2.circle(layer, tuple(int(v) for v in hand.p[i]), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.addWeighted(layer, 0.55, frame, 0.45, 0, dst=frame)
        if hand is not None:
            ax, ay = hand.anchor(self.args.anchor)
            c = (int(ax * w), int(ay * h))
            cv2.circle(frame, c, 11, accent, 2, cv2.LINE_AA)
            cv2.circle(frame, c, 4, accent, -1, cv2.LINE_AA)
            if progress > 0:
                cc = (int(hand.p[MIDDLE_MCP][0]), int(hand.p[MIDDLE_MCP][1]))
                cv2.ellipse(frame, cc, (34, 34), -90, 0, int(360 * progress), (36, 191, 251), 3,
                            cv2.LINE_AA)

    def draw(self, frame, hand, label, progress, fps):
        if self.preview_style == "modern":
            self.draw_modern(frame, hand, progress)
            return
        h, w = frame.shape[:2]
        mx, my = self.region()
        color = (80, 220, 120) if self.active else (60, 60, 230)
        cv2.rectangle(frame, (int(mx * w), int(my * h)), (int((1 - mx) * w), int((1 - my) * h)),
                      (200, 200, 200), 1)
        if hand is not None:
            for a, b in HAND_CONNECTIONS:
                pa = tuple(int(v) for v in hand.p[a])
                pb = tuple(int(v) for v in hand.p[b])
                cv2.line(frame, pa, pb, (255, 255, 255), 2)
            ax, ay = hand.anchor(self.args.anchor)
            cv2.circle(frame, (int(ax * w), int(ay * h)), 8, color, -1)
            if progress > 0:
                c = (int(hand.p[MIDDLE_MCP][0]), int(hand.p[MIDDLE_MCP][1]))
                cv2.ellipse(frame, c, (40, 40), -90, 0, int(360 * progress), (0, 200, 255), 4)
        status = "ACTIVE" if self.active else "PAUSED"
        cv2.rectangle(frame, (0, 0), (w, 34), (0, 0, 0), -1)
        cv2.putText(frame, f"{status} | {self.mode.upper()} | {label} | {fps:4.1f} fps", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        cv2.putText(frame, "Space pause  H hide  Q quit", (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)

    def make_landmarker(self):
        ensure_model()
        with open(MODEL_PATH, "rb") as f:     # buffer avoids non-ASCII path issues on Windows
            model = f.read()
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=model),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        return vision.HandLandmarker.create_from_options(options)

    def frames(self):
        """Generator: processes camera frames until stopped; yields the annotated frame."""
        landmarker = self.make_landmarker()
        cap = open_camera(self.args.camera, 640, 480)
        if cap is None:
            landmarker.close()
            raise SystemExit("Could not open the camera. Try --camera 1, and check camera permissions.")
        last_ts = 0
        fps, t_prev = 0.0, time.monotonic()
        try:
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                frame = cv2.flip(frame, 1)                 # mirror: move right = cursor right
                fh, fw = frame.shape[:2]
                self.cam_aspect = fw / fh
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ts = max(int(time.monotonic() * 1000), last_ts + 1)
                last_ts = ts
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), ts)
                hand = Hand(result.hand_landmarks[0], fw, fh) if result.hand_landmarks else None

                t = time.monotonic()
                label, progress = self.step(hand, t)

                dt = t - t_prev
                t_prev = t
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

                with self.frame_lock:
                    self.latest_raw = frame.copy()
                if self.show_preview:
                    self.draw(frame, hand, label, progress, fps)
                    pw = self.args.preview_width
                    small = cv2.resize(frame, (pw, int(fh * pw / fw)))
                    with self.frame_lock:
                        self.latest_preview = small
                    yield small
                else:
                    yield None
        finally:
            self.release_all()
            cap.release()
            landmarker.close()

    def run_worker(self):
        """Engine loop for the overlay app (runs in a background thread)."""
        try:
            for _ in self.frames():
                pass
        except SystemExit as e:
            print(e)
        except Exception as e:  # keep the UI informed instead of dying silently
            print(f"Engine stopped: {e}")
        finally:
            self.running = False

    def run(self):
        """Classic loop: mouse mode with an OpenCV preview window (no PySide6 needed)."""
        hk = self.start_hotkeys()
        print(f"Screen {self.sw}x{self.sh}. {'Active' if self.active else 'Paused'}. "
              "Ctrl+Alt+M pause/resume, Ctrl+Alt+Q quit.")
        win = "Hand Mouse"
        try:
            for small in self.frames():
                if small is not None:
                    cv2.imshow(win, small)
                    if not self.preview_open:
                        try:
                            cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
                        except Exception:
                            pass
                        self.preview_open = True
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        break
                    if key == ord(" "):
                        self.toggle()
                    if key in (ord("h"), ord("H")):
                        self.show_preview = False
                else:
                    if self.preview_open:
                        cv2.destroyWindow(win)
                        cv2.waitKey(1)
                        self.preview_open = False
                    time.sleep(0.001)
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            cv2.destroyAllWindows()
            if hk is not None:
                hk.stop()
            print("Bye.")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Control the mouse, draw and present with hand gestures.")
    p.add_argument("--mode", choices=MODES, default="mouse", help="mode to start in (default mouse)")
    p.add_argument("--classic", action="store_true",
                   help="mouse only, simple OpenCV window (no toolbar / drawing)")
    p.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    p.add_argument("--margin", type=float, default=0.18,
                   help="unused border of the camera frame, 0-0.4; bigger = smaller hand motions (default 0.18)")
    p.add_argument("--anchor", choices=["mid", "tip"], default="mid",
                   help="mid = point between thumb and index (steady clicks), tip = index fingertip")
    p.add_argument("--min-cutoff", type=float, default=1.2,
                   help="smoothing: lower = steadier but laggier (default 1.2)")
    p.add_argument("--beta", type=float, default=0.006,
                   help="smoothing: higher = less lag on fast moves (default 0.006)")
    p.add_argument("--scroll-speed", type=float, default=40.0,
                   help="scroll steps per full frame height of hand motion (default 40)")
    p.add_argument("--invert-scroll", action="store_true", help="reverse scroll direction")
    p.add_argument("--swipe-distance", type=float, default=0.18,
                   help="presentation: how far (frame fraction) a swipe must travel (default 0.18)")
    p.add_argument("--invert-swipe", action="store_true", help="presentation: swipe right = next slide")
    p.add_argument("--no-shapes", action="store_true", help="start with smart shapes off")
    p.add_argument("--pen-width", type=float, default=1.0, help="drawing line width multiplier")
    p.add_argument("--no-mic", action="store_true", help="record explainers without the microphone")
    p.add_argument("--out-dir", default=None, help="where screenshots and recordings are saved")
    p.add_argument("--screen", type=lambda s: tuple(int(v) for v in s.lower().split("x")),
                   help="override screen size, e.g. 2560x1440")
    p.add_argument("--preview-width", type=int, default=300, help="preview window width (default 300)")
    p.add_argument("--no-preview", action="store_true", help="start with the preview hidden")
    p.add_argument("--no-hotkeys", action="store_true", help="disable global hotkeys")
    p.add_argument("--start-paused", action="store_true", help="start paused")
    a = p.parse_args(argv)
    a.margin = min(max(a.margin, 0.0), 0.4)
    return a


def main():
    set_dpi_awareness()
    args = parse_args()
    if not args.classic:
        try:
            import light_board
        except ImportError as e:
            print(f"Toolbar and drawing need PySide6 ({e}). Running mouse-only mode.\n"
                  "Install with:  pip install -r requirements.txt")
        else:
            sys.exit(light_board.run(HandMouse(args)))
    if args.mode != "mouse":
        args.mode = "mouse"
    HandMouse(args).run()


if __name__ == "__main__":
    main()
