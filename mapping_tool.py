"""
Wafer Map Viewer  —  Keithley ACS KDF V1.2
Displays any KDF file as an interactive wafer map.
Subsite number = design number; switch designs via the Design selector.

Requirements:  pip install PySide6
Usage:         python wafer_mapper_light.py [file.kdf]
"""

import sys, os, re, math, statistics
from collections import defaultdict
from dataclasses import dataclass, field

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFileDialog, QTreeWidget, QTreeWidgetItem, QGroupBox,
    QLineEdit, QFormLayout, QStatusBar, QComboBox,
    QMessageBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QToolBar, QSizePolicy, QPushButton, QSpinBox,
    QStyle, QStyleOptionComboBox, QStyledItemDelegate
)
from PySide6.QtCore import Qt, QRectF, QPointF, QPoint, Signal, QSize, QRect
from PySide6.QtGui import (
    QPainter, QColor, QBrush, QPen, QFont,
    QLinearGradient, QRadialGradient, QPixmap, QIcon, QAction,
    QPolygonF, QImage, QImageWriter
)

# ─────────────────────────────────────────────────────────────────────────────
#  THEME
# ─────────────────────────────────────────────────────────────────────────────

T = {
    # Slightly warm off-white theme (less stark than pure white)
    "bg_app":        "#f2f0ec",
    "bg_panel":      "#fbfaf7",
    "bg_header":     "#efeae2",
    "bg_row_alt":    "#f6f2ec",
    "border":        "#cdd6e3",
    "border_hi":     "#a0b4cc",
    "accent":        "#1565c0",
    "accent_dim":    "#dce8fb",
    "accent_dark":   "#0d47a1",
    "pass_bg":       "#e3f6ec",
    "pass_fg":       "#1b6b3a",
    "pass_border":   "#5cb87a",
    "neutral_bg":    "#dfe8f3",
    "neutral_fg":    "#2c4a6e",
    "neutral_border":"#90a8c4",
    "fail_bg":       "#fce8e8",
    "fail_fg":       "#b71c1c",
    "fail_border":   "#e57373",
    "nodata_bg":     "#eeecf5",
    "nodata_fg":     "#7060a0",
    "nodata_border": "#b0a8d0",
    "wafer_bg":      "#eef2f8",
    "wafer_edge":    "#7a9cbd",
    "text_primary":  "#1a2537",
    "text_secondary":"#4a5c72",
    "text_dim":      "#8fa4bc",
    "selected":      "#e65100",
    "hover_border":  "#1565c0",
    "warn":          "#e65100",
}

SS = """
QMainWindow,QWidget{
    background-color:""" + T['bg_app'] + """;
    color:""" + T['text_primary'] + """;
    font-family:'Segoe UI','Calibri',sans-serif;
    font-size:13px;
}
QGroupBox{
    background-color:""" + T['bg_panel'] + """;
    border:1px solid """ + T['border'] + """;
    border-radius:10px;
    margin-top:22px;
    padding:10px 8px 8px 8px;
    font-weight:bold;
    font-size:11px;
}
QGroupBox::title{
    subcontrol-origin:margin;
    left:12px;
    padding:2px 8px;
    background:""" + T['bg_panel'] + """;
    color:""" + T['accent'] + """;
    font-size:11px;
    font-weight:bold;
    border-radius:4px;
}
QLabel{background:transparent;color:""" + T['text_primary'] + """;font-size:13px;}
QPushButton{
    background-color:""" + T['bg_panel'] + """;
    border:1px solid """ + T['border_hi'] + """;
    border-radius:8px;
    padding:6px 16px;
    color:""" + T['text_primary'] + """;
    font-weight:600;
    font-size:13px;
    min-height:28px;
}
QPushButton:hover{
    background-color:""" + T['accent_dim'] + """;
    border:1px solid """ + T['accent'] + """;
    color:""" + T['accent_dark'] + """;
}
QPushButton:pressed{background-color:""" + T['accent'] + """;color:white;}
QPushButton#primary{
    background-color:""" + T['accent'] + """;
    border:1px solid """ + T['accent_dark'] + """;
    color:white;
    font-weight:bold;
    border-radius:8px;
}
QPushButton#primary:hover{background-color:""" + T['accent_dark'] + """;}
QComboBox{
    background-color:""" + T['bg_panel'] + """;
    border:1px solid """ + T['border'] + """;
    border-radius:8px;
    padding:5px 36px 5px 12px;
    color:""" + T['text_primary'] + """;
    font-size:13px;
    min-height:32px;
}
QComboBox:hover{border:1px solid """ + T['accent'] + """;}
QComboBox:focus{border:1px solid """ + T['accent'] + """;}
QComboBox::drop-down{
    subcontrol-origin:padding;
    subcontrol-position:right center;
    width:32px;
    border:none;
    border-left:1px solid """ + T['border'] + """;
    border-top-right-radius:8px;
    border-bottom-right-radius:8px;
    background:""" + T['bg_header'] + """;
}
QComboBox::down-arrow{
    image:none;
    width:0px;
    height:0px;
}
QComboBox QAbstractItemView{
    background:""" + T['bg_panel'] + """;
    border:none;
    border-radius:0px;
    selection-background-color:""" + T['accent_dim'] + """;
    color:""" + T['text_primary'] + """;
    font-size:13px;
    padding:4px;
    outline:0;
}
/* ComboBox popup: avoid double frame (common on Windows) */
QComboBoxPrivateContainer{
    background:""" + T['bg_panel'] + """;
    border:1px solid """ + T['border_hi'] + """;
    border-radius:0px;
    padding:0px;
    margin:0px;
}
QComboBoxPrivateContainer QFrame{
    border:none;
}
QComboBoxPrivateContainer QListView{
    border:none;
    background:""" + T['bg_panel'] + """;
    outline:0;
}
QComboBoxPrivateContainer QAbstractScrollArea{
    border:none;
}
QComboBoxPrivateContainer QAbstractScrollArea::viewport{
    background:""" + T['bg_panel'] + """;
    border:none;
}
QComboBoxPrivateContainer *{
    border:none;
    outline:0;
}
QLineEdit,QSpinBox{
    background-color:""" + T['bg_panel'] + """;
    border:1px solid """ + T['border'] + """;
    border-radius:8px;
    padding:5px 10px;
    color:""" + T['text_primary'] + """;
    font-size:13px;
    min-height:28px;
}
QLineEdit:focus,QSpinBox:focus{border:1px solid """ + T['accent'] + """;}
QSpinBox::up-button,QSpinBox::down-button{
    width:20px;
    background:""" + T['bg_header'] + """;
    border:none;
    border-radius:4px;
}
QTreeWidget,QTableWidget{
    background-color:""" + T['bg_panel'] + """;
    border:1px solid """ + T['border'] + """;
    border-radius:8px;
    alternate-background-color:""" + T['bg_row_alt'] + """;
    outline:none;
    font-size:13px;
}
QTreeWidget::item,QTableWidget::item{padding:4px 5px;border:none;}
QTreeWidget::item:hover,QTableWidget::item:hover{
    background-color:""" + T['accent_dim'] + """;
    border-radius:4px;
}
QTreeWidget::item:selected,QTableWidget::item:selected{
    background-color:""" + T['accent_dim'] + """;
    color:""" + T['accent_dark'] + """;
    border-radius:4px;
}
QHeaderView::section{
    background-color:""" + T['bg_header'] + """;
    color:""" + T['accent_dark'] + """;
    border:none;
    border-right:1px solid """ + T['border'] + """;
    border-bottom:1px solid """ + T['border'] + """;
    padding:6px 10px;
    font-size:12px;
    font-weight:bold;
}
QHeaderView::section:first{border-top-left-radius:8px;}
QHeaderView::section:last{border-top-right-radius:8px;border-right:none;}
QScrollBar:vertical{
    background:""" + T['bg_app'] + """;
    width:9px;border:none;border-radius:4px;
}
QScrollBar::handle:vertical{
    background:""" + T['border_hi'] + """;
    border-radius:4px;min-height:24px;
}
QScrollBar::handle:vertical:hover{background:""" + T['accent'] + """;}
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}
QScrollBar:horizontal{
    background:""" + T['bg_app'] + """;
    height:9px;border-radius:4px;
}
QScrollBar::handle:horizontal{
    background:""" + T['border_hi'] + """;border-radius:4px;
}
QTabWidget::pane{
    border:1px solid """ + T['border'] + """;
    background:""" + T['bg_panel'] + """;
    border-radius:0 8px 8px 8px;
}
QTabBar::tab{
    background:""" + T['bg_header'] + """;
    color:""" + T['text_secondary'] + """;
    padding:7px 18px;
    border:1px solid """ + T['border'] + """;
    border-bottom:none;
    border-radius:8px 8px 0 0;
    margin-right:2px;
    font-size:12px;
}
QTabBar::tab:selected{
    background:""" + T['bg_panel'] + """;
    color:""" + T['accent'] + """;
    border-top:2px solid """ + T['accent'] + """;
    font-weight:bold;font-size:13px;
}
QStatusBar{
    background:""" + T['bg_header'] + """;
    color:""" + T['text_secondary'] + """;
    border-top:1px solid """ + T['border'] + """;
    font-size:12px;padding:3px 6px;
}
QToolBar{
    background:""" + T['bg_panel'] + """;
    border-bottom:1px solid """ + T['border'] + """;
    spacing:4px;padding:5px 10px;
}
QToolBar::separator{background:""" + T['border'] + """;width:1px;margin:4px 8px;}
QToolBar QToolButton{
    background-color:""" + T['accent'] + """;
    color:white;
    border:1px solid """ + T['accent_dark'] + """;
    border-radius:10px;
    padding:6px 14px;
    font-weight:700;
    font-size:13px;
    min-height:30px;
}
QToolBar QToolButton:hover{
    background-color:""" + T['accent_dark'] + """;
    border:1px solid """ + T['accent_dark'] + """;
}
QToolBar QToolButton:pressed{
    background-color:""" + T['accent_dark'] + """;
    padding-top:7px;   /* subtle "press" feel */
    padding-bottom:5px;
}
"""

# ─────────────────────────────────────────────────────────────────────────────
#  CUSTOM COMBOBOX  — draws its own crisp chevron arrow
# ─────────────────────────────────────────────────────────────────────────────

class ArrowComboBox(QComboBox):
    """QComboBox that paints a clean chevron arrow in the drop-down button."""

    def showPopup(self):
        super().showPopup()

        # On Windows, the popup can have native shadow/frames that QSS can't remove.
        # Force a square, no-shadow popup for a clean look.
        popup = self.view().window()
        if popup is None:
            return
        try:
            popup.setWindowFlag(Qt.FramelessWindowHint, True)
            popup.setWindowFlag(Qt.NoDropShadowWindowHint, True)
        except Exception:
            # Some Qt builds may not support NoDropShadowWindowHint; ignore gracefully.
            popup.setWindowFlag(Qt.FramelessWindowHint, True)

        popup.setAttribute(Qt.WA_TranslucentBackground, False)
        popup.setStyleSheet(
            f"background:{T['bg_panel']};"
            f"border:1px solid {T['border_hi']};"
            f"border-radius:0px;"
        )
        popup.show()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # drop-down button area (right 32 px)
        btn_x = self.width() - 32
        btn_w = 32
        cx = btn_x + btn_w / 2.0
        cy = self.height() / 2.0

        # chevron  ▾  — three points
        aw = 8.0   # half-width of chevron
        ah = 5.0   # height of chevron
        pts = QPolygonF([
            QPointF(cx - aw, cy - ah / 2),
            QPointF(cx,      cy + ah / 2),
            QPointF(cx + aw, cy - ah / 2),
        ])
        pen = QPen(QColor(T['accent']), 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawPolyline(pts)
        p.end()

# ─────────────────────────────────────────────────────────────────────────────
#  APP ICON
# ─────────────────────────────────────────────────────────────────────────────

def make_app_icon() -> QIcon:
    # Prefer a local icon asset if present (portable across machines).
    # Drop a `wafer_icon` file next to this script (e.g. wafer_icon.ico/png/icns).
    base_dir = os.path.dirname(__file__)
    preferred_exts = ('.ico', '.png', '.icns', '.jpg', '.jpeg')
    try:
        for name in os.listdir(base_dir):
            low = name.lower()
            if low.startswith('wafer_icon') and low.endswith(preferred_exts):
                pth = os.path.join(base_dir, name)
                if os.path.isfile(pth):
                    return QIcon(pth)
    except OSError:
        pass

    icon = QIcon()
    for sz in (16, 24, 32, 48, 64, 128):
        pix = QPixmap(sz, sz)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)
        cx = cy = sz / 2.0
        r  = sz / 2.0 - 1.0
        disc = QRadialGradient(cx - r*0.2, cy - r*0.25, r*1.35)
        disc.setColorAt(0, QColor("#2d5490"))
        disc.setColorAt(0.65, QColor("#1a3460"))
        disc.setColorAt(1.0, QColor("#0c1e3a"))
        p.setBrush(QBrush(disc))
        p.setPen(QPen(QColor("#5588cc"), max(1.0, sz/18.0)))
        p.drawEllipse(QPointF(cx, cy), r, r)
        nw = r*0.36; nh = max(2.0, sz*0.05)
        p.setCompositionMode(QPainter.CompositionMode_Clear)
        p.setPen(Qt.NoPen); p.setBrush(Qt.transparent)
        p.drawRect(QRectF(cx-nw/2, cy+r-nh, nw, nh+1))
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        p.setPen(QPen(QColor("#5588cc"), max(1.0, sz/20.0)))
        p.drawLine(QPointF(cx-nw/2, cy+r-nh), QPointF(cx+nw/2, cy+r-nh))
        colors = [QColor("#3dba6f"), QColor("#3dba6f"), QColor("#8a98b0"),
                  QColor("#3dba6f"), QColor("#3dba6f"), QColor("#d95555"),
                  QColor("#8a98b0"), QColor("#3dba6f"), QColor("#3dba6f")]
        ge = r*1.08; cs = ge/3.0; gap = max(0.6, cs*0.12)
        gx0 = cx - ge/2; gy0 = cy - ge/2
        p.setPen(QPen(QColor("#0c1e3a"), max(0.5, gap*0.4)))
        for row in range(3):
            for col in range(3):
                rx = gx0 + col*cs + gap; ry = gy0 + row*cs + gap
                rw = rh = cs - gap*2
                if rw > 0:
                    p.setBrush(QBrush(colors[row*3+col]))
                    p.drawRoundedRect(QRectF(rx, ry, rw, rh),
                                      max(0.5, rw*0.18), max(0.5, rw*0.18))
        p.end()
        icon.addPixmap(pix)
    return icon

# ─────────────────────────────────────────────────────────────────────────────
#  KDF PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_kdf(filepath: str):
    header: dict = {}
    sites:  list = []
    mkeys:  set  = set()
    tkeys:  set  = set()

    with open(filepath, 'r', errors='replace') as fh:
        lines = [l.rstrip('\r\n') for l in fh]

    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s == '<EOH>':
            i += 1
            break
        if ',' in s:
            k, _, v = s.partition(',')
            header[k.strip()] = v.strip()
        i += 1

    if i < len(lines) and lines[i].strip() and not lines[i].strip().startswith('Site_'):
        i += 1

    current = None
    while i < len(lines):
        s = lines[i].strip(); i += 1
        if not s:
            continue
        if s == '<EOS>':
            if current is not None:
                sites.append(current)
            current = None
            continue
        if s.startswith('Site_'):
            if current is not None:
                sites.append(current)
            parts = s.split(',')
            name  = parts[0].strip()
            try:
                x = int(parts[1].strip())
                y = int(parts[2].strip())
            except (IndexError, ValueError):
                m = re.match(r'Site_([pn])(\d+)([pn])(\d+)', name)
                if m:
                    xs, xv, ys, yv = m.groups()
                    x = int(xv) * (1 if xs == 'p' else -1)
                    y = int(yv) * (1 if ys == 'p' else -1)
                else:
                    x = y = 0
            current = {'name': name, 'x': x, 'y': y, 'subsites': {}}
            continue
        if current is None:
            continue
        if '@' in s and ',' in s:
            kp, _, vs = s.partition(',')
            parts = kp.split('@')
            if len(parts) >= 3:
                param, test, sub_raw = parts[0].strip(), parts[1].strip(), parts[2].strip()
                try:
                    sub_num = int(sub_raw.split('#')[1])
                except (IndexError, ValueError):
                    sub_num = 1
                try:
                    value = float(vs.strip())
                except ValueError:
                    value = None
                mkey = f"{param}@{test}"
                mkeys.add(mkey); tkeys.add(test)
                if sub_num not in current['subsites']:
                    current['subsites'][sub_num] = {}
                current['subsites'][sub_num][mkey] = value

    if current is not None:
        sites.append(current)

    return header, sites, sorted(mkeys), sorted(tkeys)


def get_site_value(site: dict, mkey: str, subsite: int | None = None) -> float | None:
    if subsite is not None:
        v = site['subsites'].get(subsite, {}).get(mkey)
        return v if (v is not None and math.isfinite(v)) else None

    values = [
        v for sub in site['subsites'].values()
        for k, v in sub.items()
        if k == mkey and v is not None and math.isfinite(v)
    ]
    return statistics.mean(values) if values else None


def all_subsites(sites: list) -> list[int]:
    s = set()
    for site in sites:
        s.update(site['subsites'].keys())
    return sorted(s)


def si_fmt(v: float | None) -> str:
    if v is None or not math.isfinite(v):
        return 'N/A'
    if v == 0:
        return '0'

    prefixes = [
        (1e15,'P'), (1e12,'T'), (1e9,'G'),  (1e6,'M'),  (1e3,'k'),
        (1e0,''),   (1e-3,'m'),(1e-6,'µ'),  (1e-9,'n'), (1e-12,'p'),
        (1e-15,'f'),(1e-18,'a'),
    ]
    av = abs(v)
    best_factor, best_suffix = prefixes[-1]
    for factor, suffix in prefixes:
        if 0.1 <= av / factor < 1000.0:
            best_factor = factor; best_suffix = suffix; break

    scaled = v / best_factor
    if abs(scaled) >= 100:   text = f"{scaled:.1f}"
    elif abs(scaled) >= 10:  text = f"{scaled:.2f}"
    else:                    text = f"{scaled:.3f}"
    if '.' in text:
        text = text.rstrip('0').rstrip('.')
    return f"{text}{best_suffix}"

# ─────────────────────────────────────────────────────────────────────────────
#  WAFER DATA  (holds parsed data for one wafer in a batch)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WaferData:
    filepath:    str
    wafer_id:    str          # derived from filename
    header:      dict
    sites:       list
    mkeys:       list
    tests:       list
    subsites:    list = field(default_factory=list)

    def yield_stats(self, mkey: str, sub: int | None,
                    lo: float | None, hi: float | None) -> dict:
        """Return pass/fail/nodata counts and numeric stats for this wafer."""
        vals, passed, failed, nodata = [], 0, 0, 0
        for s in self.sites:
            v = get_site_value(s, mkey, sub)
            if v is None:
                nodata += 1
            else:
                vals.append(v)
                ok = (lo is None or v >= lo) and (hi is None or v <= hi)
                if ok:
                    passed += 1
                else:
                    failed += 1
        total = passed + failed
        return dict(
            passed=passed, failed=failed, nodata=nodata,
            total=total,
            yield_pct=(passed / total * 100) if total else None,
            mean=statistics.mean(vals) if vals else None,
            stdev=statistics.pstdev(vals) if vals else None,
            min_v=min(vals) if vals else None,
            max_v=max(vals) if vals else None,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  WAFER CANVAS
# ─────────────────────────────────────────────────────────────────────────────

class WaferCanvas(QWidget):
    siteClicked = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sites         = []
        self.values        = {}
        self.low_limit     = None
        self.high_limit    = None
        self.mkey          = ''
        self.selected_site = None
        self._hover        = None
        self._rects        = {}
        self._zoom         = 1.0

        self.setMinimumSize(400, 400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

    def load(self, sites, values, lo, hi, mkey=''):
        self.sites = sites; self.values = values
        self.low_limit = lo; self.high_limit = hi; self.mkey = mkey
        self.selected_site = None; self._hover = None
        self.update()

    @property
    def _limits_active(self):
        return self.low_limit is not None or self.high_limit is not None

    def _bounds(self):
        if not self.sites:
            return -1, 1, -1, 1
        xs = [s['x'] for s in self.sites]
        ys = [s['y'] for s in self.sites]
        return min(xs), max(xs), min(ys), max(ys)

    def _layout(self, x0, x1, y0, y1, w, h):
        n_cols = x1 - x0 + 1
        n_rows = y1 - y0 + 1

        cx = w / 2.0
        cy = h / 2.0
        radius_max = min(cx, w - cx, cy, h - cy) - 0.5
        disc_margin = 2.0
        radius = max(1.0, radius_max - disc_margin)

        cell_hi = min(w / max(n_cols, 1), h / max(n_rows, 1))
        cell_lo = 1.0

        def fits(cell: float) -> bool:
            mg = max(1.0, cell * 0.03)
            half_grid_w = cell * n_cols / 2.0 - mg
            half_grid_h = cell * n_rows / 2.0 - mg
            half_grid_w = max(0.0, half_grid_w)
            half_grid_h = max(0.0, half_grid_h)
            dist = math.hypot(half_grid_w, half_grid_h)
            return dist <= (radius - 0.3)

        for _ in range(24):
            cell_mid = (cell_lo + cell_hi) / 2.0
            if fits(cell_mid):
                cell_lo = cell_mid
            else:
                cell_hi = cell_mid

        cell = cell_lo
        ox = (w - cell * n_cols) / 2.0
        oy = (h - cell * n_rows) / 2.0
        return ox, oy, cell

    # ── zoom controls ────────────────────────────────────────────────────────
    def zoom_in(self):
        self.set_zoom(self._zoom * 1.15)

    def zoom_out(self):
        self.set_zoom(self._zoom / 1.15)

    def reset_zoom(self):
        self.set_zoom(1.0)

    def set_zoom(self, z: float):
        z = max(0.6, min(3.5, float(z)))
        if abs(z - self._zoom) < 1e-6:
            return
        self._zoom = z
        self.update()

    def _to_logical(self, pos: QPointF) -> QPointF:
        # Map widget coords -> logical coords used by _rects when zoomed.
        cx = self.width() / 2.0
        cy = self.height() / 2.0
        return QPointF((pos.x() - cx) / self._zoom + cx, (pos.y() - cy) / self._zoom + cy)

    def _die_color(self, name):
        v = self.values.get(name)
        if v is None:
            return QColor(T['nodata_bg']), QColor(T['nodata_fg']), QColor(T['nodata_border'])
        if not self._limits_active:
            return QColor(T['neutral_bg']), QColor(T['neutral_fg']), QColor(T['neutral_border'])
        lo, hi  = self.low_limit, self.high_limit
        passed  = (lo is None or v >= lo) and (hi is None or v <= hi)
        if passed:
            return QColor(T['pass_bg']), QColor(T['pass_fg']), QColor(T['pass_border'])
        return QColor(T['fail_bg']), QColor(T['fail_fg']), QColor(T['fail_border'])

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        w, h = self.width(), self.height()

        p.fillRect(0, 0, w, h, QColor(T['bg_app']))
        p.setPen(QPen(QColor(T['border']), 1.0))
        for gx in range(0, w + 28, 28):
            for gy in range(0, h + 28, 28):
                p.drawPoint(gx, gy)

        if not self.sites:
            p.setPen(QColor(T['text_dim']))
            p.setFont(QFont('Segoe UI', 14))
            p.drawText(self.rect(), Qt.AlignCenter, 'Open a KDF file to begin')
            return

        # Zoom wafer content about the view center; draw legend unzoomed.
        cx_dev = w / 2.0
        cy_dev = h / 2.0
        p.save()
        p.translate(cx_dev, cy_dev)
        p.scale(self._zoom, self._zoom)
        p.translate(-cx_dev, -cy_dev)

        lw = w / self._zoom
        lh = h / self._zoom
        lx0 = cx_dev - lw / 2.0
        ly0 = cy_dev - lh / 2.0

        x0, x1, y0, y1 = self._bounds()
        ox, oy, cell    = self._layout(x0, x1, y0, y1, lw, lh)
        ox += lx0
        oy += ly0

        cx = cx_dev
        cy = cy_dev
        radius_max = min(lw / 2.0, lh / 2.0) - 0.5
        disc_margin = 2.0
        radius = max(1.0, radius_max - disc_margin)

        shad = QRadialGradient(cx+4, cy+4, radius+4)
        shad.setColorAt(0,   QColor(0, 0, 0, 24))
        shad.setColorAt(0.8, QColor(0, 0, 0,  8))
        shad.setColorAt(1.0, QColor(0, 0, 0,  0))
        p.setBrush(QBrush(shad)); p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(cx+4, cy+4), radius+4, radius+4)

        wg = QRadialGradient(cx - radius*0.15, cy - radius*0.2, radius*1.3)
        wg.setColorAt(0,    QColor('#ffffff'))
        wg.setColorAt(0.5,  QColor(T['wafer_bg']))
        wg.setColorAt(1.0,  QColor('#cfd9e8'))
        p.setBrush(QBrush(wg))
        p.setPen(QPen(QColor(T['wafer_edge']), 1.5))
        p.drawEllipse(QPointF(cx, cy), radius, radius)

        nw = radius * 0.22
        p.setPen(Qt.NoPen); p.setBrush(QColor(T['bg_app']))
        p.drawRect(QRectF(cx - nw/2, cy + radius - 6, nw, 6))
        p.setPen(QPen(QColor(T['wafer_edge']), 2))
        p.drawLine(QPointF(cx - nw/2, cy + radius - 4),
                   QPointF(cx + nw/2, cy + radius - 4))

        # die corner radius — scales with cell size, minimum 3 px
        die_radius = max(3.0, cell * 0.10)

        fs   = max(7, int(cell * 0.15))
        vfnt = QFont('Consolas', fs, QFont.Bold)
        cfnt = QFont('Consolas', max(6, fs - 2))
        self._rects = {}

        for site in self.sites:
            mg   = max(1.0, cell * 0.03)
            px_  = ox + (site['x'] - x0) * cell
            py_  = oy + (y1 - site['y']) * cell
            rect = QRectF(px_ + mg, py_ + mg, cell - 2*mg, cell - 2*mg)
            self._rects[site['name']] = rect

            bg, fg, bc = self._die_color(site['name'])
            is_sel = self.selected_site and site['name'] == self.selected_site['name']
            is_hov = self._hover         and site['name'] == self._hover['name']

            if cell > 30:
                p.setBrush(QColor(0,0,0,12)); p.setPen(Qt.NoPen)
                p.drawRoundedRect(
                    QRectF(rect.x()+2, rect.y()+2, rect.width(), rect.height()),
                    die_radius, die_radius)

            cg = QLinearGradient(rect.topLeft(), rect.bottomRight())
            cg.setColorAt(0, bg.lighter(108)); cg.setColorAt(1, bg)
            p.setBrush(QBrush(cg))
            p.setPen(QPen(QColor(T['selected']), 2.5) if is_sel
                     else QPen(QColor(T['hover_border']), 2) if is_hov
                     else QPen(bc, 1))
            p.drawRoundedRect(rect, die_radius, die_radius)

            v = self.values.get(site['name'])
            if v is not None and cell > 28:
                p.setFont(vfnt); p.setPen(fg)
                p.drawText(rect, Qt.AlignCenter, si_fmt(v))

            if cell > 58:
                p.setFont(cfnt); p.setPen(QColor(T['text_dim']))
                cr = QRectF(px_+mg+2, py_+mg+1, cell-2*mg-2, cell*0.28)
                p.drawText(cr, Qt.AlignLeft | Qt.AlignTop,
                           f"{site['x']},{site['y']}")

        p.restore()
        self._draw_legend(p, w, h)

    def _draw_legend(self, p, w, h):
        items = ([(T['pass_bg'],    T['pass_fg'],    'Pass'),
                  (T['fail_bg'],    T['fail_fg'],    'Fail'),
                  (T['nodata_bg'],  T['nodata_fg'],  'No data')]
                 if self._limits_active else
                 [(T['neutral_bg'], T['neutral_fg'], 'No limits set'),
                  (T['nodata_bg'],  T['nodata_fg'],  'No data')])

        bh = len(items) * 22 + 16
        lx = 14; ly = h - bh - 10
        # Warm translucent legend panel
        p.setBrush(QColor(255, 250, 242, 215))
        p.setPen(QPen(QColor(T['border']), 1))
        p.drawRoundedRect(QRectF(lx-6, ly-8, 178, bh), 8, 8)
        p.setFont(QFont('Segoe UI', 11))
        for bg_hex, fg_hex, label in items:
            p.setBrush(QBrush(QColor(bg_hex)))
            p.setPen(QPen(QColor(fg_hex), 1))
            p.drawRoundedRect(QRectF(lx, ly, 14, 14), 4, 4)
            p.setPen(QColor(T['text_primary']))
            p.drawText(int(lx+20), int(ly+11), label)
            ly += 22

    def mouseMoveEvent(self, e):
        pos = self._to_logical(QPointF(e.position())); self._hover = None
        for site in self.sites:
            r = self._rects.get(site['name'])
            if r and r.contains(pos):
                self._hover = site
                self.setCursor(Qt.PointingHandCursor)
                break
        else:
            self.setCursor(Qt.ArrowCursor)
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            pos = self._to_logical(QPointF(e.position()))
            for site in self.sites:
                r = self._rects.get(site['name'])
                if r and r.contains(pos):
                    self.selected_site = site
                    self.siteClicked.emit(site)
                    self.update()
                    return

    def leaveEvent(self, _e):
        self._hover = None; self.update()

# ─────────────────────────────────────────────────────────────────────────────
#  DIE DETAIL PANEL
# ─────────────────────────────────────────────────────────────────────────────

class _DesignTintDelegate(QStyledItemDelegate):
    def __init__(self, colors: list[QColor], role_design: int, parent=None):
        super().__init__(parent)
        self._colors = colors
        self._role_design = role_design

    def _color_for_design(self, design_num: int) -> QColor:
        return self._colors[(design_num - 1) % len(self._colors)]

    def paint(self, painter: QPainter, option, index):
        design_num = index.data(self._role_design)
        if isinstance(design_num, int):
            base = QColor(self._color_for_design(design_num))

            # Professional look: very subtle row wash + a clean left accent stripe.
            c = QColor(base)
            c.setAlpha(18)
            painter.save()
            painter.fillRect(option.rect, c)

            # Accent stripe only on first column to avoid looking "busy".
            if index.column() == 0:
                stripe = QColor(base)
                stripe.setAlpha(170)
                r = option.rect
                painter.fillRect(QRect(r.x(), r.y(), 4, r.height()), stripe)
            painter.restore()

        super().paint(painter, option, index)


class SiteDetailPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lo = QVBoxLayout(self); lo.setContentsMargins(8, 8, 8, 8); lo.setSpacing(8)

        self.title = QLabel('Click a die to inspect')
        self.title.setWordWrap(True)
        self.title.setStyleSheet(
            f'font-weight:bold;font-size:13px;color:{T["accent_dark"]};'
            f'padding:5px 8px;background:{T["accent_dim"]};'
            f'border-radius:8px;border-left:3px solid {T["accent"]};')
        lo.addWidget(self.title)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(['Measurement', 'Design #', 'Value'])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        # Alternating row colors fight with design tinting; keep the look clean.
        self.table.setAlternatingRowColors(False)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        # Prevent global QSS hover/selection from wiping our design tint.
        self.table.setStyleSheet(
            f"QTableWidget::item:hover{{background-color: rgba(0,0,0,0); border:1px solid {T['border']};}}"
            f"QTableWidget::item:selected{{background-color: rgba(0,0,0,0); border:1px solid {T['accent']}; color:{T['accent_dark']};}}"
        )
        lo.addWidget(self.table)

        self._design_colors = [
            QColor("#1565c0"),  # blue
            QColor("#2e7d32"),  # green
            QColor("#6a1b9a"),  # purple
            QColor("#c62828"),  # red
            QColor("#ef6c00"),  # orange
            QColor("#00838f"),  # teal
            QColor("#4e342e"),  # brown
            QColor("#455a64"),  # blue-grey
        ]
        self._role_design = int(Qt.UserRole) + 1
        self.table.setItemDelegate(
            _DesignTintDelegate(self._design_colors, self._role_design, self.table)
        )

    def _design_color(self, design_num: int) -> QColor:
        # stable color assignment per design number
        return self._design_colors[(design_num - 1) % len(self._design_colors)]

    def show_site(self, site: dict):
        self.title.setText(
            f'  {site["name"]}   ·   X = {site["x"]},  Y = {site["y"]}')

        rows = []
        for sub_num in sorted(site['subsites']):
            for mkey, val in sorted(site['subsites'][sub_num].items()):
                rows.append((mkey, sub_num, val))

        self.table.setRowCount(len(rows))
        for i, (mkey, sub, val) in enumerate(rows):
            multi = len(site.get('subsites', {})) > 1
            design_num = int(sub) if multi else None

            mi = QTableWidgetItem(mkey)
            mi.setFont(QFont('Segoe UI', 12))
            if design_num is not None:
                mi.setData(self._role_design, design_num)
            self.table.setItem(i, 0, mi)

            si = QTableWidgetItem(str(sub))
            si.setTextAlignment(Qt.AlignCenter)
            si.setForeground(QColor(T['text_secondary']))
            si.setFont(QFont('Segoe UI', 12))
            if design_num is not None:
                si.setData(self._role_design, design_num)
                si.setForeground(QColor(T['text_primary']))

            self.table.setItem(i, 1, si)

            display = si_fmt(val) if (val is not None and math.isfinite(val)) else 'N/A'
            vi = QTableWidgetItem(display)
            vi.setForeground(QColor(T['accent_dark']))
            vi.setFont(QFont('Consolas', 12, QFont.Bold))

            if design_num is not None:
                vi.setData(self._role_design, design_num)

            self.table.setItem(i, 2, vi)

# ─────────────────────────────────────────────────────────────────────────────
#  STATISTICS PANEL
# ─────────────────────────────────────────────────────────────────────────────

class StatsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lo = QVBoxLayout(self); lo.setContentsMargins(8, 8, 8, 8); lo.setSpacing(6)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(['Statistic', 'Value'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        lo.addWidget(self.table)

    def update_stats(self, values_dict: dict, lo, hi):
        vals = [v for v in values_dict.values()
                if v is not None and math.isfinite(v)]
        if not vals:
            self.table.setRowCount(0)
            return

        n      = len(vals)
        mean_v = statistics.mean(vals)
        std_v  = statistics.pstdev(vals)
        med_v  = statistics.median(vals)
        min_v  = min(vals)
        max_v  = max(vals)
        passed = sum(1 for v in vals
                     if (lo is None or v >= lo) and (hi is None or v <= hi))
        yld    = passed / n * 100

        rows = [
            ('Count',   str(n)),
            ('Mean',    si_fmt(mean_v)),
            ('Std Dev', si_fmt(std_v)),
            ('Min',     si_fmt(min_v)),
            ('Max',     si_fmt(max_v)),
            ('Median',  si_fmt(med_v)),
            ('3σ',      f'{si_fmt(mean_v - 3*std_v)}  →  {si_fmt(mean_v + 3*std_v)}'),
            ('Pass',    str(passed)),
            ('Fail',    str(n - passed)),
            ('Yield',   f'{yld:.1f}%'),
        ]
        self.table.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            ki = QTableWidgetItem(k)
            ki.setForeground(QColor(T['text_secondary']))
            ki.setFont(QFont('Segoe UI', 12))
            self.table.setItem(i, 0, ki)

            vi = QTableWidgetItem(v)
            vi.setFont(QFont('Consolas', 12))
            if k == 'Yield':
                col = (T['pass_fg'] if yld >= 90
                       else T['fail_fg'] if yld < 70
                       else T['warn'])
                vi.setForeground(QColor(col))
                vi.setFont(QFont('Segoe UI', 13, QFont.Bold))
            elif k == 'Pass': vi.setForeground(QColor(T['pass_fg']))
            elif k == 'Fail': vi.setForeground(QColor(T['fail_fg']))
            else:             vi.setForeground(QColor(T['text_primary']))
            self.table.setItem(i, 1, vi)

# ─────────────────────────────────────────────────────────────────────────────
#  BATCH WAFER THUMBNAIL  (small clickable wafer map for the batch grid)
# ─────────────────────────────────────────────────────────────────────────────

class BatchWaferThumb(QWidget):
    """A compact, clickable wafer map thumbnail used in the batch grid."""
    clicked = Signal(object)  # emits WaferData

    THUMB_SIZE = 170

    def __init__(self, wafer: WaferData, parent=None):
        super().__init__(parent)
        self.wafer      = wafer
        self._values:   dict = {}
        self._lo:       float | None = None
        self._hi:       float | None = None
        self._selected  = False
        self._yield_pct: float | None = None

        self.setFixedSize(self.THUMB_SIZE, self.THUMB_SIZE + 36)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(wafer.wafer_id)

    def update_data(self, values: dict, lo, hi, yield_pct):
        self._values    = values
        self._lo        = lo
        self._hi        = hi
        self._yield_pct = yield_pct
        self.update()

    def set_selected(self, sel: bool):
        self._selected = sel
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit(self.wafer)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)

        W = self.THUMB_SIZE
        # Card background
        card_rect = QRectF(0, 0, W, W)
        if self._selected:
            p.setBrush(QColor(T['accent_dim']))
            p.setPen(QPen(QColor(T['accent']), 2.5))
        else:
            p.setBrush(QColor(T['bg_panel']))
            p.setPen(QPen(QColor(T['border']), 1))
        p.drawRoundedRect(card_rect, 10, 10)

        # ── draw mini wafer ───────────────────────────────────────────────────
        sites  = self.wafer.sites
        margin = 10
        draw_w = W - margin * 2
        draw_h = W - margin * 2

        if sites:
            xs = [s['x'] for s in sites]; ys = [s['y'] for s in sites]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            n_cols = x1 - x0 + 1; n_rows = y1 - y0 + 1

            cx = W / 2.0; cy = W / 2.0
            radius = min(draw_w, draw_h) / 2.0 - 1.0

            # wafer disc
            wg = QRadialGradient(cx - radius*0.1, cy - radius*0.15, radius*1.2)
            wg.setColorAt(0,   QColor('#ffffff'))
            wg.setColorAt(0.5, QColor(T['wafer_bg']))
            wg.setColorAt(1.0, QColor('#cfd9e8'))
            p.setBrush(QBrush(wg))
            p.setPen(QPen(QColor(T['wafer_edge']), 1.0))
            p.drawEllipse(QPointF(cx, cy), radius, radius)

            # flat notch
            nw = radius * 0.18
            p.setPen(Qt.NoPen); p.setBrush(QColor(T['bg_app']))
            p.drawRect(QRectF(cx - nw/2, cy + radius - 4, nw, 5))
            p.setPen(QPen(QColor(T['wafer_edge']), 1.2))
            p.drawLine(QPointF(cx - nw/2, cy + radius - 3),
                       QPointF(cx + nw/2, cy + radius - 3))

            # cells
            cell = min(draw_w / max(n_cols, 1), draw_h / max(n_rows, 1))
            ox = (W - cell * n_cols) / 2.0
            oy = (W - cell * n_rows) / 2.0
            mg = max(0.5, cell * 0.06)
            limits_active = (self._lo is not None or self._hi is not None)

            for site in sites:
                v    = self._values.get(site['name'])
                px_  = ox + (site['x'] - x0) * cell
                py_  = oy + (y1 - site['y']) * cell
                rect = QRectF(px_ + mg, py_ + mg, cell - 2*mg, cell - 2*mg)

                if v is None:
                    bg = QColor(T['nodata_bg']); bc = QColor(T['nodata_border'])
                elif not limits_active:
                    bg = QColor(T['neutral_bg']); bc = QColor(T['neutral_border'])
                else:
                    ok = (self._lo is None or v >= self._lo) and \
                         (self._hi is None or v <= self._hi)
                    bg = QColor(T['pass_bg'] if ok else T['fail_bg'])
                    bc = QColor(T['pass_border'] if ok else T['fail_border'])

                p.setBrush(bg); p.setPen(QPen(bc, 0.6))
                r = max(1.0, cell * 0.12)
                p.drawRoundedRect(rect, r, r)

        # ── label strip ───────────────────────────────────────────────────────
        label_y = W + 2
        label_h = self.height() - label_y

        # wafer id
        p.setFont(QFont('Segoe UI', 9, QFont.Bold))
        p.setPen(QColor(T['accent_dark'] if self._selected else T['text_primary']))
        p.drawText(QRectF(2, label_y, W - 4, 16),
                   Qt.AlignHCenter | Qt.AlignVCenter,
                   self.wafer.wafer_id)

        # yield badge
        if self._yield_pct is not None:
            yld = self._yield_pct
            bg_c  = T['pass_bg']  if yld >= 90 else T['fail_bg']  if yld < 70 else '#fff8e1'
            fg_c  = T['pass_fg']  if yld >= 90 else T['fail_fg']  if yld < 70 else T['warn']
            bdr_c = T['pass_border'] if yld >= 90 else T['fail_border'] if yld < 70 else '#ffcc02'
            badge = QRectF(W//2 - 34, label_y + 16, 68, 16)
            p.setBrush(QColor(bg_c)); p.setPen(QPen(QColor(bdr_c), 0.8))
            p.drawRoundedRect(badge, 6, 6)
            p.setFont(QFont('Segoe UI', 8, QFont.Bold))
            p.setPen(QColor(fg_c))
            p.drawText(badge, Qt.AlignCenter, f'{yld:.1f}% yield')
        elif sites:
            p.setFont(QFont('Segoe UI', 8))
            p.setPen(QColor(T['text_dim']))
            p.drawText(QRectF(2, label_y + 16, W - 4, 16),
                       Qt.AlignHCenter | Qt.AlignVCenter, 'no limits')

        p.end()


# ─────────────────────────────────────────────────────────────────────────────
#  BATCH ANALYSIS TAB
# ─────────────────────────────────────────────────────────────────────────────

class BatchAnalysisTab(QWidget):
    """
    Full batch analysis view.
    - Top: scrollable grid of wafer thumbnails
    - Bottom-left: sortable summary table (one row per wafer)
    - Bottom-right: full-size wafer detail for the selected wafer
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._wafers:       list[WaferData]        = []
        self._thumbs:       list[BatchWaferThumb]  = []
        self._selected_idx: int | None             = None
        self._mkey:         str | None             = None
        self._sub:          int | None             = None
        self._lo:           float | None           = None
        self._hi:           float | None           = None

        self._build_ui()

    # ── build ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ── top: thumb grid inside a scroll area ──────────────────────────────
        thumb_box = QGroupBox('Batch Overview  —  click a wafer to inspect')
        thumb_box.setFixedHeight(250)
        tb_lay = QVBoxLayout(thumb_box); tb_lay.setContentsMargins(6, 18, 6, 6)

        from PySide6.QtWidgets import QScrollArea
        self._scroll = QScrollArea()
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(self._scroll.NoFrame)
        self._scroll.setStyleSheet('background:transparent;')

        self._thumb_container = QWidget()
        self._thumb_container.setStyleSheet('background:transparent;')
        self._thumb_layout = QHBoxLayout(self._thumb_container)
        self._thumb_layout.setContentsMargins(4, 4, 4, 4)
        self._thumb_layout.setSpacing(10)
        self._thumb_layout.addStretch()

        self._scroll.setWidget(self._thumb_container)
        tb_lay.addWidget(self._scroll)
        root.addWidget(thumb_box)

        # ── bottom: summary table  |  wafer detail ───────────────────────────
        bot = QHBoxLayout(); bot.setSpacing(10)

        # summary table
        tbl_box = QGroupBox('Wafer Summary')
        tbl_lay = QVBoxLayout(tbl_box); tbl_lay.setContentsMargins(6, 18, 6, 6)

        self._tbl = QTableWidget(0, 8)
        self._tbl.setHorizontalHeaderLabels(
            ['Wafer', 'Sites', 'Pass', 'Fail', 'No Data', 'Yield %', 'Mean', 'Std Dev'])
        hh = self._tbl.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, 8):
            hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self._tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        self._tbl.setAlternatingRowColors(True)
        self._tbl.verticalHeader().setVisible(False)
        self._tbl.setShowGrid(False)
        self._tbl.setSelectionBehavior(QTableWidget.SelectRows)
        self._tbl.currentRowChanged.connect(self._on_table_row_changed)
        tbl_lay.addWidget(self._tbl)

        # batch-level stats beneath the table
        self._batch_stats_lbl = QLabel('')
        self._batch_stats_lbl.setWordWrap(True)
        self._batch_stats_lbl.setStyleSheet(
            f'color:{T["text_secondary"]};font-size:12px;'
            f'padding:6px 8px;background:{T["bg_header"]};'
            f'border-radius:6px;border:1px solid {T["border"]};')
        tbl_lay.addWidget(self._batch_stats_lbl)
        bot.addWidget(tbl_box, stretch=5)

        # wafer detail (right side)
        detail_box = QGroupBox('Wafer Detail')
        detail_lay = QVBoxLayout(detail_box)
        detail_lay.setContentsMargins(6, 18, 6, 6)
        detail_lay.setSpacing(6)

        # wafer id header
        self._detail_header = QLabel('Select a wafer')
        self._detail_header.setStyleSheet(
            f'font-weight:bold;font-size:13px;color:{T["accent_dark"]};'
            f'padding:5px 8px;background:{T["accent_dim"]};'
            f'border-radius:8px;border-left:3px solid {T["accent"]};')
        detail_lay.addWidget(self._detail_header)

        # canvas + die detail in a splitter-like hbox
        canvas_die = QHBoxLayout(); canvas_die.setSpacing(8)

        self._detail_canvas = WaferCanvas()
        self._detail_canvas.setMinimumSize(320, 300)
        self._detail_canvas.siteClicked.connect(self._on_detail_die_clicked)

        canvas_wrap = QWidget()
        canvas_wrap.setStyleSheet(
            f'background:{T["bg_panel"]};border:1px solid {T["border"]};'
            f'border-radius:10px;')
        cw_lay = QVBoxLayout(canvas_wrap); cw_lay.setContentsMargins(3, 3, 3, 3)
        cw_lay.addWidget(self._detail_canvas)
        canvas_die.addWidget(canvas_wrap, stretch=3)

        right_detail = QWidget(); right_detail.setFixedWidth(230)
        rd_lay = QVBoxLayout(right_detail)
        rd_lay.setContentsMargins(0, 0, 0, 0); rd_lay.setSpacing(6)

        self._detail_stats = StatsPanel()
        rd_lay.addWidget(QLabel('Statistics'), 0)
        rd_lay.addWidget(self._detail_stats, 2)

        self._detail_die = SiteDetailPanel()
        rd_lay.addWidget(QLabel('Die Detail'), 0)
        rd_lay.addWidget(self._detail_die, 3)
        canvas_die.addWidget(right_detail, stretch=0)

        detail_lay.addLayout(canvas_die)
        bot.addWidget(detail_box, stretch=6)

        root.addLayout(bot, stretch=1)

        self._show_empty_state()

    # ── public API ────────────────────────────────────────────────────────────

    def load_wafers(self, wafers: list[WaferData]):
        self._wafers = wafers
        self._selected_idx = None
        self._rebuild_thumbs()
        self._rebuild_table()
        self._show_empty_state()

    def refresh(self, mkey: str | None, sub: int | None,
                lo: float | None, hi: float | None):
        """Called whenever measurement / limits change in the main window."""
        self._mkey = mkey
        self._sub  = sub
        self._lo   = lo
        self._hi   = hi
        self._refresh_thumbs()
        self._rebuild_table()
        if self._selected_idx is not None:
            self._refresh_detail(self._selected_idx)

    # ── internals ─────────────────────────────────────────────────────────────

    def _rebuild_thumbs(self):
        # Remove old thumbs
        while self._thumb_layout.count() > 1:          # keep trailing stretch
            item = self._thumb_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._thumbs = []

        for i, w in enumerate(self._wafers):
            thumb = BatchWaferThumb(w)
            thumb.clicked.connect(lambda ww, idx=i: self._select_wafer(idx))
            self._thumb_layout.insertWidget(i, thumb)
            self._thumbs.append(thumb)

        self._refresh_thumbs()

    def _refresh_thumbs(self):
        if not self._mkey:
            for th in self._thumbs:
                th.update_data({}, None, None, None)
            return
        for i, (th, w) in enumerate(zip(self._thumbs, self._wafers)):
            vals = {s['name']: get_site_value(s, self._mkey, self._sub)
                    for s in w.sites}
            st   = w.yield_stats(self._mkey, self._sub, self._lo, self._hi)
            th.update_data(vals, self._lo, self._hi, st['yield_pct'])
            th.set_selected(i == self._selected_idx)

    def _rebuild_table(self):
        self._tbl.blockSignals(True)
        self._tbl.setRowCount(len(self._wafers))

        batch_yields = []

        for row, w in enumerate(self._wafers):
            if self._mkey:
                st = w.yield_stats(self._mkey, self._sub, self._lo, self._hi)
            else:
                st = dict(passed=0, failed=0, nodata=len(w.sites),
                          total=0, yield_pct=None, mean=None, stdev=None,
                          min_v=None, max_v=None)

            yld = st['yield_pct']
            if yld is not None:
                batch_yields.append(yld)

            def cell(txt, align=Qt.AlignCenter, bold=False, color=None):
                it = QTableWidgetItem(txt)
                it.setTextAlignment(align)
                f  = QFont('Segoe UI', 12)
                if bold: f.setBold(True)
                it.setFont(f)
                if color: it.setForeground(QColor(color))
                return it

            # colour the row faintly based on yield
            row_bg = None
            if yld is not None:
                if yld >= 90:   row_bg = QColor(T['pass_bg'])
                elif yld < 70:  row_bg = QColor(T['fail_bg'])
                else:           row_bg = QColor('#fffde7')

            cols = [
                cell(w.wafer_id, Qt.AlignLeft | Qt.AlignVCenter, bold=True,
                     color=T['accent_dark']),
                cell(str(len(w.sites))),
                cell(str(st['passed']),  color=T['pass_fg'] if st['passed'] else None),
                cell(str(st['failed']),  color=T['fail_fg'] if st['failed'] else None),
                cell(str(st['nodata']),  color=T['nodata_fg'] if st['nodata'] else None),
                cell(f"{yld:.1f}%" if yld is not None else '—',
                     bold=True,
                     color=(T['pass_fg'] if (yld or 0) >= 90
                            else T['fail_fg'] if (yld or 0) < 70
                            else T['warn']) if yld is not None else T['text_dim']),
                cell(si_fmt(st['mean'])  if st['mean']  is not None else '—'),
                cell(si_fmt(st['stdev']) if st['stdev'] is not None else '—'),
            ]
            for c, it in enumerate(cols):
                if row_bg:
                    it.setBackground(row_bg)
                self._tbl.setItem(row, c, it)

        self._tbl.blockSignals(False)

        # batch-level summary line
        if batch_yields and self._mkey:
            avg_y  = statistics.mean(batch_yields)
            best_i = max(range(len(batch_yields)), key=lambda i: batch_yields[i])
            worst_i= min(range(len(batch_yields)), key=lambda i: batch_yields[i])
            self._batch_stats_lbl.setText(
                f'Batch: {len(self._wafers)} wafers  ·  '
                f'Avg yield: {avg_y:.1f}%  ·  '
                f'Best: {self._wafers[best_i].wafer_id} ({batch_yields[best_i]:.1f}%)  ·  '
                f'Worst: {self._wafers[worst_i].wafer_id} ({batch_yields[worst_i]:.1f}%)'
            )
        elif self._wafers:
            self._batch_stats_lbl.setText(
                f'{len(self._wafers)} wafers loaded  ·  '
                f'Set a measurement in the main controls to see yield data.')
        else:
            self._batch_stats_lbl.setText('No batch loaded.')

    def _select_wafer(self, idx: int):
        prev = self._selected_idx
        self._selected_idx = idx

        # update thumb highlight
        if prev is not None and prev < len(self._thumbs):
            self._thumbs[prev].set_selected(False)
        if idx < len(self._thumbs):
            self._thumbs[idx].set_selected(True)

        # sync table selection (without re-triggering _on_table_row_changed)
        self._tbl.blockSignals(True)
        self._tbl.setCurrentCell(idx, 0)
        self._tbl.blockSignals(False)

        self._refresh_detail(idx)

    def _on_table_row_changed(self, row: int):
        if 0 <= row < len(self._wafers):
            self._select_wafer(row)

    def _refresh_detail(self, idx: int):
        if idx < 0 or idx >= len(self._wafers):
            return
        w = self._wafers[idx]

        # header
        lot = w.header.get('LOT', '—')
        self._detail_header.setText(
            f'  {w.wafer_id}   ·   Lot: {lot}   ·   {len(w.sites)} sites')

        if not self._mkey:
            self._detail_canvas.load([], {}, None, None)
            self._detail_stats.update_stats({}, None, None)
            return

        vals = {s['name']: get_site_value(s, self._mkey, self._sub)
                for s in w.sites}
        self._detail_canvas.load(w.sites, vals, self._lo, self._hi,
                                 mkey=self._mkey)
        self._detail_stats.update_stats(vals, self._lo, self._hi)

    def _on_detail_die_clicked(self, site: dict):
        self._detail_die.show_site(site)

    def _show_empty_state(self):
        self._detail_header.setText('Select a wafer')
        self._detail_canvas.load([], {}, None, None)
        self._detail_stats.update_stats({}, None, None)
        self._batch_stats_lbl.setText('No batch loaded — use "Open Batch…" in the toolbar.')


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Wafer Map Viewer')
        self.resize(1300, 840)
        self.setStyleSheet(SS)
        self.setWindowIcon(make_app_icon())

        self._header:       dict      = {}
        self._sites:        list      = []
        self._mkeys:        list[str] = []
        self._limits:       dict      = {}
        self._current_mkey: str | None = None
        self._filepath:     str | None = None
        self._all_subsites: list[int]  = []
        self._current_sub:  int | None = None
        self._batch_wafers: list[WaferData] = []

        self._build_ui()
        self._update_ui_state()
        self.showMaximized()

    def _build_ui(self):
        tb = QToolBar('Main', self)
        tb.setIconSize(QSize(18, 18)); tb.setMovable(False)
        self.addToolBar(tb)

        # Create early so toolbar actions can reference it.
        self.canvas = WaferCanvas()

        open_act = QAction('  Open KDF…', self)
        open_act.triggered.connect(self.open_file)
        tb.addAction(open_act)
        tb.addSeparator()

        batch_act = QAction('  Open Batch…', self)
        batch_act.triggered.connect(self.open_batch)
        tb.addAction(batch_act)
        tb.addSeparator()

        exp_act = QAction('  Export Map…', self)
        exp_act.triggered.connect(self.export_map)
        tb.addAction(exp_act)
        tb.addSeparator()

        zoom_out = QAction('  −', self)
        zoom_out.setToolTip('Zoom out')
        zoom_out.triggered.connect(self.canvas.zoom_out)
        tb.addAction(zoom_out)

        zoom_in = QAction('  +', self)
        zoom_in.setToolTip('Zoom in')
        zoom_in.triggered.connect(self.canvas.zoom_in)
        tb.addAction(zoom_in)

        zoom_reset = QAction('  Reset Zoom', self)
        zoom_reset.setToolTip('Reset zoom')
        zoom_reset.triggered.connect(self.canvas.reset_zoom)
        tb.addAction(zoom_reset)
        tb.addSeparator()

        sp = QWidget(); sp.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(sp)
        self.lbl_file = QLabel('No file loaded  ')
        self.lbl_file.setStyleSheet(
            f'color:{T["accent_dark"]};'
            f'font-size:12px;'
            f'font-weight:700;'
            f'padding:4px 10px;'
            f'background:{T["accent_dim"]};'
            f'border:1px solid {T["border_hi"]};'
            f'border-radius:10px;'
            f'min-height:22px;'
        )
        tb.addWidget(self.lbl_file)

        cw = QWidget(); self.setCentralWidget(cw)
        main_v = QVBoxLayout(cw); main_v.setContentsMargins(0, 0, 0, 0); main_v.setSpacing(0)

        # top-level tab widget: Single Wafer | Batch
        self._top_tabs = QTabWidget()
        main_v.addWidget(self._top_tabs)

        # ── Single Wafer tab ──────────────────────────────────────────────────
        single_widget = QWidget()
        mh = QHBoxLayout(single_widget); mh.setSpacing(10); mh.setContentsMargins(10, 10, 10, 10)

        # ── left panel ────────────────────────────────────────────────────────
        left = QWidget(); left.setFixedWidth(320)
        lv = QVBoxLayout(left); lv.setSpacing(12); lv.setContentsMargins(0, 0, 0, 0)

        ib = QGroupBox('File Information')
        iform = QFormLayout(ib); iform.setSpacing(8)
        iform.setLabelAlignment(Qt.AlignRight)

        def mkval():
            l = QLabel('—')
            l.setStyleSheet(
                f'color:{T["text_primary"]};font-weight:700;font-size:14px;')
            return l

        self.lbl_lot   = mkval(); self.lbl_sys   = mkval()
        self.lbl_stt   = mkval(); self.lbl_cnt   = mkval()
        self.lbl_tests = mkval(); self.lbl_dsn   = mkval()

        for lbl_txt, wid in [
            ('Lot',     self.lbl_lot),
            ('System',  self.lbl_sys),
            ('Start',   self.lbl_stt),
            ('Sites',   self.lbl_cnt),
            ('Tests',   self.lbl_tests),
            ('Designs', self.lbl_dsn),
        ]:
            kl = QLabel(lbl_txt)
            kl.setStyleSheet(f'color:{T["text_secondary"]};font-size:13px;')
            iform.addRow(kl, wid)
        lv.addWidget(ib)

        # design selector — uses ArrowComboBox
        db = QGroupBox('Design')
        dv = QVBoxLayout(db); dv.setSpacing(8)
        desc = QLabel(
            'Each die may contain multiple designs.\n'
            'Select which design to display.')
        desc.setWordWrap(True)
        desc.setStyleSheet(f'color:{T["text_secondary"]};font-size:12px;')
        dv.addWidget(desc)
        self.design_combo = ArrowComboBox()
        self.design_combo.setMinimumHeight(36)
        self.design_combo.currentIndexChanged.connect(self._on_design_changed)
        dv.addWidget(self.design_combo)
        lv.addWidget(db)

        # measurement selector — uses ArrowComboBox
        mb = QGroupBox('Measurement')
        mv = QVBoxLayout(mb); mv.setSpacing(8)
        self.mkey_combo = ArrowComboBox()
        self.mkey_combo.setMinimumHeight(36)
        self.mkey_combo.currentTextChanged.connect(self._on_mkey_changed)
        mv.addWidget(self.mkey_combo)
        lv.addWidget(mb)

        # pass/fail limits
        lb = QGroupBox('Pass / Fail Limits')
        lbv = QVBoxLayout(lb); lbv.setSpacing(8)
        for lbl_txt, attr in [('Low', 'low_edit'), ('High', 'high_edit')]:
            row = QHBoxLayout()
            rl = QLabel(lbl_txt)
            rl.setStyleSheet(
                f'color:{T["text_secondary"]};font-size:12px;min-width:36px;')
            row.addWidget(rl)
            en = QLineEdit(); en.setPlaceholderText('no limit')
            setattr(self, attr, en); row.addWidget(en); lbv.addLayout(row)
        br = QHBoxLayout(); br.setSpacing(6)
        ab = QPushButton('Apply'); ab.setObjectName('primary')
        ab.clicked.connect(self._apply_limits)
        cb = QPushButton('Clear'); cb.clicked.connect(self._clear_limits)
        br.addWidget(ab); br.addWidget(cb); lbv.addLayout(br)
        lv.addWidget(lb)

        lv.addStretch()
        mh.addWidget(left)

        # ── centre canvas ─────────────────────────────────────────────────────
        canvas_wrap = QWidget()
        canvas_wrap.setStyleSheet(
            f'background:{T["bg_panel"]};border:1px solid {T["border"]};'
            f'border-radius:12px;')
        cl = QVBoxLayout(canvas_wrap); cl.setContentsMargins(4, 4, 4, 4)
        self.canvas.siteClicked.connect(self._on_die_clicked)
        cl.addWidget(self.canvas)
        mh.addWidget(canvas_wrap, stretch=3)

        # ── right panel ───────────────────────────────────────────────────────
        right = QWidget(); right.setFixedWidth(310)
        rv = QVBoxLayout(right); rv.setContentsMargins(0, 0, 0, 0)
        tabs = QTabWidget(); rv.addWidget(tabs)

        self.detail_panel = SiteDetailPanel()
        tabs.addTab(self.detail_panel, 'Die Detail')

        self.stats_panel = StatsPanel()
        tabs.addTab(self.stats_panel, 'Statistics')

        mh.addWidget(right)

        self._top_tabs.addTab(single_widget, '  Single Wafer  ')

        # ── Batch Analysis tab ────────────────────────────────────────────────
        self._batch_tab = BatchAnalysisTab()
        self._top_tabs.addTab(self._batch_tab, '  Batch Analysis  ')

        self.status = QStatusBar(); self.setStatusBar(self.status)
        self.status.showMessage('  Ready  ·  open a KDF file to begin')

    # ── file loading ──────────────────────────────────────────────────────────

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open KDF File', '',
            'KDF Files (*.kdf *.KDF);;All Files (*)')
        if path:
            self._load_kdf(path)

    def open_batch(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, 'Open Batch — select KDF files', '',
            'KDF Files (*.kdf *.KDF);;All Files (*)')
        if paths:
            self._load_batch(sorted(paths))

    def _load_batch(self, paths: list[str]):
        wafers = []
        all_mkeys: set[str] = set()
        errors = []

        for path in paths:
            try:
                header, sites, mkeys, tests = parse_kdf(path)
                wid = os.path.splitext(os.path.basename(path))[0]
                subs = all_subsites(sites)
                wd = WaferData(
                    filepath=path,
                    wafer_id=wid,
                    header=header,
                    sites=sites,
                    mkeys=mkeys,
                    tests=tests,
                    subsites=subs,
                )
                wafers.append(wd)
                all_mkeys.update(mkeys)
            except Exception as e:
                errors.append(f'{os.path.basename(path)}: {e}')

        if errors:
            QMessageBox.warning(self, 'Batch Load Warnings',
                                'Some files could not be loaded:\n\n' +
                                '\n'.join(errors))
        if not wafers:
            return

        self._batch_wafers = wafers

        # Merge mkeys into the measurement combo (union of all wafers)
        merged = sorted(all_mkeys | set(self._mkeys))
        self._mkeys = merged
        cur = self._current_mkey
        self.mkey_combo.blockSignals(True)
        self.mkey_combo.clear()
        self.mkey_combo.addItems(merged)
        if cur and cur in merged:
            self.mkey_combo.setCurrentText(cur)
        elif merged:
            self._current_mkey = merged[0]
            self.mkey_combo.setCurrentIndex(0)
        self.mkey_combo.blockSignals(False)

        self._batch_tab.load_wafers(wafers)
        self._refresh_batch()

        # Switch to batch tab automatically
        self._top_tabs.setCurrentIndex(1)

        self.status.showMessage(
            f'  Batch: {len(wafers)} wafers loaded  ·  '
            f'{sum(len(w.sites) for w in wafers)} total sites'
        )

    def _load_kdf(self, path: str):
        try:
            header, sites, params, tests = parse_kdf(path)
        except Exception as e:
            QMessageBox.critical(self, 'Parse Error',
                                 f'Could not read KDF file:\n{e}')
            return

        self._filepath     = path
        self._header       = header
        self._sites        = sites
        self._mkeys        = params
        self._limits       = {}
        self._current_mkey = None

        subs = all_subsites(sites)
        self._all_subsites = subs
        self._current_sub  = subs[0] if len(subs) == 1 else None

        self.lbl_file.setText(f'  {os.path.basename(path)}  ')
        self.lbl_lot.setText(header.get('LOT', '—'))
        self.lbl_sys.setText(header.get('SYS', '—') or header.get('TST', '—'))
        self.lbl_stt.setText(header.get('STT', '—'))
        self.lbl_cnt.setText(str(len(sites)))
        self.lbl_tests.setText(str(len(tests)))
        self.lbl_dsn.setText(str(len(subs)))

        self.setWindowTitle(
            f'Wafer Map Viewer  ·  {header.get("LOT", os.path.basename(path))}')

        # Build design selector (may be sorted by pass count once limits are active).
        self._rebuild_design_combo()

        self.mkey_combo.blockSignals(True)
        self.mkey_combo.clear()
        self.mkey_combo.addItems(params)
        self.mkey_combo.blockSignals(False)

        if params:
            self._current_mkey = params[0]
            self.mkey_combo.setCurrentText(params[0])
            self._refresh_canvas()

        self._update_ui_state()
        self.status.showMessage(
            f'  Loaded {len(sites)} sites  ·  '
            f'{len(params)} measurements  ·  '
            f'{len(tests)} tests  ·  '
            f'{len(subs)} design(s)  ·  {os.path.basename(path)}')

    # ── design / measurement / limits ─────────────────────────────────────────

    def _rebuild_design_combo(self):
        """Populate design selector, optionally sorted by pass count when limits are active."""
        subs = list(self._all_subsites or [])
        if not subs:
            return

        # Preserve current selection if possible.
        prev_sub = self._current_sub

        mkey = self._current_mkey
        lo = hi = None
        if mkey:
            lo, hi = self._limits.get(mkey, (None, None))
        limits_active = (lo is not None or hi is not None) and bool(mkey)

        def pass_and_total_for(sub_num: int) -> tuple[int, int]:
            passed = 0
            total = 0
            for s in self._sites:
                v = get_site_value(s, mkey, sub_num)
                if v is None:
                    continue
                total += 1
                if (lo is None or v >= lo) and (hi is None or v <= hi):
                    passed += 1
            return passed, total

        ordered = subs
        pass_counts: dict[int, int] = {}
        total_counts: dict[int, int] = {}
        if limits_active:
            for sn in subs:
                p, t = pass_and_total_for(sn)
                pass_counts[sn] = p
                total_counts[sn] = t
            ordered = sorted(subs, key=lambda sn: (pass_counts.get(sn, 0), -sn), reverse=True)

        self.design_combo.blockSignals(True)
        self.design_combo.clear()

        for sn in ordered:
            if limits_active:
                pc = pass_counts.get(sn, 0)
                tc = total_counts.get(sn, 0)
                pct = (pc / tc * 100.0) if tc else 0.0
                self.design_combo.addItem(f'Design {sn}  ·  Pass {pc}/{tc} ({pct:.1f}%)', sn)
            else:
                self.design_combo.addItem(f'Design {sn}', sn)

        # Restore selection if possible; otherwise pick first real design (or only).
        idx = self.design_combo.findData(prev_sub)
        if idx < 0:
            idx = 0
        self.design_combo.setCurrentIndex(idx)
        self._current_sub = self.design_combo.itemData(idx)
        self.design_combo.blockSignals(False)

    def _on_design_changed(self, idx: int):
        if idx < 0:
            return
        self._current_sub = self.design_combo.itemData(idx)
        self._refresh_canvas()
        sub = self._current_sub
        label = f'Design {sub}' if sub is not None else 'Design'
        if self._current_mkey:
            self.status.showMessage(
                f'  {self._current_mkey}  ·  {label}  ·  {len(self._sites)} sites')

    def _on_mkey_changed(self, mkey: str):
        if mkey not in self._mkeys:
            return
        if self._current_mkey:
            lo = self._parse_limit(self.low_edit.text())
            hi = self._parse_limit(self.high_edit.text())
            self._limits[self._current_mkey] = (lo, hi)

        self._current_mkey = mkey
        lo, hi = self._limits.get(mkey, (None, None))
        self.low_edit.setText('' if lo is None else str(lo))
        self.high_edit.setText('' if hi is None else str(hi))
        self._rebuild_design_combo()
        self._refresh_canvas()

    def _apply_limits(self):
        lo = self._parse_limit(self.low_edit.text(),  'Low')
        hi = self._parse_limit(self.high_edit.text(), 'High')
        if lo is None and self.low_edit.text().strip():
            return
        if hi is None and self.high_edit.text().strip():
            return
        if self._current_mkey:
            self._limits[self._current_mkey] = (lo, hi)
        self._rebuild_design_combo()
        self._refresh_canvas()
        self._refresh_batch()

    def _clear_limits(self):
        self.low_edit.clear(); self.high_edit.clear()
        if self._current_mkey:
            self._limits[self._current_mkey] = (None, None)
        self._rebuild_design_combo()
        self._refresh_canvas()
        self._refresh_batch()

    def _parse_limit(self, text: str, label: str = '') -> float | None:
        text = text.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            if label:
                QMessageBox.warning(self, 'Invalid value',
                                    f'{label} limit must be a number (e.g. 9.5 or 1e-12).')
            return None

    def _refresh_canvas(self):
        if not self._sites or not self._current_mkey:
            return
        mkey   = self._current_mkey
        sub    = self._current_sub
        lo, hi = self._limits.get(mkey, (None, None))

        values = {s['name']: get_site_value(s, mkey, sub) for s in self._sites}

        self.canvas.load(self._sites, values, lo, hi, mkey=mkey)
        self.stats_panel.update_stats(values, lo, hi)

        sub_label = f'design {sub}' if sub is not None else 'design'
        lo_s = '—' if lo is None else str(lo)
        hi_s = '—' if hi is None else str(hi)
        self.status.showMessage(
            f'  {mkey}  ·  {sub_label}'
            f'  ·  Low = {lo_s}  High = {hi_s}'
            f'  ·  {len(self._sites)} sites')
        self._refresh_batch()

    def _refresh_batch(self):
        """Push current mkey / sub / limits to the batch tab."""
        if not self._batch_wafers:
            return
        mkey = self._current_mkey
        sub  = self._current_sub
        lo, hi = self._limits.get(mkey, (None, None)) if mkey else (None, None)
        self._batch_tab.refresh(mkey, sub, lo, hi)

    def _on_die_clicked(self, site: dict):
        self.detail_panel.show_site(site)

    # ── export ────────────────────────────────────────────────────────────────

    def export_map(self):
        if not self._sites:
            QMessageBox.information(self, 'Nothing to export',
                                    'Load a KDF file first.')
            return

        default_name = 'wafer_map.png'
        if self._filepath:
            base = os.path.splitext(os.path.basename(self._filepath))[0]
            default_name = f'{base}.png'

        path, selected_filter = QFileDialog.getSaveFileName(
            self, 'Export Wafer Map', default_name,
            'PNG Image (*.png);;JPEG Image (*.jpg)')
        if not path:
            return

        # Ensure a file extension so Qt can select an image writer reliably.
        lower = path.lower()
        if not (lower.endswith('.png') or lower.endswith('.jpg') or lower.endswith('.jpeg')):
            if 'jpeg' in (selected_filter or '').lower() or 'jpg' in (selected_filter or '').lower():
                path += '.jpg'
            else:
                path += '.png'

        SCALE = 3
        lw = self.canvas.width()
        lh = self.canvas.height()
        pw = lw * SCALE
        ph = lh * SCALE

        img = QImage(pw, ph, QImage.Format_ARGB32_Premultiplied)
        img.fill(Qt.transparent)

        painter = QPainter(img)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.scale(SCALE, SCALE)
        # PySide6 compatibility: some builds require an explicit targetOffset.
        self.canvas.render(painter, QPoint(0, 0))
        painter.end()

        fmt = b'PNG'
        if path.lower().endswith(('.jpg', '.jpeg')):
            fmt = b'JPEG'
        writer = QImageWriter(path, fmt)
        if fmt == b'JPEG':
            writer.setQuality(95)

        if writer.write(img):
            self.status.showMessage(
                f'  Exported {pw}×{ph} px  ·  {path}')
        else:
            QMessageBox.critical(
                self,
                'Export Error',
                f'Failed to save image.\n\n{writer.errorString()}'
            )

    # ── ui state ──────────────────────────────────────────────────────────────

    def _update_ui_state(self):
        has = bool(self._sites)
        self.design_combo.setEnabled(has)
        self.mkey_combo.setEnabled(has)
        self.low_edit.setEnabled(has)
        self.high_edit.setEnabled(has)

# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName('Wafer Map Viewer')
    app.setStyle('Fusion')
    app.setWindowIcon(make_app_icon())
    win = MainWindow()
    win.show()
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        win._load_kdf(sys.argv[1])
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
