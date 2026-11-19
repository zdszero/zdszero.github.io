#!/usr/bin/env python3
"""股票复盘图生成器：JSON 规格 → 与 docs/WikiImage 同风格的 SVG。

用法：python3 -I stock_chart.py spec.json out.svg
图表类型与字段见 ../references/charts.md。所有坐标都由数据算出，不手工估点。
"""
import json
import math
import re
import sys
import datetime
import xml.dom.minidom
from xml.sax.saxutils import escape

# 色板与 docs/WikiImage 现有图一致：(描边, 文字, 浅底)
PALETTE = {
    "blue": ("#4c6ef5", "#3b5bdb", "#edf2ff"),
    "teal": ("#0ca678", "#0b7285", "#e6fcf5"),
    "amber": ("#f08c00", "#e8590c", "#fff4e6"),
    "grape": ("#7048e8", "#5f3dc4", "#f3f0ff"),
    "red": ("#fa5252", "#c92a2a", "#fff5f5"),
    "gray": ("#adb5bd", "#495057", "#f1f3f5"),
}
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
W, L, R = 900, 80, 860
WARN = []


def col(name, i=0):
    return PALETTE.get(name or "gray", PALETTE["gray"])[i]


def tnum(s, edge=False):
    """日期 → 小数年。支持 2021-07-07、2021-07、2021Q3、2021H1、2021，以及数字。

    数据点（edge=False）落在所属期间的中点：年度值画在年中，季度值画在季中。
    边界（edge=True，用于 x_range、bands、spans）取期间起点："2022" 即 2022-01-01。
    """
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    mid = 0 if edge else 0.5
    m = re.fullmatch(r"(\d{4})Q([1-4])", s)
    if m:
        return int(m[1]) + (int(m[2]) - 1 + mid) / 4
    m = re.fullmatch(r"(\d{4})H([12])", s)
    if m:
        return int(m[1]) + (int(m[2]) - 1 + mid) / 2
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        d = datetime.date(int(m[1]), int(m[2]), int(m[3]))
        days = 366 if d.year % 4 == 0 and (d.year % 100 or d.year % 400 == 0) else 365
        return d.year + (d.timetuple().tm_yday - 0.5) / days
    m = re.fullmatch(r"(\d{4})-(\d{1,2})", s)
    if m:
        return int(m[1]) + (int(m[2]) - 1 + mid) / 12
    m = re.fullmatch(r"(\d{4})", s)
    if m:
        return int(m[1]) + mid
    raise ValueError(f"无法解析日期：{s}")


def text_width(t, size):
    w = 0.0
    for ch in str(t):
        if ch == " ":
            w += 0.3
        elif ord(ch) < 128:
            w += 0.56
        else:
            w += 1.0
    return w * size


NO_LINE_START = set("，。、；：？！）》」』”’,.;:!?)%")


def wrap(t, size, width):
    """按估算宽度折行；ASCII 单词不拆开。"""
    tokens = re.findall(r"[A-Za-z0-9.,%+\-/×~:]+|\s|.", str(t))
    lines, cur = [], ""
    for tok in tokens:
        if cur and tok in NO_LINE_START:
            cur += tok
        elif cur and text_width(cur + tok, size) > width:
            lines.append(cur.rstrip())
            cur = tok.lstrip()
        else:
            cur += tok
    if cur.strip():
        lines.append(cur.rstrip())
    return lines or [""]


def fmt(v):
    if v is None:
        return ""
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}".rstrip("0").rstrip(".")


def nice_ticks(lo, hi, n=4):
    span = hi - lo or 1
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.floor(lo / step + 1e-9) * step
    ticks, v = [], start
    while v <= hi + step * 1e-6:
        if v >= lo - step * 1e-6:
            ticks.append(round(v, 10))
        v += step
    return ticks


class Svg:
    def __init__(self):
        self.o = []

    def add(self, s):
        self.o.append(s)

    def text(self, x: float, y: float, t, size: float = 11, fill="#868e96", weight=None, anchor="start", mono=False):
        w = f' font-weight="{weight}"' if weight else ""
        f = ' font-family="monospace"' if mono else ""
        tw = text_width(t, size)
        left = x - (tw if anchor == "end" else tw / 2 if anchor == "middle" else 0)
        if left < 4 or left + tw > W - 4:
            WARN.append(f"文字可能超出画布：{t!r} (x={x:.0f})")
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}"{w}{f} '
                 f'text-anchor="{anchor}">{escape(str(t))}</text>')

    def line(self, x1, y1, x2, y2, stroke="#e9ecef", width: float = 1, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                 f'stroke="{stroke}" stroke-width="{width}"{d}/>')

    def rect(self, x, y, w, h, fill, stroke=None, rx=0, dash=None, opacity=None):
        s = f' stroke="{stroke}"' if stroke else ""
        d = f' stroke-dasharray="{dash}"' if dash else ""
        o = f' opacity="{opacity}"' if opacity is not None else ""
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
                 f'fill="{fill}"{s}{d}{o}/>')

    def path(self, pts, stroke, width: float = 2, dash=None):
        if len(pts) < 2:
            return
        d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{width}" '
                 f'stroke-linejoin="round"{da}/>')

    def circle(self, x, y, r, fill, stroke=None):
        s = f' stroke="{stroke}" stroke-width="2"' if stroke else ""
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}"{s}/>')

    def render(self, h):
        head = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h:.0f}" font-family="{FONT}">',
                f'<rect x="0" y="0" width="{W}" height="{h:.0f}" rx="12" fill="#ffffff"/>']
        return "\n".join(head + self.o + ["</svg>"]) + "\n"


def header(svg, spec):
    svg.text(30, 34, spec["title"], 20, "#212529", 700)
    if spec.get("subtitle"):
        svg.text(30, 56, spec["subtitle"], 13, "#868e96")
    return 78


def note_box(svg, y, lines):
    if not lines:
        return y
    lines = [ln for raw in lines for ln in wrap(raw, 12, 800)]
    h = 20 + 18 * len(lines)
    svg.rect(30, y, 840, h, "#f8f9fa", "#e9ecef", rx=8)
    for i, ln in enumerate(lines):
        svg.text(48, y + 24 + 18 * i, ln, 12, "#495057")
    return y + h + 16


def outward_ticks(lo, hi, n=4):
    t = nice_ticks(lo, hi, n)
    step = t[1] - t[0] if len(t) > 1 else 1
    if t[0] > lo + 1e-9:
        t.insert(0, round(t[0] - step, 10))
    if t[-1] < hi - 1e-9:
        t.append(round(t[-1] + step, 10))
    return t


def place(x, y, opts, default_dx=8, default_dy=-8):
    """标签相对数据点的位置；opts 可带 dx / dy / anchor，用于手动避让。"""
    opts = opts or {}
    return x + opts.get("dx", default_dx), y + opts.get("dy", default_dy), opts.get("anchor", "start")


def legend(svg, y, items):
    x = L
    for it in items:
        c = col(it.get("color"))
        kind = it.get("kind", "line")
        if kind in ("line", "dash"):
            svg.line(x, y - 4, x + 24, y - 4, c, 2, "5 3" if kind == "dash" else None)
            svg.circle(x + 12, y - 4, 3.5, c)
        elif kind == "square":
            svg.add(f'<rect x="{x + 6}" y="{y - 10}" width="12" height="12" fill="#ffffff" '
                    f'stroke="{c}" stroke-width="2"/>')
        else:
            svg.rect(x + 4, y - 11, 16, 12, c if kind == "bar" else col(it.get("color"), 2), c, rx=2)
        svg.text(x + 32, y, it["name"], 12, "#495057")
        x += 32 + text_width(it["name"], 12) + 28
    return y + 26


def panel_values(p):
    vals = []
    for s in p.get("series", []):
        vals += [pt[1] for pt in s["points"] if pt[1] is not None]
    for tg in p.get("targets", []):
        vals += [f[1] for f in tg["forecasts"]]
        if tg.get("actual"):
            vals.append(tg["actual"][1])
    vals += [m["value"] for m in p.get("marks", [])]
    vals += [h["value"] for h in p.get("hlines", [])]
    for rg in p.get("ranges", []):
        vals += [rg["from"], rg["to"]]
    return vals or [0, 1]


def draw_panel(svg, p, top, bot, X, xticks):
    kind = p.get("kind", "line")
    vals = panel_values(p)
    if p.get("y_range"):
        lo, hi = p["y_range"]
        yt = p.get("y_ticks") or nice_ticks(lo, hi, p.get("y_tick_count", 4))
    else:
        lo = min(vals + ([0] if kind == "bar" or p.get("zero", True) else []))
        yt = p.get("y_ticks") or outward_ticks(lo, max(vals), p.get("y_tick_count", 4))
        lo, hi = yt[0], yt[-1]

    def Y(v):
        return bot - (v - lo) / (hi - lo) * (bot - top)

    if p.get("label"):
        svg.text(L, top - 10, p["label"], 12.5, col(p.get("label_color"), 1), 700)
    for rg in p.get("ranges", []):
        c = rg.get("color")
        svg.rect(L, Y(rg["to"]), R - L, Y(rg["from"]) - Y(rg["to"]), col(c, 2), opacity=0.8)
        if rg.get("label"):
            svg.text(R - 6, Y(rg["to"]) + 14, rg["label"], 11, col(c, 1), 700, "end")
    for v in yt:
        svg.line(L, Y(v), R, Y(v))
        svg.text(L - 8, Y(v) + 4, fmt(v) + p.get("y_suffix", ""), 11, "#868e96", anchor="end")
    for t in xticks:
        svg.line(X(t, True), top, X(t, True), bot, "#e9ecef", 1, "3 3")
    if lo < 0 < hi:
        svg.line(L, Y(0), R, Y(0), "#adb5bd", 1)
    for hl in p.get("hlines", []):
        c = hl.get("color", "gray")
        svg.line(L, Y(hl["value"]), R, Y(hl["value"]), col(c), 1.5, "6 4")
        if hl.get("label"):
            x, y, a = place(R - 6, Y(hl["value"]), hl, 0, -6)
            svg.text(x, y, hl["label"], 11, col(c, 1), anchor=hl.get("anchor", "end"))

    if kind == "line":
        for s in p["series"]:
            c = col(s.get("color", "blue"))
            segs = [[]]
            for pt in s["points"]:
                if pt[1] is None:
                    segs.append([])
                else:
                    segs[-1].append((X(pt[0]), Y(pt[1])))
            for sg in segs:
                svg.path(sg, c, s.get("width", 2), "5 4" if s.get("dashed") else None)
                if s.get("dots"):
                    for x, y in sg:
                        svg.circle(x, y, 3, c)
    elif kind == "bar":
        s = p["series"][0]
        pts = [pt for pt in s["points"] if pt[1] is not None]
        xs = sorted(tnum(pt[0]) for pt in pts)
        gap = min((b - a for a, b in zip(xs, xs[1:])), default=1)
        bw = p.get("bar_width", gap * 0.55) * (R - L) / X.span
        hl = p.get("highlight", {})
        show = p.get("value_labels", True)
        for key, v in pts:
            c = hl.get(str(key), s.get("color", "teal"))
            x, y0, yv = X(key), Y(0 if lo <= 0 else lo), Y(v)
            svg.rect(x - bw / 2, min(y0, yv), bw, abs(y0 - yv), col(c), rx=3)
            if show is True or (isinstance(show, list) and str(key) in show):
                ty = yv - 7 if v >= 0 else yv + 15
                svg.text(x, ty, fmt(v), 11, "#212529" if str(key) in hl else "#495057",
                         700 if str(key) in hl else None, "middle")
    elif kind == "forecast":
        for tg in p["targets"]:
            c = col(tg.get("color", "blue"))
            pts = [(X(f[0]), Y(f[1])) for f in tg["forecasts"]]
            svg.path(pts, c, 2)
            if tg.get("actual"):
                a = tg["actual"]
                ax, ay = X(a[0]), Y(a[1])
                svg.path([pts[-1], (ax, ay)], c, 1.5, "4 3")
                svg.add(f'<rect x="{ax - 6:.1f}" y="{ay - 6:.1f}" width="12" height="12" '
                        f'fill="#ffffff" stroke="{c}" stroke-width="2"/>')
                if len(a) > 2:
                    x, y, an = place(ax, ay, a[3] if len(a) > 3 else None, 12, 4)
                    svg.text(x, y, a[2], 11.5, "#495057", anchor=an)
            for f, (x, y) in zip(tg["forecasts"], pts):
                svg.circle(x, y, 4.5, c)
                if len(f) > 2:
                    tx, ty, an = place(x, y, f[3] if len(f) > 3 else None)
                    svg.text(tx, ty, f[2], 11.5, "#495057", anchor=an)

    for m in p.get("marks", []):
        c = m.get("color", "amber")
        x, y = X(m["date"]), Y(m["value"])
        svg.circle(x, y, 5, col(c))
        tx, ty, an = place(x, y, m)
        svg.text(tx, ty, m["label"], 12.5, "#212529", 700, an)
        if m.get("sub"):
            svg.text(tx, ty + 16, m["sub"], 11, "#495057", anchor=an)
    return Y


class XScale:
    """时间轴：数据点用期间中点，边界用期间起点。"""

    def __init__(self, start, end):
        self.x0, self.x1 = tnum(start, True), tnum(end, True)
        self.span = self.x1 - self.x0

    def __call__(self, t, edge=False):
        return L + (tnum(t, edge) - self.x0) / self.span * (R - L)


def chart_panels(spec):
    svg = Svg()
    y = header(svg, spec)
    X = XScale(*spec["x_range"])
    x0, x1 = X.x0, X.x1
    xticks = spec.get("x_ticks") or list(range(math.ceil(x0), math.floor(x1) + 1))
    bands = spec.get("bands", [])
    band_label_y = y + 10
    if bands:
        y += 22
    boxes = []
    for p in spec["panels"]:
        top = y + (26 if p.get("label") else 10)
        boxes.append((top, top + p.get("height", 200)))
        y = boxes[-1][1] + p.get("gap", 36)
    first, bottom = boxes[0][0], boxes[-1][1]

    for b in bands:
        bx0, bx1 = X(b["from"], True), X(b["to"], True)
        svg.rect(bx0, first, bx1 - bx0, bottom - first, col(b.get("color"), 2), opacity=0.6)
        if b.get("label"):
            svg.text((bx0 + bx1) / 2, band_label_y, b["label"], 11.5, col(b.get("color"), 1), 700, "middle")
    for p, (top, bot) in zip(spec["panels"], boxes):
        draw_panel(svg, p, top, bot, X, xticks)
    for e in spec.get("events", []):
        c = e.get("color", "amber")
        x = X(e["date"])
        svg.line(x, first, x, bottom, col(c, 1), 1.5, "5 4")
        if e.get("label"):
            an = e.get("anchor", "start")
            svg.text(x + (-5 if an == "end" else 5), first + 14 + e.get("dy", 0), e["label"], 12,
                     col(c, 1), 700, an)
    for sp in spec.get("spans", []):
        c = col(sp.get("color", "amber"), 1)
        i = sp.get("after_panel", 0)
        sy = boxes[i][1] + spec["panels"][i].get("gap", 36) / 2 + 4
        sx0, sx1 = X(sp.get("from"), sp.get("edge", False)), X(sp.get("to"), sp.get("edge", False))
        svg.add(f'<path d="M{sx0:.1f} {sy + 6:.1f} L{sx0:.1f} {sy:.1f} L{sx1:.1f} {sy:.1f} '
                f'L{sx1:.1f} {sy + 6:.1f}" fill="none" stroke="{c}" stroke-width="1.5"/>')
        svg.text((sx0 + sx1) / 2, sy - 6, sp["label"], 12, c, 700, "middle")
    labels = spec.get("x_tick_labels") or [str(t) for t in xticks]
    centered = spec.get("x_labels", "start") == "center"
    for t, lab in zip(xticks, labels):
        if centered:
            svg.text((X(t, True) + X(t + 1, True)) / 2, bottom + 20, lab, 12.5, "#495057", 700, "middle")
        else:
            svg.text(X(t, True) + 4, bottom + 20, lab, 12.5, "#495057", 700)
    y = bottom + 46
    if spec.get("legend"):
        y = legend(svg, y, spec["legend"])
    y = note_box(svg, y, spec.get("note"))
    return svg.render(y)


GAP_DIRS = {
    "under": ("red", "当时低估，后来上修"),
    "over": ("teal", "当时高估，后来下修"),
    "shift": ("amber", "关注点切换"),
    "match": ("gray", "基本符合"),
}
DEFAULT_COLUMNS = [
    {"key": "event", "title": "事件 / 外部条件", "width": 200, "color": "gray"},
    {"key": "inst", "title": "机构研报", "width": 190, "color": "blue"},
    {"key": "retail", "title": "散户与媒体", "width": 170, "color": "grape"},
    {"key": "gap", "title": "预期差", "width": 132, "gap": True},
]


def chart_expectation(spec):
    """竖向时间线：每行一个节点，列出事件、机构与散户的看法，以及事后看预期差的方向。"""
    svg = Svg()
    y = header(svg, spec)
    cols = spec.get("columns") or DEFAULT_COLUMNS
    spine_x, x = 132, 150
    xs = []
    for c in cols:
        xs.append(x)
        x += c["width"] + 6
    if x - 6 > 870:
        WARN.append(f"列宽合计超出画布：右边界 {x - 6}")
    y += 8
    svg.text(118, y, spec.get("date_title", "时间"), 12.5, "#212529", 700, "end")
    for c, cx in zip(cols, xs):
        color = "amber" if c.get("gap") else c.get("color", "gray")
        svg.text(cx + 10, y, c["title"], 12.5, col(color, 1), 700)
    y += 14
    first_dot = last_dot = None
    used = set()
    for row in spec["rows"]:
        cells = []
        for c in cols:
            txt = row.get(c["key"], "")
            cells.append(wrap(txt, 11.5, c["width"] - 20) if txt else [])
        h = max(40, 18 + 16 * max(len(lines) for lines in cells))
        d = row.get("gap_dir", "match")
        used.add(d)
        dc = GAP_DIRS.get(d, GAP_DIRS["match"])[0]
        svg.text(118, y + 20, row["date"], 12, "#212529", 700, "end")
        if row.get("date_sub"):
            svg.text(118, y + 36, row["date_sub"], 11, "#868e96", anchor="end")
        svg.circle(spine_x, y + 16, 5, col(dc))
        first_dot = first_dot or y + 16
        last_dot = y + 16
        for c, cx, lines in zip(cols, xs, cells):
            gap = c.get("gap")
            color = dc if gap else c.get("color", "gray")
            if not lines:
                svg.rect(cx, y, c["width"], h, "#f8f9fa", rx=8)
                continue
            svg.rect(cx, y, c["width"], h, col(color, 2), col(color), rx=8)
            for i, ln in enumerate(lines):
                bold = gap and i == 0
                svg.text(cx + 10, y + 20 + 16 * i, ln, 11.5, col(color, 1) if gap else "#495057",
                         700 if bold else None)
        y += h + 10
    svg.add(f'<line x1="{spine_x}" y1="{first_dot}" x2="{spine_x}" y2="{last_dot}" '
            f'stroke="#dee2e6" stroke-width="2"/>')
    # 让竖线压在圆点下面
    svg.o.insert(2, svg.o.pop())
    y += 8
    items = [{"name": GAP_DIRS[k][1], "color": GAP_DIRS[k][0], "kind": "box"}
             for k in GAP_DIRS if k in used]
    y = legend(svg, y + 4, items)
    y = note_box(svg, y, spec.get("note"))
    return svg.render(y)


CHARTS = {"panels": chart_panels, "expectation": chart_expectation}


def main():
    if len(sys.argv) != 3:
        sys.exit("用法：python3 -I stock_chart.py spec.json out.svg")
    spec = json.load(open(sys.argv[1], encoding="utf-8"))
    out = CHARTS[spec.get("type", "panels")](spec)
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        f.write(out)
    xml.dom.minidom.parse(sys.argv[2])
    for w in WARN:
        print("WARN", w)
    print(f"ok {sys.argv[2]}")


if __name__ == "__main__":
    main()
