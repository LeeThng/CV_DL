"""Giao diện dùng chung: màu, CSS, thẻ KPI, biểu đồ Plotly, nạp mô hình, dữ liệu mẫu."""
import base64, random
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import db
from lpr_core import CHAR_NAMES

ROOT = Path(__file__).parent
WEIGHTS = ROOT / 'weights'

# ---- Bảng màu (đã kiểm tra bằng validator: xanh dương, cam, xanh ngọc, vàng theo đúng thứ tự) ----
INK, INK2, MUTED = '#1f2328', '#52514e', '#8a8a85'
GRID, SURFACE = '#e8eaed', '#ffffff'
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
SEQ = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']     # tuần tự một màu, nhạt -> đậm
GOOD, BAD = '#137333', '#c5221f'
VEHICLE_COLOR = {'Xe máy': BLUE, 'Ô tô': ORANGE, 'Xe tải': AQUA, 'Xe buýt': YELLOW, 'Không rõ': '#a8a8a2'}
VEHICLES = ['Xe máy', 'Ô tô', 'Xe tải', 'Xe buýt', 'Không rõ']

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap');
.stApp, .stMarkdown, .kpi, .feed {{ font-family: 'Be Vietnam Pro', system-ui, sans-serif; }}
.block-container {{ padding-top: 3.6rem; max-width: 1280px; }}
h1, h2, h3 {{ font-weight: 600; letter-spacing: -0.01em; }}
[class*="st-key-card"] {{ background: {SURFACE}; border-radius: 10px; }}
.page-title {{ font-size: 1.45rem; font-weight: 600; color: {INK}; margin: 0; }}
.page-sub {{ color: {MUTED}; font-size: .85rem; margin: 2px 0 0; }}
.card-title {{ font-size: .95rem; font-weight: 600; color: {INK}; margin: 0 0 2px; }}
.card-sub {{ font-size: .78rem; color: {MUTED}; margin: 0 0 6px; }}
.kpi {{ background: {SURFACE}; border: 1px solid {GRID}; border-radius: 10px; padding: 14px 16px 10px; height: 100%; }}
.kpi .lab {{ font-size: .8rem; color: {INK2}; }}
.kpi .val {{ font-size: 1.9rem; font-weight: 600; color: {INK}; line-height: 1.25; font-variant-numeric: tabular-nums; }}
.kpi .dlt {{ font-size: .78rem; font-weight: 500; }}
.kpi .dlt small {{ color: {MUTED}; font-weight: 400; }}
.up {{ color: {GOOD}; }} .down {{ color: {BAD}; }} .flat {{ color: {MUTED}; }}
.kpi svg {{ display: block; margin-top: 6px; }}
.feed {{ display: flex; gap: 10px; align-items: center; padding: 8px 0; border-bottom: 1px solid {GRID}; }}
.feed:last-child {{ border-bottom: 0; }}
.feed .th {{ width: 64px; height: 36px; border-radius: 4px; background: #eef0f2; object-fit: cover; flex: none; }}
.feed .pl {{ font-family: 'JetBrains Mono', monospace; font-weight: 500; font-size: .92rem; color: {INK}; }}
.feed .mt {{ font-size: .74rem; color: {MUTED}; }}
.badge {{ display: inline-block; font-size: .68rem; padding: 1px 7px; border-radius: 999px; border: 1px solid {GRID}; color: {INK2}; margin-left: 6px; vertical-align: middle; }}
.badge.ok {{ color: {GOOD}; border-color: #b7dfc3; }} .badge.warn {{ color: #b45309; border-color: #f1d3a8; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: .78rem; color: {INK2}; margin: 2px 0 4px; }}
.legend i {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }}
</style>
"""


def card(key):
    return st.container(border=True, key=f'card_{key}')


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


# ---------------- định dạng số kiểu Việt ----------------
def fmt_int(n):
    return f'{int(n):,}'.replace(',', '.')


def fmt_pct(x, d=1):
    return f'{x:.{d}f}'.replace('.', ',') + '%'


def ago(ts, now=None):
    now = now or db.now_vn()
    s = max(int((now - ts).total_seconds()), 0)
    if s < 60: return 'vừa xong'
    if s < 3600: return f'{s // 60} phút trước'
    if s < 86400: return f'{s // 3600} giờ trước'
    return f'{s // 86400} ngày trước'


# ---------------- thẻ KPI ----------------
def spark_svg(vals, color, w=220, h=34):
    vals = list(vals)
    if len(vals) < 2 or max(vals) == min(vals):
        y = h / 2
        return f'<svg width="100%" height="{h}" viewBox="0 0 {w} {h}" preserveAspectRatio="none"><line x1="0" y1="{y}" x2="{w}" y2="{y}" stroke="{color}" stroke-width="2" stroke-linecap="round"/></svg>'
    lo, hi = min(vals), max(vals)
    pts = [(i * w / (len(vals) - 1), h - 3 - (v - lo) / (hi - lo) * (h - 6)) for i, v in enumerate(vals)]
    d = ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts)
    return (f'<svg width="100%" height="{h}" viewBox="0 0 {w} {h}" preserveAspectRatio="none">'
            f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/></svg>')


def kpi_card(label, value, delta, spark, color=BLUE, higher_is_good=True, delta_unit='%'):
    """delta: None (chưa có kỳ trước) hoặc số thay đổi (% hoặc điểm %)."""
    if delta is None:
        dl = '<span class="flat">— <small>chưa có dữ liệu kỳ trước</small></span>'
    else:
        up = delta > 0.05; down = delta < -0.05
        good = (up and higher_is_good) or (down and not higher_is_good)
        cls = 'flat' if not (up or down) else ('up' if good else 'down')
        arrow = '▲' if up else ('▼' if down else '■')
        num = f'{abs(delta):.1f}'.replace('.', ',')
        dl = f'<span class="{cls}">{arrow} {num}{delta_unit} <small>so với kỳ trước</small></span>'
    st.markdown(f'<div class="kpi"><div class="lab">{label}</div><div class="val">{value}</div>'
                f'<div class="dlt">{dl}</div>{spark_svg(spark, color)}</div>', unsafe_allow_html=True)


# ---------------- Plotly ----------------
def base_layout(fig, height=300, legend=True):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family='Be Vietnam Pro, system-ui, sans-serif', size=12, color=INK2),
        hoverlabel=dict(bgcolor='white', font_size=12, bordercolor=GRID),
        legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='left', x=0, font=dict(size=12)) if legend else None,
        showlegend=legend,
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=MUTED), zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED), rangemode='tozero')
    return fig


# ---------------- ảnh thu nhỏ ----------------
@lru_cache(maxsize=256)
def thumb_b64(fname):
    p = db.IMG_DIR / fname
    if not fname or not p.exists():
        return None
    return base64.b64encode(p.read_bytes()).decode()


# ---------------- nạp mô hình ----------------
def _find(patterns):
    for pat in patterns:
        hits = sorted(WEIGHTS.glob(pat))
        if hits:
            return hits[0]
    return None


def weight_files():
    return dict(
        plate=_find(['lp_detect_best.pt', '*detect*.pt', '*plate*.pt']),
        char=_find(['lp_char_best.pt', '*char*.pt']),
        vehicle=_find(['yolov8n.pt', '*vehicle*.pt']),
    )


@st.cache_resource(show_spinner='Đang tải mô hình (lần đầu mất ít phút)...')
def _load_system(plate, char, vehicle):
    from lpr_core import LPRSystem
    return LPRSystem(plate, char, vehicle)


def get_system():
    """Trả về (LPRSystem hoặc None, thông báo lỗi hoặc None)."""
    w = weight_files()
    if not w['plate'] or not w['char']:
        return None, ('Chưa tìm thấy trọng số. Hãy chép `lp_detect_best.pt` và `lp_char_best.pt` vào thư mục `weights/` '
                      '(lấy trong file `DoAn3_ketqua.zip` tải từ Kaggle).')
    try:
        sysm = _load_system(str(w['plate']), str(w['char']), str(w['vehicle']) if w['vehicle'] else None)
    except Exception as e:                          # noqa: BLE001
        return None, f'Không nạp được mô hình: {e}'
    s = settings()
    sysm.configure(s['conf_plate'], s['conf_char'], s['deskew'], s['fix'], s['vehicle'])
    return sysm, None


DEFAULTS = dict(conf_plate=0.4, conf_char=0.3, deskew=True, fix=True, vehicle=True, save_history=True)


def settings():
    # Streamlit xoá trạng thái của widget không được vẽ ở trang hiện tại -> gán lại mỗi lần chạy để cài đặt không bị mất khi đổi trang
    for k, v in DEFAULTS.items():
        key = f's_{k}'
        st.session_state[key] = st.session_state.get(key, v)
    return {k: st.session_state[f's_{k}'] for k in DEFAULTS}


# ---------------- bộ lọc thời gian ----------------
PRESETS = ['Hôm nay', '7 ngày qua', '30 ngày qua', '90 ngày qua', 'Tất cả', 'Tùy chọn']


def date_range(df, key, c1, c2):
    """Trả về (start, end, prev_start, bucket_freq). Giống bộ chọn khoảng thời gian của Google Analytics."""
    now = db.now_vn()
    preset = c1.selectbox('Khoảng thời gian', PRESETS, index=2, key=f'{key}_preset', label_visibility='collapsed')
    today = pd.Timestamp(now.date())
    if preset == 'Hôm nay':
        start = today
    elif preset == '7 ngày qua':
        start = today - pd.Timedelta(days=6)
    elif preset == '30 ngày qua':
        start = today - pd.Timedelta(days=29)
    elif preset == '90 ngày qua':
        start = today - pd.Timedelta(days=89)
    elif preset == 'Tất cả':
        start = df['ts'].min().normalize() if len(df) else today
    else:
        d = c2.date_input('Chọn khoảng', value=(today.date() - timedelta(days=13), today.date()), key=f'{key}_dates',
                          label_visibility='collapsed')
        if isinstance(d, (list, tuple)) and len(d) == 2:
            start = pd.Timestamp(d[0])
            end = pd.Timestamp(d[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        else:
            start, end = today, pd.Timestamp(now)
        return start, end, start - (end - start) - pd.Timedelta(seconds=1), ('h' if (end - start) <= pd.Timedelta(days=2) else 'D'), preset
    end = pd.Timestamp(now)
    prev_start = None if preset == 'Tất cả' else start - (end - start)
    freq = 'h' if (end - start) <= pd.Timedelta(days=2) else 'D'
    return start, end, prev_start, freq, preset


# ---------------- dữ liệu mẫu ----------------
def _plate(rng, vehicle):
    prov = rng.choice(['29', '30', '31', '33', '36', '37', '38', '43', '51', '59', '60', '61', '65', '75', '92', '99'])
    let = [c for c in CHAR_NAMES if c.isalpha()]
    if vehicle == 'Xe máy':
        return f'{prov}{rng.choice(let)}{rng.randint(1, 9)}-{rng.randint(10000, 99999)}'
    return f'{prov}{rng.choice(let)}-{rng.randint(10000, 99999)}'


def make_demo(days=90, per_day=(70, 150), seed=7):
    """Tạo dữ liệu mẫu để dashboard có số liệu khi trình diễn (đánh dấu demo=1, xoá riêng được)."""
    rng = random.Random(seed)
    now = db.now_vn()
    pool = {v: [_plate(rng, v) for _ in range(35)] for v in ('Xe máy', 'Ô tô', 'Xe tải', 'Xe buýt')}
    hour_w = [1, 1, 1, 1, 2, 4, 9, 14, 12, 8, 6, 7, 9, 7, 6, 7, 10, 14, 13, 8, 5, 3, 2, 1]
    rows = []
    for d in range(days, -1, -1):
        day = (now - timedelta(days=d)).replace(hour=0, minute=0, second=0, microsecond=0)
        wd = day.weekday()
        n = int(rng.randint(*per_day) * (0.7 if wd >= 5 else 1.0) * (1 + 0.004 * (days - d)))   # xu hướng tăng nhẹ
        for _ in range(n):
            h = rng.choices(range(24), hour_w)[0]
            ts = day + timedelta(hours=h, minutes=rng.randint(0, 59), seconds=rng.randint(0, 59))
            if ts > now:
                continue
            veh = rng.choices(['Xe máy', 'Ô tô', 'Xe tải', 'Xe buýt', 'Không rõ'], [66, 22, 6, 3, 3])[0]
            plate = rng.choice(pool[veh]) if veh in pool and rng.random() < 0.6 else _plate(rng, veh)
            valid = rng.random() < 0.93
            if not valid:
                plate = plate.replace('-', '')[:-1] if rng.random() < .5 else plate[:2] + '8' + plate[3:]
            conf = float(np.clip(rng.betavariate(9, 1.6) if valid else rng.betavariate(4, 3), .2, .99))
            rows.append((ts.strftime('%Y-%m-%d %H:%M:%S'), rng.choice(['Cổng A', 'Cổng B']), veh, plate, int(valid),
                         round(float(np.clip(rng.betavariate(12, 1.5), .4, .99)), 3), round(conf, 3), None, None, 1))
    db.insert_rows(rows)
    return len(rows)
