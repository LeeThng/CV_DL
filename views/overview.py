"""Trang Tổng quan: bố cục giống Google Analytics (thẻ KPI, biểu đồ theo thời gian, phân bổ, bản đồ nhiệt, hoạt động gần đây)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import db, ui


def _metrics(d):
    n = len(d)
    uniq = d.loc[d['valid'] & (d['plate'] != ''), 'plate'].nunique()
    rate = d['valid'].mean() * 100 if n else 0.0
    conf = d['ocr_conf'].mean() * 100 if n else 0.0
    return dict(n=n, uniq=uniq, rate=rate, conf=conf)


def _buckets(d, start, end, freq):
    idx = pd.date_range(start.floor(freq), end.floor(freq), freq=freq)
    if not len(d):
        return idx, pd.Series(0, index=idx), pd.Series(0, index=idx)
    b = d['ts'].dt.floor(freq)
    cnt = d.groupby(b).size().reindex(idx, fill_value=0)
    dv = d[d['valid'] & (d['plate'] != '')]
    uq = dv.groupby(dv['ts'].dt.floor(freq))['plate'].nunique().reindex(idx, fill_value=0)
    return idx, cnt, uq


def _delta(cur, prev, rel=True):
    if prev is None:
        return None
    if rel:
        return None if prev == 0 else (cur - prev) / prev * 100
    return cur - prev


def render():
    df_all = db.load_df()
    st.markdown('<p class="page-title">Tổng quan</p><p class="page-sub">Hoạt động nhận diện biển số xe</p>', unsafe_allow_html=True)
    if df_all.empty:
        st.info('Chưa có dữ liệu nhận diện. Hãy sang trang **Nhận diện** để thử với ảnh của bạn, '
                'hoặc tạo dữ liệu mẫu để xem dashboard.')
        if st.button('Tạo dữ liệu mẫu (90 ngày)', type='primary'):
            ui.make_demo()
            st.rerun()
        return

    f1, f2, f3 = st.columns([2, 3, 3])
    start, end, prev_start, freq, preset = ui.date_range(df_all, 'ov', f1, f2)
    vehicles = f3.multiselect('Loại xe', ui.VEHICLES, default=ui.VEHICLES, label_visibility='collapsed', placeholder='Loại xe')
    df = df_all[df_all['vehicle'].isin(vehicles)] if vehicles else df_all.iloc[0:0]
    cur = df[(df['ts'] >= start) & (df['ts'] <= end)]
    prev = None
    if prev_start is not None:
        prev = df[(df['ts'] >= prev_start) & (df['ts'] < start)]
    st.caption(f'{start:%d/%m/%Y %H:%M} – {end:%d/%m/%Y %H:%M}' + (' · so sánh với kỳ liền trước cùng độ dài' if prev is not None else ''))

    m = _metrics(cur)
    pm = _metrics(prev) if prev is not None and len(prev) else None
    idx, cnt, uq = _buckets(cur, start, end, freq)

    k = st.columns(4)
    with k[0]:
        ui.kpi_card('Lượt nhận diện', ui.fmt_int(m['n']), _delta(m['n'], pm['n'] if pm else None), cnt.values, ui.BLUE)
    with k[1]:
        ui.kpi_card('Biển số duy nhất', ui.fmt_int(m['uniq']), _delta(m['uniq'], pm['uniq'] if pm else None), uq.values, ui.BLUE)
    with k[2]:
        okr = (cur['valid'].groupby(cur['ts'].dt.floor(freq)).mean().reindex(idx).fillna(0) * 100).values if len(cur) else []
        ui.kpi_card('Tỉ lệ đọc hợp lệ', ui.fmt_pct(m['rate']), _delta(m['rate'], pm['rate'] if pm else None, rel=False), okr, ui.AQUA, delta_unit=' đ.%')
    with k[3]:
        cf = (cur['ocr_conf'].groupby(cur['ts'].dt.floor(freq)).mean().reindex(idx).fillna(0) * 100).values if len(cur) else []
        ui.kpi_card('Độ tin cậy OCR trung bình', ui.fmt_pct(m['conf']), _delta(m['conf'], pm['conf'] if pm else None, rel=False), cf, ui.YELLOW, delta_unit=' đ.%')

    st.write('')
    # ---------------- biểu đồ theo thời gian ----------------
    with ui.card('chart'):
        h1, h2, h3 = st.columns([3, 3, 2])
        h1.markdown(f'<p class="card-title">Lượt nhận diện theo {"giờ" if freq == "h" else "ngày"}</p>', unsafe_allow_html=True)
        metric = h2.radio('Chỉ số', ['Lượt nhận diện', 'Biển số duy nhất'], horizontal=True, label_visibility='collapsed')
        compare = h3.toggle('So sánh kỳ trước', value=prev is not None, disabled=prev is None)
        ser = cnt if metric == 'Lượt nhận diện' else uq
        fmt = '%H:%M %d/%m' if freq == 'h' else '%d/%m/%Y'
        fig = go.Figure()
        if compare and prev is not None:
            shift = start - prev_start
            pidx, pcnt, puq = _buckets(prev, prev_start, start - pd.Timedelta(seconds=1), freq)
            pser = (pcnt if metric == 'Lượt nhận diện' else puq)
            n = min(len(pser), len(ser))
            fig.add_trace(go.Scatter(x=idx[:n], y=pser.values[:n], name='Kỳ trước', mode='lines',
                                     line=dict(color=ui.MUTED, width=2, dash='dash'),
                                     customdata=[t.strftime(fmt) for t in pidx[:n]],
                                     hovertemplate='Kỳ trước (%{customdata}): <b>%{y}</b><extra></extra>'))
        fig.add_trace(go.Scatter(x=idx, y=ser.values, name='Kỳ này', mode='lines+markers' if len(idx) <= 45 else 'lines',
                                 line=dict(color=ui.BLUE, width=2), marker=dict(size=6, line=dict(color='white', width=2)),
                                 hovertemplate='%{x|' + fmt + '}: <b>%{y}</b><extra>Kỳ này</extra>'))
        ui.base_layout(fig, 300, legend=compare and prev is not None)
        fig.update_layout(hovermode='x')
        fig.update_xaxes(tickformat='%H:%M<br>%d/%m' if freq == 'h' else '%d/%m')
        st.plotly_chart(fig, width='stretch', config=dict(displayModeBar=False))

    # ---------------- phân bổ loại xe + biển số hay gặp ----------------
    c1, c2 = st.columns([2, 3])
    with c1, ui.card('vehicle'):
        st.markdown('<p class="card-title">Loại phương tiện</p><p class="card-sub">Tỉ trọng theo số lượt nhận diện</p>', unsafe_allow_html=True)
        vc = cur['vehicle'].value_counts().reindex(ui.VEHICLES).dropna() if len(cur) else pd.Series(dtype=int)
        if len(vc):
            fig = go.Figure(go.Pie(labels=vc.index, values=vc.values, hole=0.62, sort=False, direction='clockwise',
                                   marker=dict(colors=[ui.VEHICLE_COLOR[v] for v in vc.index], line=dict(color='white', width=2)),
                                   textinfo='none', hovertemplate='%{label}: <b>%{value}</b> (%{percent})<extra></extra>'))
            ui.base_layout(fig, 220, legend=False)
            fig.add_annotation(text=f'<b>{ui.fmt_int(vc.sum())}</b><br><span style="font-size:11px;color:{ui.MUTED}">lượt</span>',
                               showarrow=False, font=dict(size=20, color=ui.INK))
            st.plotly_chart(fig, width='stretch', config=dict(displayModeBar=False))
            leg = ''.join(f'<span><i style="background:{ui.VEHICLE_COLOR[v]}"></i>{v} · {n / vc.sum() * 100:.0f}%</span>'.replace('.', ',')
                          for v, n in vc.items())
            st.markdown(f'<div class="legend">{leg}</div>', unsafe_allow_html=True)
        else:
            st.caption('Không có dữ liệu trong khoảng này.')

    with c2, ui.card('top'):
        st.markdown('<p class="card-title">Biển số xuất hiện nhiều nhất</p><p class="card-sub">Chỉ tính các biển đọc hợp lệ</p>', unsafe_allow_html=True)
        v = cur[cur['valid'] & (cur['plate'] != '')]
        if len(v):
            top = (v.groupby('plate').agg(luot=('plate', 'size'), lan_cuoi=('ts', 'max'),
                                          loai_xe=('vehicle', lambda s: s.mode().iat[0]))
                   .sort_values('luot', ascending=False).head(8).reset_index())
            top.columns = ['Biển số', 'Lượt', 'Lần cuối', 'Loại xe']
            st.dataframe(top, hide_index=True, width='stretch', height=318, column_config={
                'Lượt': st.column_config.ProgressColumn('Lượt', min_value=0, max_value=int(top['Lượt'].max()), format='%d'),
                'Lần cuối': st.column_config.DatetimeColumn('Lần cuối', format='DD/MM HH:mm'),
            })
        else:
            st.caption('Không có dữ liệu trong khoảng này.')

    # ---------------- bản đồ nhiệt + hoạt động gần đây ----------------
    c3, c4 = st.columns([3, 2])
    with c3, ui.card('heat'):
        st.markdown('<p class="card-title">Giờ cao điểm</p><p class="card-sub">Số lượt theo thứ trong tuần và giờ trong ngày</p>', unsafe_allow_html=True)
        if len(cur):
            hm = (cur.assign(wd=cur['ts'].dt.weekday, hr=cur['ts'].dt.hour).groupby(['wd', 'hr']).size()
                  .unstack(fill_value=0).reindex(index=range(7), columns=range(24), fill_value=0))
            days = ['Thứ 2', 'Thứ 3', 'Thứ 4', 'Thứ 5', 'Thứ 6', 'Thứ 7', 'CN']
            fig = go.Figure(go.Heatmap(z=hm.values, x=[f'{h}h' for h in range(24)], y=days, xgap=2, ygap=2,
                                       colorscale=[[i / (len(ui.SEQ) - 1), c] for i, c in enumerate(ui.SEQ)],
                                       colorbar=dict(thickness=8, len=0.9, outlinewidth=0, tickfont=dict(size=10, color=ui.MUTED)),
                                       hovertemplate='%{y}, %{x}: <b>%{z}</b> lượt<extra></extra>'))
            ui.base_layout(fig, 280, legend=False)
            fig.update_yaxes(autorange='reversed', gridcolor='rgba(0,0,0,0)', rangemode='normal')
            fig.update_xaxes(tickmode='array', tickvals=[f'{h}h' for h in range(0, 24, 3)])
            st.plotly_chart(fig, width='stretch', config=dict(displayModeBar=False))
        else:
            st.caption('Không có dữ liệu trong khoảng này.')

    with c4, ui.card('recent'):
        st.markdown('<p class="card-title">Hoạt động gần đây</p><p class="card-sub">Các lượt nhận diện mới nhất</p>', unsafe_allow_html=True)
        last = cur.sort_values('ts', ascending=False).head(6)
        if len(last):
            now = db.now_vn(); html = ''
            for _, r in last.iterrows():
                b64 = ui.thumb_b64(r['crop_file']) if isinstance(r['crop_file'], str) else None
                th = f'<img class="th" src="data:image/jpeg;base64,{b64}">' if b64 else '<div class="th"></div>'
                badge = '<span class="badge ok">hợp lệ</span>' if r['valid'] else '<span class="badge warn">cần kiểm tra</span>'
                html += (f'<div class="feed">{th}<div><div class="pl">{r["plate"] or "?"}{badge}</div>'
                         f'<div class="mt">{r["vehicle"]} · {ui.ago(r["ts"], now)}</div></div></div>')
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.caption('Không có dữ liệu trong khoảng này.')
