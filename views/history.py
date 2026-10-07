"""Trang Lịch sử: tìm kiếm, lọc, xem ảnh, sửa biển số, xoá, xuất CSV."""
import pandas as pd
import streamlit as st

import db, ui
from lpr_core import is_valid_plate

PAGE = 12


def render():
    st.markdown('<p class="page-title">Lịch sử nhận diện</p><p class="page-sub">Toàn bộ biển số đã nhận diện, kèm ảnh gốc</p>', unsafe_allow_html=True)
    df_all = db.load_df()
    if df_all.empty:
        st.info('Lịch sử đang trống. Hãy nhận diện vài ảnh ở trang **Nhận diện**.')
        return

    f1, f2, f3 = st.columns([2, 3, 3])
    start, end, _, _, _ = ui.date_range(df_all, 'hi', f1, f2)
    q = f3.text_input('Tìm biển số', placeholder='Tìm biển số, ví dụ 59X1', label_visibility='collapsed')
    g1, g2, g3 = st.columns([3, 2, 2])
    vehicles = g1.multiselect('Loại xe', ui.VEHICLES, default=ui.VEHICLES, label_visibility='collapsed', placeholder='Loại xe')
    status = g2.selectbox('Trạng thái', ['Tất cả', 'Hợp lệ', 'Cần kiểm tra'], label_visibility='collapsed')
    view = g3.segmented_control('Dạng xem', ['Ảnh', 'Bảng'], default='Ảnh', label_visibility='collapsed') or 'Ảnh'

    df = df_all[(df_all['ts'] >= start) & (df_all['ts'] <= end) & df_all['vehicle'].isin(vehicles or [])]
    if q.strip():
        df = df[df['plate'].str.contains(q.strip().upper().replace(' ', ''), case=False, na=False)]
    if status == 'Hợp lệ':
        df = df[df['valid']]
    elif status == 'Cần kiểm tra':
        df = df[~df['valid']]
    df = df.sort_values('ts', ascending=False)

    m1, m2 = st.columns([4, 1])
    m1.caption(f'{ui.fmt_int(len(df))} kết quả')
    export = df[['ts', 'source', 'vehicle', 'plate', 'valid', 'det_conf', 'ocr_conf']].rename(columns={
        'ts': 'Thời gian', 'source': 'Nguồn', 'vehicle': 'Loại xe', 'plate': 'Biển số', 'valid': 'Hợp lệ',
        'det_conf': 'Độ tin cậy phát hiện', 'ocr_conf': 'Độ tin cậy OCR'})
    m2.download_button('Xuất CSV', export.to_csv(index=False).encode('utf-8-sig'), 'lich_su_nhan_dien.csv', 'text/csv',
                       icon=':material/download:', width='stretch')
    if df.empty:
        st.caption('Không có bản ghi nào khớp bộ lọc.')
        return

    if view == 'Ảnh':
        pages = max((len(df) - 1) // PAGE + 1, 1)
        page = st.number_input('Trang', 1, pages, 1, label_visibility='collapsed') if pages > 1 else 1
        chunk = df.iloc[(page - 1) * PAGE: page * PAGE]
        cols = st.columns(4)
        for i, (_, r) in enumerate(chunk.iterrows()):
            with cols[i % 4], ui.card(f'g{i}'):
                p = db.IMG_DIR / r['image_file'] if isinstance(r['image_file'], str) else None
                if p is not None and p.exists():
                    st.image(str(p), width='stretch')
                else:
                    st.markdown('<div class="th" style="width:100%;height:96px;border-radius:6px;background:#eef0f2"></div>', unsafe_allow_html=True)
                badge = '<span class="badge ok">hợp lệ</span>' if r['valid'] else '<span class="badge warn">cần kiểm tra</span>'
                st.markdown(f'<div class="pl" style="font-family:JetBrains Mono,monospace;font-weight:500">{r["plate"] or "?"}{badge}</div>'
                            f'<div class="mt" style="font-size:.74rem;color:{ui.MUTED}">{r["vehicle"]} · {r["ts"]:%d/%m/%Y %H:%M}</div>',
                            unsafe_allow_html=True)
        st.caption(f'Trang {page}/{pages}')
        return

    # ---- dạng bảng: sửa biển số và xoá ngay trên bảng ----
    st.caption('Bấm vào ô **Biển số** để sửa nếu hệ thống đọc sai; tick **Xoá** rồi bấm Lưu để xoá bản ghi.')
    ed = df[['id', 'ts', 'vehicle', 'plate', 'valid', 'ocr_conf', 'source']].copy()
    ed['xoa'] = False
    out = st.data_editor(
        ed, hide_index=True, width='stretch', height=420, key='hist_editor',
        disabled=['id', 'ts', 'vehicle', 'valid', 'ocr_conf', 'source'],
        column_config={
            'id': None,
            'ts': st.column_config.DatetimeColumn('Thời gian', format='DD/MM/YYYY HH:mm:ss'),
            'vehicle': 'Loại xe', 'plate': st.column_config.TextColumn('Biển số', max_chars=14),
            'valid': st.column_config.CheckboxColumn('Hợp lệ'),
            'ocr_conf': st.column_config.ProgressColumn('Tin cậy OCR', min_value=0, max_value=1, format='percent'),
            'source': 'Nguồn', 'xoa': st.column_config.CheckboxColumn('Xoá'),
        })
    if st.button('Lưu thay đổi', type='primary'):
        changes = {}
        for (_, a), (_, b) in zip(ed.iterrows(), out.iterrows()):
            newp = (b['plate'] or '').strip().upper().replace(' ', '')
            if newp != a['plate']:
                changes[int(a['id'])] = (newp, is_valid_plate(newp))
        dele = out.loc[out['xoa'], 'id'].tolist()
        changes = {k: v for k, v in changes.items() if k not in dele}
        if changes:
            db.update_plates(changes)
        if dele:
            db.delete_ids(dele)
        st.toast(f'Đã sửa {len(changes)} và xoá {len(dele)} bản ghi')
        st.rerun()
