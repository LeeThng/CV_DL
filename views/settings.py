"""Trang Cài đặt: ngưỡng, tuỳ chọn pipeline, trạng thái mô hình, dữ liệu mẫu và xoá dữ liệu."""
import streamlit as st

import db, ui
from lpr_core import CHAR_NAMES


def render():
    st.markdown('<p class="page-title">Cài đặt</p><p class="page-sub">Tham số nhận diện, mô hình và dữ liệu</p>', unsafe_allow_html=True)
    s = ui.settings()
    with ui.card('params'):
        st.markdown('<p class="card-title">Tham số nhận diện</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.slider('Ngưỡng phát hiện biển số', 0.1, 0.9, step=0.05, key='s_conf_plate',
                  help='Hạ thấp nếu hay bỏ sót biển; nâng cao nếu hay bắt nhầm.')
        c2.slider('Ngưỡng phát hiện ký tự', 0.1, 0.9, step=0.05, key='s_conf_char')
        d1, d2, d3 = st.columns(3)
        d1.toggle('Nắn thẳng biển (Canny + Hough)', key='s_deskew')
        d2.toggle('Sửa lỗi theo vị trí', key='s_fix', help='Chỉ sửa ký tự độ tin cậy thấp, tối đa 2 ký tự mỗi biển.')
        d3.toggle('Nhận dạng loại xe (YOLOv8n)', key='s_vehicle')

    with ui.card('models'):
        st.markdown('<p class="card-title">Trạng thái mô hình</p>', unsafe_allow_html=True)
        w = ui.weight_files()
        rows = [('Phát hiện biển số (YOLOv8s)', w['plate']), ('Phát hiện ký tự (YOLOv8s)', w['char']), ('Phát hiện phương tiện (YOLOv8n, tuỳ chọn)', w['vehicle'])]
        for name, p in rows:
            st.markdown(f'- **{name}**: ' + (f'`{p.name}` ({p.stat().st_size / 1e6:.1f} MB)' if p else '*chưa có file*'))
        st.caption(f'{len(CHAR_NAMES)} lớp ký tự: ' + ' '.join(CHAR_NAMES))

    with ui.card('data'):
        st.markdown('<p class="card-title">Dữ liệu</p><p class="card-sub">Lịch sử nằm trong thư mục <code>data/</code>. '
                    'Trên Streamlit Community Cloud, ổ đĩa là tạm thời và bị xoá khi ứng dụng khởi động lại, hãy xuất CSV thường xuyên.</p>',
                    unsafe_allow_html=True)
        df = db.load_df()
        st.write(f'Đang có **{ui.fmt_int(len(df))}** bản ghi (trong đó {ui.fmt_int(int(df["demo"].sum()))} bản ghi mẫu).')
        a, b, c = st.columns(3)
        if a.button('Tạo dữ liệu mẫu (90 ngày)', width='stretch'):
            n = ui.make_demo(); st.toast(f'Đã tạo {n} bản ghi mẫu'); st.rerun()
        if b.button('Xoá dữ liệu mẫu', width='stretch'):
            db.clear(demo_only=True); st.toast('Đã xoá dữ liệu mẫu'); st.rerun()
        with c.popover('Xoá toàn bộ lịch sử', width='stretch'):
            st.warning('Thao tác này xoá mọi bản ghi và ảnh, không hoàn tác được.')
            if st.button('Tôi chắc chắn, xoá hết', type='primary'):
                db.clear(); st.toast('Đã xoá toàn bộ lịch sử'); st.rerun()
