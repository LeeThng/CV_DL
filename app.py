"""Dashboard nhận diện biển số xe (Streamlit). Chạy: streamlit run app.py"""
import streamlit as st

import ui
from views import overview, recognize, history, settings

st.set_page_config(page_title='LPR Analytics', page_icon=':material/directions_car:', layout='wide')
ui.inject_css()
ui.settings()      # giữ cài đặt khi chuyển trang

with st.sidebar:
    st.markdown('<p class="page-title" style="font-size:1.15rem">LPR Analytics</p>'
                '<p class="page-sub">Nhận diện biển số xe Việt Nam</p>', unsafe_allow_html=True)

pg = st.navigation([
    st.Page(overview.render, title='Tổng quan', icon=':material/dashboard:', default=True),
    st.Page(recognize.render, title='Nhận diện', icon=':material/photo_camera:', url_path='nhan-dien'),
    st.Page(history.render, title='Lịch sử', icon=':material/history:', url_path='lich-su'),
    st.Page(settings.render, title='Cài đặt', icon=':material/settings:', url_path='cai-dat'),
])
pg.run()
