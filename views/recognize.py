"""Trang Nhận diện: ảnh tải lên, camera, video. Kết quả tự lưu vào lịch sử."""
import tempfile
import cv2
import pandas as pd
import streamlit as st

import db, ui
from lpr_core import decode_image


def _show_results(name, ann, recs, save, tag='x'):
    st.image(ann, channels='BGR', caption=name, width='stretch')
    if not recs:
        st.warning('Không phát hiện biển số nào trong ảnh này. Thử hạ ngưỡng phát hiện biển ở trang Cài đặt.')
        return
    cols = st.columns(min(len(recs), 4))
    for i, r in enumerate(recs):
        with cols[i % len(cols)], ui.card(f'plate_{tag}_{i}'):
            st.image(r['crop'], channels='BGR', width='stretch')
            badge = '<span class="badge ok">hợp lệ</span>' if r['valid'] else '<span class="badge warn">cần kiểm tra</span>'
            st.markdown(f'<div class="feed" style="border:0;padding:0"><div><div class="pl">{r["plate"] or "?"}{badge}</div>'
                        f'<div class="mt">{r["vehicle"]} · phát hiện {r["det_conf"]:.0%} · OCR {r["ocr_conf"]:.0%}</div></div></div>',
                        unsafe_allow_html=True)
    if save:
        n = db.add_detections(recs, ann)
        st.toast(f'Đã lưu {n} biển số vào lịch sử', icon=':material/check_circle:')


def _images(system, save):
    files = st.file_uploader('Chọn một hoặc nhiều ảnh', type=['jpg', 'jpeg', 'png', 'bmp', 'webp'], accept_multiple_files=True)
    if not files:
        st.caption('Gợi ý: ảnh chụp rõ biển số, không cần cắt sẵn. Hệ thống tự tìm biển số trong ảnh.')
        return
    for fi, f in enumerate(files):
        img = decode_image(f.getvalue())
        if img is None:
            st.error(f'Không đọc được ảnh {f.name}')
            continue
        with st.spinner(f'Đang nhận diện {f.name}...'):
            ann, recs = system.process(img, source=f'Tải ảnh: {f.name}')
        with ui.card(f'img_{fi}'):
            _show_results(f.name, ann, recs, save, f'i{fi}')


def _camera(system, save):
    shot = st.camera_input('Chụp ảnh biển số')
    if shot is None:
        return
    img = decode_image(shot.getvalue())
    ann, recs = system.process(img, source='Camera')
    _show_results('Ảnh từ camera', ann, recs, save, 'cam')


def _video(system, save):
    f = st.file_uploader('Chọn video (mp4, avi, mov)', type=['mp4', 'avi', 'mov', 'mkv'])
    c1, c2 = st.columns(2)
    per_sec = c1.slider('Số khung hình phân tích mỗi giây', 1, 5, 2)
    max_sec = c2.slider('Xử lý tối đa (giây đầu của video)', 5, 120, 30)
    if f is None or not st.button('Phân tích video', type='primary'):
        return
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        tmp.write(f.getvalue()); path = tmp.name
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0, max_sec * fps))
    step = max(int(fps / per_sec), 1)
    bar = st.progress(0.0, text='Đang xử lý video...')
    best, idx = {}, 0                       # biển số -> (độ tin cậy, ảnh đã vẽ, bản ghi)
    while idx < total or total == 0:
        ok, frame = cap.read()
        if not ok or idx / fps > max_sec:
            break
        if idx % step == 0:
            ann, recs = system.process(frame, source=f'Video: {f.name}')
            for r in recs:
                if r['valid'] and (r['plate'] not in best or r['ocr_conf'] > best[r['plate']][0]):
                    best[r['plate']] = (r['ocr_conf'], ann, r)
        idx += 1
        if total:
            bar.progress(min(idx / total, 1.0), text=f'Đang xử lý video... {idx / fps:.0f}s')
    cap.release(); bar.empty()
    if not best:
        st.warning('Không tìm thấy biển số hợp lệ nào trong đoạn video đã xử lý.')
        return
    st.success(f'Tìm thấy {len(best)} biển số khác nhau (mỗi biển giữ khung hình rõ nhất).')
    rows = []
    for plate, (_, ann, r) in best.items():
        if save:
            db.add_detections([r], ann)
        rows.append({'Biển số': plate, 'Loại xe': r['vehicle'], 'Độ tin cậy OCR': r['ocr_conf']})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch',
                 column_config={'Độ tin cậy OCR': st.column_config.ProgressColumn(min_value=0, max_value=1, format='percent')})
    for plate, (_, ann, _) in list(best.items())[:6]:
        st.image(ann, channels='BGR', caption=plate, width='stretch')


def render():
    st.markdown('<p class="page-title">Nhận diện biển số</p><p class="page-sub">Tải ảnh, chụp từ camera hoặc phân tích video</p>', unsafe_allow_html=True)
    system, err = ui.get_system()
    if system is None:
        st.error(err)
        return
    s = ui.settings()
    save = st.toggle('Tự động lưu kết quả vào lịch sử', key='s_save_history')
    if not system.use_vehicle:
        st.caption('Đang tắt nhận dạng loại xe (không có `yolov8n.pt` hoặc đã tắt trong Cài đặt): loại xe sẽ ghi là "Không rõ".')
    t1, t2, t3 = st.tabs([':material/image: Ảnh', ':material/photo_camera: Camera', ':material/movie: Video'])
    with t1:
        _images(system, save)
    with t2:
        _camera(system, save)
    with t3:
        _video(system, save)
