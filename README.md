# LPR Analytics: dashboard nhận diện biển số xe

Dashboard Streamlit kiểu Google Analytics cho đồ án "Hệ thống phát hiện và nhận dạng biển số xe" (CV + DL).

## Chức năng
- **Tổng quan**: thẻ KPI có so sánh kỳ trước, biểu đồ theo giờ/ngày, tỉ trọng loại xe, biển số hay gặp, bản đồ nhiệt giờ cao điểm, hoạt động gần đây.
- **Nhận diện**: tải ảnh (nhiều ảnh), chụp từ camera, phân tích video. Kết quả tự lưu vào lịch sử.
- **Lịch sử**: xem dạng ảnh hoặc bảng, tìm biển số, lọc, sửa biển số đọc sai, xoá, xuất CSV.
- **Cài đặt**: ngưỡng, bật/tắt nắn thẳng và sửa lỗi theo vị trí, tạo/xoá dữ liệu mẫu.

## Cấu trúc
```
app.py            # điểm vào, điều hướng
lpr_core.py       # pipeline nhận dạng (giống notebook Kaggle)
db.py             # lịch sử: SQLite + ảnh trong data/
ui.py             # giao diện dùng chung, nạp mô hình, dữ liệu mẫu
views/            # 4 trang
weights/          # CHÉP TRỌNG SỐ VÀO ĐÂY
requirements.txt, packages.txt, .streamlit/config.toml
```

## Trọng số cần chép vào `weights/`
Lấy từ `DoAn3_ketqua.zip` (Output của notebook Kaggle):

| File | Bắt buộc |
|---|---|
| `lp_detect_best.pt` | có |
| `lp_char_best.pt` | có |
| `yolov8n.pt` | không (để nhận dạng loại xe; thiếu thì loại xe ghi "Không rõ") |

## Chạy trên máy
```
pip install -r requirements.txt
streamlit run app.py
```

## Đưa lên GitHub rồi Streamlit Community Cloud
1. Tạo repo GitHub, đẩy toàn bộ thư mục này lên (kể cả `weights/*.pt`, mỗi file khoảng 22 MB nên GitHub nhận được).
2. Vào share.streamlit.io, đăng nhập GitHub, chọn **Create app**, chọn repo, nhánh, file chính `app.py`.
3. Ở **Advanced settings** chọn Python 3.11, rồi Deploy.

Lưu ý:
- Lần đầu cài `torch` và `ultralytics` mất vài phút.
- Ổ đĩa trên Streamlit Cloud là tạm thời: lịch sử bị xoá khi app khởi động lại. Hãy xuất CSV, hoặc dùng "Tạo dữ liệu mẫu" khi trình diễn.
- Bộ nhớ của gói miễn phí hạn chế. Nếu app bị sập vì hết RAM, tắt "Nhận dạng loại xe" trong Cài đặt (không có `yolov8n.pt` thì mô hình này không được nạp).
