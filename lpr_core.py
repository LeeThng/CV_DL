"""Lõi nhận dạng biển số: nắn thẳng, phát hiện ký tự, gom dòng, sửa lỗi theo vị trí.
Giống hệt logic trong notebook huấn luyện (bản Kaggle), tách ra để dùng cho ứng dụng Streamlit."""
import re
from pathlib import Path
import numpy as np
import cv2

# 30 lớp ký tự: id 0..8 = '1'..'9' | id 9..28 = 20 chữ cái | id 29 = '0'
CHAR_NAMES = list('123456789') + list('ABCDEFGHKLMNPSTUVXYZ') + ['0']
CHAR_IMGSZ = 416

VEHICLE_VI = {'car': 'Ô tô', 'motorcycle': 'Xe máy', 'bus': 'Xe buýt', 'truck': 'Xe tải'}
VEHICLE_EN = {'car': 'car', 'motorcycle': 'motorbike', 'bus': 'bus', 'truck': 'truck'}   # cv2.putText không vẽ được dấu tiếng Việt


# ---------- Nắn thẳng biển số (Canny + Hough) ----------
def deskew_plate(img, max_angle=15):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                            minLineLength=max(img.shape[1] // 3, 10), maxLineGap=10)
    if lines is None:
        return img
    angles = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(a) <= max_angle:
            angles.append(a)
    if not angles:
        return img
    angle = float(np.median(angles))
    if abs(angle) < 1.0:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# ---------- Gom ký tự thành dòng ----------
def group_lines(chars, gap_ratio=0.5):
    if not chars:
        return []
    med_h = float(np.median([c['h'] for c in chars]))
    srt = sorted(chars, key=lambda c: c['cy'])
    lines = [srt]
    if len(srt) > 1:
        gaps = [srt[i + 1]['cy'] - srt[i]['cy'] for i in range(len(srt) - 1)]
        k = int(np.argmax(gaps))
        if gaps[k] > gap_ratio * med_h:
            lines = [srt[:k + 1], srt[k + 1:]]
            lines = [l for l in lines if len(l) >= 2] or [srt]
    return [sorted(l, key=lambda c: c['cx']) for l in lines]


# ---------- Sửa lỗi theo vị trí (chỉ với ký tự độ tin cậy thấp) ----------
TO_DIGIT = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'T': '1', 'Z': '2', 'S': '5', 'G': '6', 'B': '8', 'A': '4'}
TO_LETTER = {'0': 'D', '2': 'Z', '5': 'S', '6': 'G', '8': 'B', '4': 'A'}
FIX_CONF = 0.7
MAX_FIX = 2
PLATE_RE = re.compile(r'^\d{2}[A-Z]{1,2}\d?-\d{4,5}$')


def _expected(i, line_idx, n_lines):
    if n_lines == 1:
        return 'D' if (i < 2 or i >= 4) else ('L' if i == 2 else None)
    if line_idx == 0:
        return 'D' if i < 2 else ('L' if i == 2 else None)
    return 'D'


def fix_by_position(groups, conf_thr=None, max_changes=None):
    conf_thr = FIX_CONF if conf_thr is None else conf_thr
    max_changes = MAX_FIX if max_changes is None else max_changes
    raw, fixed, changes = [], [], 0
    for li, g in enumerate(groups):
        r, f = [], []
        for i, c in enumerate(g):
            ch, new, want = c['ch'], c['ch'], _expected(i, li, len(groups))
            if c['conf'] < conf_thr:
                if want == 'D' and not ch.isdigit():
                    new = TO_DIGIT.get(ch, ch)
                elif want == 'L' and ch.isdigit():
                    new = TO_LETTER.get(ch, ch)
            changes += (new != ch)
            r.append(ch); f.append(new)
        raw.append(''.join(r)); fixed.append(''.join(f))
    return raw if changes > max_changes else fixed


def format_plate(lines):
    if not lines:
        return ''
    if len(lines) == 2:
        return lines[0] + '-' + lines[1]
    s = lines[0]
    if len(s) > 3:
        cut = 4 if s[3].isalpha() else 3
        return s[:cut] + '-' + s[cut:]
    return s


def is_valid_plate(text):
    return bool(PLATE_RE.match(text or ''))


def read_text_from_chars(chars, use_fix=True, fix_thr=None):
    groups = group_lines(chars)
    lines = [''.join(c['ch'] for c in g) for g in groups]
    raw = ''.join(lines)
    if use_fix:
        lines = fix_by_position(groups, fix_thr)
    text = format_plate(lines)
    conf = float(np.mean([c['conf'] for g in groups for c in g])) if groups else 0.0
    return dict(raw=raw, lines=lines, text=text, valid=is_valid_plate(text), conf=conf)


def predict_chars(model, img, conf=0.3, imgsz=CHAR_IMGSZ):
    # Dùng CHAR_NAMES cố định, không dùng model.names (tên ghi trong file trọng số có thể sai thứ tự)
    r = model.predict(img, imgsz=imgsz, conf=conf, iou=0.5, agnostic_nms=True, verbose=False)[0]
    chars = []
    if r.boxes is None or len(r.boxes) == 0:
        return chars
    xywh = r.boxes.xywh.cpu().numpy(); cls = r.boxes.cls.cpu().numpy().astype(int); cf = r.boxes.conf.cpu().numpy()
    for (cx, cy, w, h), c, s in zip(xywh, cls, cf):
        if int(c) >= len(CHAR_NAMES):
            continue
        chars.append(dict(cx=float(cx), cy=float(cy), w=float(w), h=float(h), ch=CHAR_NAMES[int(c)], conf=float(s)))
    return chars


def decode_image(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR) if arr.size else None


class LPRSystem:
    """Pipeline 3 tầng: phương tiện (tuỳ chọn) -> biển số -> ký tự."""

    def __init__(self, plate_w, char_w, vehicle_w=None):
        from ultralytics import YOLO
        self.plate = YOLO(str(plate_w))
        self.char = YOLO(str(char_w))
        self.veh = YOLO(str(vehicle_w)) if vehicle_w else None
        self.conf_plate, self.conf_char = 0.4, 0.3
        self.use_deskew, self.use_fix, self.use_vehicle = True, True, vehicle_w is not None

    def configure(self, conf_plate=0.4, conf_char=0.3, use_deskew=True, use_fix=True, use_vehicle=True):
        self.conf_plate, self.conf_char = conf_plate, conf_char
        self.use_deskew, self.use_fix = use_deskew, use_fix
        self.use_vehicle = use_vehicle and self.veh is not None

    def read_plate(self, crop):
        if self.use_deskew:
            crop = deskew_plate(crop)
        return read_text_from_chars(predict_chars(self.char, crop, self.conf_char), self.use_fix)

    def _vehicles(self, img):
        if not self.use_vehicle:
            return []
        r = self.veh.predict(img, classes=[2, 3, 5, 7], conf=0.35, verbose=False)[0]
        return [(b, self.veh.names[int(c)]) for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy())]

    @staticmethod
    def _match_vehicle(center, vehicles):
        best, best_area = None, None
        for (x1, y1, x2, y2), name in vehicles:
            if x1 <= center[0] <= x2 and y1 <= center[1] <= y2:
                area = (x2 - x1) * (y2 - y1)
                if best is None or area < best_area:
                    best, best_area = name, area
        return best

    def process(self, img, source='image'):
        """Trả về (ảnh đã vẽ, danh sách bản ghi). Mỗi bản ghi có thêm khoá 'crop' (ảnh biển số cắt ra)."""
        H, W = img.shape[:2]
        ann, recs = img.copy(), []
        fs = max(0.6, W / 1500); th = max(2, W // 700)
        vehicles = self._vehicles(img)
        for (x1, y1, x2, y2), name in vehicles:
            cv2.rectangle(ann, (int(x1), int(y1)), (int(x2), int(y2)), (255, 128, 0), 1)
            cv2.putText(ann, VEHICLE_EN.get(name, name), (int(x1), max(int(y1) - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs * 0.6, (255, 128, 0), 1)
        r = self.plate.predict(img, imgsz=640, conf=self.conf_plate, verbose=False)[0]
        for b, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = [int(v) for v in b]
            pw, ph = int((x2 - x1) * 0.05), int((y2 - y1) * 0.08)
            crop = img[max(y1 - ph, 0):min(y2 + ph, H), max(x1 - pw, 0):min(x2 + pw, W)]
            if crop.size == 0:
                continue
            res = self.read_plate(crop)
            veh = self._match_vehicle(((x1 + x2) / 2, (y1 + y2) / 2), vehicles)
            color = (0, 200, 0) if res['valid'] else (0, 140, 255)
            cv2.rectangle(ann, (x1, y1), (x2, y2), color, th)
            label = res['text'] or '?'
            (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            cv2.rectangle(ann, (x1, max(y1 - tht - 12, 0)), (x1 + tw + 6, max(y1, tht + 12)), color, -1)
            cv2.putText(ann, label, (x1 + 3, max(y1 - 6, tht + 4)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th)
            recs.append(dict(source=source, vehicle=VEHICLE_VI.get(veh, 'Không rõ') if veh else 'Không rõ',
                             plate=res['text'], valid=bool(res['valid']), det_conf=round(float(cf), 3),
                             ocr_conf=round(res['conf'], 3), crop=crop))
        return ann, recs
