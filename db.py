"""Lưu lịch sử nhận diện: SQLite (history.db) + ảnh JPEG trong thư mục data/images."""
import os, sqlite3, uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import cv2
import pandas as pd

DATA_DIR = Path(os.environ.get('LPR_DATA_DIR', Path(__file__).parent / 'data'))
IMG_DIR = DATA_DIR / 'images'
DB_PATH = DATA_DIR / 'history.db'
TZ = ZoneInfo('Asia/Ho_Chi_Minh')
COLS = ['id', 'ts', 'source', 'vehicle', 'plate', 'valid', 'det_conf', 'ocr_conf', 'image_file', 'crop_file', 'demo']


def now_vn():
    return datetime.now(TZ).replace(tzinfo=None)


def _con():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, source TEXT, vehicle TEXT, plate TEXT,
        valid INTEGER, det_conf REAL, ocr_conf REAL, image_file TEXT, crop_file TEXT, demo INTEGER DEFAULT 0)""")
    con.execute("CREATE INDEX IF NOT EXISTS idx_ts ON detections(ts)")
    return con


def load_df():
    with _con() as con:
        df = pd.read_sql_query('SELECT * FROM detections ORDER BY ts', con)
    df['ts'] = pd.to_datetime(df['ts'])
    df['valid'] = df['valid'].astype(bool)
    df['demo'] = df['demo'].astype(bool)
    return df


def save_jpg(img, prefix, max_side=1280, quality=85):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    name = f'{prefix}_{uuid.uuid4().hex[:10]}.jpg'
    cv2.imwrite(str(IMG_DIR / name), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return name


def add_detections(recs, ann=None, ts=None):
    """recs: bản ghi từ LPRSystem.process (có khoá 'crop'). Lưu ảnh đã vẽ (dùng chung) và ảnh biển cắt riêng."""
    if not recs:
        return 0
    ts = ts or now_vn()
    img_name = save_jpg(ann, 'scene') if ann is not None else None
    rows = []
    for r in recs:
        crop = r.get('crop')
        crop_name = save_jpg(crop, 'plate', max_side=320) if crop is not None and crop.size else None
        rows.append((ts.strftime('%Y-%m-%d %H:%M:%S'), r.get('source'), r.get('vehicle'), r.get('plate'), int(bool(r.get('valid'))),
                     r.get('det_conf'), r.get('ocr_conf'), img_name, crop_name, 0))
    with _con() as con:
        con.executemany('INSERT INTO detections (ts, source, vehicle, plate, valid, det_conf, ocr_conf, image_file, crop_file, demo) '
                        'VALUES (?,?,?,?,?,?,?,?,?,?)', rows)
    return len(rows)


def insert_rows(rows):
    with _con() as con:
        con.executemany('INSERT INTO detections (ts, source, vehicle, plate, valid, det_conf, ocr_conf, image_file, crop_file, demo) '
                        'VALUES (?,?,?,?,?,?,?,?,?,?)', rows)


def update_plates(changes):
    """changes: {id: (plate, valid)}"""
    with _con() as con:
        con.executemany('UPDATE detections SET plate=?, valid=? WHERE id=?', [(p, int(v), i) for i, (p, v) in changes.items()])


def _unlink_unused(con, files):
    for f in set(x for x in files if x):
        left = con.execute('SELECT 1 FROM detections WHERE image_file=? OR crop_file=? LIMIT 1', (f, f)).fetchone()
        if not left:
            (IMG_DIR / f).unlink(missing_ok=True)


def delete_ids(ids):
    ids = [int(i) for i in ids]
    if not ids:
        return
    with _con() as con:
        q = ','.join('?' * len(ids))
        files = [x for row in con.execute(f'SELECT image_file, crop_file FROM detections WHERE id IN ({q})', ids) for x in row]
        con.execute(f'DELETE FROM detections WHERE id IN ({q})', ids)
        _unlink_unused(con, files)


def clear(demo_only=False):
    with _con() as con:
        where = ' WHERE demo=1' if demo_only else ''
        files = [x for row in con.execute(f'SELECT image_file, crop_file FROM detections{where}') for x in row]
        con.execute(f'DELETE FROM detections{where}')
        _unlink_unused(con, files)
