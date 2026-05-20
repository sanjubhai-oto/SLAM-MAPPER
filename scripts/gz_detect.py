#!/usr/bin/env python3.14
"""
YOLO object detection on the rover's fisheye streams.

Consumes the C++ webui MJPEG feeds (front + rear), runs YOLOv11n, draws
boxes, serves annotated MJPEGs back to the browser.

Endpoints:
  http://localhost:8081/front/detect.mjpeg
  http://localhost:8081/front/detections.json
  http://localhost:8081/rear/detect.mjpeg
  http://localhost:8081/rear/detections.json
  http://localhost:8081/detect.mjpeg       (alias for front, back-compat)
  http://localhost:8081/detections.json    (alias for front)

NB: gz.transport13 and cv2 segfault when loaded in the same Python process
on macOS arm64. This script avoids gz entirely by pulling MJPEG over HTTP.

Model: yolo11n.pt (Ultralytics, COCO-80). Same arch family DJI / Skydio
ship for on-board object detection.
"""
import argparse
import io
import json
import os
import socket
import sys
import threading
import time
import urllib.request

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
from PIL import Image as PILImage, ImageDraw
from ultralytics import YOLO, YOLOWorld


class CamState:
    def __init__(self, name):
        self.name = name
        self.lock = threading.Lock()
        self.jpeg = None
        self.seq = 0
        self.detections = []


def mjpeg_frames(url, timeout=10):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        boundary = None
        ct = resp.headers.get('Content-Type', '')
        if 'boundary=' in ct:
            boundary = ct.split('boundary=', 1)[1].strip().encode()
        if not boundary:
            boundary = b'frame'
        sep = b'--' + boundary
        buf = b''
        while True:
            chunk = resp.read(8192)
            if not chunk:
                return
            buf += chunk
            while True:
                i = buf.find(sep)
                if i < 0:
                    break
                j = buf.find(b'\r\n\r\n', i)
                if j < 0:
                    break
                header_block = buf[i:j].decode(errors='ignore')
                clen = None
                for line in header_block.split('\r\n'):
                    if line.lower().startswith('content-length:'):
                        clen = int(line.split(':', 1)[1].strip())
                if clen is None:
                    k = buf.find(sep, j + 4)
                    if k < 0:
                        break
                    yield buf[j + 4:k - 2]
                    buf = buf[k:]
                else:
                    start = j + 4
                    end = start + clen
                    if len(buf) < end:
                        break
                    yield buf[start:end]
                    buf = buf[end + 2:]


# Single model shared across reader threads. ultralytics predict is
# generally thread-safe on CPU but we serialize with a lock to be safe.
_model_lock = threading.Lock()


def reader_loop(state: CamState, source_url, model, conf, device):
    while True:
        try:
            print(f'[detect/{state.name}] connecting MJPEG {source_url}', flush=True)
            for jpg_bytes in mjpeg_frames(source_url):
                try:
                    pil = PILImage.open(io.BytesIO(jpg_bytes)).convert('RGB')
                except Exception:
                    continue
                try:
                    with _model_lock:
                        # Use track() so detections carry stable IDs across
                        # frames (ByteTrack). persist=True keeps state between
                        # calls per source (we pass a unique key per cam).
                        res = model.track(pil, conf=conf, device=device,
                                          verbose=False, imgsz=512,
                                          tracker='bytetrack.yaml',
                                          persist=True)
                except Exception as e:
                    print(f'[detect/{state.name}] infer err: {e}', flush=True)
                    continue
                r = res[0]
                dets = []
                draw = ImageDraw.Draw(pil)
                if r.boxes is not None and len(r.boxes) > 0:
                    ids_t = getattr(r.boxes, 'id', None)
                    ids = (ids_t.cpu().numpy().astype(int)
                           if ids_t is not None else [-1] * len(r.boxes))
                    for box, cls, c, tid in zip(
                            r.boxes.xyxy.cpu().numpy(),
                            r.boxes.cls.cpu().numpy().astype(int),
                            r.boxes.conf.cpu().numpy(),
                            ids):
                        x1, y1, x2, y2 = [int(v) for v in box]
                        name = model.names[int(cls)]
                        color = (60, 220, 60)
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                        label = f'#{tid} {name} {c:.2f}'
                        draw.rectangle([x1, max(0, y1 - 14), x1 + 8 * len(label), y1],
                                       fill=color)
                        draw.text((x1 + 2, max(0, y1 - 13)), label, fill=(0, 0, 0))
                        dets.append({'id': int(tid), 'cls': name,
                                     'conf': float(c),
                                     'box': [x1, y1, x2 - x1, y2 - y1]})
                buf = io.BytesIO()
                pil.save(buf, format='JPEG', quality=70)
                with state.lock:
                    state.jpeg = buf.getvalue()
                    state.seq += 1
                    state.detections = dets
        except Exception as e:
            print(f'[detect/{state.name}] stream err: {e}; reconnecting in 2s',
                  flush=True)
            time.sleep(2)


def http_server(cams: dict, port: int):
    """cams: dict name -> CamState."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('0.0.0.0', port))
    srv.listen(16)
    print(f'[detect] http on :{port}  cams={list(cams.keys())}', flush=True)

    def resolve(path):
        # /front/detect.mjpeg, /rear/detections.json, /detect.mjpeg (front), ...
        if path.startswith('/front/'):
            return cams.get('front'), path[len('/front'):]
        if path.startswith('/rear/'):
            return cams.get('rear'), path[len('/rear'):]
        # back-compat aliases -> front
        return cams.get('front'), path

    def serve_mjpeg(conn, state: CamState):
        hdr = (b'HTTP/1.1 200 OK\r\n'
               b'Access-Control-Allow-Origin: *\r\n'
               b'Cache-Control: no-cache, private\r\n'
               b'Content-Type: multipart/x-mixed-replace; '
               b'boundary=frame\r\n\r\n')
        conn.sendall(hdr)
        last = -1
        conn.settimeout(None)
        while True:
            with state.lock:
                jpg = state.jpeg
                seq = state.seq
            if jpg is not None and seq != last:
                last = seq
                head = (b'--frame\r\nContent-Type: image/jpeg\r\n'
                        b'Content-Length: ' + str(len(jpg)).encode()
                        + b'\r\n\r\n')
                conn.sendall(head + jpg + b'\r\n')
            time.sleep(0.05)

    def serve_json(conn, state: CamState):
        with state.lock:
            body = json.dumps({'cam': state.name, 'seq': state.seq,
                               'detections': state.detections})
        body_b = body.encode()
        hdr = (b'HTTP/1.1 200 OK\r\n'
               b'Access-Control-Allow-Origin: *\r\n'
               b'Content-Type: application/json\r\n'
               b'Content-Length: ' + str(len(body_b)).encode()
               + b'\r\n\r\n')
        conn.sendall(hdr + body_b)

    def handle(conn):
        try:
            conn.settimeout(5.0)
            data = b''
            while b'\r\n\r\n' not in data:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                data += chunk
            req_line = data.split(b'\r\n', 1)[0].decode(errors='ignore')
            path = req_line.split(' ')[1] if ' ' in req_line else '/'
            state, sub = resolve(path)
            if state is None:
                conn.sendall(b'HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n')
                return
            if sub.startswith('/detect.mjpeg'):
                serve_mjpeg(conn, state)
            elif sub.startswith('/detections.json'):
                serve_json(conn, state)
            else:
                conn.sendall(b'HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n')
        except Exception:
            pass
        finally:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            conn.close()

    while True:
        c, _ = srv.accept()
        threading.Thread(target=handle, args=(c,), daemon=True).start()


DEFAULT_CLASSES = ['orange traffic cone', 'standing human person',
                   'warehouse worker in orange vest',
                   'cardboard box obstacle', 'wooden crate', 'wooden pallet',
                   'oil drum barrel', 'industrial shelving rack', 'forklift',
                   'cylindrical pillar', 'tree',
                   'wall', 'floor', 'sky']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='yolov8s-world.pt',
                    help='YOLO weights (default yolov8s-world.pt — open vocab)')
    ap.add_argument('--front',
                    default='http://localhost:8080/front.mjpeg')
    ap.add_argument('--rear',
                    default='http://localhost:8080/rear.mjpeg')
    ap.add_argument('--port', type=int, default=8081)
    ap.add_argument('--conf', type=float, default=0.20)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--classes', nargs='+', default=DEFAULT_CLASSES,
                    help='whitelist of classes (YOLO-World accepts any text)')
    args = ap.parse_args()

    print(f'[detect] loading {args.model} on {args.device} ...', flush=True)
    is_world = 'world' in args.model.lower()
    if is_world:
        model = YOLOWorld(args.model)
        model.set_classes(args.classes)
        print(f'[detect] YOLO-World classes={args.classes}', flush=True)
    else:
        model = YOLO(args.model)

    cams = {
        'front': CamState('front'),
        'rear': CamState('rear'),
    }
    for name, url in (('front', args.front), ('rear', args.rear)):
        threading.Thread(
            target=reader_loop,
            args=(cams[name], url, model, args.conf, args.device),
            daemon=True).start()
    http_server(cams, args.port)
    return 0


if __name__ == '__main__':
    sys.exit(main())
