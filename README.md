# SAM3 Human Analysis API

Phát hiện người trong ảnh/video/camera, phân loại tư thế (đứng/ngồi/nằm) và nhận diện chuyển động sử dụng [SAM3](https://huggingface.co/facebook/sam3) của Meta AI.

## Tính năng

- Phát hiện người bằng SAM3 với text prompt `"person"`
- Phân loại tư thế: **đứng** / **ngồi** / **nằm**
- Nhận diện chuyển động (moving / static) cho video và camera
- Trả về JSON kết quả + file ảnh/video đã được vẽ annotation
- Hỗ trợ: ảnh tĩnh (jpg/png), video (mp4/avi), camera thật và RTSP stream

## Yêu cầu

- [Anaconda](https://www.anaconda.com/download) hoặc [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Python 3.10
- GPU (khuyến nghị) hoặc CPU

> **Lưu ý:** SAM3 yêu cầu chấp nhận Meta Privacy Policy trên HuggingFace trước khi tải model.
> Truy cập [https://huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3) và đăng nhập để chấp nhận điều khoản.

---

## Cài đặt

### 1. Tạo môi trường Conda

```bash
conda create -n sam3-demo python=3.10 -y
conda activate sam3-demo
```

### 2. Cài PyTorch

**Nếu có GPU (CUDA 11.8):**
```bash
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

**Nếu có GPU (CUDA 12.1):**
```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

**Nếu chỉ có CPU (hoặc Mac M1/M2):**
```bash
conda install pytorch torchvision cpuonly -c pytorch -y
```

### 3. Cài các thư viện còn lại

```bash
pip install -r requirements.txt
```

### 4. Cấu hình môi trường

```bash
cp .env.example .env
```

Mở file `.env` và chỉnh theo máy của bạn:

```dotenv
# Chọn thiết bị: cuda (GPU), mps (Apple Silicon), hoặc cpu
DEVICE=cuda

# Ngưỡng tin cậy khi phát hiện người (0.0 - 1.0)
DETECTION_THRESHOLD=0.5

# Số frame/giây phân tích trong video (thấp hơn = nhanh hơn)
ANALYZE_FPS=2

# Số giây thu từ camera/RTSP stream
STREAM_CAPTURE_SECONDS=10

# Ngưỡng pixel dịch chuyển để xác định đang di chuyển
MOVEMENT_THRESHOLD_PX=15
```

---

## Chạy server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Sau khi khởi động, mở trình duyệt tại:
- **Swagger UI (thử API trực tiếp):** `http://localhost:8000/docs`
- **Health check:** `http://localhost:8000/health`

> Lần đầu chạy, SAM3 sẽ tự động tải model (~1-2 GB). Cần kết nối internet và đã đăng nhập HuggingFace.

---

## Sử dụng API

### Phân tích ảnh

```bash
curl -X POST http://localhost:8000/analyze/image \
  -F "file=@/path/to/image.jpg"
```

**Response:**
```json
{
  "source": "image.jpg",
  "type": "image",
  "has_humans": true,
  "analyzed_at": "2026-05-14T10:00:00Z",
  "output_file": "/files/result_20260514_100000_image.jpg",
  "persons": [
    {
      "id": 1,
      "confidence": 0.93,
      "bbox": [45.0, 12.0, 210.0, 480.0],
      "pose": "standing",
      "is_moving": null
    }
  ]
}
```

### Phân tích video

```bash
curl -X POST http://localhost:8000/analyze/video \
  -F "file=@/path/to/video.mp4"
```

**Response:**
```json
{
  "source": "video.mp4",
  "type": "video",
  "has_humans": true,
  "output_file": "/files/result_20260514_100000_video.mp4",
  "duration_sec": 10.5,
  "source_fps": 30.0,
  "total_frames": 315,
  "analyzed_frames": 21,
  "frames": [
    {
      "frame_idx": 0,
      "timestamp_sec": 0.0,
      "persons": [
        {
          "id": 1,
          "confidence": 0.91,
          "bbox": [50.0, 20.0, 200.0, 460.0],
          "pose": "standing",
          "is_moving": false
        }
      ]
    }
  ],
  "person_summary": [
    {
      "id": 1,
      "dominant_pose": "standing",
      "was_moving": true,
      "frames_detected": 18
    }
  ]
}
```

### Phân tích camera / RTSP stream

```bash
# Webcam (index 0)
curl -X POST http://localhost:8000/analyze/stream \
  -F "url=0" \
  -F "duration_sec=10"

# RTSP camera
curl -X POST http://localhost:8000/analyze/stream \
  -F "url=rtsp://username:password@192.168.1.100:554/stream" \
  -F "duration_sec=15"
```

### Tải file kết quả

```bash
# Tải ảnh/video đã được vẽ annotation
curl -O http://localhost:8000/files/result_20260514_100000_image.jpg
```

---

## Cấu trúc project

```
sam3-demo/
├── app/
│   ├── main.py              # FastAPI app, các endpoint
│   ├── schemas.py           # Pydantic models cho JSON response
│   ├── detector.py          # SAM3 phát hiện người
│   ├── pose.py              # Phân loại tư thế (standing/sitting/lying)
│   ├── tracker.py           # IoU tracker + phát hiện chuyển động
│   ├── video_processor.py   # Xử lý video/stream theo frame
│   └── visualizer.py        # Vẽ bounding box và nhãn lên ảnh/video
├── outputs/                 # File kết quả được lưu tại đây (tự tạo khi chạy)
├── requirements.txt
├── .env.example
└── README.md
```

## Màu sắc annotation

| Màu | Tư thế |
|-----|--------|
| Xanh lá | Đứng (standing) |
| Cam | Ngồi (sitting) |
| Đỏ | Nằm (lying) |

Label hiển thị: `ID:1 standing 0.93 [MOVING]`

---

## Gỡ lỗi thường gặp

**Lỗi `OSError: facebook/sam3 is not a local folder and is not a valid model identifier`**
- Kiểm tra kết nối internet
- Đăng nhập HuggingFace và chấp nhận điều khoản tại trang model

**Lỗi `CUDA out of memory`**
- Giảm `DETECTION_THRESHOLD` lên cao hơn (ví dụ `0.7`) để ít object hơn
- Tăng `ANALYZE_FPS` xuống thấp hơn (ví dụ `1`) để giảm tần suất xử lý
- Hoặc đặt `DEVICE=cpu`

**Video output không có âm thanh**
- Đây là giới hạn của OpenCV VideoWriter, chỉ ghi được hình ảnh
