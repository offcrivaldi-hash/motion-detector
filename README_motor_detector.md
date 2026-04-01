# Motor Detector untuk Video Lokal

Program ini memakai 2 tahap supaya hemat CPU:

1. `Background subtraction` hanya di ROI merah untuk mencari area gerakan.
2. `YOLO` hanya dijalankan pada crop area gerakan itu untuk memastikan objek benar-benar motor.

Program ini sekarang mendukung:

- `single video mode`
- `batch folder mode` untuk scan semua video dalam satu folder
- urutan proses berdasarkan waktu awal video yang di-resolve otomatis

## Output

Setiap motor yang lolos konfirmasi akan menghasilkan:

- `motor_HHMMSS_mmm_fFRAME_crop.jpg` -> close-up motor + timestamp di header
- `motor_HHMMSS_mmm_fFRAME_full.jpg` -> full frame + kotak hijau
- `detections.csv` -> log timestamp, frame, confidence, bbox, dan nama file

Kalau pakai mode folder, output root juga akan berisi:

- `videos_summary.csv` -> urutan video yang diproses + jumlah event tiap video
- `all_detections.csv` -> gabungan seluruh event dari semua video

## Install

```powershell
pip install -r requirements-motor-detector.txt
```

## Contoh pemakaian

```powershell
python motor_detector.py `
  --video "C:\Users\IT\Desktop\1 parkir\hiv00224.mp4" `
  --output-dir ".\output_hiv00224" `
  --start-datetime "2026-03-31 20:55:31" `
  --show
```

## Catatan ROI

Default ROI sudah diset mengikuti area merah pada screenshot:

- `0.115,0.163,0.837,0.906`

Kalau posisi area deteksi berubah, ganti dengan argumen:

```powershell
python motor_detector.py --video "video.mp4" --roi-norm "0.10,0.15,0.85,0.92"
```

## Testing segmen tertentu

Kalau ingin tuning di bagian video yang sudah ada motor, loncat ke frame tertentu:

```powershell
python motor_detector.py `
  --video "C:\Users\IT\Desktop\1 parkir\hiv00224.mp4" `
  --output-dir ".\output_segment" `
  --start-datetime "2026-03-31 20:55:31" `
  --start-frame 4200 `
  --max-frames 500 `
  --warmup-frames 30
```

## Scan 1 folder video

```powershell
python motor_detector.py `
  --input-dir "C:\Users\IT\Desktop\1 parkir" `
  --output-dir ".\output_semua_video"
```

Kalau ingin progress lebih kelihatan di terminal:

```powershell
python motor_detector.py `
  --input-dir "C:\Users\IT\Desktop\1 parkir" `
  --output-dir ".\output_semua_video" `
  --progress-step 1
```

Mode folder akan:

- scan semua file video di folder
- urutkan berdasarkan waktu awal video
- simpan hasil tiap video ke subfolder sendiri
- buat `videos_summary.csv` dan `all_detections.csv` di folder output root
- tampilkan progress persen selama proses berjalan

## Cara urutan waktu video ditentukan

Default `--video-time-source auto`:

1. pakai `--video-time-map` kalau ada
2. coba baca datetime dari nama file kalau format nama mendukung
3. fallback ke `LastWriteTime` file

## CPU dan GPU

Sekarang `--device` default adalah `auto`.

Artinya:

- `OpenCV/background subtraction/tracking` tetap jalan di CPU
- `YOLO` akan pindah ke GPU otomatis **kalau** PyTorch CUDA tersedia
- kalau GPU tidak tersedia, YOLO fallback ke CPU

Contoh pakai otomatis:

```powershell
python motor_detector.py `
  --input-dir "C:\Users\IT\Desktop\1 parkir" `
  --output-dir ".\output_semua_video" `
  --device auto
```

Kalau mau paksa CPU:

```powershell
python motor_detector.py --video "video.mp4" --device cpu
```

Untuk folder video Anda, fallback `LastWriteTime` cocok dengan watermark video. Contoh:

- file `hiv00223.mp4` punya waktu file `2026-03-31 20:55:30`
- watermark video yang Anda kirim menunjukkan `2026-03-31 20:55:31`

Kalau ingin kontrol penuh per video, pakai CSV mapping:

```csv
file,start_datetime
hiv00193.mp4,2026-03-31 00:14:52
hiv00194.mp4,2026-03-31 00:59:08
```

Lalu jalankan:

```powershell
python motor_detector.py `
  --input-dir "C:\Users\IT\Desktop\1 parkir" `
  --output-dir ".\output_semua_video" `
  --video-time-source map `
  --video-time-map ".\video_times.csv"
```

## Tuning cepat kalau terlalu sensitif / kurang sensitif

- Kurangi noise gerakan kecil: naikkan `--min-motion-area` misalnya ke `3500`
- Kalau motor sering tidak tertangkap: turunkan `--min-motion-area` ke `1800`
- Kalau YOLO terlalu pelit: turunkan `--yolo-conf` misalnya ke `0.25`
- Kalau event dobel: naikkan `--track-min-hits` ke `3`
- Kalau event dobel masih muncul: naikkan `--event-dedupe-ms` atau `--event-dedupe-distance`
- Kalau motor diam ikut tersimpan: naikkan `--track-min-travel` misalnya ke `60`
