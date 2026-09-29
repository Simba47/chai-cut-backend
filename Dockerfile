FROM node:22-slim
WORKDIR /app

# System deps. fontconfig: config files the ffmpeg build's caption/text renderer (libass) expects.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl ca-certificates xz-utils fontconfig \
    && rm -rf /var/lib/apt/lists/*

# FFmpeg 9.0, static build — the same version used in development. Debian's package is 5.1,
# which fails on renders with looping inputs (frame photo/video lanes): "Failed to inject
# frame into filter network … Error reinitializing filters". The build includes ffprobe,
# libass (captions), freetype/fontconfig (text) and the movie filter (rounded corners).
RUN curl -fL https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n9.0-latest-linux64-gpl-9.0.tar.xz \
      -o /tmp/ffmpeg.tar.xz \
    && tar -xJf /tmp/ffmpeg.tar.xz -C /tmp \
    && mv /tmp/ffmpeg-n9.0-latest-linux64-gpl-9.0/bin/ffmpeg /tmp/ffmpeg-n9.0-latest-linux64-gpl-9.0/bin/ffprobe /usr/local/bin/ \
    && rm -rf /tmp/ffmpeg* \
    && ffmpeg -hide_banner -version | head -1 \
    && for f in subtitles drawtext movie zoompan; do \
         ffmpeg -hide_banner -filters | grep -q " $f " || { echo "ffmpeg build is missing the $f filter"; exit 1; }; \
       done

RUN curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp \
    -o /usr/local/bin/yt-dlp && chmod +x /usr/local/bin/yt-dlp

# Python deps
COPY src/python/requirements.txt src/python/requirements-nodeps.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt --break-system-packages \
    && pip3 install --no-cache-dir --no-deps -r /tmp/requirements-nodeps.txt --break-system-packages \
    && MPLBACKEND=Agg python3 -c "from mediapipe.tasks.python import vision; import cv2; cv2.CascadeClassifier"

# Node deps
COPY package.json package-lock.json* ./
RUN npm install

# Source
COPY . .
RUN npm run build

# Fonts are committed to src/python/fonts/ — no download needed.

ENV NODE_ENV=production

CMD ["node", "dist/index.js"]
