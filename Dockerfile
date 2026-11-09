FROM node:22-slim
WORKDIR /app

# System deps. fontconfig: config files the ffmpeg build's caption/text renderer (libass) expects.
# libfribidi0: Pillow joins Telugu and Hindi letters only with it (text with a look is drawn by Pillow).
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl ca-certificates xz-utils fontconfig libfribidi0 \
    && rm -rf /var/lib/apt/lists/*

# FFmpeg 9.0, static build. Debian's package is 5.1, which fails on renders with looping inputs
# (frame photo/video lanes): "Failed to inject frame into filter network … Error reinitializing
# filters". The build includes ffprobe, libass (captions), freetype/fontconfig (text) and the movie
# filter (rounded corners).
# One fixed build, checked by its checksum: "latest" changed with every rebuild, and the 2026-11-04
# build (n9.0.2-47) lost the main video from the 10th part/slot of a frames clip on — the same clip
# exported right with 9.0.2 in development (user, 2026-11-05). This is the nearest build to 9.0.2
# (BtbN keeps its month-end builds). Change it on purpose, after an export of a frames clip with
# 8+ parts looks right.
ARG FFMPEG_BUILD=autobuild-2026-09-30-13-08
ARG FFMPEG_NAME=ffmpeg-n9.0.2-17-g2a571b6068-linux64-gpl-9.0
ARG FFMPEG_SHA256=68ee646831adaae2495618346f3bba94ff207ff83bbd34d643e7004730d66269
RUN curl -fL https://github.com/BtbN/FFmpeg-Builds/releases/download/${FFMPEG_BUILD}/${FFMPEG_NAME}.tar.xz \
      -o /tmp/ffmpeg.tar.xz \
    && echo "${FFMPEG_SHA256}  /tmp/ffmpeg.tar.xz" | sha256sum -c - \
    && tar -xJf /tmp/ffmpeg.tar.xz -C /tmp \
    && mv /tmp/${FFMPEG_NAME}/bin/ffmpeg /tmp/${FFMPEG_NAME}/bin/ffprobe /usr/local/bin/ \
    && rm -rf /tmp/ffmpeg* \
    && ffmpeg -hide_banner -version | head -1 \
    && for f in subtitles drawtext movie zoompan sendcmd; do \
         ffmpeg -hide_banner -filters | grep -q " $f " || { echo "ffmpeg build is missing the $f filter"; exit 1; }; \
       done \
    && { ffmpeg -hide_banner -buildconf | grep -q enable-libharfbuzz || { echo "ffmpeg build lacks harfbuzz (drawtext would break Telugu/Hindi text)"; exit 1; }; }

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
