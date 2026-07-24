FROM node:22-slim
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg python3 python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp \
    -o /usr/local/bin/yt-dlp && chmod +x /usr/local/bin/yt-dlp

# Python deps
COPY src/python/requirements.txt /tmp/req.txt
RUN pip3 install --no-cache-dir -r /tmp/req.txt --break-system-packages

# Node deps
COPY package.json package-lock.json* ./
RUN npm install -g pnpm && pnpm install --frozen-lockfile

# Source
COPY . .
RUN pnpm build

RUN mkdir -p /app/fonts
ENV FONTS_DIR=/app/fonts
ENV NODE_ENV=production

CMD ["node", "dist/index.js"]
