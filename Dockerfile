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
RUN npm install

# Source
COPY . .
RUN npm run build

# Download fonts so caption burn-in works without a volume mount
RUN mkdir -p /app/src/python/fonts && \
    curl -fsSL -o /app/src/python/fonts/NotoSansTelugu-Regular.ttf \
      "https://github.com/notofonts/notofonts.github.io/raw/main/fonts/NotoSansTelugu/unhinted/ttf/NotoSansTelugu-Regular.ttf" && \
    curl -fsSL -o /app/src/python/fonts/NotoSansDevanagari-Regular.ttf \
      "https://github.com/notofonts/notofonts.github.io/raw/main/fonts/NotoSansDevanagari/unhinted/ttf/NotoSansDevanagari-Regular.ttf" && \
    curl -fsSL -o /app/src/python/fonts/Roboto-Regular.ttf \
      "https://github.com/googlefonts/roboto/raw/main/src/hinted/Roboto-Regular.ttf" && \
    curl -fsSL -o /app/src/python/fonts/Montserrat-Bold.ttf \
      "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Bold.ttf"

ENV NODE_ENV=production

CMD ["node", "dist/index.js"]
