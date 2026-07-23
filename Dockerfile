FROM node:22-slim AS base
WORKDIR /app

# Install system deps: ffmpeg, yt-dlp, python3 + pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o /usr/local/bin/yt-dlp \
    && chmod +x /usr/local/bin/yt-dlp

# Python deps
COPY apps/worker/src/python/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# Node deps (monorepo context)
COPY package.json pnpm-workspace.yaml ./
COPY packages/shared/package.json ./packages/shared/package.json
COPY apps/worker/package.json ./apps/worker/package.json
RUN npm install -g pnpm && pnpm install --filter worker --frozen-lockfile

# Source
COPY packages/shared ./packages/shared
COPY apps/worker ./apps/worker
COPY tsconfig.base.json ./

# Build shared, then worker
RUN pnpm --filter @chai-cut/shared build
RUN pnpm --filter worker build

# Fonts directory (mount at runtime or COPY here)
RUN mkdir -p /app/fonts
ENV FONTS_DIR=/app/fonts

CMD ["node", "apps/worker/dist/index.js"]
