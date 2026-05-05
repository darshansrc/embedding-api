FROM oven/bun:latest

WORKDIR /app

COPY package.json bun.lock* ./
RUN bun install --frozen-lockfile

COPY . .

RUN bun scripts/download-model.mjs

EXPOSE 3000
CMD ["bun", "run", "src/index.ts"]
