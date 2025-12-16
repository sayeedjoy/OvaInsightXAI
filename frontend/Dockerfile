FROM node:20-alpine AS base

# Install pnpm
RUN corepack enable && corepack prepare pnpm@10.12.1 --activate

# Set working directory
WORKDIR /app

# Copy package files
COPY package.json ./
# Copy lockfile if it exists (optional)
COPY pnpm-lock.yaml* ./

# Install dependencies
# Try frozen lockfile first, fallback to regular install if lockfile doesn't exist
RUN if [ -f pnpm-lock.yaml ]; then \
        pnpm install --frozen-lockfile || pnpm install; \
    else \
        echo "Warning: pnpm-lock.yaml not found, installing without lockfile"; \
        pnpm install; \
    fi

# Copy source code
COPY . .

# Build the application
RUN pnpm build

# Production image
FROM node:20-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production

# Copy necessary files from builder
COPY --from=base /app/public ./public
COPY --from=base /app/.next/standalone ./
COPY --from=base /app/.next/static ./.next/static

EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["node", "server.js"]

