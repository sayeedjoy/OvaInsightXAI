import type { NextConfig } from "next"

const nextConfig: NextConfig = {
    /* config options here */
    output: "standalone",
    images: {
        remotePatterns: [
            {
                protocol: "https",
                hostname: "images.unsplash.com"
            }
        ]
    },
    async headers() {
        return [
            {
                source: "/(.*)",
                headers: [
                    {
                        key: "Content-Security-Policy",
                        value: [
                            "default-src 'self'",
                            "connect-src 'self'",
                            "img-src 'self' data: https:",
                            "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
                            "style-src 'self' 'unsafe-inline'"
                        ].join("; ")
                    },
                    {
                        key: "Permissions-Policy",
                        value: [
                            "accelerometer=()",
                            "autoplay=()",
                            "camera=()",
                            "cross-origin-isolated=()",
                            "display-capture=()",
                            "encrypted-media=()",
                            "fullscreen=(self)",
                            "geolocation=()",
                            "gyroscope=()",
                            "keyboard-map=()",
                            "magnetometer=()",
                            "microphone=()",
                            "midi=()",
                            "payment=()",
                            "picture-in-picture=()",
                            "publickey-credentials-get=()",
                            "screen-wake-lock=()",
                            "sync-xhr=()",
                            "usb=()",
                            "web-share=()",
                            "xr-spatial-tracking=()"
                        ].join(", ")
                    }
                ]
            }
        ]
    }
}

export default nextConfig
