import type { NextConfig } from "next"

const nextConfig: NextConfig = {
    /* config options here */
    output: "standalone",
    experimental: {
        optimizePackageImports: ["lucide-react", "@radix-ui/react-accordion", "@radix-ui/react-dialog", "@radix-ui/react-dropdown-menu", "@radix-ui/react-navigation-menu", "@radix-ui/react-popover", "@radix-ui/react-select", "@radix-ui/react-tabs", "@radix-ui/react-tooltip"]
    },
    images: {
        remotePatterns: [
            {
                protocol: "https",
                hostname: "images.unsplash.com"
            },
            {
                protocol: "https",
                hostname: "images.pexels.com"
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
                            "img-src 'self' blob: data: https:",
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
