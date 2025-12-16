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
                            "connect-src 'self' https://api.sayeedjoy.com",
                            "img-src 'self' data:",
                            "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
                            "style-src 'self' 'unsafe-inline'"
                        ].join("; ")
                    }
                ]
            }
        ]
    }
}

export default nextConfig
