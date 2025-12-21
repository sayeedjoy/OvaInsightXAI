import { Inter } from "next/font/google"
import { Providers } from "./providers"
import type { ReactNode } from "react"
import type { Metadata } from "next"
import "@/styles/globals.css"
import { site } from "@/config/site"

const inter = Inter({
    subsets: ["latin"],
    variable: "--font-inter",
    display: "swap"
})

export const metadata: Metadata = {
    metadataBase: new URL(site.url),
    title: {
        default: site.name,
        template: `%s | ${site.name}`
    },
    description: site.description,
    keywords: ["SaaS", "Next.js", "Drizzle", "Better Auth", "modern web app"],
    authors: [{ name: site.name }],
    creator: site.name,
    openGraph: {
        type: "website",
        locale: "en_US",
        url: site.url,
        siteName: site.name,
        title: site.name,
        description: site.description,
        images: [
            {
                url: site.ogImage,
                width: 1200,
                height: 630,
                alt: site.name
            }
        ]
    },
    twitter: {
        card: "summary_large_image",
        title: site.name,
        description: site.description,
        images: [site.ogImage],
        creator: site.links.twitter
    },
    robots: {
        index: true,
        follow: true,
        googleBot: {
            index: true,
            follow: true,
            "max-video-preview": -1,
            "max-image-preview": "large",
            "max-snippet": -1
        }
    },
    icons: {
        icon: "/logo.png",
        shortcut: "/logo.png",
        apple: "/logo.png"
    }
}

export default function RootLayout({
    children
}: Readonly<{
    children: ReactNode
}>) {
    return (
        <html lang="en" suppressHydrationWarning className={inter.variable}>
            <body className="flex min-h-svh flex-col antialiased">
                <Providers>{children}</Providers>
            </body>
        </html>
    )
}
