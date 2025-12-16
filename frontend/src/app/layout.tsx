import { Inter } from "next/font/google"
import { Providers } from "./providers"
import type { ReactNode } from "react"
import "@/styles/globals.css"

const inter = Inter({
    subsets: ["latin"],
    variable: "--font-inter",
    display: "swap"
})

export default function RootLayout({
    children
}: Readonly<{
    children: ReactNode
}>) {
    return (
        <html lang="en" suppressHydrationWarning className={inter.variable}>
            <head>
                <script
                    async
                    src="/seline.js"
                    data-token="24cc7b65ecf3469"
                />
            </head>
            <body className="flex min-h-svh flex-col antialiased">
                <Providers>{children}</Providers>
            </body>
        </html>
    )
}
