"use client"

import { ThemeProvider } from "next-themes"
import type { ReactNode } from "react"
import NextTopLoader from 'nextjs-toploader';
import { Toaster } from "sonner"

export function Providers({ children }: { children: ReactNode }) {
    return (
        <ThemeProvider
            attribute="class"
            defaultTheme="dark"
            disableTransitionOnChange
        >
            <NextTopLoader color="var(--primary)" showSpinner={false} />
            {children}
            <Toaster />
        </ThemeProvider>
    )
}
