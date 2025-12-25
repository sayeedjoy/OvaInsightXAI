"use client"

import { Loader2 } from "lucide-react"

export function XAILoading() {
    return (
        <div className="flex flex-col items-center justify-center space-y-4 py-8">
            <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            <p className="text-sm text-muted-foreground">Computing explanations...</p>
        </div>
    )
}

