"use client"

import { useEffect, useState } from "react"
import { Loader2 } from "lucide-react"

const ML_TERMS = [
    "Analyzing biomarkers...",
    "Running ML model...",
    "Processing neural network...",
    "Calculating predictions...",
    "Evaluating risk factors...",
    "Computing confidence scores...",
    "Training data analysis...",
    "Feature extraction in progress...",
    "Model inference running...",
    "Pattern recognition active..."
]

interface PredictionLoadingProps {
    isVisible: boolean
}

export function PredictionLoading({ isVisible }: PredictionLoadingProps) {
    const [currentTermIndex, setCurrentTermIndex] = useState(0)

    useEffect(() => {
        if (!isVisible) return

        const interval = setInterval(() => {
            setCurrentTermIndex((prev) => (prev + 1) % ML_TERMS.length)
        }, 1500) // Change term every 1.5 seconds

        return () => clearInterval(interval)
    }, [isVisible])

    if (!isVisible) return null

    return (
        <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-background/80 backdrop-blur-md">
            {/* Spinner */}
            <div className="flex items-center justify-center">
                <Loader2 className="h-16 w-16 animate-spin text-primary sm:h-20 sm:w-20" />
            </div>

            {/* Rotating ML/AI Terms */}
            <div className="absolute bottom-12 left-0 right-0 flex justify-center px-4 sm:bottom-16">
                <div className="min-h-[2rem] text-center">
                    <p className="text-sm font-medium text-muted-foreground transition-opacity duration-500 sm:text-base">
                        {ML_TERMS[currentTermIndex]}
                    </p>
                </div>
            </div>
        </div>
    )
}

