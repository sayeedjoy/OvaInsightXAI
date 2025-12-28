"use client"

import { useState } from "react"
import { BrainTumorPredictionForm } from "@/components/brain-tumor-components"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
    PredictionResultCard,
    type PredictionResult,
    PredictionLoading
} from "@/components/prediction-components"

export default function BrainTumorPredictPage() {
    const [selectedFile, setSelectedFile] = useState<File | null>(null)
    const [previewUrl, setPreviewUrl] = useState<string | null>(null)
    const [isSubmitting, setIsSubmitting] = useState(false)
    const [result, setResult] = useState<PredictionResult | null>(null)
    const [error, setError] = useState<string | null>(null)

    const resetForm = () => {
        setSelectedFile(null)
        setPreviewUrl(null)
        setResult(null)
        setError(null)
    }

    const handleFileSelect = (file: File | null) => {
        setSelectedFile(file)
        setError(null)
        setResult(null)

        // Create preview URL
        if (file) {
            const url = URL.createObjectURL(file)
            setPreviewUrl(url)
        } else {
            if (previewUrl) {
                URL.revokeObjectURL(previewUrl)
            }
            setPreviewUrl(null)
        }
    }

    const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault()
        
        if (!selectedFile) {
            setError("Please select an image file")
            return
        }

        setIsSubmitting(true)
        setError(null)

        try {
            // Create FormData
            const formData = new FormData()
            formData.append("file", selectedFile)

            // Minimum delay of 2 seconds for better UX
            const minDelayPromise = new Promise((resolve) => {
                setTimeout(resolve, 2000)
            })

            // API call promise
            const apiPromise = fetch("/api/predict/brain-tumor", {
                method: "POST",
                body: formData
            }).then(async (response) => {
                const data = await response.json()

                if (!response.ok) {
                    throw new Error(data?.detail ?? "Prediction failed.")
                }

                return data
            })

            // Wait for both API call and minimum delay
            const [data] = await Promise.all([apiPromise, minDelayPromise])

            setResult({
                prediction: data.prediction,
                confidence: data.confidence ?? null,
                xai: data.xai ?? null
            })
        } catch (fetchError) {
            const message =
                fetchError instanceof Error
                    ? fetchError.message
                    : "Unable to get prediction."
            setError(message)
            setResult(null)
        } finally {
            setIsSubmitting(false)
        }
    }

    // Cleanup preview URL on unmount
    if (previewUrl && !selectedFile) {
        URL.revokeObjectURL(previewUrl)
        setPreviewUrl(null)
    }

    return (
        <>
            <div className="container mx-auto max-w-7xl px-4 py-6 sm:px-6 sm:py-8 md:px-8 md:py-10 lg:px-10 xl:px-12">
                <div className="mb-6 space-y-2 text-center sm:mb-8 lg:mb-12">
                    <h1 className="text-2xl font-semibold sm:text-3xl md:text-4xl lg:text-5xl">
                        Brain Tumor Classification
                    </h1>
                    <p className="text-sm text-muted-foreground sm:text-base lg:text-lg max-w-2xl mx-auto">
                        Upload a brain MRI image to classify it as glioma, meningioma, or tumor.
                    </p>
                </div>
                <div className="relative grid gap-6 lg:grid-cols-[1.4fr_1fr] lg:gap-10 xl:gap-12">
                    <div className="relative">
                        <PredictionLoading isVisible={isSubmitting} />
                        <BrainTumorPredictionForm
                            selectedFile={selectedFile}
                            previewUrl={previewUrl}
                            isSubmitting={isSubmitting}
                            onFileSelect={handleFileSelect}
                            onSubmit={handleSubmit}
                            onReset={resetForm}
                        />
                    </div>
                    <PredictionResultCard 
                        result={result} 
                        error={error}
                        conditionName="Brain Tumor"
                        displayClass={true}
                    />
                </div>
            </div>
        </>
    )
}

