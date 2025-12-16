"use client"

import { useState } from "react"

import {
    PredictionForm,
    FEATURE_FIELDS,
    type FeatureKey,
    type FormValues,
    PredictionResultCard,
    type PredictionResult
} from "@/components/prediction-components"

const emptyFormState: FormValues = FEATURE_FIELDS.reduce(
    (acc, field) => {
        acc[field.key] = ""
        return acc
    },
    {} as FormValues
)

export default function PredictPage() {
    const [formValues, setFormValues] = useState<FormValues>(emptyFormState)
    const [isSubmitting, setIsSubmitting] = useState(false)
    const [result, setResult] = useState<PredictionResult | null>(null)
    const [error, setError] = useState<string | null>(null)

    const resetForm = () => {
        setFormValues(emptyFormState)
        setResult(null)
        setError(null)
    }

    const handleInputChange = (key: FeatureKey, value: string) => {
        setFormValues((prev) => ({
            ...prev,
            [key]: value
        }))
    }

    const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault()
        setIsSubmitting(true)
        setError(null)

        const payload: Record<FeatureKey, number> = {} as Record<
            FeatureKey,
            number
        >

        for (const field of FEATURE_FIELDS) {
            const value = formValues[field.key].trim()
            const numericValue = Number.parseFloat(value)
            if (Number.isNaN(numericValue)) {
                setIsSubmitting(false)
                setResult(null)
                setError(
                    `Please provide a valid number for ${field.label.toUpperCase()}.`
                )
                return
            }
            payload[field.key] = numericValue
        }

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            })

            const data = await response.json()

            if (!response.ok) {
                throw new Error(data?.detail ?? "Prediction failed.")
            }

            setResult({
                prediction: data.prediction,
                confidence: data.confidence ?? null
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

    return (
        <div className="container mx-auto max-w-7xl px-4 py-6 sm:px-6 sm:py-8 md:px-8 md:py-10 lg:px-10">
            <div className="mb-6 space-y-2 text-center sm:mb-8">
                <h1 className="text-2xl font-semibold sm:text-3xl md:text-4xl">
                    Ovarian Cancer Prediction
                </h1>
                <p className="text-sm text-muted-foreground sm:text-base">
                    Provide the 12 biomarkers below to run inference against the FastAPI
                    backend.
                </p>
            </div>
            <div className="grid gap-6 lg:grid-cols-[2fr_1fr] lg:gap-8">
                <PredictionForm
                    formValues={formValues}
                    isSubmitting={isSubmitting}
                    onInputChange={handleInputChange}
                    onSubmit={handleSubmit}
                    onReset={resetForm}
                />
                <PredictionResultCard result={result} error={error} />
            </div>
        </div>
    )
}


