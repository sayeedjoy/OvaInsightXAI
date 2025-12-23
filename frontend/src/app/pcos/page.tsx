"use client"

import { useState } from "react"

import {
    PcosPredictionForm,
    PCOS_FEATURE_FIELDS,
    type PcosFeatureKey,
    type PcosFormValues
} from "@/components/pcos-components"
import {
    PredictionLoading,
    PredictionResultCard,
    type PredictionResult
} from "@/components/prediction-components"
import {
    generatePcosNegativeTestCase,
    generatePcosPositiveTestCase
} from "@/lib/pcos-test-case-generator"

const emptyFormState: PcosFormValues = PCOS_FEATURE_FIELDS.reduce(
    (acc, field) => {
        acc[field.key] = ""
        return acc
    },
    {} as PcosFormValues
)

export default function PcosPredictPage() {
    const [formValues, setFormValues] = useState<PcosFormValues>(emptyFormState)
    const [isSubmitting, setIsSubmitting] = useState(false)
    const [result, setResult] = useState<PredictionResult | null>(null)
    const [error, setError] = useState<string | null>(null)

    const resetForm = () => {
        setFormValues(emptyFormState)
        setResult(null)
        setError(null)
    }

    const handleInputChange = (key: PcosFeatureKey, value: string) => {
        setFormValues((prev) => ({
            ...prev,
            [key]: value
        }))
    }

    const handleFillTestCase = (type: "negative" | "positive") => {
        if (isSubmitting) return

        setError(null)
        setResult(null)

        const testCase =
            type === "negative"
                ? generatePcosNegativeTestCase()
                : generatePcosPositiveTestCase()

        setFormValues(testCase)
    }

    const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault()
        setIsSubmitting(true)
        setError(null)

        const payload: Record<PcosFeatureKey, number> = {} as Record<
            PcosFeatureKey,
            number
        >

        for (const field of PCOS_FEATURE_FIELDS) {
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
            const minDelayPromise = new Promise((resolve) => {
                setTimeout(resolve, 5500)
            })

            const apiPromise = fetch("/api/predict/pcos", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            }).then(async (response) => {
                const data = await response.json()

                if (!response.ok) {
                    throw new Error(data?.detail ?? "Prediction failed.")
                }

                return data
            })

            const [data] = await Promise.all([apiPromise, minDelayPromise])

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
        <>
            <div className="container mx-auto max-w-7xl px-4 py-6 sm:px-6 sm:py-8 md:px-8 md:py-10 lg:px-10 xl:px-12">
                <div className="mb-6 space-y-2 text-center sm:mb-8 lg:mb-12">
                    <h1 className="text-2xl font-semibold sm:text-3xl md:text-4xl lg:text-5xl">
                        PCOS Prediction
                    </h1>
                    <p className="text-sm text-muted-foreground sm:text-base lg:text-lg max-w-2xl mx-auto">
                        Provide the 20 discretized PCOS features below to run inference against the FastAPI backend.
                    </p>
                </div>
                <div className="relative grid gap-6 lg:grid-cols-[1.4fr_1fr] lg:gap-10 xl:gap-12">
                    <div className="relative">
                        <PredictionLoading isVisible={isSubmitting} />
                        <PcosPredictionForm
                            formValues={formValues}
                            isSubmitting={isSubmitting}
                            onInputChange={handleInputChange}
                            onSubmit={handleSubmit}
                            onReset={resetForm}
                            onFillTestCase={handleFillTestCase}
                        />
                    </div>
                    <PredictionResultCard result={result} error={error} />
                </div>
            </div>
        </>
    )
}

