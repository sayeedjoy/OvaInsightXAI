"use client"

import { useState } from "react"

import {
    HepatitisPredictionForm,
    HEPATITIS_FEATURE_FIELDS,
    type HepatitisFeatureKey,
    type HepatitisFormValues
} from "@/components/hepatitis-components"
import {
    PredictionLoading,
    PredictionResultCard,
    type PredictionResult
} from "@/components/prediction-components"
import {
    generateHepatitisNegativeTestCase,
    generateHepatitisPositiveTestCase
} from "@/lib/hepatitis-test-case-generator"

const emptyFormState: HepatitisFormValues = HEPATITIS_FEATURE_FIELDS.reduce(
    (acc, field) => {
        acc[field.key] = ""
        return acc
    },
    {} as HepatitisFormValues
)

export default function HepatitisPredictPage() {
    const [formValues, setFormValues] = useState<HepatitisFormValues>(emptyFormState)
    const [isSubmitting, setIsSubmitting] = useState(false)
    const [result, setResult] = useState<PredictionResult | null>(null)
    const [error, setError] = useState<string | null>(null)

    const resetForm = () => {
        setFormValues(emptyFormState)
        setResult(null)
        setError(null)
    }

    const handleInputChange = (key: HepatitisFeatureKey, value: string) => {
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
                ? generateHepatitisNegativeTestCase()
                : generateHepatitisPositiveTestCase()

        setFormValues(testCase)
    }

    const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault()
        setIsSubmitting(true)
        setError(null)

        const payload: Record<HepatitisFeatureKey, number> = {} as Record<
            HepatitisFeatureKey,
            number
        >

        for (const field of HEPATITIS_FEATURE_FIELDS) {
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
                setTimeout(resolve, 2000)
            })

            const apiPromise = fetch("/api/predict/hepatitis_b", {
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

    return (
        <>
            <div className="container mx-auto max-w-7xl px-4 py-6 sm:px-6 sm:py-8 md:px-8 md:py-10 lg:px-10 xl:px-12">
                <div className="mb-6 space-y-2 text-center sm:mb-8 lg:mb-12">
                    <h1 className="text-2xl font-semibold sm:text-3xl md:text-4xl lg:text-5xl">
                        Hepatitis B Prediction
                    </h1>
                    <p className="text-sm text-muted-foreground sm:text-base lg:text-lg max-w-2xl mx-auto">
                        Provide the 15 clinical and laboratory features below to run inference against the FastAPI backend.
                    </p>
                </div>
                <div className="relative grid gap-6 lg:grid-cols-[1.4fr_1fr] lg:gap-10 xl:gap-12">
                    <div className="relative">
                        <PredictionLoading isVisible={isSubmitting} />
                        <HepatitisPredictionForm
                            formValues={formValues}
                            isSubmitting={isSubmitting}
                            onInputChange={handleInputChange}
                            onSubmit={handleSubmit}
                            onReset={resetForm}
                            onFillTestCase={handleFillTestCase}
                        />
                    </div>
                    <PredictionResultCard
                        result={result}
                        error={error}
                        conditionName="Hepatitis B"
                    />
                </div>
            </div>
        </>
    )
}

