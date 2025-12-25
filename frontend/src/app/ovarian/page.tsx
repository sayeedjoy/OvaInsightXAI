"use client"

import { useState } from "react"
import { BarChart3 } from "lucide-react"

import {
    PredictionForm,
    FEATURE_FIELDS,
    type FeatureKey,
    type FormValues,
    PredictionResultCard,
    type PredictionResult,
    PredictionLoading
} from "@/components/prediction-components"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { XAIContainer } from "@/components/xai"
import {
    generateNegativeTestCase,
    generatePositiveTestCase
} from "@/lib/test-case-generator"

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

    const handleFillTestCase = (type: "negative" | "positive") => {
        if (isSubmitting) return

        setError(null)
        setResult(null)

        // Generate test case using frontend generator
        const testCase =
            type === "negative"
                ? generateNegativeTestCase()
                : generatePositiveTestCase()

        setFormValues(testCase)
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
            // Minimum delay of 2 seconds for better UX
            const minDelayPromise = new Promise((resolve) => {
                setTimeout(resolve, 2000)
            })

            // API call promise
            const apiPromise = fetch("/api/predict", {
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

    return (
        <>
            <div className="container mx-auto max-w-7xl px-4 py-6 sm:px-6 sm:py-8 md:px-8 md:py-10 lg:px-10 xl:px-12">
                <div className="mb-6 space-y-2 text-center sm:mb-8 lg:mb-12">
                    <h1 className="text-2xl font-semibold sm:text-3xl md:text-4xl lg:text-5xl">
                        Ovarian Cancer Prediction
                    </h1>
                    <p className="text-sm text-muted-foreground sm:text-base lg:text-lg max-w-2xl mx-auto">
                        Provide the 12 biomarkers below to run inference against the FastAPI
                        backend.
                    </p>
                </div>
                <div className="relative grid gap-6 lg:grid-cols-[1.4fr_1fr] lg:gap-10 xl:gap-12">
                    <div className="relative">
                        <PredictionLoading isVisible={isSubmitting} />
                        <PredictionForm
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

                {/* Model Explanations Section */}
                {result?.xai && (
                    <div className="mt-6 lg:mt-10">
                        <Card>
                            <CardHeader>
                                <div className="flex items-center gap-2">
                                    <BarChart3 className="h-5 w-5 text-muted-foreground" />
                                    <CardTitle className="text-lg font-semibold sm:text-xl">
                                        Model Explanations
                                    </CardTitle>
                                </div>
                            </CardHeader>
                            <CardContent>
                                <XAIContainer xaiData={result.xai} />
                            </CardContent>
                        </Card>
                    </div>
                )}
            </div>
        </>
    )
}


