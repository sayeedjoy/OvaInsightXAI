"use client"

import { useState } from "react"

import { Button } from "@/components/ui/button"
import {
    Card,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

const FEATURE_FIELDS = [
    { key: "age", label: "Age", placeholder: "Years" },
    { key: "alb", label: "ALB", placeholder: "g/dL" },
    { key: "alp", label: "ALP", placeholder: "U/L" },
    { key: "bun", label: "BUN", placeholder: "mg/dL" },
    { key: "ca125", label: "CA125", placeholder: "U/mL" },
    { key: "eo_abs", label: "EO#", placeholder: "10^9/L" },
    { key: "ggt", label: "GGT", placeholder: "U/L" },
    { key: "he4", label: "HE4", placeholder: "pmol/L" },
    { key: "mch", label: "MCH", placeholder: "pg" },
    { key: "mono_abs", label: "MONO#", placeholder: "10^9/L" },
    { key: "na", label: "Na", placeholder: "mmol/L" },
    { key: "pdw", label: "PDW", placeholder: "%" }
] as const

type FeatureKey = (typeof FEATURE_FIELDS)[number]["key"]

type PredictionResult = {
    prediction: number | string
    confidence?: number | null
}

const emptyFormState: Record<FeatureKey, string> = FEATURE_FIELDS.reduce(
    (acc, field) => {
        acc[field.key] = ""
        return acc
    },
    {} as Record<FeatureKey, string>
)

export default function PredictPage() {
    const [formValues, setFormValues] = useState<Record<FeatureKey, string>>(
        emptyFormState
    )
    const [isSubmitting, setIsSubmitting] = useState(false)
    const [result, setResult] = useState<PredictionResult | null>(null)
    const [error, setError] = useState<string | null>(null)

    const resetForm = () => {
        setFormValues(emptyFormState)
        setResult(null)
        setError(null)
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

    const confidenceDisplay =
        result?.confidence !== undefined && result?.confidence !== null
            ? `${(result.confidence * 100).toFixed(1)}%`
            : null

    return (
        <div className="container mx-auto max-w-5xl py-10">
            <div className="mb-8 space-y-2 text-center">
                <h1 className="text-3xl font-semibold">Prediction Form</h1>
                <p className="text-muted-foreground">
                    Provide the 12 biomarkers below to run inference against the FastAPI
                    backend.
                </p>
            </div>
            <div className="grid gap-8 md:grid-cols-[2fr_1fr]">
                <Card>
                    <CardHeader>
                        <CardTitle>Patient Metrics</CardTitle>
                        <CardDescription>
                            All values are required. Use numeric values only.
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <form
                            className="grid gap-4 md:grid-cols-2"
                            onSubmit={handleSubmit}
                        >
                            {FEATURE_FIELDS.map((field) => (
                                <div className="flex flex-col space-y-2" key={field.key}>
                                    <Label htmlFor={field.key}>{field.label}</Label>
                                    <Input
                                        id={field.key}
                                        type="number"
                                        inputMode="decimal"
                                        placeholder={field.placeholder}
                                        value={formValues[field.key]}
                                        onChange={(event) =>
                                            setFormValues((prev) => ({
                                                ...prev,
                                                [field.key]: event.target.value
                                            }))
                                        }
                                        step="any"
                                        required
                                    />
                                </div>
                            ))}
                            <div className="md:col-span-2 flex gap-3 pt-2">
                                <Button type="submit" disabled={isSubmitting}>
                                    {isSubmitting ? "Predicting..." : "Run Prediction"}
                                </Button>
                                <Button
                                    type="button"
                                    variant="outline"
                                    onClick={resetForm}
                                    disabled={isSubmitting}
                                >
                                    Reset
                                </Button>
                            </div>
                        </form>
                    </CardContent>
                </Card>

                <Card className="h-fit">
                    <CardHeader>
                        <CardTitle>Result</CardTitle>
                        <CardDescription>
                            {"Shows the latest prediction and confidence score."}
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        {error && (
                            <p className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                                {error}
                            </p>
                        )}
                        {result ? (
                            <div className="space-y-2">
                                <p className="text-sm text-muted-foreground">Prediction</p>
                                <p className="text-3xl font-semibold">{result.prediction}</p>
                                {confidenceDisplay && (
                                    <>
                                        <p className="text-sm text-muted-foreground">
                                            Confidence
                                        </p>
                                        <p className="text-xl font-medium">
                                            {confidenceDisplay}
                                        </p>
                                    </>
                                )}
                            </div>
                        ) : (
                            <p className="text-muted-foreground">
                                Submit the form to see predictions here.
                            </p>
                        )}
                    </CardContent>
                </Card>
            </div>
        </div>
    )
}


