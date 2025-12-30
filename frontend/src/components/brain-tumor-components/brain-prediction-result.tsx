"use client"

import {
    AlertCircle,
    AlertTriangle,
    BarChart3,
    Gauge,
    Info
} from "lucide-react"

import {
    Card,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle
} from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { XAIResponse } from "@/types/xai"

export type PredictionResult = {
    prediction: number | string
    confidence?: number | null
    xai?: XAIResponse | null
}

interface BrainPredictionResultCardProps {
    result: PredictionResult | null
    error: string | null
}

export function BrainPredictionResultCard({
    result,
    error
}: BrainPredictionResultCardProps) {
    const getPredictionDisplay = (prediction: number | string) => {
        // Handle string class names (e.g., "glioma", "meningioma", "tumor")
        if (typeof prediction === "string") {
            const classLabel = prediction.charAt(0).toUpperCase() + prediction.slice(1)
            return {
                label: `Classified as: ${classLabel} (Tumor Detected)`,
                description: `The AI model detected brain MRI patterns consistent with a ${prediction}.`,
                variant: "default" as const,
                icon: Info,
                iconColor: "text-blue-600 dark:text-blue-500",
                bgColor: "bg-blue-50 dark:bg-blue-950/20",
                borderColor: "border-blue-200 dark:border-blue-800"
            }
        }
        
        // Invalid prediction
        return {
            label: "Invalid Result",
            description: "The prediction value could not be interpreted.",
            variant: "outline" as const,
            icon: AlertTriangle,
            iconColor: "text-yellow-600 dark:text-yellow-500",
            bgColor: "bg-yellow-50 dark:bg-yellow-950/20",
            borderColor: "border-yellow-200 dark:border-yellow-800"
        }
    }

    const confidenceDisplay =
        result?.confidence !== undefined && result?.confidence !== null
            ? `${(result.confidence * 100).toFixed(1)}%`
            : null

    const getConfidenceColor = (confidence: number) => {
        if (confidence >= 0.9) return "text-green-600 dark:text-green-400"
        if (confidence >= 0.75) return "text-blue-600 dark:text-blue-400"
        if (confidence >= 0.6) return "text-yellow-600 dark:text-yellow-400"
        return "text-orange-600 dark:text-orange-400"
    }

    const getProgressBarColor = (confidence: number) => {
        if (confidence >= 0.9) return "bg-green-500 dark:bg-green-600"
        if (confidence >= 0.75) return "bg-blue-500 dark:bg-blue-600"
        if (confidence >= 0.6) return "bg-yellow-500 dark:bg-yellow-600"
        return "bg-orange-500 dark:bg-orange-600"
    }

    const predictionDisplay = result ? getPredictionDisplay(result.prediction) : null
    const IconComponent = predictionDisplay?.icon

    return (
        <Card className="h-fit w-full lg:sticky lg:top-6">
            <CardHeader className="pb-4 sm:pb-6 lg:pb-6">
                <CardTitle className="text-lg font-semibold sm:text-xl lg:text-xl">
                    Prediction Result
                </CardTitle>
                <CardDescription className="text-xs sm:text-sm lg:text-sm">
                    Analysis based on brain MRI image classification
                </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 sm:space-y-6">
                {error && (
                    <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-3 sm:p-4">
                        <div className="flex items-start gap-3">
                            <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-destructive sm:h-6 sm:w-6" />
                            <div className="flex-1 space-y-1 min-w-0">
                                <p className="text-sm font-semibold text-destructive sm:text-base">
                                    Error
                                </p>
                                <p className="text-xs leading-relaxed text-destructive/90 sm:text-sm">
                                    {error}
                                </p>
                            </div>
                        </div>
                    </div>
                )}

                {result && predictionDisplay && IconComponent ? (
                    <div className="space-y-4 sm:space-y-6">
                        {/* Prediction Status */}
                        <div className={`rounded-xl border-2 ${predictionDisplay.borderColor} ${predictionDisplay.bgColor} p-4 sm:p-5`}>
                            <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:gap-4">
                                <div className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-background/80 sm:h-14 sm:w-14`}>
                                    <IconComponent className={`h-6 w-6 sm:h-7 sm:w-7 ${predictionDisplay.iconColor}`} />
                                </div>
                                <div className="flex-1 space-y-3 min-w-0 lg:space-y-3">
                                    <div className="flex flex-wrap items-center gap-2 lg:gap-2">
                                        <h3 className="text-base font-semibold leading-tight sm:text-lg lg:text-lg">
                                            {predictionDisplay.label}
                                        </h3>
                                    </div>
                                    <p className="text-xs leading-relaxed text-muted-foreground sm:text-sm lg:text-sm lg:leading-relaxed">
                                        {predictionDisplay.description}
                                    </p>
                                </div>
                            </div>
                        </div>

                        <Separator className="hidden sm:block" />

                        {/* Confidence Score */}
                        {confidenceDisplay && result?.confidence !== undefined && result?.confidence !== null && (
                            <div className="space-y-3 sm:space-y-4">
                                <div className="flex items-center gap-2">
                                    <Gauge className="h-4 w-4 text-muted-foreground sm:h-5 sm:w-5" />
                                    <p className="text-xs font-medium text-muted-foreground sm:text-sm">
                                        Model Confidence
                                    </p>
                                </div>
                                <div className="space-y-3">
                                    <div className="flex items-baseline gap-2">
                                        <span className={`text-2xl font-bold sm:text-3xl lg:text-3xl ${getConfidenceColor(result.confidence)}`}>
                                            {confidenceDisplay}
                                        </span>
                                    </div>
                                    <div className="space-y-2">
                                        <div className="h-2.5 w-full overflow-hidden rounded-full bg-muted sm:h-3">
                                            <div
                                                className={`h-full transition-all duration-700 ease-out ${getProgressBarColor(result.confidence)}`}
                                                style={{
                                                    width: `${(result.confidence * 100).toFixed(1)}%`
                                                }}
                                            />
                                        </div>
                                        <p className="text-xs text-muted-foreground sm:text-sm">
                                            The model's confidence in this prediction result
                                        </p>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Disclaimer */}
                        <div className="rounded-lg border border-border bg-muted/50 p-3 sm:p-4">
                            <div className="flex items-start gap-2 sm:gap-3">
                                <Info className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground sm:h-5 sm:w-5" />
                                <p className="text-xs leading-relaxed text-muted-foreground sm:text-sm lg:leading-relaxed">
                                    This output is not a medical diagnosis.
                                </p>
                            </div>
                        </div>
                    </div>
                ) : (
                    <div className="flex flex-col items-center justify-center space-y-4 py-8 sm:py-12 text-center">
                        <div className="flex h-16 w-16 items-center justify-center rounded-full bg-muted sm:h-20 sm:w-20">
                            <BarChart3 className="h-8 w-8 text-muted-foreground sm:h-10 sm:w-10" />
                        </div>
                        <div className="space-y-1.5 px-4">
                            <p className="text-sm font-semibold text-foreground sm:text-base">
                                No prediction yet
                            </p>
                            <p className="text-xs text-muted-foreground sm:text-sm">
                                Submit the form to see prediction results here
                            </p>
                        </div>
                    </div>
                )}
            </CardContent>
        </Card>
    )
}

