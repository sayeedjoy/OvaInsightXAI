"use client"

import {
    AlertCircle,
    AlertTriangle,
    BarChart3,
    CheckCircle2,
    Gauge,
    Info,
    XCircle
} from "lucide-react"

import { Badge } from "@/components/ui/badge"
import {
    Card,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle
} from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"

export type PredictionResult = {
    prediction: number | string
    confidence?: number | null
}

interface PredictionResultCardProps {
    result: PredictionResult | null
    error: string | null
}

export function PredictionResultCard({ result, error }: PredictionResultCardProps) {
    const getPredictionDisplay = (prediction: number | string) => {
        const predValue = typeof prediction === "string" 
            ? Number.parseFloat(prediction) 
            : prediction
        
        if (Number.isNaN(predValue)) {
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

        // Assuming 1 = cancer, 0 = no cancer
        if (predValue === 1 || predValue > 0.5) {
            return {
                label: "Possible Ovarian Cancer",
                description: "The model indicates a potential risk of ovarian cancer. Please consult with a healthcare professional for further evaluation and diagnostic testing.",
                variant: "destructive" as const,
                icon: XCircle,
                iconColor: "text-red-600 dark:text-red-500",
                bgColor: "bg-red-50 dark:bg-red-950/20",
                borderColor: "border-red-200 dark:border-red-800"
            }
        } else {
            return {
                label: "No Ovarian Cancer",
                description: "The model indicates no signs of ovarian cancer based on the provided biomarkers. However, regular check-ups and screenings are still recommended.",
                variant: "default" as const,
                icon: CheckCircle2,
                iconColor: "text-green-600 dark:text-green-500",
                bgColor: "bg-green-50 dark:bg-green-950/20",
                borderColor: "border-green-200 dark:border-green-800"
            }
        }
    }

    const confidenceDisplay =
        result?.confidence !== undefined && result?.confidence !== null
            ? `${(result.confidence * 100).toFixed(1)}%`
            : null

    const getConfidenceLevel = (confidence: number) => {
        if (confidence >= 0.9) return { label: "Very High", color: "text-green-600 dark:text-green-400" }
        if (confidence >= 0.75) return { label: "High", color: "text-blue-600 dark:text-blue-400" }
        if (confidence >= 0.6) return { label: "Moderate", color: "text-yellow-600 dark:text-yellow-400" }
        return { label: "Low", color: "text-orange-600 dark:text-orange-400" }
    }

    const confidenceInfo = result?.confidence !== undefined && result?.confidence !== null
        ? getConfidenceLevel(result.confidence)
        : null

    const predictionDisplay = result ? getPredictionDisplay(result.prediction) : null
    const IconComponent = predictionDisplay?.icon

    return (
        <Card className="h-fit w-full lg:sticky lg:top-6">
            <CardHeader className="pb-4 sm:pb-6">
                <CardTitle className="text-lg font-semibold sm:text-xl">
                    Prediction Result
                </CardTitle>
                <CardDescription className="text-xs sm:text-sm">
                    Analysis based on the provided biomarker values
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
                        <div className={`rounded-lg border ${predictionDisplay.borderColor} ${predictionDisplay.bgColor} p-4 sm:p-5`}>
                            <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:gap-4">
                                <div className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-background/80 sm:h-14 sm:w-14`}>
                                    <IconComponent className={`h-6 w-6 sm:h-7 sm:w-7 ${predictionDisplay.iconColor}`} />
                                </div>
                                <div className="flex-1 space-y-3 min-w-0">
                                    <div className="flex flex-wrap items-center gap-2">
                                        <h3 className="text-base font-semibold leading-tight sm:text-lg lg:text-xl">
                                            {predictionDisplay.label}
                                        </h3>
                                        <Badge 
                                            variant={predictionDisplay.variant}
                                            className="text-xs shrink-0"
                                        >
                                            {predictionDisplay.variant === "destructive" 
                                                ? "High Risk" 
                                                : "Low Risk"}
                                        </Badge>
                                    </div>
                                    <p className="text-xs leading-relaxed text-muted-foreground sm:text-sm">
                                        {predictionDisplay.description}
                                    </p>
                                </div>
                            </div>
                        </div>

                        <Separator className="hidden sm:block" />

                        {/* Confidence Score */}
                        {confidenceDisplay && confidenceInfo && (
                            <div className="space-y-3 sm:space-y-4">
                                <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                                    <div className="flex items-center gap-2">
                                        <Gauge className="h-4 w-4 text-muted-foreground sm:h-5 sm:w-5" />
                                        <p className="text-xs font-medium text-muted-foreground sm:text-sm">
                                            Model Confidence
                                        </p>
                                    </div>
                                    <Badge variant="outline" className="w-fit text-xs">
                                        {confidenceInfo.label}
                                    </Badge>
                                </div>
                                <div className="space-y-3">
                                    <div className="flex items-baseline gap-2">
                                        <span className={`text-2xl font-bold sm:text-3xl lg:text-4xl ${confidenceInfo.color}`}>
                                            {confidenceDisplay}
                                        </span>
                                    </div>
                                    <div className="space-y-2">
                                        <div className="h-2.5 w-full overflow-hidden rounded-full bg-muted sm:h-3">
                                            <div
                                                className={`h-full transition-all duration-700 ease-out ${
                                                    confidenceInfo.label === "Very High" 
                                                        ? "bg-green-500 dark:bg-green-600"
                                                        : confidenceInfo.label === "High"
                                                        ? "bg-blue-500 dark:bg-blue-600"
                                                        : confidenceInfo.label === "Moderate"
                                                        ? "bg-yellow-500 dark:bg-yellow-600"
                                                        : "bg-orange-500 dark:bg-orange-600"
                                                }`}
                                                style={{
                                                    width: `${(result.confidence! * 100).toFixed(1)}%`
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
                                <p className="text-xs leading-relaxed text-muted-foreground sm:text-sm">
                                    <strong className="font-semibold text-foreground">Disclaimer:</strong>{" "}
                                    This prediction is based on machine learning analysis and should not replace 
                                    professional medical diagnosis. Always consult with qualified healthcare 
                                    professionals for medical decisions.
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

