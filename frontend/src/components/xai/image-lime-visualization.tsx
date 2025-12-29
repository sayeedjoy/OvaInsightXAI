"use client"

import { useState } from "react"
import { Info, AlertCircle } from "lucide-react"
import type { ImageLIMEExplanation } from "@/types/xai"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

interface ImageLIMEVisualizationProps {
    explanation: ImageLIMEExplanation
    originalImageUrl?: string | null  // Optional: original image URL
}

export function ImageLIMEVisualization({ 
    explanation, 
    originalImageUrl 
}: ImageLIMEVisualizationProps) {
    const [selectedFeature, setSelectedFeature] = useState<number | null>(null)

    if (explanation.error) {
        return (
            <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-4">
                <div className="flex items-start gap-3">
                    <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-destructive" />
                    <div className="flex-1 space-y-1 min-w-0">
                        <p className="text-sm font-semibold text-destructive">Error</p>
                        <p className="text-xs leading-relaxed text-destructive/90">
                            {explanation.error}
                        </p>
                    </div>
                </div>
            </div>
        )
    }

    if (!explanation.visualization_image && !explanation.feature_importance) {
        return (
            <div className="rounded-lg border border-muted p-4">
                <p className="text-sm text-muted-foreground">No LIME explanation available</p>
            </div>
        )
    }

    // Sort features by absolute importance
    const sortedFeatures = explanation.feature_importance
        ? [...explanation.feature_importance].sort(
              (a, b) => Math.abs(b.importance) - Math.abs(a.importance)
          )
        : []

    return (
        <div className="space-y-4">
            {/* Visualization Image */}
            {explanation.visualization_image && (
                <div className="relative w-full rounded-lg border overflow-hidden bg-muted/50">
                    <img
                        src={explanation.visualization_image}
                        alt="LIME Explanation"
                        className="w-full h-auto"
                    />
                </div>
            )}

            {/* Feature Importance List */}
            {sortedFeatures.length > 0 && (
                <Card>
                    <CardHeader className="pb-3">
                        <CardTitle className="text-sm font-semibold">Superpixel Importance</CardTitle>
                        <CardDescription className="text-xs">
                            Click on a superpixel to see its contribution to the prediction
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <div className="space-y-2 max-h-64 overflow-y-auto">
                            {sortedFeatures.map((feature) => {
                                const isPositive = feature.importance > 0
                                return (
                                    <div
                                        key={feature.feature_index}
                                        className={`flex items-center justify-between p-2 rounded-lg border cursor-pointer transition-colors ${
                                            selectedFeature === feature.feature_index
                                                ? "bg-primary/10 border-primary"
                                                : "hover:bg-muted"
                                        }`}
                                        onClick={() => setSelectedFeature(
                                            selectedFeature === feature.feature_index 
                                                ? null 
                                                : feature.feature_index
                                        )}
                                    >
                                        <div className="flex items-center gap-2">
                                            <span className="text-sm font-medium">
                                                Superpixel {feature.feature_index}
                                            </span>
                                            <Badge
                                                variant={isPositive ? "default" : "secondary"}
                                                className="text-xs"
                                            >
                                                {isPositive ? "Positive" : "Negative"}
                                            </Badge>
                                        </div>
                                        <span
                                            className={`text-sm font-semibold ${
                                                isPositive
                                                    ? "text-green-600 dark:text-green-400"
                                                    : "text-red-600 dark:text-red-400"
                                            }`}
                                        >
                                            {feature.importance > 0 ? "+" : ""}
                                            {feature.importance.toFixed(4)}
                                        </span>
                                    </div>
                                )
                            })}
                        </div>
                    </CardContent>
                </Card>
            )}

            {/* Information */}
            <Card>
                <CardHeader className="pb-3">
                    <div className="flex items-center gap-2">
                        <Info className="h-4 w-4 text-muted-foreground" />
                        <CardTitle className="text-sm font-semibold">About LIME Explanation</CardTitle>
                    </div>
                </CardHeader>
                <CardContent>
                    <CardDescription className="text-xs leading-relaxed">
                        LIME (Local Interpretable Model-agnostic Explanations) divides the image into 
                        superpixels (similar regions) and shows how each region affects the prediction. 
                        Positive values (green) indicate regions that increase the prediction confidence, 
                        while negative values (red) indicate regions that decrease it. This helps identify 
                        which specific areas of the brain scan are most relevant to the model's decision.
                    </CardDescription>
                </CardContent>
            </Card>

            {/* Prediction Probabilities */}
            {explanation.probabilities && explanation.probabilities.length > 0 && (
                <Card>
                    <CardHeader className="pb-3">
                        <CardTitle className="text-sm font-semibold">Class Probabilities</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="space-y-2">
                            {explanation.probabilities.map((prob, idx) => {
                                const classes = ["Glioma", "Meningioma", "Tumor"]
                                const className = classes[idx] || `Class ${idx}`
                                return (
                                    <div key={idx} className="flex items-center justify-between">
                                        <span className="text-sm text-muted-foreground">{className}</span>
                                        <div className="flex items-center gap-2">
                                            <div className="w-32 h-2 bg-muted rounded-full overflow-hidden">
                                                <div
                                                    className="h-full bg-primary transition-all"
                                                    style={{ width: `${prob * 100}%` }}
                                                />
                                            </div>
                                            <span className="text-sm font-medium w-12 text-right">
                                                {(prob * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                    </div>
                                )
                            })}
                        </div>
                    </CardContent>
                </Card>
            )}
        </div>
    )
}

