"use client"

import { useState } from "react"
import { Info, AlertCircle } from "lucide-react"
import type { ImageSHAPExplanation } from "@/types/xai"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

interface ImageSHAPVisualizationProps {
    explanation: ImageSHAPExplanation
    originalImageUrl?: string | null  // Optional: original image URL for overlay
}

export function ImageSHAPVisualization({
    explanation,
    originalImageUrl
}: ImageSHAPVisualizationProps) {
    const [overlayOpacity, setOverlayOpacity] = useState(0.5)
    const [showOverlay, setShowOverlay] = useState(true)

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

    if (!explanation.heatmap_image && !explanation.heatmap_data) {
        return (
            <div className="rounded-lg border border-muted p-4">
                <p className="text-sm text-muted-foreground">No SHAP heatmap available</p>
            </div>
        )
    }

    return (
        <div className="space-y-4">
            {/* Controls */}
            {originalImageUrl && (
                <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                    <div className="flex items-center gap-2">
                        <input
                            type="checkbox"
                            id="show-overlay"
                            checked={showOverlay}
                            onChange={(e) => setShowOverlay(e.target.checked)}
                            className="h-4 w-4 rounded border-gray-300"
                        />
                        <label htmlFor="show-overlay" className="text-sm font-medium">
                            Overlay on original image
                        </label>
                    </div>
                    {showOverlay && (
                        <div className="flex items-center gap-2">
                            <label htmlFor="opacity" className="text-sm font-medium">
                                Opacity:
                            </label>
                            <input
                                type="range"
                                id="opacity"
                                min="0"
                                max="1"
                                step="0.1"
                                value={overlayOpacity}
                                onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
                                className="w-32"
                            />
                            <span className="text-sm text-muted-foreground">
                                {Math.round(overlayOpacity * 100)}%
                            </span>
                        </div>
                    )}
                </div>
            )}

            {/* Heatmap Display */}
            <div className="relative w-full rounded-lg border overflow-hidden bg-muted/50">
                {showOverlay && originalImageUrl ? (
                    <div className="relative w-full">
                        {/* Original image */}
                        <img
                            src={originalImageUrl}
                            alt="Original MRI"
                            className="w-full h-auto"
                        />
                        {/* Heatmap overlay */}
                        {explanation.heatmap_image && (
                            <img
                                src={explanation.heatmap_image}
                                alt="SHAP Heatmap"
                                className="absolute inset-0 w-full h-full object-cover mix-blend-screen"
                                style={{ opacity: overlayOpacity }}
                            />
                        )}
                    </div>
                ) : (
                    explanation.heatmap_image && (
                        <img
                            src={explanation.heatmap_image}
                            alt="SHAP Heatmap"
                            className="w-full h-auto"
                        />
                    )
                )}
            </div>

            {/* Information */}
            <Card>
                <CardHeader className="pb-3">
                    <div className="flex items-center gap-2">
                        <Info className="h-4 w-4 text-muted-foreground" />
                        <CardTitle className="text-sm font-semibold">About SHAP Heatmap</CardTitle>
                    </div>
                </CardHeader>
                <CardContent>
                    <CardDescription className="text-xs leading-relaxed">
                        The heatmap shows which regions of the MRI image contributed most to the prediction.
                        Red/yellow areas indicate high importance (strong influence on the prediction),
                        while blue areas indicate low importance. This helps identify which parts of the
                        brain scan the model focuses on when making its classification.
                    </CardDescription>
                </CardContent>
            </Card>

        </div>
    )
}

