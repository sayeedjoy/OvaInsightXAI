"use client"

import { useState, useMemo } from "react"
import { AlertCircle, Info } from "lucide-react"
import type { ImageALEExplanation } from "@/types/xai"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    ReferenceLine
} from "recharts"

interface ImageALEVisualizationProps {
    explanation: ImageALEExplanation
    originalImageUrl?: string | null
}

export function ImageALEVisualization({
    explanation,
    originalImageUrl
}: ImageALEVisualizationProps) {
    const [selectedPatch, setSelectedPatch] = useState<number | null>(null)

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

    if (!explanation.ale_plots || explanation.ale_plots.length === 0) {
        return (
            <div className="rounded-lg border border-muted p-4">
                <p className="text-sm text-muted-foreground">No ALE plots available</p>
            </div>
        )
    }

    const plots = explanation.ale_plots
    const displayPatch = selectedPatch !== null ? plots[selectedPatch] : plots[0]

    // Calculate min/max for scale
    const aleValues = displayPatch?.ale_values || []
    const minAle = aleValues.length > 0 ? Math.min(...aleValues) : 0
    const maxAle = aleValues.length > 0 ? Math.max(...aleValues) : 0

    // Transform data for recharts
    const chartData = useMemo(() => {
        if (!displayPatch) return []
        return displayPatch.bin_centers.map((binCenter, i) => ({
            intensity: binCenter.toFixed(2),
            ale: displayPatch.ale_values[i],
        }))
    }, [displayPatch])

    return (
        <div className="space-y-4">
            <div className="flex items-start gap-2 rounded-lg border border-blue-200 bg-blue-50/50 p-3 dark:border-blue-900 dark:bg-blue-950/30">
                <Info className="mt-0.5 h-4 w-4 shrink-0 text-blue-600 dark:text-blue-400" />
                <div className="flex-1 space-y-1 text-xs text-blue-900 dark:text-blue-100">
                    <p className="font-medium">Patch-based Accumulated Local Effects</p>
                    <p>
                        This visualization shows accumulated local effects for each patch.
                        ALE measures how the prediction changes as patch intensity varies, accumulated across bins.
                    </p>
                </div>
            </div>

            {/* Patch selector */}
            {plots.length > 1 && (
                <div className="space-y-2">
                    <label className="text-sm font-medium">Select Patch:</label>
                    <div className="flex flex-wrap gap-2">
                        {plots.map((plot, idx) => (
                            <button
                                key={plot.patch_index}
                                onClick={() => setSelectedPatch(idx)}
                                className={`rounded-md border px-3 py-1 text-xs transition-colors ${selectedPatch === idx || (selectedPatch === null && idx === 0)
                                        ? "border-primary bg-primary/10 text-primary"
                                        : "border-muted bg-background hover:bg-muted"
                                    }`}
                            >
                                Patch {plot.patch_index} ({plot.patch_row}, {plot.patch_col})
                            </button>
                        ))}
                    </div>
                </div>
            )}

            {/* ALE Plot with Recharts */}
            {displayPatch && (
                <Card>
                    <CardHeader>
                        <CardTitle className="text-base">
                            Patch {displayPatch.patch_index} - Row {displayPatch.patch_row}, Col {displayPatch.patch_col}
                        </CardTitle>
                        <CardDescription>
                            Coordinates: ({displayPatch.patch_coords.x_start}, {displayPatch.patch_coords.y_start}) to ({displayPatch.patch_coords.x_end}, {displayPatch.patch_coords.y_end})
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <div className="space-y-4">
                            {/* Recharts Line Chart */}
                            <div className="h-64 w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                    <LineChart
                                        data={chartData}
                                        margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                                    >
                                        <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                                        <XAxis
                                            dataKey="intensity"
                                            label={{ value: 'Intensity', position: 'insideBottom', offset: -5 }}
                                            tick={{ fontSize: 11 }}
                                        />
                                        <YAxis
                                            domain={[minAle - 0.01, maxAle + 0.01]}
                                            label={{ value: 'ALE', angle: -90, position: 'insideLeft' }}
                                            tick={{ fontSize: 11 }}
                                            tickFormatter={(value) => value.toFixed(3)}
                                        />
                                        <Tooltip
                                            contentStyle={{
                                                backgroundColor: 'hsl(var(--card))',
                                                border: '1px solid hsl(var(--border))',
                                                borderRadius: '8px'
                                            }}
                                            formatter={(value: number) => [value.toFixed(5), 'ALE Value']}
                                        />
                                        <ReferenceLine y={0} stroke="hsl(var(--muted-foreground))" strokeDasharray="5 5" />
                                        <Line
                                            type="monotone"
                                            dataKey="ale"
                                            stroke="hsl(var(--primary))"
                                            strokeWidth={2}
                                            dot={{ fill: 'hsl(var(--primary))', strokeWidth: 2, r: 3 }}
                                            activeDot={{ r: 5, fill: 'hsl(var(--primary))' }}
                                        />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>

                            {/* Stats */}
                            <div className="grid grid-cols-3 gap-4 rounded-lg border bg-muted/50 p-3">
                                <div>
                                    <p className="text-xs text-muted-foreground">Min ALE</p>
                                    <p className="text-sm font-semibold">
                                        {minAle.toFixed(5)}
                                    </p>
                                </div>
                                <div>
                                    <p className="text-xs text-muted-foreground">Max ALE</p>
                                    <p className="text-sm font-semibold">
                                        {maxAle.toFixed(5)}
                                    </p>
                                </div>
                                <div>
                                    <p className="text-xs text-muted-foreground">Bins</p>
                                    <p className="text-sm font-semibold">
                                        {displayPatch.bin_centers.length}
                                    </p>
                                </div>
                            </div>
                        </div>
                    </CardContent>
                </Card>
            )}
        </div>
    )
}
