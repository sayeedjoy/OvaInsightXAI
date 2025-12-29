"use client"

import { useState, useMemo } from "react"
import { AlertCircle, Info } from "lucide-react"
import type { ImageICEExplanation } from "@/types/xai"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Legend
} from "recharts"

interface ImageICEVisualizationProps {
    explanation: ImageICEExplanation
    originalImageUrl?: string | null
}

// Color palette for multiple curves - theme-aware colors
const CURVE_COLORS = [
    "#6366f1",  // Indigo
    "#8b5cf6",  // Purple
    "#10b981",  // Emerald
    "#f59e0b",  // Amber
    "#ec4899",  // Pink
]

export function ImageICEVisualization({
    explanation,
    originalImageUrl
}: ImageICEVisualizationProps) {
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

    if (!explanation.ice_plots || explanation.ice_plots.length === 0) {
        return (
            <div className="rounded-lg border border-muted p-4">
                <p className="text-sm text-muted-foreground">No ICE plots available</p>
            </div>
        )
    }

    const plots = explanation.ice_plots
    const displayPatch = selectedPatch !== null ? plots[selectedPatch] : plots[0]

    // Transform data for recharts (merge all curves into single data array)
    const chartData = useMemo(() => {
        if (!displayPatch) return []
        return displayPatch.intensity_values.map((intensity, i) => {
            const dataPoint: Record<string, number | string> = {
                intensity: intensity.toFixed(2),
            }
            displayPatch.curves.forEach((curve, curveIdx) => {
                dataPoint[`curve_${curveIdx}`] = curve.predictions[i]
            })
            return dataPoint
        })
    }, [displayPatch])

    return (
        <div className="space-y-4">
            <div className="flex items-start gap-2 rounded-lg border border-blue-200 bg-blue-50/50 p-3 dark:border-blue-900 dark:bg-blue-950/30">
                <Info className="mt-0.5 h-4 w-4 shrink-0 text-blue-600 dark:text-blue-400" />
                <div className="flex-1 space-y-1 text-xs text-blue-900 dark:text-blue-100">
                    <p className="font-medium">Patch-based Individual Conditional Expectation</p>
                    <p>
                        This visualization shows individual prediction curves for each patch variation.
                        Each curve represents how the prediction changes when varying patch intensity.
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

            {/* ICE Plot with Recharts */}
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
                            {/* Recharts Multi-Line Chart */}
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
                                            domain={[0, 1]}
                                            label={{ value: 'Prediction', angle: -90, position: 'insideLeft' }}
                                            tick={{ fontSize: 11 }}
                                        />
                                        <Tooltip
                                            contentStyle={{
                                                backgroundColor: 'var(--color-card, hsl(var(--card)))',
                                                border: '1px solid var(--color-border, hsl(var(--border)))',
                                                borderRadius: '8px',
                                                color: 'var(--color-foreground, hsl(var(--foreground)))'
                                            }}
                                            labelStyle={{ color: 'var(--color-foreground, hsl(var(--foreground)))' }}
                                            formatter={(value: number, name: string) => [
                                                value.toFixed(4),
                                                `Sample ${name.replace('curve_', '')}`
                                            ]}
                                        />
                                        {displayPatch.curves.map((curve, curveIdx) => (
                                            <Line
                                                key={curve.sample_index}
                                                type="monotone"
                                                dataKey={`curve_${curveIdx}`}
                                                stroke={CURVE_COLORS[curveIdx % CURVE_COLORS.length]}
                                                strokeWidth={1.5}
                                                dot={false}
                                                opacity={0.7}
                                            />
                                        ))}
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>

                            {/* Stats */}
                            <div className="grid grid-cols-2 gap-4 rounded-lg border bg-muted/50 p-3">
                                <div>
                                    <p className="text-xs text-muted-foreground">Number of Curves</p>
                                    <p className="text-sm font-semibold">
                                        {displayPatch.curves.length}
                                    </p>
                                </div>
                                <div>
                                    <p className="text-xs text-muted-foreground">Grid Points</p>
                                    <p className="text-sm font-semibold">
                                        {displayPatch.intensity_values.length}
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
