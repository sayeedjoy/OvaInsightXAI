"use client"

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts"
import type { LIMEExplanation } from "@/types/xai"

interface LIMEVisualizationProps {
    explanation: LIMEExplanation
}

export function LIMEVisualization({ explanation }: LIMEVisualizationProps) {
    if (explanation.error) {
        return (
            <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-4">
                <p className="text-sm text-destructive">Error: {explanation.error}</p>
            </div>
        )
    }

    if (!explanation.feature_importance || explanation.feature_importance.length === 0) {
        return (
            <div className="rounded-lg border border-muted p-4">
                <p className="text-sm text-muted-foreground">No LIME explanation available</p>
            </div>
        )
    }

    // Sort by absolute importance
    const sortedFeatures = [...explanation.feature_importance].sort(
        (a, b) => Math.abs(b.importance) - Math.abs(a.importance)
    )

    const data = sortedFeatures.map((feat) => ({
        feature: feat.feature,
        importance: feat.importance,
    }))

    // Color bars based on positive/negative importance
    const getColor = (value: number) => {
        return value >= 0 ? "hsl(var(--chart-1))" : "hsl(var(--chart-2))"
    }

    return (
        <div className="space-y-4">
            <ResponsiveContainer width="100%" height={400}>
                <BarChart data={data} layout="vertical" margin={{ top: 5, right: 30, left: 100, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" label={{ value: "Importance", position: "insideBottom", offset: -5 }} />
                    <YAxis
                        type="category"
                        dataKey="feature"
                        width={90}
                        tick={{ fontSize: 12 }}
                        angle={-45}
                        textAnchor="end"
                    />
                    <Tooltip
                        content={({ active, payload }) => {
                            if (active && payload && payload.length > 0) {
                                const data = payload[0].payload as typeof data[number]
                                return (
                                    <div className="rounded-lg border bg-background p-3 shadow-md">
                                        <p className="font-medium">{data.feature}</p>
                                        <p className="text-sm">
                                            Importance: <span className="font-mono">{data.importance.toFixed(4)}</span>
                                        </p>
                                    </div>
                                )
                            }
                            return null
                        }}
                    />
                    <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                        {data.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={getColor(entry.importance)} />
                        ))}
                    </Bar>
                </BarChart>
            </ResponsiveContainer>
            <div className="text-xs text-muted-foreground">
                <p>
                    LIME shows how each feature locally affects the prediction. Positive values (red) increase the
                    prediction, negative values (blue) decrease it.
                </p>
            </div>
        </div>
    )
}

