"use client"

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts"
import type { SHAPExplanation } from "@/types/xai"

interface SHAPVisualizationProps {
    explanation: SHAPExplanation
}

export function SHAPVisualization({ explanation }: SHAPVisualizationProps) {
    if (explanation.error) {
        return (
            <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-4">
                <p className="text-sm text-destructive">Error: {explanation.error}</p>
            </div>
        )
    }

    if (!explanation.contributions || explanation.contributions.length === 0) {
        return (
            <div className="rounded-lg border border-muted p-4">
                <p className="text-sm text-muted-foreground">No SHAP values available</p>
            </div>
        )
    }

    // Sort by absolute SHAP value for better visualization
    const sortedContributions = [...explanation.contributions].sort(
        (a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value)
    )

    const data = sortedContributions.map((contrib) => ({
        feature: contrib.feature,
        shapValue: contrib.shap_value,
        value: contrib.value,
    }))

    // Color bars based on positive/negative SHAP values
    const getColor = (value: number) => {
        return value >= 0 ? "hsl(var(--chart-1))" : "hsl(var(--chart-2))"
    }

    return (
        <div className="space-y-4">
            {explanation.base_value !== null && explanation.base_value !== undefined && (
                <div className="text-sm text-muted-foreground">
                    Base value: <span className="font-medium">{explanation.base_value.toFixed(4)}</span>
                </div>
            )}
            <ResponsiveContainer width="100%" height={400}>
                <BarChart data={data} layout="vertical" margin={{ top: 5, right: 30, left: 100, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" label={{ value: "SHAP Value", position: "insideBottom", offset: -5 }} />
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
                                            SHAP Value: <span className="font-mono">{data.shapValue.toFixed(4)}</span>
                                        </p>
                                        <p className="text-sm">
                                            Feature Value: <span className="font-mono">{data.value.toFixed(2)}</span>
                                        </p>
                                    </div>
                                )
                            }
                            return null
                        }}
                    />
                    <Bar dataKey="shapValue" radius={[0, 4, 4, 0]}>
                        {data.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={getColor(entry.shapValue)} />
                        ))}
                    </Bar>
                </BarChart>
            </ResponsiveContainer>
            <div className="text-xs text-muted-foreground">
                <p>
                    Positive SHAP values (red) push the prediction toward the positive class, while negative values
                    (blue) push toward the negative class.
                </p>
            </div>
        </div>
    )
}

