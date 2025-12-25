"use client"

import { useState } from "react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import type { PDP1DResponse } from "@/types/xai"

interface PDPVisualizationProps {
    explanation: PDP1DResponse
}

export function PDPVisualization({ explanation }: PDPVisualizationProps) {
    if (explanation.error) {
        return (
            <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-4">
                <p className="text-sm text-destructive">Error: {explanation.error}</p>
            </div>
        )
    }

    if (!explanation.pdp_plots || explanation.pdp_plots.length === 0) {
        return (
            <div className="rounded-lg border border-muted p-4">
                <p className="text-sm text-muted-foreground">No PDP plots available</p>
            </div>
        )
    }

    const [selectedFeature, setSelectedFeature] = useState(explanation.pdp_plots[0]?.feature || "")

    const selectedPlot = explanation.pdp_plots.find((plot) => plot.feature === selectedFeature)

    if (!selectedPlot) {
        return null
    }

    const data = selectedPlot.grid_values.map((value, index) => ({
        featureValue: value,
        prediction: selectedPlot.predictions[index],
    }))

    return (
        <div className="space-y-4">
            <div className="flex items-center gap-4">
                <label className="text-sm font-medium">Feature:</label>
                <Select value={selectedFeature} onValueChange={setSelectedFeature}>
                    <SelectTrigger className="w-[200px]">
                        <SelectValue placeholder="Select a feature" />
                    </SelectTrigger>
                    <SelectContent>
                        {explanation.pdp_plots.map((plot) => (
                            <SelectItem key={plot.feature} value={plot.feature}>
                                {plot.feature}
                            </SelectItem>
                        ))}
                    </SelectContent>
                </Select>
            </div>
            <ResponsiveContainer width="100%" height={400}>
                <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                        dataKey="featureValue"
                        label={{ value: selectedFeature, position: "insideBottom", offset: -5 }}
                    />
                    <YAxis label={{ value: "Average Prediction", angle: -90, position: "insideLeft" }} />
                    <Tooltip
                        content={({ active, payload }) => {
                            if (active && payload && payload.length > 0) {
                                const data = payload[0].payload as typeof data[number]
                                return (
                                    <div className="rounded-lg border bg-background p-3 shadow-md">
                                        <p className="font-medium">{selectedFeature}</p>
                                        <p className="text-sm">
                                            Value: <span className="font-mono">{data.featureValue.toFixed(2)}</span>
                                        </p>
                                        <p className="text-sm">
                                            Prediction: <span className="font-mono">{data.prediction.toFixed(4)}</span>
                                        </p>
                                    </div>
                                )
                            }
                            return null
                        }}
                    />
                    <Legend />
                    <Line
                        type="monotone"
                        dataKey="prediction"
                        stroke="hsl(var(--chart-1))"
                        strokeWidth={2}
                        dot={{ r: 3 }}
                        name="PDP"
                    />
                </LineChart>
            </ResponsiveContainer>
            <div className="text-xs text-muted-foreground">
                <p>
                    Partial Dependence Plot shows the average effect of a feature on the model prediction, marginalizing
                    over all other features.
                </p>
            </div>
        </div>
    )
}

