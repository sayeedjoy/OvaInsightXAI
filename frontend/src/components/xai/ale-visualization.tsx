"use client"

import { useState } from "react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import type { ALE1DResponse } from "@/types/xai"

interface ALEVisualizationProps {
    explanation: ALE1DResponse
}

export function ALEVisualization({ explanation }: ALEVisualizationProps) {
    if (explanation.error) {
        return (
            <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-4">
                <p className="text-sm text-destructive">Error: {explanation.error}</p>
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

    const [selectedFeature, setSelectedFeature] = useState(explanation.ale_plots[0]?.feature || "")

    const selectedPlot = explanation.ale_plots.find((plot) => plot.feature === selectedFeature)

    if (!selectedPlot) {
        return null
    }

    const data = selectedPlot.bin_centers.map((center, index) => ({
        binCenter: center,
        aleValue: selectedPlot.ale_values[index],
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
                        {explanation.ale_plots.map((plot) => (
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
                        dataKey="binCenter"
                        label={{ value: selectedFeature, position: "insideBottom", offset: -5 }}
                    />
                    <YAxis label={{ value: "ALE Value", angle: -90, position: "insideLeft" }} />
                    <Tooltip
                        content={({ active, payload }) => {
                            if (active && payload && payload.length > 0) {
                                const data = payload[0].payload as typeof data[number]
                                return (
                                    <div className="rounded-lg border bg-background p-3 shadow-md">
                                        <p className="font-medium">{selectedFeature}</p>
                                        <p className="text-sm">
                                            Bin Center: <span className="font-mono">{data.binCenter.toFixed(2)}</span>
                                        </p>
                                        <p className="text-sm">
                                            ALE Value: <span className="font-mono">{data.aleValue.toFixed(4)}</span>
                                        </p>
                                    </div>
                                )
                            }
                            return null
                        }}
                    />
                    <Line
                        type="monotone"
                        dataKey="aleValue"
                        stroke="hsl(var(--chart-1))"
                        strokeWidth={2}
                        dot={{ r: 3 }}
                    />
                </LineChart>
            </ResponsiveContainer>
            <div className="text-xs text-muted-foreground">
                <p>
                    Accumulated Local Effects (ALE) plots show the accumulated local effect of a feature on the
                    prediction, accounting for feature interactions. Values are centered around zero.
                </p>
            </div>
        </div>
    )
}

