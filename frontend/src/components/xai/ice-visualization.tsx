"use client"

import { useState } from "react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import type { ICE1DResponse } from "@/types/xai"

interface ICEVisualizationProps {
    explanation: ICE1DResponse
}

export function ICEVisualization({ explanation }: ICEVisualizationProps) {
    if (explanation.error) {
        return (
            <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-4">
                <p className="text-sm text-destructive">Error: {explanation.error}</p>
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

    const [selectedFeature, setSelectedFeature] = useState(explanation.ice_plots[0]?.feature || "")

    const selectedPlot = explanation.ice_plots.find((plot) => plot.feature === selectedFeature)

    if (!selectedPlot) {
        return null
    }

    // Prepare data for all ICE curves
    const gridValues = selectedPlot.grid_values
    const curves = selectedPlot.curves

    // Create data array where each point has grid value and all curve predictions
    const data = gridValues.map((gridValue, index) => {
        const point: Record<string, number> = { featureValue: gridValue }
        curves.forEach((curve, curveIndex) => {
            point[`curve_${curveIndex}`] = curve.predictions[index]
        })
        return point
    })

    // Generate colors for curves (use muted colors for many curves)
    const getCurveColor = (index: number, total: number) => {
        const hue = (index * 360) / total
        return `hsl(${hue}, 70%, 50%)`
    }

    return (
        <div className="space-y-4">
            <div className="flex items-center gap-4">
                <label className="text-sm font-medium">Feature:</label>
                <Select value={selectedFeature} onValueChange={setSelectedFeature}>
                    <SelectTrigger className="w-[200px]">
                        <SelectValue placeholder="Select a feature" />
                    </SelectTrigger>
                    <SelectContent>
                        {explanation.ice_plots.map((plot) => (
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
                    <YAxis label={{ value: "Prediction", angle: -90, position: "insideLeft" }} />
                    <Tooltip
                        content={({ active, payload, label }) => {
                            if (active && payload && payload.length > 0) {
                                return (
                                    <div className="rounded-lg border bg-background p-3 shadow-md">
                                        <p className="font-medium">{selectedFeature}</p>
                                        <p className="text-sm">
                                            Value: <span className="font-mono">{Number(label).toFixed(2)}</span>
                                        </p>
                                        {payload.map((entry, index) => (
                                            <p key={index} className="text-sm">
                                                Curve {index + 1}:{" "}
                                                <span className="font-mono">
                                                    {Number(entry.value).toFixed(4)}
                                                </span>
                                            </p>
                                        ))}
                                    </div>
                                )
                            }
                            return null
                        }}
                    />
                    <Legend />
                    {curves.map((curve, index) => (
                        <Line
                            key={index}
                            type="monotone"
                            dataKey={`curve_${index}`}
                            stroke={getCurveColor(index, curves.length)}
                            strokeWidth={1.5}
                            dot={false}
                            name={`Sample ${curve.sample_index + 1}`}
                            connectNulls
                        />
                    ))}
                </LineChart>
            </ResponsiveContainer>
            <div className="text-xs text-muted-foreground">
                <p>
                    Individual Conditional Expectation plots show how the prediction changes for individual samples as a
                    feature varies. Each line represents one sample.
                </p>
            </div>
        </div>
    )
}

