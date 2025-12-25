"use client"

import { useEffect, useRef, useState } from "react"
import { useTheme } from "next-themes"
import * as echarts from "echarts"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import type { PDP1DResponse } from "@/types/xai"
import { getEChartsTheme, getEChartsBaseOption } from "@/lib/echarts-theme"

interface PDPVisualizationProps {
    explanation: PDP1DResponse
}

export function PDPVisualization({ explanation }: PDPVisualizationProps) {
    const chartRef = useRef<HTMLDivElement>(null)
    const chartInstanceRef = useRef<echarts.ECharts | null>(null)
    const { theme: currentTheme } = useTheme()
    const [selectedFeature, setSelectedFeature] = useState(explanation.pdp_plots[0]?.feature || "")

    const selectedPlot = explanation.pdp_plots.find((plot) => plot.feature === selectedFeature)

    useEffect(() => {
        if (!chartRef.current || !selectedPlot) return

        if (explanation.error) {
            return
        }

        // Initialize chart
        if (!chartInstanceRef.current) {
            chartInstanceRef.current = echarts.init(chartRef.current)
        }

        const chart = chartInstanceRef.current
        const isDark = currentTheme === "dark"
        const theme = getEChartsTheme(isDark)

        const data = selectedPlot.grid_values.map((value, index) => [
            value,
            selectedPlot.predictions[index],
        ])

        const option: echarts.EChartsOption = {
            ...getEChartsBaseOption(),
            grid: {
                left: 60,
                right: 40,
                top: 20,
                bottom: 40,
                containLabel: true,
            },
            xAxis: {
                type: "value",
                name: selectedFeature,
                nameLocation: "middle",
                nameGap: 30,
                nameTextStyle: {
                    color: theme.textColor,
                    fontSize: 12,
                },
                axisLabel: {
                    color: theme.textColor,
                },
                axisLine: {
                    lineStyle: {
                        color: theme.gridColor,
                    },
                },
                splitLine: {
                    show: true,
                    lineStyle: {
                        color: theme.gridColor,
                        type: "dashed",
                    },
                },
            },
            yAxis: {
                type: "value",
                name: "Average Prediction",
                nameLocation: "middle",
                nameGap: 50,
                nameTextStyle: {
                    color: theme.textColor,
                    fontSize: 12,
                },
                axisLabel: {
                    color: theme.textColor,
                },
                axisLine: {
                    lineStyle: {
                        color: theme.gridColor,
                    },
                },
                splitLine: {
                    show: true,
                    lineStyle: {
                        color: theme.gridColor,
                        type: "dashed",
                    },
                },
            },
            tooltip: {
                ...getEChartsBaseOption().tooltip,
                trigger: "axis",
                formatter: (params) => {
                    const param = Array.isArray(params) ? params[0] : params
                    const value = param.value as [number, number]
                    return `
                        <div style="margin: 4px 0;">
                            <strong>${selectedFeature}</strong>
                        </div>
                        <div style="margin: 4px 0;">
                            Value: <code>${value[0].toFixed(2)}</code>
                        </div>
                        <div style="margin: 4px 0;">
                            Prediction: <code>${value[1].toFixed(4)}</code>
                        </div>
                    `
                },
            },
            series: [
                {
                    type: "line",
                    data: data,
                    smooth: true,
                    symbol: "circle",
                    symbolSize: 6,
                    lineStyle: {
                        color: theme.primaryColor,
                        width: 2,
                    },
                    itemStyle: {
                        color: theme.primaryColor,
                        borderColor: theme.isDark ? "#1f2937" : "#ffffff",
                        borderWidth: 2,
                    },
                    areaStyle: {
                        color: {
                            type: "linear",
                            x: 0,
                            y: 0,
                            x2: 0,
                            y2: 1,
                            colorStops: [
                                {
                                    offset: 0,
                                    color: theme.isDark
                                        ? "rgba(96, 165, 250, 0.3)"
                                        : "rgba(37, 99, 235, 0.2)",
                                },
                                {
                                    offset: 1,
                                    color: theme.isDark
                                        ? "rgba(96, 165, 250, 0.05)"
                                        : "rgba(37, 99, 235, 0.05)",
                                },
                            ],
                        },
                    },
                    emphasis: {
                        focus: "series",
                        itemStyle: {
                            borderWidth: 3,
                        },
                    },
                },
            ],
        }

        chart.setOption(option)

        // Handle resize
        const handleResize = () => {
            chart.resize()
        }
        window.addEventListener("resize", handleResize)

        return () => {
            window.removeEventListener("resize", handleResize)
        }
    }, [explanation, selectedFeature, selectedPlot, currentTheme])

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (chartInstanceRef.current) {
                chartInstanceRef.current.dispose()
                chartInstanceRef.current = null
            }
        }
    }, [])

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

    if (!selectedPlot) {
        return null
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
                        {explanation.pdp_plots.map((plot) => (
                            <SelectItem key={plot.feature} value={plot.feature}>
                                {plot.feature}
                            </SelectItem>
                        ))}
                    </SelectContent>
                </Select>
            </div>
            <div ref={chartRef} className="h-[400px] w-full" />
            <div className="text-xs text-muted-foreground">
                <p>
                    Partial Dependence Plot shows the average effect of a feature on the model prediction, marginalizing
                    over all other features.
                </p>
            </div>
        </div>
    )
}
