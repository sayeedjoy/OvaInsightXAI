"use client"

import { useEffect, useRef, useState } from "react"
import { useTheme } from "next-themes"
import * as echarts from "echarts"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import type { ICE1DResponse } from "@/types/xai"
import { getEChartsTheme, getEChartsBaseOption } from "@/lib/echarts-theme"

interface ICEVisualizationProps {
    explanation: ICE1DResponse
}

export function ICEVisualization({ explanation }: ICEVisualizationProps) {
    const chartRef = useRef<HTMLDivElement>(null)
    const chartInstanceRef = useRef<echarts.ECharts | null>(null)
    const { theme: currentTheme } = useTheme()
    const [selectedFeature, setSelectedFeature] = useState(explanation.ice_plots[0]?.feature || "")

    const selectedPlot = explanation.ice_plots.find((plot) => plot.feature === selectedFeature)

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

        const gridValues = selectedPlot.grid_values
        const curves = selectedPlot.curves

        // Prepare series data for each curve
        const series = curves.map((curve, index) => {
            const data = gridValues.map((gridValue, gridIndex) => [
                gridValue,
                curve.predictions[gridIndex],
            ])

            // Use colors from theme palette, cycling through if needed
            const colorIndex = index % theme.lineColors.length
            const lineColor = theme.lineColors[colorIndex]

            return {
                type: "line" as const,
                data: data,
                smooth: true,
                symbol: "none",
                lineStyle: {
                    color: lineColor,
                    width: 1.5,
                    opacity: curves.length > 10 ? 0.6 : 0.8,
                },
                name: `Sample ${curve.sample_index + 1}`,
                emphasis: {
                    focus: "series",
                    lineStyle: {
                        width: 2.5,
                        opacity: 1,
                    },
                },
            }
        })

        const option: echarts.EChartsOption = {
            ...getEChartsBaseOption(),
            grid: {
                left: 60,
                right: 40,
                top: 20,
                bottom: 40,
                containLabel: true,
            },
            legend: {
                show: curves.length <= 10,
                type: "scroll",
                orient: "horizontal",
                bottom: 0,
                textStyle: {
                    color: theme.textColor,
                    fontSize: 11,
                },
                itemWidth: 14,
                itemHeight: 14,
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
                name: "Prediction",
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
                    if (!Array.isArray(params)) return ""
                    const value = params[0].value as [number, number]
                    let content = `
                        <div style="margin: 4px 0;">
                            <strong>${selectedFeature}</strong>
                        </div>
                        <div style="margin: 4px 0;">
                            Value: <code>${value[0].toFixed(2)}</code>
                        </div>
                    `
                    params.forEach((param) => {
                        const seriesName = param.seriesName || ""
                        const val = Array.isArray(param.value) ? param.value[1] : param.value
                        content += `
                            <div style="margin: 4px 0;">
                                ${seriesName}: <code>${Number(val).toFixed(4)}</code>
                            </div>
                        `
                    })
                    return content
                },
            },
            series: series,
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

    if (!explanation.ice_plots || explanation.ice_plots.length === 0) {
        return (
            <div className="rounded-lg border border-muted p-4">
                <p className="text-sm text-muted-foreground">No ICE plots available</p>
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
                        {explanation.ice_plots.map((plot) => (
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
                    Individual Conditional Expectation plots show how the prediction changes for individual samples as a
                    feature varies. Each line represents one sample.
                </p>
            </div>
        </div>
    )
}
