"use client"

import { useEffect, useRef } from "react"
import { useTheme } from "next-themes"
import * as echarts from "echarts"
import type { SHAPExplanation } from "@/types/xai"
import { getEChartsTheme, getEChartsBaseOption } from "@/lib/echarts-theme"

interface SHAPVisualizationProps {
    explanation: SHAPExplanation
}

export function SHAPVisualization({ explanation }: SHAPVisualizationProps) {
    const chartRef = useRef<HTMLDivElement>(null)
    const chartInstanceRef = useRef<echarts.ECharts | null>(null)
    const { theme: currentTheme } = useTheme()

    useEffect(() => {
        if (!chartRef.current) return

        if (explanation.error) {
            return
        }

        if (!explanation.contributions || explanation.contributions.length === 0) {
            return
        }

        // Initialize chart
        if (!chartInstanceRef.current) {
            chartInstanceRef.current = echarts.init(chartRef.current)
        }

        const chart = chartInstanceRef.current
        const isDark = currentTheme === "dark"
        const theme = getEChartsTheme(isDark)

        // Sort by absolute SHAP value for better visualization
        const sortedContributions = [...explanation.contributions].sort(
            (a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value)
        )

        const data = sortedContributions.map((contrib) => ({
            value: contrib.shap_value,
            feature: contrib.feature,
            originalValue: contrib.value,
        }))

        const option: echarts.EChartsOption = {
            ...getEChartsBaseOption(),
            grid: {
                left: 120,
                right: 40,
                top: 20,
                bottom: 20,
                containLabel: false,
            },
            xAxis: {
                type: "value",
                name: "SHAP Value",
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
                type: "category",
                data: data.map((d) => d.feature),
                axisLabel: {
                    color: theme.textColor,
                    fontSize: 11,
                    interval: 0,
                },
                axisLine: {
                    lineStyle: {
                        color: theme.gridColor,
                    },
                },
            },
            tooltip: {
                ...getEChartsBaseOption().tooltip,
                trigger: "axis",
                axisPointer: {
                    type: "shadow",
                },
                formatter: (params) => {
                    const param = Array.isArray(params) ? params[0] : params
                    // ECharts passes data in param.data for bar charts
                    const dataPoint = param.data as { value: number; feature?: string; originalValue?: number }
                    const featureName = dataPoint?.feature || param.name || "Unknown"
                    const shapValue = dataPoint?.value ?? param.value ?? 0
                    const originalValue = dataPoint?.originalValue
                    
                    let content = `
                        <div style="margin: 4px 0;">
                            <strong>${featureName}</strong>
                        </div>
                        <div style="margin: 4px 0;">
                            SHAP Value: <code>${typeof shapValue === "number" ? shapValue.toFixed(4) : shapValue}</code>
                        </div>
                    `
                    
                    if (originalValue !== undefined && originalValue !== null && typeof originalValue === "number") {
                        content += `
                            <div style="margin: 4px 0;">
                                Feature Value: <code>${originalValue.toFixed(2)}</code>
                            </div>
                        `
                    }
                    
                    return content
                },
            },
            series: [
                {
                    type: "bar",
                    data: data.map((d) => ({
                        value: d.value,
                        feature: d.feature,
                        originalValue: d.originalValue,
                        itemStyle: {
                            color: d.value >= 0 ? theme.positiveColor : theme.negativeColor,
                        },
                    })),
                    barWidth: "60%",
                    label: {
                        show: false,
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
    }, [explanation, currentTheme])

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

    if (!explanation.contributions || explanation.contributions.length === 0) {
        return (
            <div className="rounded-lg border border-muted p-4">
                <p className="text-sm text-muted-foreground">No SHAP values available</p>
            </div>
        )
    }

    return (
        <div className="space-y-4">
            {explanation.base_value !== null && explanation.base_value !== undefined && (
                <div className="text-sm text-muted-foreground">
                    Base value: <span className="font-medium">{explanation.base_value.toFixed(4)}</span>
                </div>
            )}
            <div ref={chartRef} className="h-[400px] w-full" />
            <div className="text-xs text-muted-foreground">
                <p>
                    Positive SHAP values (green) push the prediction toward the positive class, while negative values
                    (red) push toward the negative class.
                </p>
            </div>
        </div>
    )
}
