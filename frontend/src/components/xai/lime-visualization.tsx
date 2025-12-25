"use client"

import { useEffect, useRef } from "react"
import { useTheme } from "next-themes"
import * as echarts from "echarts"
import type { LIMEExplanation } from "@/types/xai"
import { getEChartsTheme, getEChartsBaseOption } from "@/lib/echarts-theme"

interface LIMEVisualizationProps {
    explanation: LIMEExplanation
}

export function LIMEVisualization({ explanation }: LIMEVisualizationProps) {
    const chartRef = useRef<HTMLDivElement>(null)
    const chartInstanceRef = useRef<echarts.ECharts | null>(null)
    const { theme: currentTheme } = useTheme()

    useEffect(() => {
        if (!chartRef.current) return

        if (explanation.error) {
            return
        }

        if (!explanation.feature_importance || explanation.feature_importance.length === 0) {
            return
        }

        // Initialize chart
        if (!chartInstanceRef.current) {
            chartInstanceRef.current = echarts.init(chartRef.current)
        }

        const chart = chartInstanceRef.current
        const isDark = currentTheme === "dark"
        const theme = getEChartsTheme(isDark)

        // Sort by absolute importance
        const sortedFeatures = [...explanation.feature_importance].sort(
            (a, b) => Math.abs(b.importance) - Math.abs(a.importance)
        )

        const data = sortedFeatures.map((feat) => ({
            value: feat.importance,
            feature: feat.feature,
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
                name: "Importance",
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
                    const data = param.data as typeof data[number]
                    return `
                        <div style="margin: 4px 0;">
                            <strong>${data.feature}</strong>
                        </div>
                        <div style="margin: 4px 0;">
                            Importance: <code>${data.value.toFixed(4)}</code>
                        </div>
                    `
                },
            },
            series: [
                {
                    type: "bar",
                    data: data.map((d) => ({
                        value: d.value,
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

    if (!explanation.feature_importance || explanation.feature_importance.length === 0) {
        return (
            <div className="rounded-lg border border-muted p-4">
                <p className="text-sm text-muted-foreground">No LIME explanation available</p>
            </div>
        )
    }

    return (
        <div className="space-y-4">
            <div ref={chartRef} className="h-[400px] w-full" />
            <div className="text-xs text-muted-foreground">
                <p>
                    LIME shows how each feature locally affects the prediction. Positive values (green) increase the
                    prediction, negative values (red) decrease it.
                </p>
            </div>
        </div>
    )
}
