/**
 * ECharts theme utility for light and dark mode support
 * Provides theme-aware colors that work well in both modes
 */

export interface EChartsTheme {
    isDark: boolean
    backgroundColor: string
    textColor: string
    gridColor: string
    positiveColor: string
    negativeColor: string
    primaryColor: string
    lineColors: string[]
    tooltip: {
        backgroundColor: string
        borderColor: string
        textColor: string
    }
}

/**
 * Get theme-aware colors for ECharts
 * Detects theme by checking if dark class exists on document
 * @param isDark - Optional boolean to explicitly set dark mode. If not provided, detects from DOM.
 */
export function getEChartsTheme(isDark?: boolean): EChartsTheme {
    // Check if dark mode is active
    const darkMode =
        isDark !== undefined
            ? isDark
            : typeof document !== "undefined" &&
              document.documentElement.classList.contains("dark")

    if (darkMode) {
        // Dark mode colors - lighter, vibrant
        return {
            isDark: darkMode,
            backgroundColor: "transparent",
            textColor: "#e5e7eb", // light gray
            gridColor: "rgba(255, 255, 255, 0.1)",
            positiveColor: "#34d399", // green-400
            negativeColor: "#f87171", // red-400
            primaryColor: "#60a5fa", // blue-400
            lineColors: [
                "#60a5fa", // blue-400
                "#34d399", // green-400
                "#fbbf24", // amber-400
                "#a78bfa", // violet-400
                "#fb7185", // rose-400
                "#2dd4bf", // teal-400
                "#f472b6", // pink-400
                "#818cf8", // indigo-400
            ],
            tooltip: {
                backgroundColor: "rgba(31, 41, 55, 0.95)", // gray-800 with opacity
                borderColor: "rgba(75, 85, 99, 0.5)", // gray-600
                textColor: "#f3f4f6", // gray-100
            },
        }
    } else {
        // Light mode colors - darker, more saturated
        return {
            isDark: darkMode,
            backgroundColor: "transparent",
            textColor: "#1f2937", // gray-800
            gridColor: "rgba(0, 0, 0, 0.1)",
            positiveColor: "#059669", // green-600
            negativeColor: "#dc2626", // red-600
            primaryColor: "#2563eb", // blue-600
            lineColors: [
                "#2563eb", // blue-600
                "#059669", // green-600
                "#d97706", // amber-600
                "#7c3aed", // violet-600
                "#e11d48", // rose-600
                "#0d9488", // teal-600
                "#db2777", // pink-600
                "#4f46e5", // indigo-600
            ],
            tooltip: {
                backgroundColor: "rgba(255, 255, 255, 0.95)",
                borderColor: "rgba(209, 213, 219, 0.8)", // gray-300
                textColor: "#1f2937", // gray-800
            },
        }
    }
}

/**
 * Get ECharts option with theme applied
 * This is a base configuration that can be extended
 */
export function getEChartsBaseOption() {
    const theme = getEChartsTheme()

    return {
        backgroundColor: theme.backgroundColor,
        textStyle: {
            color: theme.textColor,
            fontFamily: "var(--font-sans)",
        },
        grid: {
            left: "10%",
            right: "10%",
            top: "15%",
            bottom: "15%",
            containLabel: true,
        },
        tooltip: {
            backgroundColor: theme.tooltip.backgroundColor,
            borderColor: theme.tooltip.borderColor,
            borderWidth: 1,
            textStyle: {
                color: theme.tooltip.textColor,
            },
            padding: [8, 12],
            extraCssText: "box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);",
        },
    }
}

