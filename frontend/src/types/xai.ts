/**
 * TypeScript types for XAI (Explainable AI) explanations
 */

export interface SHAPContribution {
    feature: string
    value: number
    shap_value: number
}

export interface SHAPExplanation {
    base_value?: number | null
    contributions: SHAPContribution[]
    prediction?: number | null
    error?: string | null
}

export interface LIMEFeatureImportance {
    feature: string
    importance: number
}

export interface LIMEExplanation {
    feature_importance: LIMEFeatureImportance[]
    prediction?: number | null
    error?: string | null
}

export interface PDP1DPlot {
    feature: string
    feature_index: number
    grid_values: number[]
    predictions: number[]
}

export interface PDP1DResponse {
    pdp_plots: PDP1DPlot[]
    error?: string | null
}

export interface ICE1DCurve {
    sample_index: number
    predictions: number[]
}

export interface ICE1DPlot {
    feature: string
    feature_index: number
    grid_values: number[]
    curves: ICE1DCurve[]
}

export interface ICE1DResponse {
    ice_plots: ICE1DPlot[]
    error?: string | null
}

export interface ALE1DPlot {
    feature: string
    feature_index: number
    bin_centers: number[]
    ale_values: number[]
}

export interface ALE1DResponse {
    ale_plots: ALE1DPlot[]
    error?: string | null
}

export interface XAIResponse {
    shap: SHAPExplanation | Record<string, unknown>
    lime: LIMEExplanation | Record<string, unknown>
    pdp_1d: PDP1DResponse | Record<string, unknown>
    ice_1d: ICE1DResponse | Record<string, unknown>
    ale_1d: ALE1DResponse | Record<string, unknown>
}

// Type guards
export function isSHAPExplanation(obj: unknown): obj is SHAPExplanation {
    return (
        typeof obj === "object" &&
        obj !== null &&
        "contributions" in obj &&
        Array.isArray((obj as SHAPExplanation).contributions)
    )
}

export function isLIMEExplanation(obj: unknown): obj is LIMEExplanation {
    return (
        typeof obj === "object" &&
        obj !== null &&
        "feature_importance" in obj &&
        Array.isArray((obj as LIMEExplanation).feature_importance)
    )
}

export function isPDP1DResponse(obj: unknown): obj is PDP1DResponse {
    return (
        typeof obj === "object" &&
        obj !== null &&
        "pdp_plots" in obj &&
        Array.isArray((obj as PDP1DResponse).pdp_plots)
    )
}

export function isICE1DResponse(obj: unknown): obj is ICE1DResponse {
    return (
        typeof obj === "object" &&
        obj !== null &&
        "ice_plots" in obj &&
        Array.isArray((obj as ICE1DResponse).ice_plots)
    )
}

export function isALE1DResponse(obj: unknown): obj is ALE1DResponse {
    return (
        typeof obj === "object" &&
        obj !== null &&
        "ale_plots" in obj &&
        Array.isArray((obj as ALE1DResponse).ale_plots)
    )
}

