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

// Image-based XAI types
export interface ImageSHAPExplanation {
    heatmap_image?: string | null  // Base64 encoded image
    heatmap_data?: number[] | null  // Flattened array for custom rendering
    heatmap_shape?: number[] | null  // [H, W]
    probabilities?: number[] | null
    predicted_class?: number | null
    error?: string | null
}

export interface ImageLIMEFeatureImportance {
    feature_index: number
    importance: number
    superpixel_mask?: number[] | null
    superpixel_shape?: number[] | null
}

export interface ImageLIMEExplanation {
    visualization_image?: string | null  // Base64 encoded image
    feature_importance?: ImageLIMEFeatureImportance[]
    top_label?: number | null
    probabilities?: number[] | null
    mask_shape?: number[] | null
    error?: string | null
}

// Image PDP types
export interface ImagePDPPlot {
    patch_index: number
    patch_row: number
    patch_col: number
    patch_coords: {
        y_start: number
        y_end: number
        x_start: number
        x_end: number
    }
    intensity_values: number[]
    predictions: number[]
}

export interface ImagePDPExplanation {
    pdp_plots?: ImagePDPPlot[]
    grid_size?: number | null
    patch_size?: { height: number; width: number } | null
    image_size?: { height: number; width: number } | null
    predicted_class?: number | null
    error?: string | null
}

// Image ICE types
export interface ImageICECurve {
    sample_index: number
    predictions: number[]
}

export interface ImageICEPlot {
    patch_index: number
    patch_row: number
    patch_col: number
    patch_coords: {
        y_start: number
        y_end: number
        x_start: number
        x_end: number
    }
    intensity_values: number[]
    curves: ImageICECurve[]
}

export interface ImageICEExplanation {
    ice_plots?: ImageICEPlot[]
    grid_size?: number | null
    patch_size?: { height: number; width: number } | null
    image_size?: { height: number; width: number } | null
    predicted_class?: number | null
    error?: string | null
}

// Image ALE types
export interface ImageALEPlot {
    patch_index: number
    patch_row: number
    patch_col: number
    patch_coords: {
        y_start: number
        y_end: number
        x_start: number
        x_end: number
    }
    bin_centers: number[]
    ale_values: number[]
}

export interface ImageALEExplanation {
    ale_plots?: ImageALEPlot[]
    grid_size?: number | null
    patch_size?: { height: number; width: number } | null
    image_size?: { height: number; width: number } | null
    predicted_class?: number | null
    error?: string | null
}

export interface XAIResponse {
    shap: SHAPExplanation | ImageSHAPExplanation | Record<string, unknown>
    lime: LIMEExplanation | ImageLIMEExplanation | Record<string, unknown>
    pdp_1d: PDP1DResponse | ImagePDPExplanation | Record<string, unknown>
    ice_1d: ICE1DResponse | ImageICEExplanation | Record<string, unknown>
    ale_1d: ALE1DResponse | ImageALEExplanation | Record<string, unknown>
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

// Image XAI type guards
export function isImageSHAPExplanation(obj: unknown): obj is ImageSHAPExplanation {
    return (
        typeof obj === "object" &&
        obj !== null &&
        ("heatmap_image" in obj || "heatmap_data" in obj || "error" in obj) &&
        !("contributions" in obj)  // Not tabular SHAP
    )
}

export function isImageLIMEExplanation(obj: unknown): obj is ImageLIMEExplanation {
    return (
        typeof obj === "object" &&
        obj !== null &&
        ("visualization_image" in obj || "feature_importance" in obj || "error" in obj) &&
        !("feature_importance" in obj && 
          Array.isArray((obj as any).feature_importance) && 
          (obj as any).feature_importance.length > 0 &&
          "feature" in (obj as any).feature_importance[0])  // Not tabular LIME (which has "feature" string)
    )
}

export function isImagePDPExplanation(obj: unknown): obj is ImagePDPExplanation {
    return (
        typeof obj === "object" &&
        obj !== null &&
        ("pdp_plots" in obj || "error" in obj) &&
        !("pdp_plots" in obj && 
          Array.isArray((obj as any).pdp_plots) && 
          (obj as any).pdp_plots.length > 0 &&
          "feature" in (obj as any).pdp_plots[0])  // Not tabular PDP (which has "feature" string)
    )
}

export function isImageICEExplanation(obj: unknown): obj is ImageICEExplanation {
    return (
        typeof obj === "object" &&
        obj !== null &&
        ("ice_plots" in obj || "error" in obj) &&
        !("ice_plots" in obj && 
          Array.isArray((obj as any).ice_plots) && 
          (obj as any).ice_plots.length > 0 &&
          "feature" in (obj as any).ice_plots[0])  // Not tabular ICE (which has "feature" string)
    )
}

export function isImageALEExplanation(obj: unknown): obj is ImageALEExplanation {
    return (
        typeof obj === "object" &&
        obj !== null &&
        ("ale_plots" in obj || "error" in obj) &&
        !("ale_plots" in obj && 
          Array.isArray((obj as any).ale_plots) && 
          (obj as any).ale_plots.length > 0 &&
          "feature" in (obj as any).ale_plots[0])  // Not tabular ALE (which has "feature" string)
    )
}

