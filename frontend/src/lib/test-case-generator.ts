/**
 * Test case generator for ovarian cancer prediction form.
 * 
 * Generates clinically realistic test cases for both negative (normal/healthy)
 * and positive (ovarian cancer) scenarios. All values are within medically
 * valid ranges based on clinical guidelines.
 */

import type { FormValues } from "@/components/prediction-components"

/**
 * Generate a random number between min and max (inclusive)
 */
function randomInRange(min: number, max: number): number {
    return Math.random() * (max - min) + min
}

/**
 * Generate a random number with normal distribution approximation
 * Using Box-Muller transform for better distribution
 */
function randomNormal(mean: number, stdDev: number): number {
    const u1 = Math.random()
    const u2 = Math.random()
    const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
    return z0 * stdDev + mean
}

/**
 * Clamp a value between min and max
 */
function clamp(value: number, min: number, max: number): number {
    return Math.max(min, Math.min(max, value))
}

/**
 * Generate a negative (normal/healthy) test case.
 * 
 * Negative cases represent patients without ovarian cancer:
 * - CA125 < 35 U/mL (normal threshold)
 * - HE4 within normal range (30-70 pmol/L)
 * - Normal albumin, ALP, GGT, sodium levels
 * - All other markers within healthy ranges
 * 
 * @returns FormValues object with all fields as strings
 */
export function generateNegativeTestCase(): FormValues {
    // Age: 20-90 years (realistic patient age range)
    // Using normal distribution centered around 45
    const age = clamp(randomNormal(45, 15), 20, 90)

    // ALB (Albumin): 3.5-5.0 g/dL (normal albumin range)
    const alb = clamp(randomNormal(4.2, 0.4), 3.5, 5.0)

    // ALP (Alkaline Phosphatase): 30-130 U/L (normal range)
    const alp = clamp(randomNormal(75, 20), 30, 130)

    // BUN (Blood Urea Nitrogen): 7-20 mg/dL (normal range)
    const bun = clamp(randomNormal(14, 3), 7, 20)

    // CA125: 5-34.9 U/mL (normal, must be < 35)
    // Using exponential-like distribution favoring lower values
    const ca125 = clamp(Math.random() * 30 + 5, 5, 34.9)

    // EO# (Eosinophil Absolute Count): 0.05-0.5 x 10^9/L (normal)
    const eo_abs = clamp(randomNormal(0.2, 0.1), 0.05, 0.5)

    // GGT (Gamma-glutamyl Transferase): 5-40 U/L (normal range)
    const ggt = clamp(randomNormal(25, 10), 5, 40)

    // HE4: 30-70 pmol/L (normal range)
    const he4 = clamp(randomNormal(50, 12), 30, 70)

    // MCH (Mean Corpuscular Hemoglobin): 27-31 pg (normal)
    const mch = clamp(randomNormal(29, 1.5), 27, 31)

    // MONO# (Monocyte Absolute Count): 0.2-0.8 x 10^9/L (normal)
    const mono_abs = clamp(randomNormal(0.5, 0.15), 0.2, 0.8)

    // Na (Sodium): 135-145 mmol/L (normal sodium range)
    const na = clamp(randomNormal(140, 2.5), 135, 145)

    // PDW (Platelet Distribution Width): 9-14% (normal)
    const pdw = clamp(randomNormal(11.5, 1.5), 9, 14)

    return {
        age: age.toFixed(1),
        alb: alb.toFixed(2),
        alp: alp.toFixed(1),
        bun: bun.toFixed(1),
        ca125: ca125.toFixed(1),
        eo_abs: eo_abs.toFixed(2),
        ggt: ggt.toFixed(1),
        he4: he4.toFixed(1),
        mch: mch.toFixed(1),
        mono_abs: mono_abs.toFixed(2),
        na: na.toFixed(1),
        pdw: pdw.toFixed(1)
    }
}

/**
 * Generate a positive (ovarian cancer) test case.
 * 
 * Positive cases represent patients with ovarian cancer:
 * - CA125 > 35 U/mL, often > 100 U/mL (elevated marker)
 * - HE4 > 70 pmol/L, often > 150 pmol/L (elevated marker)
 * - ALB often low (2.5-3.8 g/dL) due to malnutrition/cachexia
 * - ALP often elevated (100-300 U/L) due to liver involvement
 * - GGT often elevated (40-200 U/L) due to liver involvement
 * - Na may be low (130-140 mmol/L) due to hyponatremia
 * - Age typically 40-85 years (higher risk age group)
 * - PDW often elevated (12-20%) due to platelet abnormalities
 * 
 * @returns FormValues object with all fields as strings
 */
export function generatePositiveTestCase(): FormValues {
    // Age: 40-85 years (higher risk age group for ovarian cancer)
    const age = clamp(randomNormal(62, 12), 40, 85)

    // ALB (Albumin): Often low, 2.5-3.8 g/dL (malnutrition/cachexia in cancer)
    const alb = clamp(randomNormal(3.2, 0.4), 2.5, 3.8)

    // ALP (Alkaline Phosphatase): Often elevated, 100-300 U/L
    // Liver involvement or bone metastasis can cause elevation
    const alp = clamp(randomNormal(180, 50), 100, 300)

    // BUN (Blood Urea Nitrogen): Variable, 10-30 mg/dL
    const bun = clamp(randomNormal(18, 5), 10, 30)

    // CA125: 36-500 U/mL (elevated, > 35)
    // Most ovarian cancer patients have significantly elevated CA125
    // Using exponential-like distribution favoring higher values
    const ca125 = clamp(Math.random() * 464 + 36, 36, 500)

    // EO# (Eosinophil Absolute Count): Variable, 0.1-0.6 x 10^9/L
    const eo_abs = clamp(randomNormal(0.3, 0.15), 0.1, 0.6)

    // GGT (Gamma-glutamyl Transferase): Often elevated, 40-200 U/L
    // Liver involvement can cause elevation
    const ggt = clamp(randomNormal(90, 40), 40, 200)

    // HE4: 71-500 pmol/L (elevated marker)
    // Strong indicator of ovarian cancer, often very elevated
    const he4 = clamp(Math.random() * 429 + 71, 71, 500)

    // MCH (Mean Corpuscular Hemoglobin): Variable, 25-30 pg
    // Can be low with anemia, which is common in cancer patients
    const mch = clamp(randomNormal(27.5, 2), 25, 30)

    // MONO# (Monocyte Absolute Count): Variable, 0.3-1.0 x 10^9/L
    // Can be elevated in cancer patients
    const mono_abs = clamp(randomNormal(0.6, 0.2), 0.3, 1.0)

    // Na (Sodium): May be low, 130-140 mmol/L (hyponatremia in cancer patients)
    const na = clamp(randomNormal(137, 3), 130, 140)

    // PDW (Platelet Distribution Width): Often elevated, 12-20%
    // Platelet abnormalities are common in cancer patients
    const pdw = clamp(randomNormal(15, 2.5), 12, 20)

    return {
        age: age.toFixed(1),
        alb: alb.toFixed(2),
        alp: alp.toFixed(1),
        bun: bun.toFixed(1),
        ca125: ca125.toFixed(1),
        eo_abs: eo_abs.toFixed(2),
        ggt: ggt.toFixed(1),
        he4: he4.toFixed(1),
        mch: mch.toFixed(1),
        mono_abs: mono_abs.toFixed(2),
        na: na.toFixed(1),
        pdw: pdw.toFixed(1)
    }
}

