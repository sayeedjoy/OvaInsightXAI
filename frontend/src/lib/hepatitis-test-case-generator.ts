import { HEPATITIS_FEATURE_FIELDS, type HepatitisFeatureKey, type HepatitisFormValues } from "@/components/hepatitis-components"

type FeatureOrder = HepatitisFeatureKey[]

const featureOrder: FeatureOrder = HEPATITIS_FEATURE_FIELDS.map((field) => field.key)

const lastPositiveSignature: { current: string | null } = { current: null }
const lastNegativeSignature: { current: string | null } = { current: null }

/**
 * Generate a random float between min and max (inclusive)
 */
function randomFloat(min: number, max: number): number {
    return Math.random() * (max - min) + min
}

/**
 * Round a float to 2-3 decimal places for precision
 */
function roundFloat(value: number): number {
    return Math.round(value * 1000) / 1000
}

/**
 * Pick a binary value (0 or 1) based on weighted probability
 * @param probabilityOfOne Probability of returning 1 (0.0 to 1.0)
 * @returns "0" or "1" as a string
 */
function pickBinary(probabilityOfOne: number): string {
    return Math.random() < probabilityOfOne ? "1" : "0"
}

/**
 * Generate a positive (high-risk/severe hepatitis) test case.
 * Based on Class 0 (Negative/Severe Hepatitis) characteristics from CSV.
 * Binary fields are strictly 0 or 1. Fractional fields use CSV-observed ranges.
 */
function generatePositiveCase(): HepatitisFormValues {
    const values = {} as HepatitisFormValues

    // Binary features (strictly 0 or 1)
    // High probability symptoms for Class 0 (severe hepatitis):
    values.fatigue = pickBinary(0.94) // ~94% probability
    values.malaise = pickBinary(0.72) // ~72% probability
    values.spiders = pickBinary(0.69) // ~69% probability
    values.ascites = pickBinary(0.44) // ~44% probability
    
    // Other binary fields - set with moderate probabilities for severe case
    values.liver_big = pickBinary(0.60)
    values.spleen_palpable = pickBinary(0.50)
    values.varices = pickBinary(0.35)
    values.histology = pickBinary(0.70)

    // Sex: random 0 or 1
    values.sex = pickBinary(0.5)

    // Fractional features (Class 0 ranges - severe hepatitis)
    // Age: Mean 0.56
    values.age = String(roundFloat(randomFloat(0.50, 0.65)))
    
    // Bilirubin: Typically higher (Mean: 0.28, Range: 0.01 to 1.0)
    values.bilirubin = String(roundFloat(randomFloat(0.20, 0.50)))
    
    // Alk_phosphate: Use moderate-high range for severe case
    values.alk_phosphate = String(roundFloat(randomFloat(0.25, 0.40)))
    
    // SGOT: Use higher range for severe case
    values.sgot = String(roundFloat(randomFloat(0.15, 0.30)))
    
    // Albumin: Typically lower (Mean: 0.27, Range: 0.0 to 0.49)
    values.albumin = String(roundFloat(randomFloat(0.20, 0.45)))
    
    // Protime: Mean: 0.52, Range: 0.29 to 0.90
    values.protime = String(roundFloat(randomFloat(0.45, 0.75)))

    return values
}

/**
 * Generate a negative (low-risk/mild hepatitis) test case.
 * Based on Class 1 (Positive Hepatitis - mild) characteristics from CSV.
 * Binary fields are strictly 0 or 1. Fractional fields use CSV-observed ranges.
 */
function generateNegativeCase(): HepatitisFormValues {
    const values = {} as HepatitisFormValues

    // Binary features (strictly 0 or 1)
    // Low probability symptoms for Class 1 (mild hepatitis):
    values.ascites = pickBinary(0.05) // ~5% probability
    values.spiders = pickBinary(0.24) // ~24% probability
    values.malaise = pickBinary(0.31) // ~31% probability
    
    // Other binary fields - set with low probabilities for mild case
    values.fatigue = pickBinary(0.20)
    values.liver_big = pickBinary(0.15)
    values.spleen_palpable = pickBinary(0.10)
    values.varices = pickBinary(0.08)
    values.histology = pickBinary(0.25)

    // Sex: random 0 or 1
    values.sex = pickBinary(0.5)

    // Fractional features (Class 1 ranges - mild/positive hepatitis)
    // Age: Mean: 0.46
    values.age = String(roundFloat(randomFloat(0.40, 0.55)))
    
    // Bilirubin: Typically lower (Mean: 0.11, Range: 0.0 to 0.55)
    values.bilirubin = String(roundFloat(randomFloat(0.05, 0.20)))
    
    // Alk_phosphate: Use lower-moderate range for mild case
    values.alk_phosphate = String(roundFloat(randomFloat(0.10, 0.25)))
    
    // SGOT: Use lower range for mild case
    values.sgot = String(roundFloat(randomFloat(0.05, 0.15)))
    
    // Albumin: Typically higher (Mean: 0.43, Range: 0.14 to 1.0)
    values.albumin = String(roundFloat(randomFloat(0.35, 0.60)))
    
    // Protime: Mean: 0.65, Range: 0.0 to 1.0
    values.protime = String(roundFloat(randomFloat(0.55, 0.75)))

    return values
}

/**
 * Generate a case with signature tracking to avoid duplicates
 */
function generateCase(
    generator: () => HepatitisFormValues,
    lastSignature: { current: string | null }
): HepatitisFormValues {
    let attempt = 0
    let candidate: HepatitisFormValues | null = null

    while (attempt < 4) {
        const values = generator()

        const signature = JSON.stringify(values)
        candidate = values
        if (signature !== lastSignature.current) {
            lastSignature.current = signature
            return values
        }

        attempt += 1
    }

    return candidate ?? ({} as HepatitisFormValues)
}

export function generateHepatitisPositiveTestCase(): HepatitisFormValues {
    return generateCase(generatePositiveCase, lastPositiveSignature)
}

export function generateHepatitisNegativeTestCase(): HepatitisFormValues {
    return generateCase(generateNegativeCase, lastNegativeSignature)
}

