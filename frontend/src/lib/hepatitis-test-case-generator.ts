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
 * Generate a positive (high-risk/advanced hepatitis) test case.
 * Uses scaled clinical intensity values in 2.0-3.0 range for severe features.
 */
function generatePositiveCase(): HepatitisFormValues {
    const values = {} as HepatitisFormValues

    // Clinical features (originally binary, now scaled)
    values.fatigue = String(roundFloat(randomFloat(2.0, 3.0)))
    values.malaise = String(roundFloat(randomFloat(2.0, 3.0)))
    values.spiders = String(roundFloat(randomFloat(2.0, 3.0)))
    values.liver_big = String(roundFloat(randomFloat(2.0, 3.0)))
    values.spleen_palpable = String(roundFloat(randomFloat(2.0, 3.0)))

    // Severe features
    values.ascites = String(roundFloat(randomFloat(2.2, 3.0)))
    values.varices = String(roundFloat(randomFloat(2.2, 3.0)))

    // Laboratory features (fractional in CSV)
    values.bilirubin = String(roundFloat(randomFloat(2.0, 3.0)))
    values.alk_phosphate = String(roundFloat(randomFloat(2.0, 3.0)))
    values.sgot = String(roundFloat(randomFloat(2.0, 3.0)))
    values.protime = String(roundFloat(randomFloat(2.0, 3.0)))

    // Albumin drops in liver damage
    values.albumin = String(roundFloat(randomFloat(0.3, 1.2)))

    // Demographics
    values.age = String(roundFloat(randomFloat(1.5, 3.0)))
    values.sex = String(roundFloat(randomFloat(0.0, 1.0)))

    // Histology (severe for positive case)
    values.histology = String(roundFloat(randomFloat(2.0, 3.0)))

    return values
}

/**
 * Generate a negative (low-risk/mild/healthy) test case.
 * Uses scaled clinical intensity values in 0.0-1.0 range for normal features.
 */
function generateNegativeCase(): HepatitisFormValues {
    const values = {} as HepatitisFormValues

    // Clinical features
    values.fatigue = String(roundFloat(randomFloat(0.0, 1.0)))
    values.malaise = String(roundFloat(randomFloat(0.0, 1.0)))
    values.spiders = String(roundFloat(randomFloat(0.0, 1.0)))
    values.liver_big = String(roundFloat(randomFloat(0.0, 1.0)))
    values.spleen_palpable = String(roundFloat(randomFloat(0.0, 1.0)))

    // Mild features
    values.ascites = String(roundFloat(randomFloat(0.0, 0.8)))
    values.varices = String(roundFloat(randomFloat(0.0, 0.8)))

    // Laboratory features
    values.bilirubin = String(roundFloat(randomFloat(0.0, 1.0)))
    values.alk_phosphate = String(roundFloat(randomFloat(0.0, 1.0)))
    values.sgot = String(roundFloat(randomFloat(0.0, 1.0)))
    values.protime = String(roundFloat(randomFloat(0.0, 1.0)))

    // Albumin normal/preserved
    values.albumin = String(roundFloat(randomFloat(2.0, 3.0)))

    // Demographics
    values.age = String(roundFloat(randomFloat(0.5, 2.0)))
    values.sex = String(roundFloat(randomFloat(0.0, 1.0)))

    // Histology (normal for negative case)
    values.histology = String(roundFloat(randomFloat(0.0, 1.0)))

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

