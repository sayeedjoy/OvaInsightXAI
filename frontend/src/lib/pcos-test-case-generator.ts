import { PCOS_FEATURE_FIELDS, type PcosFeatureKey, type PcosFormValues } from "@/components/pcos-components"

type BucketValue = 0 | 1 | 2 | 3
type BucketOption = { value: BucketValue; weight: number }
type FeatureBuckets = Record<PcosFeatureKey, BucketOption[]>

const featureOrder: PcosFeatureKey[] = PCOS_FEATURE_FIELDS.map((field) => field.key)

const lastPositiveSignature: { current: string | null } = { current: null }
const lastNegativeSignature: { current: string | null } = { current: null }

function pickBucket(options: BucketOption[]): BucketValue {
    const totalWeight = options.reduce((sum, option) => sum + option.weight, 0)
    const roll = Math.random() * totalWeight

    let cumulative = 0
    for (const option of options) {
        cumulative += option.weight
        if (roll <= cumulative) {
            return option.value
        }
    }

    return options[options.length - 1]!.value
}

function buildPositiveBuckets(): FeatureBuckets {
    return {
        marriage_status_years: [
            { value: 1, weight: 0.45 },
            { value: 2, weight: 0.4 },
            { value: 3, weight: 0.15 }
        ],
        cycle_regular_irregular: [
            { value: 2, weight: 0.6 },
            { value: 3, weight: 0.4 }
        ],
        pulse_rate_bpm: [
            { value: 1, weight: 0.5 },
            { value: 2, weight: 0.4 },
            { value: 0, weight: 0.1 }
        ],
        fsh_miu_ml: [
            { value: 1, weight: 0.55 },
            { value: 2, weight: 0.45 }
        ],
        age_years: [
            { value: 1, weight: 0.4 },
            { value: 2, weight: 0.45 },
            { value: 0, weight: 0.1 },
            { value: 3, weight: 0.05 }
        ],
        follicle_no_left: [
            { value: 2, weight: 0.55 },
            { value: 3, weight: 0.45 }
        ],
        bmi: [
            { value: 2, weight: 0.6 },
            { value: 3, weight: 0.4 }
        ],
        skin_darkening_yn: [
            { value: 2, weight: 0.6 },
            { value: 3, weight: 0.4 }
        ],
        ii_beta_hcg_miu_ml: [
            { value: 1, weight: 0.45 },
            { value: 2, weight: 0.45 },
            { value: 0, weight: 0.1 }
        ],
        bp_diastolic_mmhg: [
            { value: 1, weight: 0.45 },
            { value: 2, weight: 0.45 },
            { value: 0, weight: 0.1 }
        ],
        hair_growth_yn: [
            { value: 2, weight: 0.6 },
            { value: 3, weight: 0.4 }
        ],
        avg_f_size_left_mm: [
            { value: 1, weight: 0.55 },
            { value: 2, weight: 0.45 }
        ],
        avg_f_size_right_mm: [
            { value: 1, weight: 0.55 },
            { value: 2, weight: 0.45 }
        ],
        waist_hip_ratio: [
            { value: 2, weight: 0.55 },
            { value: 3, weight: 0.45 }
        ],
        weight_kg: [
            { value: 2, weight: 0.5 },
            { value: 3, weight: 0.35 },
            { value: 1, weight: 0.15 }
        ],
        weight_gain_yn: [
            { value: 2, weight: 0.6 },
            { value: 3, weight: 0.4 }
        ],
        lh_miu_ml: [
            { value: 2, weight: 0.6 },
            { value: 3, weight: 0.4 }
        ],
        follicle_no_right: [
            { value: 2, weight: 0.55 },
            { value: 3, weight: 0.45 }
        ],
        hip_inch: [
            { value: 1, weight: 0.45 },
            { value: 2, weight: 0.45 },
            { value: 0, weight: 0.05 },
            { value: 3, weight: 0.05 }
        ],
        waist_inch: [
            { value: 1, weight: 0.45 },
            { value: 2, weight: 0.45 },
            { value: 3, weight: 0.05 },
            { value: 0, weight: 0.05 }
        ]
    }
}

function buildNegativeBuckets(): FeatureBuckets {
    return {
        marriage_status_years: [
            { value: 0, weight: 0.35 },
            { value: 1, weight: 0.4 },
            { value: 2, weight: 0.25 }
        ],
        cycle_regular_irregular: [
            { value: 0, weight: 0.55 },
            { value: 1, weight: 0.45 }
        ],
        pulse_rate_bpm: [
            { value: 1, weight: 0.45 },
            { value: 0, weight: 0.3 },
            { value: 2, weight: 0.25 }
        ],
        fsh_miu_ml: [
            { value: 1, weight: 0.55 },
            { value: 2, weight: 0.45 }
        ],
        age_years: [
            { value: 1, weight: 0.4 },
            { value: 2, weight: 0.25 },
            { value: 0, weight: 0.35 }
        ],
        follicle_no_left: [
            { value: 0, weight: 0.55 },
            { value: 1, weight: 0.45 }
        ],
        bmi: [
            { value: 0, weight: 0.55 },
            { value: 1, weight: 0.45 }
        ],
        skin_darkening_yn: [
            { value: 0, weight: 0.55 },
            { value: 1, weight: 0.45 }
        ],
        ii_beta_hcg_miu_ml: [
            { value: 1, weight: 0.45 },
            { value: 0, weight: 0.35 },
            { value: 2, weight: 0.2 }
        ],
        bp_diastolic_mmhg: [
            { value: 1, weight: 0.45 },
            { value: 0, weight: 0.35 },
            { value: 2, weight: 0.2 }
        ],
        hair_growth_yn: [
            { value: 0, weight: 0.6 },
            { value: 1, weight: 0.4 }
        ],
        avg_f_size_left_mm: [
            { value: 1, weight: 0.6 },
            { value: 2, weight: 0.4 }
        ],
        avg_f_size_right_mm: [
            { value: 1, weight: 0.6 },
            { value: 2, weight: 0.4 }
        ],
        waist_hip_ratio: [
            { value: 0, weight: 0.55 },
            { value: 1, weight: 0.45 }
        ],
        weight_kg: [
            { value: 1, weight: 0.45 },
            { value: 0, weight: 0.35 },
            { value: 2, weight: 0.2 }
        ],
        weight_gain_yn: [
            { value: 0, weight: 0.6 },
            { value: 1, weight: 0.4 }
        ],
        lh_miu_ml: [
            { value: 0, weight: 0.55 },
            { value: 1, weight: 0.45 }
        ],
        follicle_no_right: [
            { value: 0, weight: 0.55 },
            { value: 1, weight: 0.45 }
        ],
        hip_inch: [
            { value: 1, weight: 0.45 },
            { value: 0, weight: 0.35 },
            { value: 2, weight: 0.2 }
        ],
        waist_inch: [
            { value: 1, weight: 0.45 },
            { value: 0, weight: 0.35 },
            { value: 2, weight: 0.2 }
        ]
    }
}

function generateCase(buckets: FeatureBuckets, lastSignature: { current: string | null }): PcosFormValues {
    let attempt = 0
    let candidate: PcosFormValues | null = null

    while (attempt < 4) {
        const values = {} as PcosFormValues

        for (const key of featureOrder) {
            values[key] = String(pickBucket(buckets[key]))
        }

        const signature = JSON.stringify(values)
        candidate = values
        if (signature !== lastSignature.current) {
            lastSignature.current = signature
            return values
        }

        attempt += 1
    }

    return candidate ?? ({} as PcosFormValues)
}

export function generatePcosPositiveTestCase(): PcosFormValues {
    const buckets = buildPositiveBuckets()
    return generateCase(buckets, lastPositiveSignature)
}

export function generatePcosNegativeTestCase(): PcosFormValues {
    const buckets = buildNegativeBuckets()
    return generateCase(buckets, lastNegativeSignature)
}

