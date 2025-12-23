import type { PcosFormValues } from "@/components/pcos-components"

function randomBin(): string {
    return String(Math.floor(Math.random() * 3)) // 0, 1, or 2
}

export function generatePcosNegativeTestCase(): PcosFormValues {
    const value = randomBin()
    return {
        marriage_status_years: value,
        cycle_regular_irregular: value,
        pulse_rate_bpm: value,
        fsh_miu_ml: value,
        age_years: value,
        follicle_no_left: value,
        bmi: value,
        skin_darkening_yn: value,
        ii_beta_hcg_miu_ml: value,
        bp_diastolic_mmhg: value,
        hair_growth_yn: value,
        avg_f_size_left_mm: value,
        avg_f_size_right_mm: value,
        waist_hip_ratio: value,
        weight_kg: value,
        weight_gain_yn: value,
        lh_miu_ml: value,
        follicle_no_right: value,
        hip_inch: value,
        waist_inch: value
    }
}

export function generatePcosPositiveTestCase(): PcosFormValues {
    const value = randomBin()
    return {
        marriage_status_years: value,
        cycle_regular_irregular: value,
        pulse_rate_bpm: value,
        fsh_miu_ml: value,
        age_years: value,
        follicle_no_left: value,
        bmi: value,
        skin_darkening_yn: value,
        ii_beta_hcg_miu_ml: value,
        bp_diastolic_mmhg: value,
        hair_growth_yn: value,
        avg_f_size_left_mm: value,
        avg_f_size_right_mm: value,
        waist_hip_ratio: value,
        weight_kg: value,
        weight_gain_yn: value,
        lh_miu_ml: value,
        follicle_no_right: value,
        hip_inch: value,
        waist_inch: value
    }
}

