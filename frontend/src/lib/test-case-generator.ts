/**
 * Test case generator for ovarian cancer prediction form.
 * 
 * Generates test cases with discretized/binned values matching the model's expected input.
 */

import type { FormValues } from "@/components/prediction-components"

/**
 * Generate a random integer between min and max (inclusive)
 */
function randomInt(min: number, max: number): number {
    return Math.floor(Math.random() * (max - min + 1)) + min
}

/**
 * Generate a negative (normal/healthy) test case.
 * Uses bin values that produce a "No Ovarian Cancer" prediction.
 * 
 * @returns FormValues object with all fields as strings
 */
export function generateNegativeTestCase(): FormValues {
    return {
        age: String(randomInt(1, 2)),      // Higher age bins
        alb: String(randomInt(0, 1)),      // Lower albumin
        alp: String(randomInt(1, 2)),      // Higher ALP
        bun: String(randomInt(1, 2)),      // Higher BUN
        ca125: String(2),                  // High CA125
        eo_abs: String(randomInt(1, 2)),   // Variable eosinophils
        ggt: String(randomInt(1, 2)),      // Higher GGT
        he4: String(2),                    // High HE4
        mch: String(randomInt(0, 1)),      // Lower MCH
        mono_abs: String(randomInt(1, 2)), // Elevated monocytes
        na: String(randomInt(0, 1)),       // Lower sodium
        pdw: String(randomInt(1, 2))       // Elevated PDW
    }
}

/**
 * Generate a positive (ovarian cancer) test case.
 * Uses bin values that produce a "Possible Ovarian Cancer" prediction.
 * 
 * @returns FormValues object with all fields as strings
 */
export function generatePositiveTestCase(): FormValues {
    return {
        age: String(randomInt(0, 1)),      // Lower age bins
        alb: String(randomInt(1, 2)),      // Higher albumin (healthy)
        alp: String(randomInt(0, 1)),      // Lower ALP (normal)
        bun: String(randomInt(0, 1)),      // Lower BUN (normal)
        ca125: String(0),                  // Low CA125 (normal)
        eo_abs: String(randomInt(0, 1)),   // Normal eosinophils
        ggt: String(randomInt(0, 1)),      // Lower GGT (normal)
        he4: String(0),                    // Low HE4 (normal)
        mch: String(randomInt(1, 2)),      // Normal MCH
        mono_abs: String(randomInt(0, 1)), // Normal monocytes
        na: String(randomInt(1, 2)),       // Normal sodium
        pdw: String(randomInt(0, 1))       // Normal PDW
    }
}

