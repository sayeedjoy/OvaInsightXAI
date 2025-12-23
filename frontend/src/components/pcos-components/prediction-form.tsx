"use client"

import { Button } from "@/components/ui/button"
import {
    Card,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

export const PCOS_FEATURE_FIELDS = [
    { key: "marriage_status_years", label: "Marraige Status (Yrs)", placeholder: "0-2" },
    { key: "cycle_regular_irregular", label: "Cycle(R/I)", placeholder: "0-2" },
    { key: "pulse_rate_bpm", label: "Pulse rate(bpm)", placeholder: "0-2" },
    { key: "fsh_miu_ml", label: "FSH(mIU/mL)", placeholder: "0-2" },
    { key: "age_years", label: "Age (yrs)", placeholder: "0-2" },
    { key: "follicle_no_left", label: "Follicle No. (L)", placeholder: "0-2" },
    { key: "bmi", label: "BMI", placeholder: "0-2" },
    { key: "skin_darkening_yn", label: "Skin darkening (Y/N)", placeholder: "0-2" },
    { key: "ii_beta_hcg_miu_ml", label: "II beta-HCG(mIU/mL)", placeholder: "0-2" },
    { key: "bp_diastolic_mmhg", label: "BP _Diastolic (mmHg)", placeholder: "0-2" },
    { key: "hair_growth_yn", label: "hair growth(Y/N)", placeholder: "0-2" },
    { key: "avg_f_size_left_mm", label: "Avg. F size (L) (mm)", placeholder: "0-2" },
    { key: "avg_f_size_right_mm", label: "Avg. F size (R) (mm)", placeholder: "0-2" },
    { key: "waist_hip_ratio", label: "Waist:Hip Ratio", placeholder: "0-2" },
    { key: "weight_kg", label: "Weight (Kg)", placeholder: "0-2" },
    { key: "weight_gain_yn", label: "Weight gain(Y/N)", placeholder: "0-2" },
    { key: "lh_miu_ml", label: "LH(mIU/mL)", placeholder: "0-2" },
    { key: "follicle_no_right", label: "Follicle No. (R)", placeholder: "0-2" },
    { key: "hip_inch", label: "Hip(inch)", placeholder: "0-2" },
    { key: "waist_inch", label: "Waist(inch)", placeholder: "0-2" }
] as const

export type PcosFeatureKey = (typeof PCOS_FEATURE_FIELDS)[number]["key"]

export type PcosFormValues = Record<PcosFeatureKey, string>

interface PcosPredictionFormProps {
    formValues: PcosFormValues
    isSubmitting: boolean
    onInputChange: (key: PcosFeatureKey, value: string) => void
    onSubmit: (event: React.FormEvent<HTMLFormElement>) => void
    onReset: () => void
    onFillTestCase?: (type: "negative" | "positive") => void
}

export function PcosPredictionForm({
    formValues,
    isSubmitting,
    onInputChange,
    onSubmit,
    onReset,
    onFillTestCase
}: PcosPredictionFormProps) {
    return (
        <Card>
            <CardHeader className="pb-4 sm:pb-6">
                <CardTitle className="text-lg sm:text-xl lg:text-2xl">Patient Metrics</CardTitle>
                <CardDescription className="text-xs sm:text-sm lg:text-base">
                    All values are required. Use numeric bins (0, 1, 2) as provided.
                </CardDescription>
            </CardHeader>
            <CardContent className="pt-4 sm:pt-6">
                <form
                    className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 sm:gap-5 lg:gap-6"
                    onSubmit={onSubmit}
                >
                    {PCOS_FEATURE_FIELDS.map((field) => (
                        <div className="flex flex-col space-y-2" key={field.key}>
                            <Label 
                                htmlFor={field.key}
                                className="text-sm sm:text-base lg:text-sm"
                            >
                                {field.label}
                            </Label>
                            <Input
                                id={field.key}
                                type="number"
                                inputMode="decimal"
                                placeholder={field.placeholder}
                                value={formValues[field.key]}
                                onChange={(event) =>
                                    onInputChange(field.key, event.target.value)
                                }
                                step="any"
                                required
                                disabled={isSubmitting}
                                className="h-9 sm:h-10 lg:h-11 text-sm sm:text-base"
                            />
                        </div>
                    ))}
                    <div className="flex flex-col gap-3 pt-2 sm:col-span-2 lg:col-span-3">
                        <div className="flex flex-col gap-3 sm:flex-row">
                            <Button 
                                type="submit" 
                                disabled={isSubmitting}
                                className="w-full sm:w-auto sm:flex-1 lg:text-base lg:h-11"
                            >
                                {isSubmitting ? "Predicting..." : "Run Prediction"}
                            </Button>
                            <Button
                                type="button"
                                variant="outline"
                                onClick={onReset}
                                disabled={isSubmitting}
                                className="w-full sm:w-auto lg:text-base lg:h-11"
                            >
                                Reset
                            </Button>
                        </div>
                        {onFillTestCase && (
                            <div className="flex flex-col gap-2 sm:flex-row sm:gap-3">
                                <Button
                                    type="button"
                                    variant="default"
                                    onClick={() => onFillTestCase("negative")}
                                    disabled={isSubmitting}
                                    className="w-full sm:w-auto sm:flex-1 lg:text-base lg:h-11"
                                >
                                    Fill Negative Case
                                </Button>
                                <Button
                                    type="button"
                                    variant="default"
                                    onClick={() => onFillTestCase("positive")}
                                    disabled={isSubmitting}
                                    className="w-full sm:w-auto sm:flex-1 lg:text-base lg:h-11"
                                >
                                    Fill Positive Case
                                </Button>
                            </div>
                        )}
                    </div>
                </form>
            </CardContent>
        </Card>
    )
}

