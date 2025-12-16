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

export const FEATURE_FIELDS = [
    { key: "age", label: "Age", placeholder: "Years" },
    { key: "alb", label: "ALB", placeholder: "g/dL" },
    { key: "alp", label: "ALP", placeholder: "U/L" },
    { key: "bun", label: "BUN", placeholder: "mg/dL" },
    { key: "ca125", label: "CA125", placeholder: "U/mL" },
    { key: "eo_abs", label: "EO#", placeholder: "10^9/L" },
    { key: "ggt", label: "GGT", placeholder: "U/L" },
    { key: "he4", label: "HE4", placeholder: "pmol/L" },
    { key: "mch", label: "MCH", placeholder: "pg" },
    { key: "mono_abs", label: "MONO#", placeholder: "10^9/L" },
    { key: "na", label: "Na", placeholder: "mmol/L" },
    { key: "pdw", label: "PDW", placeholder: "%" }
] as const

export type FeatureKey = (typeof FEATURE_FIELDS)[number]["key"]

export type FormValues = Record<FeatureKey, string>

interface PredictionFormProps {
    formValues: FormValues
    isSubmitting: boolean
    onInputChange: (key: FeatureKey, value: string) => void
    onSubmit: (event: React.FormEvent<HTMLFormElement>) => void
    onReset: () => void
    onFillTestCase?: (type: "negative" | "positive") => void
}

export function PredictionForm({
    formValues,
    isSubmitting,
    onInputChange,
    onSubmit,
    onReset,
    onFillTestCase
}: PredictionFormProps) {
    return (
        <Card>
            <CardHeader className="pb-4 sm:pb-6">
                <CardTitle className="text-lg sm:text-xl lg:text-2xl">Patient Metrics</CardTitle>
                <CardDescription className="text-xs sm:text-sm lg:text-base">
                    All values are required. Use numeric values only.
                </CardDescription>
            </CardHeader>
            <CardContent className="pt-4 sm:pt-6">
                <form
                    className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 sm:gap-5 lg:gap-6"
                    onSubmit={onSubmit}
                >
                    {FEATURE_FIELDS.map((field) => (
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
                                    variant="secondary"
                                    onClick={() => onFillTestCase("negative")}
                                    disabled={isSubmitting}
                                    className="w-full sm:w-auto sm:flex-1 lg:text-base lg:h-11"
                                >
                                    Fill Negative Case
                                </Button>
                                <Button
                                    type="button"
                                    variant="secondary"
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

