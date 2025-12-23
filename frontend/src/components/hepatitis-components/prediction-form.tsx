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

export const HEPATITIS_FEATURE_FIELDS = [
    { key: "age", label: "Age", placeholder: "0.0-3.0" },
    { key: "sex", label: "Sex", placeholder: "0.0-1.0" },
    { key: "fatigue", label: "Fatigue", placeholder: "0.0-3.0" },
    { key: "malaise", label: "Malaise", placeholder: "0.0-3.0" },
    { key: "liver_big", label: "Liver Big", placeholder: "0.0-3.0" },
    { key: "spleen_palpable", label: "Spleen Palpable", placeholder: "0.0-3.0" },
    { key: "spiders", label: "Spiders", placeholder: "0.0-3.0" },
    { key: "ascites", label: "Ascites", placeholder: "0.0-3.0" },
    { key: "varices", label: "Varices", placeholder: "0.0-3.0" },
    { key: "bilirubin", label: "Bilirubin", placeholder: "0.0-3.0" },
    { key: "alk_phosphate", label: "Alk Phosphate", placeholder: "0.0-3.0" },
    { key: "sgot", label: "SGOT", placeholder: "0.0-3.0" },
    { key: "albumin", label: "Albumin", placeholder: "0.0-3.0" },
    { key: "protime", label: "Protime", placeholder: "0.0-3.0" },
    { key: "histology", label: "Histology", placeholder: "0.0-3.0" }
] as const

export type HepatitisFeatureKey = (typeof HEPATITIS_FEATURE_FIELDS)[number]["key"]

export type HepatitisFormValues = Record<HepatitisFeatureKey, string>

interface HepatitisPredictionFormProps {
    formValues: HepatitisFormValues
    isSubmitting: boolean
    onInputChange: (key: HepatitisFeatureKey, value: string) => void
    onSubmit: (event: React.FormEvent<HTMLFormElement>) => void
    onReset: () => void
    onFillTestCase?: (type: "negative" | "positive") => void
}

export function HepatitisPredictionForm({
    formValues,
    isSubmitting,
    onInputChange,
    onSubmit,
    onReset,
    onFillTestCase
}: HepatitisPredictionFormProps) {
    return (
        <Card>
            <CardHeader className="pb-4 sm:pb-6">
                <CardTitle className="text-lg sm:text-xl lg:text-2xl">Patient Metrics</CardTitle>
                <CardDescription className="text-xs sm:text-sm lg:text-base">
                    All values are required. Use numeric values in 0.0-3.0 range (floats allowed).
                </CardDescription>
            </CardHeader>
            <CardContent className="pt-4 sm:pt-6">
                <form
                    className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 sm:gap-5 lg:gap-6"
                    onSubmit={onSubmit}
                >
                    {HEPATITIS_FEATURE_FIELDS.map((field) => (
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

