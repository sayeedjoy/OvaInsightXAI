"use client"

import { useRef, useState } from "react"
import Image from "next/image"
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

interface BrainTumorPredictionFormProps {
    selectedFile: File | null
    previewUrl: string | null
    isSubmitting: boolean
    onFileSelect: (file: File | null) => void
    onSubmit: (event: React.FormEvent<HTMLFormElement>) => void
    onReset: () => void
}

export function BrainTumorPredictionForm({
    selectedFile,
    previewUrl,
    isSubmitting,
    onFileSelect,
    onSubmit,
    onReset
}: BrainTumorPredictionFormProps) {
    const fileInputRef = useRef<HTMLInputElement>(null)

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0] || null
        if (file) {
            // Validate file type
            if (!file.type.startsWith("image/")) {
                alert("Please select a valid image file (JPEG or PNG)")
                return
            }
            // Validate file size (10MB)
            if (file.size > 10 * 1024 * 1024) {
                alert("File size must be less than 10MB")
                return
            }
            onFileSelect(file)
        }
    }

    const handleReset = () => {
        onFileSelect(null)
        if (fileInputRef.current) {
            fileInputRef.current.value = ""
        }
        onReset()
    }

    return (
        <Card>
            <CardHeader className="pb-4 sm:pb-6">
                <CardTitle className="text-lg sm:text-xl lg:text-2xl">MRI Image Upload</CardTitle>
                <CardDescription className="text-xs sm:text-sm lg:text-base">
                    Upload a brain MRI image (JPEG or PNG, max 10MB) for tumor classification
                </CardDescription>
            </CardHeader>
            <CardContent className="pt-4 sm:pt-6">
                <form onSubmit={onSubmit} className="space-y-6">
                    <div className="space-y-4">
                        <div className="flex flex-col space-y-2">
                            <Label 
                                htmlFor="mri-image"
                                className="text-sm sm:text-base lg:text-sm"
                            >
                                MRI Image
                            </Label>
                            <Input
                                id="mri-image"
                                ref={fileInputRef}
                                type="file"
                                accept="image/jpeg,image/jpg,image/png"
                                onChange={handleFileChange}
                                disabled={isSubmitting}
                                className="h-10 sm:h-11 lg:h-12 text-sm sm:text-base cursor-pointer"
                            />
                            <p className="text-xs text-muted-foreground">
                                Accepted formats: JPEG, PNG. Maximum file size: 10MB
                            </p>
                        </div>

                        {previewUrl && (
                            <div className="space-y-2">
                                <Label className="text-sm sm:text-base lg:text-sm">
                                    Image Preview
                                </Label>
                                <div className="relative w-full h-64 sm:h-80 lg:h-96 border rounded-lg overflow-hidden bg-muted">
                                    <Image
                                        src={previewUrl}
                                        alt="MRI Preview"
                                        fill
                                        className="object-contain"
                                    />
                                </div>
                            </div>
                        )}
                    </div>

                    <div className="flex flex-col gap-3 sm:flex-row">
                        <Button 
                            type="submit" 
                            disabled={isSubmitting || !selectedFile}
                            className="w-full sm:w-auto sm:flex-1 lg:text-base lg:h-11"
                        >
                            {isSubmitting ? "Analyzing..." : "Run Prediction"}
                        </Button>
                        <Button
                            type="button"
                            variant="outline"
                            onClick={handleReset}
                            disabled={isSubmitting}
                            className="w-full sm:w-auto lg:text-base lg:h-11"
                        >
                            Reset
                        </Button>
                    </div>
                </form>
            </CardContent>
        </Card>
    )
}

