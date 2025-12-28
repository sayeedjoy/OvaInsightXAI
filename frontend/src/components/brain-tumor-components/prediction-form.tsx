"use client"

import { useRef, useEffect, useState } from "react"
import { Upload } from "lucide-react"
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
    const dropZoneRef = useRef<HTMLDivElement>(null)
    const pasteZoneRef = useRef<HTMLFormElement>(null)
    const [isDragging, setIsDragging] = useState(false)

    const validateFile = (file: File): string | null => {
        // Validate file type
        if (!file.type.startsWith("image/")) {
            return "Please select a valid image file (JPEG or PNG)"
        }
        // Validate file size (10MB)
        if (file.size > 10 * 1024 * 1024) {
            return "File size must be less than 10MB"
        }
        return null
    }

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0] || null
        if (file) {
            const error = validateFile(file)
            if (error) {
                alert(error)
                return
            }
            onFileSelect(file)
        }
    }

    const handlePaste = (event: React.ClipboardEvent) => {
        const items = event.clipboardData.items
        for (let i = 0; i < items.length; i++) {
            const item = items[i]
            if (item.type.startsWith("image/")) {
                event.preventDefault()
                const file = item.getAsFile()
                if (file) {
                    const error = validateFile(file)
                    if (error) {
                        alert(error)
                        return
                    }
                    onFileSelect(file)
                    // Update the file input to reflect the pasted file
                    const dataTransfer = new DataTransfer()
                    dataTransfer.items.add(file)
                    if (fileInputRef.current) {
                        fileInputRef.current.files = dataTransfer.files
                    }
                }
                break
            }
        }
    }

    const handleDragOver = (event: React.DragEvent) => {
        event.preventDefault()
        event.stopPropagation()
        setIsDragging(true)
    }

    const handleDragLeave = (event: React.DragEvent) => {
        event.preventDefault()
        event.stopPropagation()
        setIsDragging(false)
    }

    const handleDrop = (event: React.DragEvent) => {
        event.preventDefault()
        event.stopPropagation()
        setIsDragging(false)

        const file = event.dataTransfer.files?.[0]
        if (file) {
            const error = validateFile(file)
            if (error) {
                alert(error)
                return
            }
            onFileSelect(file)
            // Update the file input to reflect the dropped file
            const dataTransfer = new DataTransfer()
            dataTransfer.items.add(file)
            if (fileInputRef.current) {
                fileInputRef.current.files = dataTransfer.files
            }
        }
    }

    const handleReset = () => {
        onFileSelect(null)
        if (fileInputRef.current) {
            fileInputRef.current.value = ""
        }
        onReset()
    }

    // Handle paste events on the entire form
    useEffect(() => {
        const handleGlobalPaste = (event: ClipboardEvent) => {
            // Only handle paste if the form is visible and not submitting
            if (isSubmitting) return
            
            const items = event.clipboardData?.items
            if (!items) return

            for (let i = 0; i < items.length; i++) {
                const item = items[i]
                if (item.type.startsWith("image/")) {
                    event.preventDefault()
                    const file = item.getAsFile()
                    if (file) {
                        const error = validateFile(file)
                        if (error) {
                            alert(error)
                            return
                        }
                        onFileSelect(file)
                        // Update the file input to reflect the pasted file
                        const dataTransfer = new DataTransfer()
                        dataTransfer.items.add(file)
                        if (fileInputRef.current) {
                            fileInputRef.current.files = dataTransfer.files
                        }
                    }
                    break
                }
            }
        }

        // Add paste listener to document
        document.addEventListener("paste", handleGlobalPaste)
        return () => {
            document.removeEventListener("paste", handleGlobalPaste)
        }
    }, [isSubmitting, onFileSelect])

    return (
        <Card>
            <CardHeader className="pb-4 sm:pb-6">
                <CardTitle className="text-lg sm:text-xl lg:text-2xl">MRI Image Upload</CardTitle>
                <CardDescription className="text-xs sm:text-sm lg:text-base">
                    Upload a brain MRI image (JPEG or PNG, max 10MB) for tumor classification. 
                    You can paste an image from your clipboard (Ctrl+V / Cmd+V) or drag and drop.
                </CardDescription>
            </CardHeader>
            <CardContent className="pt-4 sm:pt-6">
                <form 
                    ref={pasteZoneRef}
                    onSubmit={onSubmit} 
                    onPaste={handlePaste}
                    className="space-y-6"
                >
                    <div className="space-y-4">
                        <div className="flex flex-col space-y-2">
                            <Label 
                                htmlFor="mri-image"
                                className="text-sm sm:text-base lg:text-sm"
                            >
                                MRI Image
                            </Label>
                            <div
                                ref={dropZoneRef}
                                onDragOver={handleDragOver}
                                onDragLeave={handleDragLeave}
                                onDrop={handleDrop}
                                className={`relative border-2 border-dashed rounded-lg p-6 transition-colors cursor-pointer bg-muted/30 ${
                                    isDragging 
                                        ? "border-primary bg-primary/10" 
                                        : "border-muted-foreground/25 hover:border-primary/50"
                                }`}
                                onClick={() => fileInputRef.current?.click()}
                            >
                                <input
                                    id="mri-image"
                                    ref={fileInputRef}
                                    type="file"
                                    accept="image/jpeg,image/jpg,image/png"
                                    onChange={handleFileChange}
                                    disabled={isSubmitting}
                                    className="hidden"
                                />
                                <div className="flex flex-col items-center justify-center space-y-2 text-center">
                                    <Upload className="h-8 w-8 text-muted-foreground" />
                                    <div className="space-y-1">
                                        <p className="text-sm font-medium">
                                            {selectedFile 
                                                ? selectedFile.name 
                                                : "Click to upload or drag and drop"}
                                        </p>
                                        <p className="text-xs text-muted-foreground">
                                            JPEG, PNG (max 10MB) • Or paste from clipboard (Ctrl+V)
                                        </p>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {previewUrl && (
                            <div className="space-y-2">
                                <div className="flex items-center justify-between">
                                    <Label className="text-sm sm:text-base lg:text-sm">
                                        Image Preview
                                    </Label>
                                    {selectedFile && (
                                        <div className="text-xs text-muted-foreground">
                                            {selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
                                        </div>
                                    )}
                                </div>
                                <div className="relative w-full border rounded-lg overflow-hidden bg-muted/50">
                                    <div className="relative w-full min-h-[300px] max-h-[600px] flex items-center justify-center p-4">
                                        {previewUrl ? (
                                            <img
                                                key={previewUrl}
                                                src={previewUrl}
                                                alt="MRI Preview"
                                                className="max-w-full max-h-[600px] object-contain"
                                                onError={(e) => {
                                                    console.error("Failed to load preview image from URL:", previewUrl)
                                                    // Hide the broken image and show error message
                                                    const target = e.currentTarget
                                                    target.style.display = "none"
                                                    const parent = target.parentElement
                                                    if (parent) {
                                                        // Remove existing error message if any
                                                        const existingError = parent.querySelector(".preview-error")
                                                        if (existingError) {
                                                            existingError.remove()
                                                        }
                                                        // Add error message
                                                        const errorDiv = document.createElement("div")
                                                        errorDiv.className = "preview-error text-muted-foreground text-sm text-center"
                                                        errorDiv.textContent = "Failed to load image preview. Please try selecting the image again."
                                                        parent.appendChild(errorDiv)
                                                    }
                                                }}
                                                onLoad={(e) => {
                                                    // Remove any error messages when image loads successfully
                                                    const target = e.currentTarget
                                                    const parent = target.parentElement
                                                    const errorMsg = parent?.querySelector(".preview-error")
                                                    if (errorMsg) {
                                                        errorMsg.remove()
                                                    }
                                                    target.style.display = ""
                                                }}
                                            />
                                        ) : (
                                            <div className="flex items-center justify-center text-muted-foreground">
                                                <p>No preview available</p>
                                            </div>
                                        )}
                                    </div>
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

