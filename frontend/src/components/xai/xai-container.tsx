"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { SHAPVisualization } from "./shap-visualization"
import { LIMEVisualization } from "./lime-visualization"
import { PDPVisualization } from "./pdp-visualization"
import { ICEVisualization } from "./ice-visualization"
import { ALEVisualization } from "./ale-visualization"
import { XAILoading } from "./xai-loading"
import type { XAIResponse } from "@/types/xai"
import { isSHAPExplanation, isLIMEExplanation, isPDP1DResponse, isICE1DResponse, isALE1DResponse } from "@/types/xai"

interface XAIContainerProps {
    xaiData: XAIResponse | null
    isLoading?: boolean
}

export function XAIContainer({ xaiData, isLoading }: XAIContainerProps) {
    const [activeTab, setActiveTab] = useState("shap")

    if (isLoading) {
        return <XAILoading />
    }

    if (!xaiData) {
        return (
            <div className="rounded-lg border border-muted p-4">
                <p className="text-sm text-muted-foreground">No XAI explanations available</p>
            </div>
        )
    }

    // Validate and extract explanations
    const shapExplanation = isSHAPExplanation(xaiData.shap) ? xaiData.shap : null
    const limeExplanation = isLIMEExplanation(xaiData.lime) ? xaiData.lime : null
    const pdpResponse = isPDP1DResponse(xaiData.pdp_1d) ? xaiData.pdp_1d : null
    const iceResponse = isICE1DResponse(xaiData.ice_1d) ? xaiData.ice_1d : null
    const aleResponse = isALE1DResponse(xaiData.ale_1d) ? xaiData.ale_1d : null

    return (
        <div className="space-y-4">
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
                <TabsList className="grid w-full grid-cols-5">
                    <TabsTrigger value="shap">SHAP</TabsTrigger>
                    <TabsTrigger value="lime">LIME</TabsTrigger>
                    <TabsTrigger value="pdp">PDP</TabsTrigger>
                    <TabsTrigger value="ice">ICE</TabsTrigger>
                    <TabsTrigger value="ale">ALE</TabsTrigger>
                </TabsList>
                <TabsContent value="shap" className="mt-4">
                    {shapExplanation ? (
                        <SHAPVisualization explanation={shapExplanation} />
                    ) : (
                        <div className="rounded-lg border border-muted p-4">
                            <p className="text-sm text-muted-foreground">SHAP explanation not available</p>
                        </div>
                    )}
                </TabsContent>
                <TabsContent value="lime" className="mt-4">
                    {limeExplanation ? (
                        <LIMEVisualization explanation={limeExplanation} />
                    ) : (
                        <div className="rounded-lg border border-muted p-4">
                            <p className="text-sm text-muted-foreground">LIME explanation not available</p>
                        </div>
                    )}
                </TabsContent>
                <TabsContent value="pdp" className="mt-4">
                    {pdpResponse ? (
                        <PDPVisualization explanation={pdpResponse} />
                    ) : (
                        <div className="rounded-lg border border-muted p-4">
                            <p className="text-sm text-muted-foreground">PDP plots not available</p>
                        </div>
                    )}
                </TabsContent>
                <TabsContent value="ice" className="mt-4">
                    {iceResponse ? (
                        <ICEVisualization explanation={iceResponse} />
                    ) : (
                        <div className="rounded-lg border border-muted p-4">
                            <p className="text-sm text-muted-foreground">ICE plots not available</p>
                        </div>
                    )}
                </TabsContent>
                <TabsContent value="ale" className="mt-4">
                    {aleResponse ? (
                        <ALEVisualization explanation={aleResponse} />
                    ) : (
                        <div className="rounded-lg border border-muted p-4">
                            <p className="text-sm text-muted-foreground">ALE plots not available</p>
                        </div>
                    )}
                </TabsContent>
            </Tabs>
        </div>
    )
}

