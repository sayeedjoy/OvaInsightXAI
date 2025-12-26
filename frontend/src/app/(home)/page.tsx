"use client"

import { FAQSection } from "@/components/layout/sections/faq"
import Hero from "@/components/hero"
import Features from "@/components/features"
import ModelFeaturesSection from "@/components/model-features"
import Team from "@/components/team"
import { site } from "@/config/site"
import dynamic from "next/dynamic"
import { Suspense } from "react"

// Lazy load below-the-fold components for better initial load performance
const LazyModelFeaturesSection = dynamic(
    () => import("@/components/model-features"),
    {
        loading: () => (
            <div className="bg-background py-12 sm:py-16 lg:py-20">
                <div className="mx-auto max-w-5xl px-6">
                    <div className="mb-12 space-y-4 animate-pulse">
                        <div className="h-10 bg-muted rounded w-3/4"></div>
                        <div className="h-6 bg-muted rounded w-1/2"></div>
                    </div>
                    <div className="grid grid-cols-6 gap-3">
                        <div className="col-span-full lg:col-span-2 h-64 bg-muted rounded-lg"></div>
                        <div className="col-span-full sm:col-span-3 lg:col-span-2 h-64 bg-muted rounded-lg"></div>
                        <div className="col-span-full sm:col-span-3 lg:col-span-2 h-64 bg-muted rounded-lg"></div>
                    </div>
                </div>
            </div>
        ),
        ssr: true
    }
)

const LazyTeam = dynamic(() => import("@/components/team"), {
    loading: () => (
        <div className="flex flex-col items-center justify-center py-14 px-4 animate-pulse">
            <div className="h-12 bg-muted rounded w-64 mb-20"></div>
            <div className="h-20 w-20 bg-muted rounded-full mb-12"></div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-12 max-w-4xl">
                {[...Array(4)].map((_, i) => (
                    <div key={i} className="flex flex-col items-center">
                        <div className="h-20 w-20 bg-muted rounded-full"></div>
                        <div className="h-6 bg-muted rounded w-24 mt-4"></div>
                    </div>
                ))}
            </div>
        </div>
    ),
    ssr: true
})

const LazyFAQSection = dynamic(
    () => import("@/components/layout/sections/faq").then((mod) => ({ default: mod.FAQSection })),
    {
        loading: () => (
            <section className="container mx-auto px-4 py-12 sm:py-16 lg:py-20 md:w-[700px] animate-pulse">
                <div className="mb-8 text-center">
                    <div className="h-6 bg-muted rounded w-16 mx-auto mb-2"></div>
                    <div className="h-10 bg-muted rounded w-64 mx-auto"></div>
                </div>
                <div className="space-y-4">
                    {[...Array(5)].map((_, i) => (
                        <div key={i} className="h-14 bg-muted rounded"></div>
                    ))}
                </div>
            </section>
        ),
        ssr: true
    }
)

export default function Home() {
    return (
        <>
            <Hero />
            <Features />
            <LazyModelFeaturesSection />
            <LazyTeam />
            <LazyFAQSection />
        </>
    )
}
