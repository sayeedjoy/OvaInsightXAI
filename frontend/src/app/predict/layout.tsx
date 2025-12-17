import { Navbar } from "@/components/layout/navbar"
import { FooterSection } from "@/components/layout/sections/footer"
import type { Metadata } from "next"
import { site } from "@/config/site"

export const metadata: Metadata = {
    title: "Ovarian Cancer Prediction",
    description:
        "Predict ovarian cancer risk using advanced machine learning models. Input 12 biomarkers to get instant predictions with confidence scores.",
    keywords: [
        "ovarian cancer",
        "cancer prediction",
        "machine learning",
        "biomarkers",
        "health prediction",
        "medical AI"
    ],
    openGraph: {
        type: "website",
        url: `${site.url}/predict`,
        title: `Ovarian Cancer Prediction | ${site.name}`,
        description:
            "Predict ovarian cancer risk using advanced machine learning models. Input 12 biomarkers to get instant predictions with confidence scores.",
        images: [
            {
                url: site.ogImage,
                width: 1200,
                height: 630,
                alt: "Ovarian Cancer Prediction Tool"
            }
        ]
    },
    twitter: {
        card: "summary_large_image",
        title: `Ovarian Cancer Prediction | ${site.name}`,
        description:
            "Predict ovarian cancer risk using advanced machine learning models. Input 12 biomarkers to get instant predictions with confidence scores.",
        images: [site.ogImage]
    }
}

export default function PredictLayout({
    children
}: {
    children: React.ReactNode
}) {
    return (
        <>
            <Navbar />
            {children}
            <FooterSection />
        </>
    )
}

