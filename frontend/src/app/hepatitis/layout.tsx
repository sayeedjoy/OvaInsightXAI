import { Navbar } from "@/components/layout/navbar"
import { FooterSection } from "@/components/layout/sections/footer"
import type { Metadata } from "next"
import { site } from "@/config/site"
import "@/components/hepatitis-components/index.css"

export const metadata: Metadata = {
    title: "Hepatitis B Prediction",
    description:
        "Predict Hepatitis B likelihood using machine learning models. Input 15 clinical and laboratory features to get instant predictions.",
    keywords: [
        "hepatitis b",
        "hepatitis",
        "hepatitis b prediction",
        "machine learning",
        "health prediction",
        "medical AI",
        "liver disease"
    ],
    openGraph: {
        type: "website",
        url: `${site.url}/hepatitis`,
        title: `Hepatitis B Prediction | ${site.name}`,
        description:
            "Predict Hepatitis B likelihood using advanced machine learning models. Input 15 clinical and laboratory features to get predictions with confidence scores.",
        images: [
            {
                url: site.ogImage,
                width: 1200,
                height: 630,
                alt: "Hepatitis B Prediction Tool"
            }
        ]
    },
    twitter: {
        card: "summary_large_image",
        title: `Hepatitis B Prediction | ${site.name}`,
        description:
            "Predict Hepatitis B likelihood using advanced machine learning models. Input 15 clinical and laboratory features to get predictions with confidence scores.",
        images: [site.ogImage]
    }
}

export default function HepatitisLayout({
    children
}: {
    children: React.ReactNode
}) {
    return (
        <div style={{ fontFamily: "Inter, ui-sans-serif, sans-serif, system-ui" }}>
            <Navbar />
            {children}
            <FooterSection />
        </div>
    )
}

