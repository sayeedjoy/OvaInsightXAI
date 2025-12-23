import { Navbar } from "@/components/layout/navbar"
import { FooterSection } from "@/components/layout/sections/footer"
import type { Metadata } from "next"
import { site } from "@/config/site"
import "@/components/pcos-components/index.css"

export const metadata: Metadata = {
    title: "PCOS Prediction",
    description:
        "Predict polycystic ovary syndrome (PCOS) likelihood using machine learning models. Input 20 discretized clinical features to get instant predictions.",
    keywords: [
        "pcos",
        "polycystic ovary syndrome",
        "pcos prediction",
        "machine learning",
        "health prediction",
        "medical AI"
    ],
    openGraph: {
        type: "website",
        url: `${site.url}/pcos`,
        title: `PCOS Prediction | ${site.name}`,
        description:
            "Predict PCOS likelihood using advanced machine learning models. Input 20 discretized features to get predictions with confidence scores.",
        images: [
            {
                url: site.ogImage,
                width: 1200,
                height: 630,
                alt: "PCOS Prediction Tool"
            }
        ]
    },
    twitter: {
        card: "summary_large_image",
        title: `PCOS Prediction | ${site.name}`,
        description:
            "Predict PCOS likelihood using advanced machine learning models. Input 20 discretized features to get predictions with confidence scores.",
        images: [site.ogImage]
    }
}

export default function PcosLayout({
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

