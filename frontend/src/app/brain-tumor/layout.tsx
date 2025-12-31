import { Navbar } from "@/components/layout/navbar"
import { FooterSection } from "@/components/layout/sections/footer"
import type { Metadata } from "next"
import { site } from "@/config/site"
import "@/components/brain-tumor-components/index.css"

export const metadata: Metadata = {
    title: "Brain Tumor Classification",
    description:
        "Classify brain MRI images to detect glioma, meningioma, pituitary tumor, or no tumor using advanced deep learning models. Upload an MRI image to get instant predictions with confidence scores.",
    keywords: [
        "brain tumor",
        "brain tumor classification",
        "glioma",
        "meningioma",
        "pituitary tumor",
        "MRI classification",
        "medical AI",
        "deep learning",
        "image classification"
    ],
    openGraph: {
        type: "website",
        url: `${site.url}/brain-tumor`,
        title: `Brain Tumor Classification | ${site.name}`,
        description:
            "Classify brain MRI images to detect glioma, meningioma, pituitary tumor, or no tumor using advanced deep learning models. Upload an MRI image to get instant predictions with confidence scores.",
        images: [
            {
                url: site.ogImage,
                width: 1200,
                height: 630,
                alt: "Brain Tumor Classification Tool"
            }
        ]
    },
    twitter: {
        card: "summary_large_image",
        title: `Brain Tumor Classification | ${site.name}`,
        description:
            "Classify brain MRI images to detect glioma, meningioma, pituitary tumor, or no tumor using advanced deep learning models. Upload an MRI image to get instant predictions with confidence scores.",
        images: [site.ogImage]
    }
}

export default function BrainTumorLayout({
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

