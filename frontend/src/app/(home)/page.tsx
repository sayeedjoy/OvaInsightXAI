import { BenefitsSection } from "@/components/layout/sections/benefits"
import { CommunitySection } from "@/components/layout/sections/community"
import { ContactSection } from "@/components/layout/sections/contact"
import { FAQSection } from "@/components/layout/sections/faq"
import { FeaturesSection } from "@/components/layout/sections/features"
import { HeroSection } from "@/components/layout/sections/hero"
import { PricingSection } from "@/components/layout/sections/pricing"
import { ServicesSection } from "@/components/layout/sections/services"
import { TeamSection } from "@/components/layout/sections/team"
import { TestimonialSection } from "@/components/layout/sections/testimonial"
import LogoCloud from "@/components/logo-cloud"
import { site } from "@/config/site"

export const metadata = {
    title: "Home",
    description: site.description,
    openGraph: {
        type: "website",
        url: site.url,
        title: site.name,
        description: site.description,
        images: [
            {
                url: site.ogImage,
                width: 1200,
                height: 630,
                alt: site.name
            }
        ]
    },
    twitter: {
        card: "summary_large_image",
        title: site.name,
        description: site.description,
        images: [site.ogImage]
    }
}

export default function Home() {
    return (
        <>
            <HeroSection />
            <FAQSection />
        </>
    )
}
