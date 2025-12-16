import { Mail } from "lucide-react"
import XIcon from "@/components/icons/x-icon"
import GithubIcon from "@/components/icons/github-icon"
import LinkedInIcon from "@/components/icons/linkedin-icon"

import Image from "next/image"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { site } from "@/config/site"

interface FooterLinkProps {
    href: string
    label: string
    icon?: React.ReactNode
    external?: boolean
}

const socialLinks: FooterLinkProps[] = [
    {
        href: site.links.github,
        label: "GitHub",
        icon: <GithubIcon className="size-5 fill-foreground" />,
        external: true
    },
    {
        href: site.links.twitter,
        label: "Twitter",
        icon: <XIcon className="size-5 fill-foreground" />,
        external: true
    },
    {
        href: "https://linkedin.com",
        label: "LinkedIn",
        icon: <LinkedInIcon className="size-5 fill-foreground" />,
        external: true
    },
    {
        href: `mailto:${site.mailSupport}`,
        label: "Email",
        icon: <Mail className="size-5" />
    }
]

export const FooterSection = () => {
    return (
        <footer id="footer">
            <div className="mx-auto max-w-7xl border-t border-border/50 pt-16 pb-0 lg:pb-12">
                <div className="p-8 lg:p-12">
                    {/* Main Footer Content */}
                    <div className="flex flex-col items-center text-center lg:items-start lg:text-left">
                        {/* Brand Section */}
                        <div className="mb-8 max-w-2xl lg:mb-12">
                            <Link
                                href="/"
                                className="group mb-4 inline-flex items-center gap-2 font-bold"
                            >
                                <Image
                                    src={site.logo}
                                    alt={site.name}
                                    width={30}
                                    height={30}
                                />
                                <h3 className="font-bold text-2xl">
                                    {site.name}
                                </h3>
                            </Link>
                            <p className="mb-6 text-muted-foreground leading-relaxed text-sm sm:text-base lg:text-lg">
                                Advanced ovarian cancer prediction using machine learning 
                                and biomarker analysis. Our AI-powered system helps healthcare 
                                professionals assess risk through comprehensive biomarker evaluation 
                                for early detection and improved patient outcomes.
                            </p>

                            {/* Social Links */}
                            <div className="flex justify-center gap-2 lg:justify-start">
                                {socialLinks.map((social) => (
                                    <Button
                                        key={social.label}
                                        asChild
                                        variant="ghost"
                                        size="sm"
                                        className="p-2 hover:bg-accent/50"
                                    >
                                        <Link
                                            href={social.href}
                                            target={
                                                social.external
                                                    ? "_blank"
                                                    : undefined
                                            }
                                            rel={
                                                social.external
                                                    ? "noopener noreferrer"
                                                    : undefined
                                            }
                                            aria-label={social.label}
                                        >
                                            {social.icon}
                                        </Link>
                                    </Button>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Bottom Section */}
                    <div className="mt-12 flex flex-col items-center justify-between gap-4 text-muted-foreground text-sm lg:flex-row">
                        <p>
                            &copy; 2025 {site.name}. All rights reserved.
                        </p>
                        <p className="text-xs">
                            This tool is for research and educational purposes. 
                            Consult a healthcare professional for medical advice.
                        </p>
                    </div>
                </div>
            </div>
        </footer>
    )
}
