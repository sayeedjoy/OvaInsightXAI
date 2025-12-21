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
        href: site.links.linkedin,
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
        <footer id="footer" className="bg-background">
            <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
                <div className="flex flex-col items-center justify-center gap-6 text-center">
                    {/* Brand and Social Links */}
                    <div className="flex flex-col items-center gap-4">
                        <Link
                            href="/"
                            className="inline-flex items-center gap-2 font-bold transition-opacity hover:opacity-80"
                        >
                            <Image
                                src={site.logo}
                                alt={site.name}
                                width={24}
                                height={24}
                                className="h-6 w-6"
                            />
                            <span className="text-lg font-semibold">
                                {site.name}
                            </span>
                        </Link>
                        
                        {/* Social Links */}
                        <div className="flex items-center gap-1">
                            {socialLinks.map((social) => (
                                <Button
                                    key={social.label}
                                    asChild
                                    variant="ghost"
                                    size="sm"
                                    className="h-9 w-9 p-0 hover:bg-accent"
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

                    {/* Copyright and Disclaimer */}
                    <div className="flex flex-col items-center gap-2 text-sm text-muted-foreground">
                        <p>
                            &copy; {new Date().getFullYear()} {site.name}. All rights reserved.
                        </p>
                    </div>
                </div>
            </div>
        </footer>
    )
}
