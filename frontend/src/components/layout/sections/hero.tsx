"use client"
import { ArrowRight, Heart, Shield } from "lucide-react"
import Image from "next/image"
import Link from "next/link"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"

export const HeroSection = () => {
    return (
        <section className="container mx-auto w-full px-4">
            <div className="grid gap-12 py-24 md:grid-cols-2 md:items-center md:gap-14 lg:grid-cols-[0.8fr,1.2fr] lg:gap-20 xl:gap-24 xl:py-32">
                {/* Left side - Copy */}
                <div className="space-y-8 text-center md:space-y-10 md:text-left">
                    <Badge
                        variant="outline"
                        className="rounded-2xl py-2 text-sm"
                    >
                        <span className="mr-2 text-primary">
                            <Badge>
                                <Heart className="mr-1 size-3" />
                                Healthcare AI
                            </Badge>
                        </span>
                
                    </Badge>

                    <div className="font-bold text-4xl md:text-5xl lg:text-6xl">
                        <h1>
                            Advanced
                            <span className="bg-gradient-to-r from-[#7033ff] to-primary bg-clip-text px-2 text-transparent">
                                Ovarian Cancer
                            </span>
                            Prediction
                        </h1>
                    </div>

                    <p className="mx-auto max-w-lg text-muted-foreground text-lg leading-relaxed md:mx-0 lg:text-xl xl:max-w-xl">
                        {`Powered by machine learning, our AI-driven platform analyzes 12 key biomarkers to provide accurate early-stage ovarian cancer risk assessment. Get instant predictions with confidence scores to support clinical decision-making.`}
                    </p>

                    <div className="flex flex-col items-center space-y-4 md:flex-row md:space-x-4 md:space-y-0 md:justify-start">
                        <Button
                            asChild
                            size="lg"
                            className="group/arrow rounded-full"
                        >
                            <Link href="/predict">
                                Get Prediction
                                <ArrowRight className="ml-2 size-5 transition-transform group-hover/arrow:translate-x-1" />
                            </Link>
                        </Button>

                        <Button
                            asChild
                            variant="outline"
                            size="lg"
                            className="rounded-full"
                        >
                            <Link
                                href="#features"
                                className="flex items-center gap-2"
                            >
                                <Shield className="size-5" />
                                Learn More
                            </Link>
                        </Button>
                    </div>
                </div>

                {/* Right side - Image */}
                <div className="group relative lg:scale-105">
                    {/* Enhanced animated glow effect */}
                    <div className="absolute inset-0 -z-10">
                        <div className="-translate-x-1/2 -translate-y-1/2 absolute top-1/2 left-1/2 h-[75%] w-[85%] animate-pulse bg-gradient-to-r from-primary/30 via-purple-500/30 to-primary/30 blur-3xl" />
                    </div>

                    {/* Image Container */}
                    <div className="relative mx-auto w-full overflow-hidden rounded-xl border border-border/50 bg-background shadow-2xl transition-all duration-500 group-hover:shadow-primary/20">
                        <Image
                            width={800}
                            height={800}
                            className="relative w-full h-auto object-cover"
                            src="/riajulsins.jpg"
                            alt="Ovarian cancer prediction healthcare technology"
                            priority
                        />
                    </div>

                    {/* Decorative elements */}
                    <div className="-right-8 -bottom-8 absolute -z-10 size-32 rounded-full bg-primary/30 blur-3xl lg:size-40" />
                    <div className="-top-8 -left-8 absolute -z-10 size-28 rounded-full bg-primary/30 blur-3xl lg:size-36" />
                </div>
            </div>
        </section>
    )
}

