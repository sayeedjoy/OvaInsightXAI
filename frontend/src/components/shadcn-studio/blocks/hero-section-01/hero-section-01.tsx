import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { ArrowRight } from 'lucide-react'

export default function HeroSection() {
  return (
    <section className="container space-y-6 py-8 md:py-12 lg:py-24">
      <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-4 text-center">
        <h1 className="font-heading text-3xl sm:text-5xl md:text-6xl lg:text-7xl">
          Advanced Medical AI Diagnostics
        </h1>
        <p className="max-w-[85%] leading-normal text-muted-foreground sm:text-lg sm:leading-7">
          Leverage cutting-edge machine learning models to predict medical conditions with high accuracy. 
          From ovarian cancer to hepatitis B and PCOS detection.
        </p>
        <div className="flex gap-4">
          <Button asChild size="lg" className="rounded-full">
            <Link href="/">
              Get Started
              <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
          <Button asChild variant="outline" size="lg" className="rounded-full">
            <Link href="#features">Learn More</Link>
          </Button>
        </div>
      </div>
    </section>
  )
}

