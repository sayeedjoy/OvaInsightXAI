import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ArrowUpRight, CirclePlay, Github } from "lucide-react";
import Link from "next/link";
import Image from "next/image";

export default function Hero() {
  return (
    <div className="min-h-screen lg:min-h-0 flex items-start justify-center pt-8 lg:pt-20 pb-0 relative w-full">
      {/* Dashed Top Left Fade Grid - Light Mode */}
      <div
        className="absolute inset-0 z-0 dark:hidden"
        style={{
          backgroundImage: `
            linear-gradient(to right, #e7e5e4 1px, transparent 1px),
            linear-gradient(to bottom, #e7e5e4 1px, transparent 1px)
          `,
          backgroundSize: "20px 20px",
          backgroundPosition: "0 0, 0 0",
          maskImage: `
            repeating-linear-gradient(
              to right,
              black 0px,
              black 3px,
              transparent 3px,
              transparent 8px
            ),
            repeating-linear-gradient(
              to bottom,
              black 0px,
              black 3px,
              transparent 3px,
              transparent 8px
            ),
            radial-gradient(ellipse 80% 80% at 0% 0%, #000 50%, transparent 90%)
          `,
          WebkitMaskImage: `
            repeating-linear-gradient(
              to right,
              black 0px,
              black 3px,
              transparent 3px,
              transparent 8px
            ),
            repeating-linear-gradient(
              to bottom,
              black 0px,
              black 3px,
              transparent 3px,
              transparent 8px
            ),
            radial-gradient(ellipse 80% 80% at 0% 0%, #000 50%, transparent 90%)
          `,
          maskComposite: "intersect",
          WebkitMaskComposite: "source-in",
        }}
      />
      {/* Dashed Top Left Fade Grid - Dark Mode */}
      <div
        className="absolute inset-0 z-0 hidden dark:block"
        style={{
          backgroundImage: `
            linear-gradient(to right, rgb(64, 64, 64) 1px, transparent 1px),
            linear-gradient(to bottom, rgb(64, 64, 64) 1px, transparent 1px)
          `,
          backgroundSize: "20px 20px",
          backgroundPosition: "0 0, 0 0",
          maskImage: `
            repeating-linear-gradient(
              to right,
              black 0px,
              black 3px,
              transparent 3px,
              transparent 8px
            ),
            repeating-linear-gradient(
              to bottom,
              black 0px,
              black 3px,
              transparent 3px,
              transparent 8px
            ),
            radial-gradient(ellipse 80% 80% at 0% 0%, #000 50%, transparent 90%)
          `,
          WebkitMaskImage: `
            repeating-linear-gradient(
              to right,
              black 0px,
              black 3px,
              transparent 3px,
              transparent 8px
            ),
            repeating-linear-gradient(
              to bottom,
              black 0px,
              black 3px,
              transparent 3px,
              transparent 8px
            ),
            radial-gradient(ellipse 80% 80% at 0% 0%, #000 50%, transparent 90%)
          `,
          maskComposite: "intersect",
          WebkitMaskComposite: "source-in",
        }}
      />
      <div className="max-w-(--breakpoint-xl) w-full mx-auto grid grid-cols-1 lg:grid-cols-2 items-start gap-8 lg:gap-12 px-4 sm:px-6 pt-6 lg:pt-0 pb-0 relative z-10">
        <div>
          <Badge
            variant="secondary"
            className="rounded-full py-1 border-border"
            asChild
          >
            <Link href="#">
              v1.0.0 Released <ArrowUpRight className="ml-1 size-4" />
            </Link>
          </Badge>
          <h1 className="mt-6 text-4xl md:text-5xl lg:text-[2.75rem] xl:text-[3.25rem] font-semibold leading-[1.2]! tracking-[-0.035em]">
            Advanced Medical AI Diagnostics for Early Detection!
          </h1>
          <p className="mt-6 max-w-[60ch] sm:text-lg text-foreground/80">
            Leverage cutting-edge machine learning models to predict medical conditions with high accuracy. From ovarian cancer to hepatitis B and PCOS detection
          </p>
          <div className="mt-12 flex flex-row items-center gap-4">
            <Button size="lg" className="rounded-full text-base">
              Start Prediction <ArrowUpRight className="h-5! w-5!" />
            </Button>
            <Button
              variant="outline"
              size="lg"
              className="rounded-full text-base shadow-none"
            >
              <Github className="h-5 w-5" />
              Source Code
            </Button>
          </div>
        </div>
        <div className="w-full h-[400px] md:h-[500px] lg:h-[600px] rounded-xl overflow-hidden relative -mt-4 lg:-mt-8">
          <Image
            src="/hero.webp"
            alt="Hero image"
            fill
            className="object-contain"
            priority
          />
        </div>
      </div>
    </div>
  );
}
