import { Button } from "@/components/ui/button";
import { ArrowRight } from "lucide-react";
import Link from "next/link";
import Image from "next/image";

const features = [
  {
    category: "Medical Conditions",
    title: "Ovarian Cancer Prediction",
    details:
      "Advanced AI model trained to predict ovarian cancer risk using comprehensive medical data. Get accurate predictions to aid in early detection and treatment planning.",
    tutorialLink: "/ovarian",
    image: "/model-ova.webp",
  },
  {
    category: "Medical Conditions",
    title: "Hepatitis B Detection",
    details:
      "State-of-the-art machine learning model for Hepatitis B diagnosis. Analyze patient symptoms and lab results to provide reliable diagnostic insights.",
    tutorialLink: "/hepatitis",
    image: "/model-hepa.webp",
  },
  {
    category: "Medical Conditions",
    title: "PCOS Prediction",
    details:
      "Comprehensive AI-powered model for Polycystic Ovary Syndrome prediction. Evaluate multiple health parameters to assist in accurate diagnosis and management.",
    tutorialLink: "/pcos",
    image: "/model-pcos.webp",
  }
];

const Features = () => {
  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="max-w-(--breakpoint-lg) w-full py-10 px-6">
        <h2 className="text-4xl md:text-[2.75rem] md:leading-[1.2] font-semibold tracking-[-0.03em] sm:max-w-xl text-pretty sm:mx-auto sm:text-center">
        Medical Predictions Models
        </h2>
        <p className="mt-2 text-muted-foreground text-lg sm:text-xl sm:text-center">
          Explore our advanced machine learning models designed to predict various medical conditions with high accuracy.
        </p>
        <div className="mt-8 md:mt-16 w-full mx-auto space-y-20">
          {features.map((feature) => (
            <div
              key={feature.title}
              className="flex flex-col md:flex-row items-center gap-x-12 gap-y-6 md:even:flex-row-reverse"
            >
                <div className="relative w-full md:basis-1/2 max-w-[200px] sm:max-w-[250px] md:max-w-[280px] lg:max-w-[320px] mx-auto">
                <div className="relative w-full aspect-[4/3] max-h-[180px] sm:max-h-[220px] md:max-h-[250px] lg:max-h-[280px]">
                  <Image
                    src={feature.image}
                    alt={feature.title}
                    fill
                    loading="lazy"
                    className="object-contain"
                    sizes="(max-width: 640px) 200px, (max-width: 768px) 250px, (max-width: 1024px) 280px, 320px"
                  />
                </div>
              </div>
              <div className="basis-1/2 shrink-0">
                <span className="uppercase font-medium text-sm text-muted-foreground">
                  {feature.category}
                </span>
                <h4 className="my-3 text-3xl font-semibold tracking-[-0.02em]">
                  {feature.title}
                </h4>
                <p className="text-muted-foreground">{feature.details}</p>
                <Button asChild size="lg" className="mt-6 rounded-full gap-3">
                  <Link href={feature.tutorialLink}>
                    Start Prediction <ArrowRight />
                  </Link>
                </Button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Features;
