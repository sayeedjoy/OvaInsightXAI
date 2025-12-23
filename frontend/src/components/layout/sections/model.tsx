import { Brain, Activity, Heart } from 'lucide-react'

import Features from '@/components/shadcn-studio/blocks/features-section-01/features-section-01'

const featuresList = [
  {
    icon: Brain,
    title: 'Ovarian Cancer Prediction',
    description:
      'Advanced AI model trained to predict ovarian cancer risk using comprehensive medical data. Get accurate predictions to aid in early detection and treatment planning.',
    cardBorderColor: 'border-primary/40 hover:border-primary',
    avatarTextColor: 'text-primary',
    avatarBgColor: 'bg-primary/10',
    href: '/ovarian'
  },
  {
    icon: Activity,
    title: 'Hepatitis B Detection',
    description:
      'State-of-the-art machine learning model for Hepatitis B diagnosis. Analyze patient symptoms and lab results to provide reliable diagnostic insights.',
    cardBorderColor: 'border-green-600/40 hover:border-green-600 dark:border-green-400/40 dark:hover:border-green-400',
    avatarTextColor: 'text-green-600 dark:text-green-400',
    avatarBgColor: 'bg-green-600/10 dark:bg-green-400/10',
    href: '/hepatitis'
  },
  {
    icon: Heart,
    title: 'PCOS Prediction',
    description:
      'Comprehensive AI-powered model for Polycystic Ovary Syndrome prediction. Evaluate multiple health parameters to assist in accurate diagnosis and management.',
    cardBorderColor: 'border-amber-600/40 hover:border-amber-600 dark:border-amber-400/40 dark:hover:border-amber-400',
    avatarTextColor: 'text-amber-600 dark:text-amber-400',
    avatarBgColor: 'bg-amber-600/10 dark:bg-amber-400/10',
    href: '/pcos'
  }
]

export const ModelSection = () => {
  return (
    <Features 
      featuresList={featuresList}
      title="AI-Powered Medical Predictions"
      description="Choose from our advanced machine learning models to get accurate predictions for various medical conditions. Click on any model below to start your prediction."
      showButton={false}
    />
  )
}

