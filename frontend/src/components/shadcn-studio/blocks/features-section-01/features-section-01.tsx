import type { ComponentType } from 'react'
import Link from 'next/link'

import { ArrowRightIcon, ArrowUpRightIcon } from 'lucide-react'

import { Avatar, AvatarFallback } from '@/components/ui/avatar'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'

import { cn } from '@/lib/utils'

type Features = {
  icon: ComponentType
  title: string
  description: string
  cardBorderColor: string
  avatarTextColor: string
  avatarBgColor: string
  href?: string
}[]

const Features = ({ 
  featuresList,
  title = 'Discover the Exclusive Perks Today',
  description = 'Explore key features designed to enhance your shopping experience with intuitive navigation, robust security, and seamless functionality.',
  showButton = true
}: { 
  featuresList: Features
  title?: string
  description?: string
  showButton?: boolean
}) => {
  return (
    <section className='py-8 sm:py-16 lg:py-24'>
      <div className='mx-auto max-w-7xl px-4 sm:px-6 lg:px-8'>
        {/* Header */}
        <div className='mb-12 space-y-4 sm:mb-16 lg:mb-24'>
          <h2 className='text-2xl font-semibold md:text-3xl lg:text-4xl'>{title}</h2>
          <p className='text-muted-foreground text-xl'>
            {description}
          </p>
          {showButton && (
            <Button variant='outline' className='rounded-lg text-base shadow-none has-[>svg]:px-6' size='lg' asChild>
              <a href='#'>
                See all features
                <ArrowRightIcon />
              </a>
            </Button>
          )}
        </div>

        <div className='grid gap-6 sm:grid-cols-2 lg:grid-cols-3'>
          {featuresList.map((features, index) => (
            <Card key={index} className={cn('shadow-none transition-colors duration-300', features.cardBorderColor, 'h-full flex flex-col')}>
              <CardContent className='flex flex-col flex-1'>
                <Avatar className={cn('mb-6 size-10 rounded-md', features.avatarTextColor)}>
                  <AvatarFallback className={cn('rounded-md [&>svg]:size-6', features.avatarBgColor)}>
                    <features.icon />
                  </AvatarFallback>
                </Avatar>
                <h6 className='mb-2 text-lg font-semibold'>{features.title}</h6>
                <p className='text-muted-foreground mb-4 flex-1'>{features.description}</p>
                {features.href && (
                  <Link 
                    href={features.href} 
                    className='inline-flex items-center gap-1 text-sm font-medium underline decoration-dotted underline-offset-4 hover:opacity-80 transition-opacity mt-auto'
                  >
                    Visit page
                    <ArrowUpRightIcon className='size-4' />
                  </Link>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  )
}

export default Features
