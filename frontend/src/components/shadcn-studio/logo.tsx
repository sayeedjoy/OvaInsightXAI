import { cn } from '@/lib/utils'
import { site } from '@/config/site'

const Logo = ({ className }: { className?: string }) => {
  return (
    <div className={cn('flex items-center gap-2.5', className)}>
      <span className='text-sm font-semibold lg:text-base'>{site.name}</span>
    </div>
  )
}

export default Logo
