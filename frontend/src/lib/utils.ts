import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"
import * as React from "react"

export function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs))
}

/**
 * Utility type to ensure components that accept children have it properly typed.
 * This prevents TypeScript errors in strict mode or different TS versions.
 */
export type ComponentPropsWithChildren<T extends React.ElementType> = 
    React.ComponentPropsWithoutRef<T> & {
        children?: React.ReactNode
    }
