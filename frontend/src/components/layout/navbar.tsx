"use client"
import { Menu, X } from "lucide-react"
import Image from "next/image"
import Link from "next/link"
import React from "react"
import { ModeToggle } from "./mode-toggle"
import { Button } from "../ui/button"
import {
    NavigationMenu,
    NavigationMenuItem,
    NavigationMenuLink,
    NavigationMenuList,
    NavigationMenuTrigger,
    NavigationMenuContent
} from "../ui/navigation-menu"
import { Separator } from "../ui/separator"
import {
    Sheet,
    SheetContent,
    SheetHeader,
    SheetTitle,
    SheetTrigger
} from "../ui/sheet"
import { site } from "@/config/site"

interface RouteProps {
    href: string
    label: string
}

interface ModelMenuProps {
    href: string
    label: string
    logo: string
    color: string
    hoverColor: string
    bgColor: string
}

const routeList: RouteProps[] = [
    {
        href: "/",
        label: "Home"
    },
    {
        href: "#about",
        label: "About Us"
    },
    {
        href: "#contact",
        label: "Contact"
    }
]

const modelMenuList: ModelMenuProps[] = [
    {
        href: "/ovarian",
        label: "Ovarian",
        logo: "/model-ova.webp",
        color: "text-primary",
        hoverColor: "hover:text-primary",
        bgColor: "hover:bg-primary/10"
    },
    {
        href: "/hepatitis",
        label: "Hepatitis",
        logo: "/model-hepa.webp",
        color: "text-green-600 dark:text-green-400",
        hoverColor: "hover:text-green-600 dark:hover:text-green-400",
        bgColor: "hover:bg-green-600/10 dark:hover:bg-green-400/10"
    },
    {
        href: "/pcos",
        label: "PCOS",
        logo: "/model-pcos.webp",
        color: "text-amber-600 dark:text-amber-400",
        hoverColor: "hover:text-amber-600 dark:hover:text-amber-400",
        bgColor: "hover:bg-amber-600/10 dark:hover:bg-amber-400/10"
    }
]

export const Navbar = () => {
    const [isOpen, setIsOpen] = React.useState(false)

    return (
        <div className="sticky top-2 z-50 mx-auto w-[98%] max-w-7xl px-4">
            <nav className="rounded-xl border border-border bg-card/50 shadow-black/2 shadow-sm backdrop-blur-sm">
                <div className="flex items-center justify-between px-4 py-3 lg:px-6">
                    {/* Logo */}
                    <Link
                        href="/"
                        className="group flex items-center gap-2 font-bold"
                    >
                        <div className="relative">
                            <Image
                                src={site.logo}
                                alt={site.name}
                                width={30}
                                height={30}
                            />
                        </div>
                        <h3 className="font-bold text-sm text-black dark:text-white lg:text-base">
                            {site.name}
                        </h3>
                    </Link>

                    {/* Desktop Navigation */}
                    <div className="hidden items-center space-x-1 lg:flex">
                        <NavigationMenu>
                            <NavigationMenuList className="space-x-2">
                                <NavigationMenuItem key={routeList[0].href}>
                                    <NavigationMenuLink asChild>
                                        <Link
                                            href={routeList[0].href}
                                            className="rounded-lg px-4 py-2 font-medium text-sm text-black dark:text-white transition-colors hover:bg-accent/50 hover:text-primary"
                                        >
                                            {routeList[0].label}
                                        </Link>
                                    </NavigationMenuLink>
                                </NavigationMenuItem>
                                <NavigationMenuItem>
                                    <NavigationMenuTrigger className="rounded-lg px-4 py-2 font-medium text-sm text-black dark:text-white transition-colors hover:bg-accent/50 hover:text-primary">
                                        Models
                                    </NavigationMenuTrigger>
                                    <NavigationMenuContent>
                                        <div className="w-[200px] p-2">
                                            {modelMenuList.map(({ href, label, logo, color, hoverColor, bgColor }) => (
                                                <NavigationMenuLink key={href} asChild>
                                                    <Link
                                                        href={href}
                                                        className={`flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors ${bgColor}`}
                                                    >
                                                        <Image
                                                            src={logo}
                                                            alt={label}
                                                            width={24}
                                                            height={24}
                                                            className="object-contain"
                                                        />
                                                        <span className={`${color} ${hoverColor}`}>{label}</span>
                                                    </Link>
                                                </NavigationMenuLink>
                                            ))}
                                        </div>
                                    </NavigationMenuContent>
                                </NavigationMenuItem>
                                {routeList.slice(1).map(({ href, label }) => (
                                    <NavigationMenuItem key={href}>
                                        <NavigationMenuLink asChild>
                                            <Link
                                                href={href}
                                                className="rounded-lg px-4 py-2 font-medium text-sm text-black dark:text-white transition-colors hover:bg-accent/50 hover:text-primary"
                                            >
                                                {label}
                                            </Link>
                                        </NavigationMenuLink>
                                    </NavigationMenuItem>
                                ))}
                            </NavigationMenuList>
                        </NavigationMenu>
                    </div>

                    {/* Desktop Actions */}
                    <div className="hidden items-center gap-2 lg:flex">
                        <ModeToggle />
                    </div>

                    {/* Mobile Menu Button */}
                    <div className="flex items-center gap-2 lg:hidden">
                        <ModeToggle />
                        <Sheet open={isOpen} onOpenChange={setIsOpen}>
                            <SheetTrigger asChild>
                                <Button
                                    variant="outline"
                                    size="sm"
                                    className="rounded-lg hover:bg-accent/50"
                                    aria-label="Toggle menu"
                                >
                                    {isOpen ? (
                                        <X className="size-4" />
                                    ) : (
                                        <Menu className="size-4" />
                                    )}
                                </Button>
                            </SheetTrigger>

                            <SheetContent
                                side="right"
                                className="w-full max-w-sm border-border/50 border-l bg-background/95 backdrop-blur-md"
                            >
                                <div className="flex h-full flex-col">
                                    <SheetHeader className="pb-4">
                                        <SheetTitle>
                                            <Link
                                                href="/"
                                                className="flex items-center gap-2"
                                                onClick={() => setIsOpen(false)}
                                            >
                                                <Image
                                                    src={site.logo}
                                                    alt={site.name}
                                                    width={32}
                                                    height={32}
                                                />
                                                <span className="font-bold text-lg text-black dark:text-white">
                                                    {site.name}
                                                </span>
                                            </Link>
                                        </SheetTitle>
                                    </SheetHeader>

                                    <Separator className="mb-4" />

                                    {/* Mobile Navigation Links */}
                                    <div className="flex flex-1 flex-col">
                                        <div className="space-y-1">
                                            <Button
                                                key={routeList[0].href}
                                                onClick={() =>
                                                    setIsOpen(false)
                                                }
                                                asChild
                                                variant="ghost"
                                                className="h-auto w-full justify-start px-3 py-2.5 font-medium text-black dark:text-white hover:bg-accent/50"
                                            >
                                                <Link href={routeList[0].href}>
                                                    {routeList[0].label}
                                                </Link>
                                            </Button>
                                            <div className="px-3 py-2 text-sm font-semibold text-black dark:text-white">
                                                Models
                                            </div>
                                            {modelMenuList.map(
                                                ({ href, label, logo, color, hoverColor, bgColor }) => (
                                                    <Button
                                                        key={href}
                                                        onClick={() =>
                                                            setIsOpen(false)
                                                        }
                                                        asChild
                                                        variant="ghost"
                                                        className={`h-auto w-full justify-start gap-3 px-3 py-2.5 font-medium ${bgColor}`}
                                                    >
                                                        <Link href={href}>
                                                            <Image
                                                                src={logo}
                                                                alt={label}
                                                                width={24}
                                                                height={24}
                                                                className="object-contain"
                                                            />
                                                            <span className={`${color} ${hoverColor}`}>{label}</span>
                                                        </Link>
                                                    </Button>
                                                )
                                            )}
                                            {routeList.slice(1).map(
                                                ({ href, label }) => (
                                                    <Button
                                                        key={href}
                                                        onClick={() =>
                                                            setIsOpen(false)
                                                        }
                                                        asChild
                                                        variant="ghost"
                                                        className="h-auto w-full justify-start px-3 py-2.5 font-medium text-black dark:text-white hover:bg-accent/50"
                                                    >
                                                        <Link href={href}>
                                                            {label}
                                                        </Link>
                                                    </Button>
                                                )
                                            )}
                                        </div>
                                    </div>
                                </div>
                            </SheetContent>
                        </Sheet>
                    </div>
                </div>
            </nav>
        </div>
    )
}
