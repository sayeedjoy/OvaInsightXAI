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
    NavigationMenuList
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

const routeList: RouteProps[] = [
    {
        href: "/",
        label: "Home"
    },
    {
        href: "/predict",
        label: "Prediction Test"
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
                        <h3 className="font-bold text-xl lg:text-2xl">
                            {site.name}
                        </h3>
                    </Link>

                    {/* Desktop Navigation */}
                    <div className="hidden items-center space-x-1 lg:flex">
                        <NavigationMenu>
                            <NavigationMenuList className="space-x-2">
                                {routeList.map(({ href, label }) => (
                                    <NavigationMenuItem key={href}>
                                        <NavigationMenuLink asChild>
                                            <Link
                                                href={href}
                                                className="rounded-lg px-4 py-2 font-medium text-sm transition-colors hover:bg-accent/50 hover:text-primary"
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
                                                <span className="font-bold text-lg">
                                                    {site.name}
                                                </span>
                                            </Link>
                                        </SheetTitle>
                                    </SheetHeader>

                                    <Separator className="mb-4" />

                                    {/* Mobile Navigation Links */}
                                    <div className="flex flex-1 flex-col">
                                        <div className="space-y-1">
                                            {routeList.map(
                                                ({ href, label }) => (
                                                    <Button
                                                        key={href}
                                                        onClick={() =>
                                                            setIsOpen(false)
                                                        }
                                                        asChild
                                                        variant="ghost"
                                                        className="h-auto w-full justify-start px-3 py-2.5 font-medium hover:bg-accent/50"
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
