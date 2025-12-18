"use client"
import { ArrowRight, Activity, CheckCircle2, AlertCircle } from "lucide-react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { motion } from "framer-motion"

// Biomarker data - Clinical Dataset style as per the realistic AI design
const biomarkers = [
    { name: "CA125", value: "85.2", unit: "U/mL", status: "high" },
    { name: "HE4", value: "72.4", unit: "pmol/L", status: "normal" },
    { name: "ALB", value: "4.2", unit: "g/dL", status: "normal" },
    { name: "ALP", value: "58.0", unit: "IU/L", status: "normal" },
    { name: "BUN", value: "12.8", unit: "mg/dL", status: "normal" },
    { name: "GGT", value: "45.1", unit: "U/L", status: "low" },
    { name: "Na+", value: "135", unit: "mEq/L", status: "normal" },
    { name: "Neut%", value: "73.5", unit: "%", status: "high" },
]

export const HeroSection = () => {
    return (
        <section className="relative w-full overflow-hidden bg-[#fcfcfc] py-16 md:py-24 lg:py-32 dark:bg-[#121212]">
            {/* Minimalist Background - Technical Grid */}
            <div className="absolute inset-0 z-0">
                <div className="absolute inset-0 bg-[linear-gradient(to_right,#e2e8f0_1px,transparent_1px),linear-gradient(to_bottom,#e2e8f0_1px,transparent_1px)] bg-[size:40px_40px] opacity-30 dark:bg-[linear-gradient(to_right,#2a2a2a_1px,transparent_1px),linear-gradient(to_bottom,#2a2a2a_1px,transparent_1px)] dark:opacity-40" />
                <div className="absolute inset-x-0 bottom-0 h-40 bg-gradient-to-t from-[#fcfcfc] to-transparent dark:from-[#121212]" />
            </div>

            <div className="container relative z-10 mx-auto px-4 md:px-6">
                <div className="grid grid-cols-1 items-center gap-12 lg:grid-cols-2 lg:gap-16">
                    {/* Left Column: Typography & Messaging */}
                    <div className="flex flex-col items-center text-center space-y-6 lg:items-start lg:text-left lg:space-y-8 max-w-4xl mx-auto lg:mx-0">
                        {/* Status Badge */}
                        <div className="inline-flex items-center rounded-full border border-teal-200 bg-teal-50 px-3 py-1 text-sm font-medium text-teal-900 dark:border-teal-800 dark:bg-teal-950 dark:text-teal-100">
                            <span className="mr-2 flex h-2 w-2">
                                <span className="absolute inline-flex h-2 w-2 animate-ping rounded-full bg-teal-600 opacity-75" />
                                <span className="relative inline-flex h-2 w-2 rounded-full bg-teal-600" />
                            </span>
                            Clinical AI Engine v1.0
                        </div>

                        {/* Heading */}
                        <h1 className="text-4xl font-extrabold tracking-tight text-slate-900 dark:text-white sm:text-6xl lg:text-7xl">
                            <span className="text-slate-400 dark:text-slate-500">Advanced</span><br />
                            Ovarian Cancer<br />
                            <span className="text-teal-600 dark:text-teal-400">Prediction</span>
                        </h1>

                        <p className="max-w-[42rem] text-lg leading-relaxed text-slate-600 dark:text-slate-400 sm:text-xl">
                            Our proprietary machine learning model analyzes 12 clinical biomarkers in real-time,
                            delivering a diagnostic confidence score of 94.7% for early-stage detection.
                        </p>

                        {/* Action Buttons */}
                        <div className="flex w-full flex-col items-center gap-3 sm:flex-row justify-center lg:justify-start">
                            <Button
                                asChild
                                size="lg"
                                className="h-12 bg-slate-900 text-white shadow-lg shadow-slate-200 lg:shadow-none hover:bg-slate-800 dark:bg-white dark:text-slate-900 dark:hover:bg-slate-200 px-8 text-base font-semibold"
                            >
                                <Link href="/predict">
                                    Start Risk Assessment
                                    <ArrowRight className="ml-2 size-4" />
                                </Link>
                            </Button>
                            <Button
                                asChild
                                variant="outline"
                                size="lg"
                                className="h-12 border-slate-200 bg-transparent text-slate-900 hover:bg-slate-50 dark:border-slate-800 dark:text-slate-300 dark:hover:bg-slate-800 px-8 text-base font-semibold"
                            >
                                <Link href="#features">
                                    View Validation Data
                                </Link>
                            </Button>
                        </div>

                        {/* Validation Proof Points */}
                        <div className="flex flex-wrap items-center justify-center lg:justify-start gap-x-6 gap-y-2 text-sm font-medium text-slate-500 dark:text-slate-400">
                            <div className="flex items-center gap-1.5">
                                <CheckCircle2 className="size-4 text-teal-600" />
                                <span>HIPAA Compliant</span>
                            </div>
                            <div className="flex items-center gap-1.5">
                                <CheckCircle2 className="size-4 text-teal-600" />
                                <span>IRB Approved Dataset</span>
                            </div>
                            <div className="flex items-center gap-1.5">
                                <CheckCircle2 className="size-4 text-teal-600" />
                                <span>Clinical Accuracy 94.7%</span>
                            </div>
                        </div>
                    </div>

                    {/* Right Column: Dashboard Widget UI */}
                    <div className="relative w-full mx-auto lg:mx-0">
                        <Card className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl shadow-slate-200/50 lg:shadow-none dark:border-slate-800 dark:bg-slate-900 transition-shadow">
                            {/* Widget Header */}
                            <div className="flex items-center justify-between border-b border-slate-100 bg-slate-50/50 px-6 py-4 dark:border-slate-800 dark:bg-slate-900/50">
                                <div className="flex items-center gap-3 text-left">
                                    <div className="flex h-10 w-10 items-center justify-center rounded-lg border border-slate-200 bg-white shadow-sm dark:border-slate-700 dark:bg-slate-800">
                                        <Activity className="size-5 text-teal-600" />
                                    </div>
                                    <div>
                                        <h3 className="font-bold text-slate-900 dark:text-slate-100">Live Prediction Dashboard</h3>
                                        <p className="text-xs text-slate-500 font-mono">ENCRYPTED_ID: #OVA-2024-X92</p>
                                    </div>
                                </div>
                                <div className="flex items-center gap-3">
                                    <div className="hidden sm:flex items-center gap-2 px-3 py-1 rounded-full bg-teal-50 border border-teal-100 dark:bg-teal-950/30 dark:border-teal-900">
                                        <span className="flex h-2 w-2 rounded-full bg-teal-500 animate-pulse" />
                                        <span className="text-[10px] font-bold text-teal-700 dark:text-teal-400 uppercase tracking-wider">AI Analysis Active</span>
                                    </div>
                                </div>
                            </div>

                            {/* Widget Content - Data Grid */}
                            <div className="p-0">
                                <div className="grid grid-cols-2 divide-x divide-slate-100 dark:divide-slate-800 border-b border-slate-100 dark:border-slate-800 bg-white dark:bg-slate-900">
                                    {biomarkers.map((item, i) => (
                                        <div key={item.name} className="flex flex-col p-5 group hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors">
                                            <div className="mb-2 flex items-center justify-between">
                                                <span className="text-xs font-bold text-slate-400 dark:text-slate-500 uppercase tracking-tighter">{item.name}</span>
                                                {item.status === 'high' && <AlertCircle className="size-3.5 text-red-500" />}
                                                {item.status === 'low' && <AlertCircle className="size-3.5 text-amber-500" />}
                                            </div>
                                            <div className="flex items-baseline gap-1">
                                                <span className="text-2xl font-black text-slate-900 dark:text-slate-100 tabular-nums">{item.value}</span>
                                                <span className="text-[10px] font-bold text-slate-400 uppercase">{item.unit}</span>
                                            </div>
                                        </div>
                                    ))}
                                </div>

                                {/* Prediction Result Area */}
                                <div className="bg-slate-50 p-8 dark:bg-slate-900/80">
                                    <div className="space-y-4">
                                        <div className="flex items-center justify-between text-sm">
                                            <span className="font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wider">Malignancy Probability</span>
                                            <span className="font-mono text-slate-400 text-xs text-right">0.85s</span>
                                        </div>
                                        <div className="h-3 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
                                            <motion.div
                                                initial={{ width: "0%" }}
                                                animate={{ width: "12%" }}
                                                transition={{ duration: 1.5, ease: "easeOut", delay: 0.5 }}
                                                className="h-full bg-teal-600"
                                            />
                                        </div>
                                        <div className="flex items-center justify-between">
                                            <span className="text-sm font-bold text-slate-500">Classification: <span className="text-emerald-600 dark:text-emerald-400 uppercase">Benign</span></span>
                                            <span className="text-3xl font-black text-teal-700 dark:text-teal-400">12%</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </Card>

                        {/* Depth elements hidden on desktop if they contribute to "shadowy" look */}
                        <div className="absolute -left-16 top-1/2 -z-10 h-80 w-80 -translate-y-1/2 rounded-full bg-teal-100/30 blur-[100px] dark:bg-teal-900/10 lg:hidden" />
                        <div className="absolute -right-16 bottom-0 -z-10 h-80 w-80 rounded-full bg-blue-100/30 blur-[100px] dark:bg-blue-900/10 lg:hidden" />
                    </div>
                </div>
            </div>
        </section>
    )
}

