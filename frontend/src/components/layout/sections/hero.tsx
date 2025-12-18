"use client"
import { ArrowRight, CheckCircle2, FileText, Brain, Shield } from "lucide-react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { motion } from "framer-motion"

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

                    {/* Right Column: Process Flow Visualization */}
                    <div className="relative w-full mx-auto lg:mx-0">
                        <div className="relative flex flex-col items-center justify-center space-y-8 p-8">
                            {/* Process Steps */}
                            <div className="relative w-full space-y-6">
                                {/* Step 1: Input Biomarkers */}
                                <motion.div
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ duration: 0.5, delay: 0.1 }}
                                    className="flex items-center gap-4"
                                >
                                    <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-xl bg-teal-100 dark:bg-teal-900/30 border border-teal-200 dark:border-teal-800">
                                        <FileText className="size-6 text-teal-600 dark:text-teal-400" />
                                    </div>
                                    <div className="flex-1">
                                        <h3 className="text-base font-semibold text-slate-900 dark:text-slate-100 mb-1">
                                            Input Biomarkers
                                        </h3>
                                        <p className="text-sm text-slate-600 dark:text-slate-400">
                                            12 clinical biomarkers analyzed
                                        </p>
                                    </div>
                                </motion.div>

                                {/* Step 2: AI Analysis */}
                                <motion.div
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ duration: 0.5, delay: 0.4 }}
                                    className="flex items-center gap-4"
                                >
                                    <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-xl bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700">
                                        <Brain className="size-6 text-slate-700 dark:text-slate-300" />
                                    </div>
                                    <div className="flex-1">
                                        <h3 className="text-base font-semibold text-slate-900 dark:text-slate-100 mb-1">
                                            AI Analysis
                                        </h3>
                                        <p className="text-sm text-slate-600 dark:text-slate-400">
                                            Machine learning model processes data
                                        </p>
                                    </div>
                                </motion.div>

                                {/* Step 3: Risk Assessment */}
                                <motion.div
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ duration: 0.5, delay: 0.7 }}
                                    className="flex items-center gap-4"
                                >
                                    <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-xl bg-emerald-100 dark:bg-emerald-900/30 border border-emerald-200 dark:border-emerald-800">
                                        <Shield className="size-6 text-emerald-600 dark:text-emerald-400" />
                                    </div>
                                    <div className="flex-1">
                                        <h3 className="text-base font-semibold text-slate-900 dark:text-slate-100 mb-1">
                                            Risk Assessment
                                        </h3>
                                        <p className="text-sm text-slate-600 dark:text-slate-400">
                                            Instant results with 94.7% accuracy
                                        </p>
                                    </div>
                                </motion.div>
                            </div>

                            {/* Key Metrics */}
                            <div className="grid grid-cols-3 gap-4 w-full pt-4 border-t border-slate-200 dark:border-slate-700">
                                <motion.div
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ duration: 0.5, delay: 0.9 }}
                                    className="text-center"
                                >
                                    <div className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-1">12</div>
                                    <div className="text-xs text-slate-600 dark:text-slate-400">Biomarkers</div>
                                </motion.div>
                                <motion.div
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ duration: 0.5, delay: 1 }}
                                    className="text-center"
                                >
                                    <div className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-1">94.7%</div>
                                    <div className="text-xs text-slate-600 dark:text-slate-400">Accuracy</div>
                                </motion.div>
                                <motion.div
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ duration: 0.5, delay: 1.1 }}
                                    className="text-center"
                                >
                                    <div className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-1">&lt;1s</div>
                                    <div className="text-xs text-slate-600 dark:text-slate-400">Response</div>
                                </motion.div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    )
}

