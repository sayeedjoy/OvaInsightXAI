import {
    Accordion,
    AccordionContent,
    AccordionItem,
    AccordionTrigger
} from "@/components/ui/accordion"

interface FAQProps {
    question: string
    answer: string
    value: string
}

const FAQList: FAQProps[] = [
    {
        question: "What medical conditions can OvaInsightXAI predict?",
        answer: "OvaInsightXAI can predict three medical conditions: Ovarian Cancer, Hepatitis B, and Polycystic Ovary Syndrome (PCOS). Each model is trained on clinical data to provide accurate predictions based on your input parameters.",
        value: "item-1"
    },
    {
        question: "How accurate are the predictions?",
        answer: "Our AI models are trained on validated clinical datasets and undergo rigorous testing. However, predictions should be used as a screening tool and not replace professional medical diagnosis. Always consult with healthcare professionals for definitive diagnosis and treatment.",
        value: "item-2"
    },
    {
        question: "Is my medical data secure and private?",
        answer: "Yes, we take data privacy seriously. Your input data is processed securely and is not stored permanently. We use industry-standard security measures to protect your information. However, we recommend not entering sensitive personal identifiers.",
        value: "item-3"
    },
    {
        question: "How do I use the prediction models?",
        answer: "Simply navigate to the specific model page (Ovarian, Hepatitis B, or PCOS), fill in the required clinical parameters, and submit. The AI will analyze your inputs and provide a prediction along with explainable AI visualizations to help you understand the factors influencing the result.",
        value: "item-4"
    },
    {
        question: "Can I use this tool for medical diagnosis?",
        answer: "No. OvaInsightXAI is designed as a research and screening tool to assist healthcare professionals and researchers. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical decisions.",
        value: "item-5"
    }
]

export const FAQSection = () => {
    return (
        <section id="faq" className="container mx-auto px-4 py-12 sm:py-16 lg:py-20 md:w-[700px]">
            <div className="mb-8 text-center">
                <h2 className="mb-2 text-center text-lg text-primary tracking-wider">
                    FAQs
                </h2>

                <h2 className="text-center font-bold text-3xl md:text-4xl">
                    Common Questions
                </h2>
            </div>

            <Accordion type="single" collapsible className="AccordionRoot">
                {FAQList.map(({ question, answer, value }) => (
                    <AccordionItem key={value} value={value}>
                        <AccordionTrigger className="text-left">
                            {question}
                        </AccordionTrigger>

                        <AccordionContent>{answer}</AccordionContent>
                    </AccordionItem>
                ))}
            </Accordion>
        </section>
    )
}
